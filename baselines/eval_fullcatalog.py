# baselines/eval_fullcatalog.py
"""Full-catalog ranking eval reusing the single-positive metrics. Produces aggregate metrics
(R@k, MRR, nDCG@k) + per-sample hit@k arrays + a per_user_hits.npz writer for significance.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from baselines.metrics import rank_metrics_single, KS, NDCG_KS

def _target_ranks(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """0-based rank of each target = (#items scoring strictly higher) + (#items tied with the
    target but at a smaller catalog index). Ties are broken DETERMINISTICALLY by ascending item
    index — equivalent to a stable descending sort, but computed without sorting (fully
    vectorized; ~17x faster than argsort+loop and avoids materialising an (N, |catalog|) order
    array). NOT the optimistic strictly-greater count (which would hand every tie the best slot
    and over-state the metrics)."""
    n = scores.shape[0]
    tgt = scores[np.arange(n), targets][:, None]            # (n, 1) each target's score
    idx = np.arange(scores.shape[1])[None, :]               # (1, C) column indices
    higher = (scores > tgt).sum(axis=1)
    tied_before = ((scores == tgt) & (idx < targets[:, None])).sum(axis=1)
    return (higher + tied_before).astype(np.int64)

def eval_full_catalog(scores: np.ndarray, targets: np.ndarray, user_ids: np.ndarray):
    """scores: (N, |catalog|); targets: (N,) item indices; user_ids: (N,).
    Returns (agg dict with R@k/MRR/nDCG@k, hits dict {k: bool array (N,)})."""
    ranks = _target_ranks(scores, targets)
    n = scores.shape[0]
    hits = {k: np.zeros(n, dtype=bool) for k in KS}
    mrr_arr = np.zeros(n, dtype=np.float64)
    ndcg_arr = {k: np.zeros(n, dtype=np.float64) for k in NDCG_KS}
    for i in range(n):
        hit, mrr, ndcg = rank_metrics_single(int(ranks[i]))
        for k in KS:
            hits[k][i] = hit[k]
        mrr_arr[i] = mrr
        for k in NDCG_KS:
            ndcg_arr[k][i] = ndcg[k]
    agg = {f"R@{k}": float(hits[k].mean()) for k in KS}
    agg["MRR"] = float(mrr_arr.mean())
    for k in NDCG_KS:
        agg[f"nDCG@{k}"] = float(ndcg_arr[k].mean())
    _assert_self_consistent(agg)
    return agg, hits


def _assert_self_consistent(agg: dict) -> None:
    """Enforced self-consistency checks on the aggregate metrics (fail loudly)."""
    for k in KS:
        v = agg[f"R@{k}"]
        assert 0.0 <= v <= 1.0, f"R@{k}={v} out of [0,1]"
    for key, v in agg.items():
        assert np.isfinite(v), f"{key}={v} is not finite"
    # R@k is monotone non-decreasing in k (a hit@k is also a hit@k' for k'>k).
    sk = sorted(KS)
    for a, b in zip(sk, sk[1:]):
        assert agg[f"R@{a}"] <= agg[f"R@{b}"] + 1e-12, (
            f"R@{a}={agg[f'R@{a}']} > R@{b}={agg[f'R@{b}']} (not monotone in k)")

def write_per_user_hits(out_dir: Path, user_ids: np.ndarray, baseline_hits: dict,
                        vanilla_hits: dict) -> None:
    """Persist baseline + the matching-setting generative vanilla hits (already aligned by user
    order) for run_statistical_significance.py's baseline_vs_vanilla comparison.

    Persists the cutoffs present in BOTH dicts (keys ``baseline_hit{k}`` / ``vanilla_hit{k}``).
    The generative vanilla anchor only persists @1/@10, so the paired comparison covers those."""
    assert len(user_ids) > 0, "write_per_user_hits: empty user_ids (no paired users)"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"user_ids": np.array(user_ids)}
    for k in sorted(set(baseline_hits) & set(vanilla_hits)):
        arrays[f"vanilla_hit{k}"] = vanilla_hits[k]
        arrays[f"baseline_hit{k}"] = baseline_hits[k]
    np.savez(out_dir / "per_user_hits.npz", **arrays)


def align_vanilla(base_uid, base_hits: dict, vanilla_npz):
    """Pair baseline per-sample hits with the same-setting generative vanilla hits.

    Returns (kept_user_ids, baseline_hits, vanilla_hits) over the cutoffs the vanilla npz persists.
    - **LOO settings** (UNIQUE user_ids in the npz, e.g. Amazon): align by user_id — orders may
      differ (SASRec reads user_sequence.txt; the scorer iterates TestDatasetGRAM).
    - **Per-sample settings** (REPEATED user_ids, e.g. MIND mode-B has up to ~98 samples/user):
      user_id is NOT a key, so a user_id dict silently collapses to one arbitrary sample/user and
      fabricates the McNemar table (review-fix 2026-06-14). Both lists are the scorer's `tsl` by
      construction, so require element-wise-equal user_id order and pair POSITIONALLY; fail loudly.
    """
    van = np.load(vanilla_npz, allow_pickle=True)
    van_uid = np.asarray(van["user_ids"])
    base_uid = np.asarray(base_uid)
    avail = [k for k in KS if f"vanilla_hit{k}" in van.files]
    if len(set(van_uid.tolist())) == len(van_uid):       # unique → align by user_id
        posn = {u: i for i, u in enumerate(van_uid)}
        keep = np.array([i for i, u in enumerate(base_uid) if u in posn], dtype=np.int64)
        # Require EVERY baseline sample to pair (full coverage) — catches a SMOKE-vs-full N mismatch
        # (scorer slices by user, baselines by sample) that would otherwise silently under-power the
        # paired test on the intersection (review-fix 2026-06-15).
        assert keep.size == base_uid.shape[0], (
            f"only {keep.size}/{base_uid.shape[0]} baseline users pair with vanilla npz "
            f"{vanilla_npz} — user-set mismatch (smoke vs full? different filter?)")
        sel = np.array([posn[base_uid[i]] for i in keep])
        vh = {k: van[f"vanilla_hit{k}"][sel] for k in avail}
        bh = {k: base_hits[k][keep] for k in avail}
        return base_uid[keep], bh, vh
    # Repeated user_ids (per-sample, e.g. MIND mode-B). CANNOT pair positionally: both lists group
    # users into contiguous blocks via `[s for uid in set(test_uids) for s in groups[uid]]`, and
    # SET iteration order differs ACROSS PROCESSES (Python string-hash randomization), so the block
    # order of the scorer's npz ≠ this run's. But within a user the rows are contiguous and in
    # deterministic build_samples order, so the stable cross-process key is (user_id, within-user
    # occurrence index). Pair by that key (review-fix 2026-06-14).
    van_keys = _occurrence_keys(van_uid)
    base_keys = _occurrence_keys(base_uid)
    posn = {k: i for i, k in enumerate(van_keys)}
    keep = np.array([i for i, k in enumerate(base_keys) if k in posn], dtype=np.int64)
    assert keep.size == len(base_keys) == len(van_keys), (
        f"per-sample pairing is not a full bijection (kept {keep.size} of base {len(base_keys)} / "
        f"vanilla {len(van_keys)}) — baseline & scorer (user,occurrence) sample sets differ")
    sel = np.array([posn[base_keys[i]] for i in keep])
    vh = {k: van[f"vanilla_hit{k}"][sel] for k in avail}
    bh = {k: base_hits[k][keep] for k in avail}
    return base_uid[keep], bh, vh


def _occurrence_keys(uids) -> list:
    """(user_id, k) where k = 0-based occurrence index of that user_id so far. Stable across
    processes (independent of set-iteration block order); within-user order is deterministic."""
    seen: dict = {}
    keys = []
    for u in uids:
        u = u.item() if hasattr(u, "item") else u
        c = seen.get(u, 0)
        keys.append((u, c))
        seen[u] = c + 1
    return keys
