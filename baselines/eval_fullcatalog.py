# baselines/eval_fullcatalog.py
"""Full-catalog ranking eval reusing the single-positive metrics. Produces aggregate metrics
(R@k, MRR, nDCG@k) + per-sample hit@k arrays + a per_user_hits.npz writer for significance.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from baselines.metrics import rank_metrics_single, KS, NDCG_KS

def _target_ranks(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """0-based rank of each target under a STABLE descending argsort.

    This MUST match the scorer's tie semantics (genrec_v2 ``eval_peruser``):
    ``torch.argsort(scores, descending=True)`` is stable, so among items with an
    equal score the one with the SMALLER catalog index is ranked first. The
    target's rank is therefore its position in that stable descending order — NOT
    the optimistic "strictly-greater" count (which would hand every tie the best
    possible slot and over-state the metrics).

    We replicate the stable descending order with ``(-scores).argsort(kind='stable')``
    (numpy stable sort preserves the original index order among equal keys), then
    read off where each target index lands.
    """
    n = scores.shape[0]
    order = np.argsort(-scores, axis=1, kind="stable")  # stable descending order
    # position of column `targets[i]` in row i's order → its 0-based rank.
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        # np.where on the (small) order row; argmax of the equality mask is the rank.
        ranks[i] = int(np.flatnonzero(order[i] == targets[i])[0])
    return ranks

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

    Persists ALL ``KS`` (1,5,10,50) for BOTH baseline and vanilla so significance can be tested
    at every reported cutoff (keys ``baseline_hit{k}`` / ``vanilla_hit{k}``)."""
    assert len(user_ids) > 0, "write_per_user_hits: empty user_ids (no paired users)"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"user_ids": np.array(user_ids)}
    for k in KS:
        arrays[f"vanilla_hit{k}"] = vanilla_hits[k]
        arrays[f"baseline_hit{k}"] = baseline_hits[k]
    np.savez(out_dir / "per_user_hits.npz", **arrays)
