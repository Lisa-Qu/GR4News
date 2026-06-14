# baselines/eval_fullcatalog.py
"""Full-catalog ranking eval reusing the single-positive metrics. Produces aggregate metrics
(R@k, MRR, nDCG@k) + per-sample hit@k arrays + a per_user_hits.npz writer for significance.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from baselines.metrics import rank_metrics_single, KS, NDCG_KS

def _target_ranks(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """0-based rank of each target under descending score (stable, ties broken by index)."""
    # rank = number of items strictly scoring higher than the target's score.
    n = scores.shape[0]
    tgt_scores = scores[np.arange(n), targets]
    return (scores > tgt_scores[:, None]).sum(axis=1)

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
    return agg, hits

def write_per_user_hits(out_dir: Path, user_ids: np.ndarray, baseline_hits: dict,
                        vanilla_hits: dict) -> None:
    """Persist baseline + the matching-setting generative vanilla hits (already aligned by user
    order) for run_statistical_significance.py's baseline_vs_vanilla comparison."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "per_user_hits.npz",
             user_ids=np.array(user_ids),
             vanilla_hit1=vanilla_hits[1], vanilla_hit10=vanilla_hits[10],
             baseline_hit1=baseline_hits[1], baseline_hit10=baseline_hits[10])
