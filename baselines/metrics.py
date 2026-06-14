# baselines/metrics.py
"""Single-positive specialization of the scorer's _rank_metrics (full-catalog retrieval).

Full-catalog next-item retrieval has exactly ONE relevant item (the held-out target). Given its
0-based rank (or None if not retrieved/out of range), reproduce the scorer's hit@k / MRR / nDCG@k
EXACTLY (n_rel=1 ⇒ IDCG=1, so nDCG@k = 1/log2(rank+2) when rank<k). Proven equal to
genrec_v2.run_main_table._rank_metrics by tests/baselines/test_metrics.py.
"""
from __future__ import annotations
import math

KS = (1, 5, 10, 50)
NDCG_KS = (5, 10)

def rank_metrics_single(rank: int | None) -> tuple[dict, float, dict]:
    if rank is None:
        return ({k: False for k in KS}, 0.0, {k: 0.0 for k in NDCG_KS})
    hit = {k: bool(rank < k) for k in KS}
    mrr = 1.0 / (rank + 1)
    ndcg = {k: (1.0 / math.log2(rank + 2) if rank < k else 0.0) for k in NDCG_KS}
    return hit, mrr, ndcg
