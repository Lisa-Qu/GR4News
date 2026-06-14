# tests/baselines/test_metrics.py
import numpy as np
import pytest
from baselines.metrics import rank_metrics_single, KS, NDCG_KS

def _ref_rank_metrics(rel_ordered):
    """Byte-identical copy of genrec_v2.run_main_table._rank_metrics for the oracle comparison."""
    hit = {k: bool(rel_ordered[:k].max() > 0) for k in KS}
    nz = np.flatnonzero(rel_ordered)
    mrr = float(1.0 / (nz[0] + 1)) if nz.size else 0.0
    n_rel = int(rel_ordered.sum())
    ndcg = {}
    for k in NDCG_KS:
        topk = rel_ordered[:k]
        dcg = float(np.sum(topk / np.log2(np.arange(2, topk.size + 2))))
        idcg = float(np.sum(1.0 / np.log2(np.arange(2, min(k, n_rel) + 2)))) if n_rel else 0.0
        ndcg[k] = dcg / idcg if idcg > 0 else 0.0
    return hit, mrr, ndcg

@pytest.mark.parametrize("rank", [0, 1, 4, 9, 49, 50, 100])
def test_single_matches_reference(rank):
    n_catalog = 200
    rel = np.zeros(n_catalog, dtype=np.int64)
    if rank < n_catalog:
        rel[rank] = 1
    ref_hit, ref_mrr, ref_ndcg = _ref_rank_metrics(rel)
    hit, mrr, ndcg = rank_metrics_single(rank if rank < n_catalog else None)
    assert hit == ref_hit
    assert mrr == pytest.approx(ref_mrr)
    for k in NDCG_KS:
        assert ndcg[k] == pytest.approx(ref_ndcg[k])

def test_target_absent():
    hit, mrr, ndcg = rank_metrics_single(None)
    assert hit == {1: False, 5: False, 10: False, 50: False}
    assert mrr == 0.0
    assert ndcg == {5: 0.0, 10: 0.0}
