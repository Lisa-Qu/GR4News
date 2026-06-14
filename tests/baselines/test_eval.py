# tests/baselines/test_eval.py
import numpy as np
from baselines.eval_fullcatalog import eval_full_catalog

def test_toy_ranking():
    # 3 users, catalog of 5 items. scores[u] high → ranked first. targets = item index.
    scores = np.array([
        [9, 1, 2, 3, 4],   # target 0 → rank 0 (hit@1)
        [1, 2, 3, 4, 9],   # target 4 → rank 0 (hit@1)
        [5, 4, 3, 2, 1],   # target 3 → rank 3 (hit@5, not hit@1)
    ], dtype=np.float32)
    targets = np.array([0, 4, 3])
    uids = np.array(["u0", "u1", "u2"])
    agg, hits = eval_full_catalog(scores, targets, uids)
    assert agg["R@1"] == 2/3
    assert agg["R@5"] == 1.0
    assert hits[1].tolist() == [True, True, False]
    assert hits[10].tolist() == [True, True, True]
    assert agg["MRR"] == (1.0 + 1.0 + 0.25) / 3
