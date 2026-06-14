# tests/baselines/test_eval.py
import numpy as np
from baselines.eval_fullcatalog import eval_full_catalog, _target_ranks

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


def _argsort_rank(scores_row, target):
    """Reference: rank under the scorer's STABLE descending argsort tie convention.
    torch.argsort(descending=True) and np.argsort(-x, kind='stable') agree (verified on
    torch 2.1 server + torch 2.9 local): among equal scores the SMALLER index ranks first."""
    order = np.argsort(-scores_row, kind="stable")
    return int(np.flatnonzero(order == target)[0])


def test_target_rank_tie_matches_argsort():
    # Row with ties around the target. Catalog indices 0..4, scores have a 3-way tie at 5.0.
    #   scores = [5, 3, 5, 3, 5]; stable desc order = [0, 2, 4, 1, 3].
    # target=2 ties with 0 and 4 → strictly-greater count would give optimistic rank 0,
    # but the stable argsort puts index 0 first ⇒ TRUE rank of target 2 is 1.
    # target=4 ⇒ rank 2 (after 0 and 2). target=0 ⇒ rank 0.
    scores = np.array([
        [5.0, 3.0, 5.0, 3.0, 5.0],
        [5.0, 3.0, 5.0, 3.0, 5.0],
        [5.0, 3.0, 5.0, 3.0, 5.0],
        [1.0, 2.0, 2.0, 9.0, 2.0],   # target 2 ties with 1 and 4 below the 9 at idx3 → rank 2
    ])
    targets = np.array([0, 2, 4, 2])
    ranks = _target_ranks(scores, targets)
    expected = np.array([_argsort_rank(scores[i], targets[i]) for i in range(len(targets))])
    assert ranks.tolist() == expected.tolist()
    assert ranks.tolist() == [0, 1, 2, 2]  # explicit: ties NOT given the optimistic best slot


def test_self_consistency_asserts_hold():
    # eval_full_catalog must produce monotone, finite, in-range metrics (enforced asserts).
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((50, 30)).astype(np.float32)
    targets = rng.integers(0, 30, size=50)
    uids = np.array([f"u{i}" for i in range(50)])
    agg, _ = eval_full_catalog(scores, targets, uids)  # would raise if inconsistent
    assert 0.0 <= agg["R@1"] <= agg["R@5"] <= agg["R@10"] <= agg["R@50"] <= 1.0
