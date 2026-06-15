# tests/baselines/test_eval.py
import numpy as np
import pytest
from baselines.eval_fullcatalog import (
    eval_full_catalog, _target_ranks, align_vanilla, _occurrence_keys, write_per_user_hits,
)

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


def _write_van(tmp_path, user_ids, h1, h10):
    p = tmp_path / "per_user_hits.npz"
    np.savez(p, user_ids=np.array(user_ids),
             vanilla_hit1=np.array(h1, bool), vanilla_hit10=np.array(h10, bool))
    return p


def test_align_vanilla_loo_by_userid_reorders(tmp_path):
    # LOO (unique user_ids): vanilla stored in a DIFFERENT order than the baseline; must pair by id.
    van = _write_van(tmp_path, ["a", "b", "c"], [1, 0, 1], [1, 1, 0])
    base_uid = np.array(["c", "a", "b"])
    base_hits = {1: np.array([0, 1, 0], bool), 10: np.array([1, 1, 0], bool)}
    kept, bh, vh = align_vanilla(base_uid, base_hits, van)
    assert kept.tolist() == ["c", "a", "b"]
    # vanilla realigned to baseline's order: c→1, a→1, b→0 for hit1
    assert vh[1].tolist() == [1, 1, 0]
    assert vh[10].tolist() == [0, 1, 1]
    assert bh[1].tolist() == [0, 1, 0]


def test_align_vanilla_persample_occurrence_key_block_permuted(tmp_path):
    # Per-sample (repeated user_ids): vanilla user-BLOCKS in a different order than the baseline
    # (simulates cross-process set-iteration), within-user order identical. Must pair by
    # (user_id, occurrence), NOT position.
    # vanilla order: u0,u0,u1 ; baseline order: u1,u0,u0  (block-permuted)
    van = _write_van(tmp_path, ["u0", "u0", "u1"], [1, 0, 1], [1, 1, 0])
    base_uid = np.array(["u1", "u0", "u0"])
    base_hits = {1: np.array([9, 7, 8], int) > 0, 10: np.array([1, 1, 1], bool)}  # distinct markers
    # keys: van (u0,0)=row0,(u0,1)=row1,(u1,0)=row2 ; base (u1,0)=row0,(u0,0)=row1,(u0,1)=row2
    assert _occurrence_keys(np.array(["u0", "u0", "u1"])) == [("u0", 0), ("u0", 1), ("u1", 0)]
    kept, bh, vh = align_vanilla(base_uid, base_hits, van)
    # baseline order kept; vanilla pulled by matching (user,occurrence):
    #   base row0 (u1,0) → van row2 → vanilla_hit1=1 ; base row1 (u0,0)→van row0→1 ; row2 (u0,1)→van row1→0
    assert vh[1].tolist() == [1, 1, 0]
    assert kept.tolist() == ["u1", "u0", "u0"]


def test_align_vanilla_persample_full_bijection_required(tmp_path):
    # Mismatched per-user counts (smoke vs full) must fail loudly, not silently sub-pair.
    van = _write_van(tmp_path, ["u0", "u0", "u1"], [1, 0, 1], [1, 1, 0])
    base_uid = np.array(["u0", "u1"])  # missing one u0 occurrence
    base_hits = {1: np.array([1, 1], bool), 10: np.array([1, 1], bool)}
    with pytest.raises(AssertionError):
        align_vanilla(base_uid, base_hits, van)
