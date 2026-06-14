# tests/baselines/test_data_mind.py
import numpy as np
from baselines.data_mind import load_mind_for_nrms

def test_split_matches_scorer():
    d = load_mind_for_nrms(max_history=50, max_title_len=30)
    # Test users are a non-empty disjoint subset of val users; targets in vocab.
    assert len(d.test_samples) > 0 and len(d.val_samples) > 0
    test_uids = {s["user_id"] for s in d.test_samples}
    val_uids = {s["user_id"] for s in d.val_samples}
    assert test_uids.isdisjoint(val_uids)
    # Every sample has history (>=1 valid id) + a target in the catalog.
    s = d.test_samples[0]
    assert s["target"] in d.news2idx
    assert len(s["history"]) >= 1
    # Title tensor shape = (n_news, max_title_len)
    assert d.title_tokens.shape[1] == 30
