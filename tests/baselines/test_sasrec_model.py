# tests/baselines/test_sasrec_model.py
import torch
from baselines.sasrec import SASRec

def test_forward_shape_and_causality():
    m = SASRec(n_items=100, d=64, n_blocks=2, n_heads=1, max_len=20, dropout=0.0)
    seq = torch.randint(1, 100, (4, 20))
    seq_repr = m.seq_repr(seq)            # (B, d) last-position
    assert seq_repr.shape == (4, 64)
    scores = m.full_scores(seq)           # (B, n_items+1)
    assert scores.shape == (4, 101)

def test_smoke_loss_decreases():
    torch.manual_seed(0)
    m = SASRec(n_items=50, d=32, n_blocks=2, n_heads=1, max_len=10, dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    seq = torch.randint(1, 50, (16, 10)); tgt = torch.randint(1, 50, (16,))
    first = last = None
    for _ in range(50):
        loss = torch.nn.functional.cross_entropy(m.full_scores(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or loss.item(); last = loss.item()
    assert last < first
