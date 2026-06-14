# baselines/sasrec.py
"""SASRec (Kang & McAuley 2018) — causal self-attention over item-id sequence; score = last
hidden · item embeddings (full catalog). Standard config (d=64, 2 blocks, 1 head); no tuning.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, n_items: int, d: int = 64, n_blocks: int = 2, n_heads: int = 1,
                 max_len: int = 20, dropout: float = 0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)  # +1 for PAD=0
        self.pos_emb = nn.Embedding(max_len, d)
        self.max_len = max_len
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=d, dropout=dropout,
                                           batch_first=True, activation="relu")
        self.blocks = nn.TransformerEncoder(layer, n_blocks)
        self.ln = nn.LayerNorm(d)
    def seq_hidden(self, seq):  # seq: (B, L) left-padded item ids, 0=PAD → (B, L, d)
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).clamp(max=self.max_len - 1)
        x = self.drop(self.item_emb(seq) + self.pos_emb(pos)[None])
        x = x * (seq != 0).unsqueeze(-1)  # zero PAD positions entirely (item+pos) → no PAD signal
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1)
        # NOTE: causal mask ONLY — do NOT add src_key_padding_mask. With left-padding a
        # front PAD query position would attend solely to itself, and key-padding-masking that
        # one key yields an all-masked softmax row → NaN that propagates to the last position.
        # PAD tokens are neutralised via padding_idx=0 (zero item embedding); the last position
        # (a real item) attends causally over the history. Standard SASRec convention.
        h = self.blocks(x, mask=causal)
        return self.ln(h)  # (B, L, d) — per-position next-item query reps (LN is per-position)
    def seq_repr(self, seq):  # (B, d) — last position only (eval query)
        return self.seq_hidden(seq)[:, -1]
    def full_scores(self, seq):  # eval: (B, n_items+1) from the last position
        return self.seq_repr(seq) @ self.item_emb.weight.T
    def all_scores(self, seq):  # train: (B, L, n_items+1) — score every position (all-position SASRec)
        return self.seq_hidden(seq) @ self.item_emb.weight.T
