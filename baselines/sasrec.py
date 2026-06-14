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
    def seq_repr(self, seq):  # seq: (B, L) right-aligned item ids, 0=PAD
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).clamp(max=self.max_len - 1)
        x = self.drop(self.item_emb(seq) + self.pos_emb(pos)[None])
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1)
        pad = seq == 0
        h = self.blocks(x, mask=causal, src_key_padding_mask=pad)
        return self.ln(h[:, -1])  # last position (the next-item query)
    def full_scores(self, seq):  # (B, n_items+1)
        return self.seq_repr(seq) @ self.item_emb.weight.T
