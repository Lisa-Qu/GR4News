# baselines/nrms.py
"""NRMS (Wu et al. 2019) — news encoder (multi-head self-attn + additive attn over title words)
and user encoder (additive attn over history news vectors). Standard published config; no tuning.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, dim: int, hidden: int = 200):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)
    def forward(self, x, mask=None):  # x: (B, L, dim)
        a = self.query(torch.tanh(self.proj(x))).squeeze(-1)  # (B, L)
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        w = F.softmax(a, dim=-1).unsqueeze(-1)
        return (w * x).sum(dim=1)  # (B, dim)

class NewsEncoder(nn.Module):
    def __init__(self, vocab: int, emb_dim: int = 300, heads: int = 16, head_dim: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        d = heads * head_dim  # NRMS self-attention output dim (16*16=256), NOT emb_dim (300).
        # The 300-dim GloVe-style word embeddings feed a multi-head self-attention whose model
        # dim is heads*head_dim=256 (Wu et al. 2019). emb_dim need not be divisible by heads,
        # so the attention runs in the 256-d space (kdim/vdim = emb_dim for the K/V projection).
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True,
                                          kdim=emb_dim, vdim=emb_dim)
        self.q_proj = nn.Linear(emb_dim, d)  # query side must match the attention model dim
        self.add = AdditiveAttention(d)
        self.drop = nn.Dropout(dropout)
        self.out_dim = d
    def forward(self, title_tokens):  # (B, L)
        mask = title_tokens != 0
        x = self.drop(self.emb(title_tokens))                 # (B, L, emb)
        q = self.q_proj(x)                                    # (B, L, d)
        x, _ = self.attn(q, x, x, key_padding_mask=~mask)     # (B, L, d)
        return self.add(self.drop(x), mask)                   # (B, d)

class NRMS(nn.Module):
    def __init__(self, vocab: int, emb_dim: int = 300, heads: int = 16, head_dim: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        self.news = NewsEncoder(vocab, emb_dim, heads, head_dim, dropout)
        self.user_attn = AdditiveAttention(self.news.out_dim)
    def encode_news(self, titles):  # (M, L) → (M, dim)
        return self.news(titles)
    def encode_user(self, hist_titles, hist_mask):  # (B, H, L), (B, H)
        B, H, L = hist_titles.shape
        nv = self.news(hist_titles.reshape(B * H, L)).reshape(B, H, -1)
        return self.user_attn(nv, hist_mask)  # (B, dim)
    def forward(self, hist_titles, hist_mask, cand_titles):  # cand_titles: (B, C, L)
        u = self.encode_user(hist_titles, hist_mask)          # (B, dim)
        B, C, L = cand_titles.shape
        cv = self.news(cand_titles.reshape(B * C, L)).reshape(B, C, -1)
        return (u.unsqueeze(1) * cv).sum(-1)                  # (B, C) scores
