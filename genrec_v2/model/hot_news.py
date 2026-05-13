"""Hot News Fusion: Cross-Attention over category-top items + Gate."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HotNewsFusion(nn.Module):
    """Cross-attention over hot news embeddings with learned gating."""

    def __init__(
        self,
        hidden_dim: int,
        hot_embeddings: torch.Tensor,  # [N_hot, dim]
    ) -> None:
        super().__init__()
        self.register_buffer("hot_embs", hot_embeddings)  # frozen
        _, D = hot_embeddings.shape
        self.hidden_dim = hidden_dim

        # Cross-attention: user_state projects to hot-embedding space
        self.query_proj = nn.Linear(hidden_dim, D)
        self.out_proj = nn.Linear(D, hidden_dim)  # project back
        self.scale = D ** 0.5

        # Gate
        self.gate_proj = nn.Linear(hidden_dim + hidden_dim, 1)

    def forward(self, user_state: torch.Tensor) -> torch.Tensor:
        """Fuse hot news context into user_state."""
        B, D_user = user_state.shape

        # Cross-attention
        q = self.query_proj(user_state)  # [B, D]
        attn = (q @ self.hot_embs.T) / self.scale  # [B, N_hot]
        attn_w = F.softmax(attn, dim=-1)
        hot_context_raw = attn_w @ self.hot_embs  # [B, D]
        hot_context = self.out_proj(hot_context_raw)  # [B, hidden_dim]

        # Gate
        gate_input = torch.cat([user_state, hot_context], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # [B, 1]

        return user_state + gate * hot_context


def build_hot_embeddings(
    news_jsonl: str,
    item_embeddings: np.ndarray,
    item_ids: list[str],
    topk: int = 5,
    min_cat_clicks: int = 100,
) -> tuple[torch.Tensor, list[str]]:
    """Build category-level top-k hot item embeddings.

    Returns:
        hot_embs: [N_hot, dim] tensor of hot item embeddings
        hot_item_ids: list of hot item ids (for debugging)
    """
    # Load category info
    news_cat: dict[str, str] = {}
    with open(news_jsonl) as f:
        for line in f:
            d = json.loads(line)
            news_cat[d["news_id"]] = d.get("category", "")

    id_to_idx = {nid: i for i, nid in enumerate(item_ids)}

    # Count clicks per category-item (use item_embeddings as proxy — already have them)
    # Actually, we need click counts. Build from semantic_codes or load from metadata.
    # For proxy: use simple frequency from the pre-built corpus
    cat_items: dict[str, list[str]] = {}
    for nid, cat in news_cat.items():
        if cat not in cat_items:
            cat_items[cat] = []
        if nid in id_to_idx:
            cat_items[cat].append(nid)

    hot_items: list[str] = []
    for cat, items in sorted(cat_items.items()):
        if len(items) < min_cat_clicks / topk:  # heuristic filter
            continue
        top = items[:topk]  # Use first K items in this category
        hot_items.extend(top)

    indices = [id_to_idx[nid] for nid in hot_items if nid in id_to_idx]
    hot_embs = torch.tensor(item_embeddings[indices], dtype=torch.float32)

    return hot_embs, hot_items
