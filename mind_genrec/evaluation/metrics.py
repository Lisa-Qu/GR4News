"""Ranking metrics for offline recommendation evaluation."""

from __future__ import annotations

import math


def rank_of_first_hit(target_item: str, ranked_items: list[str], *, top_k: int) -> int | None:
    """Return 1-based rank of the first hit within top-k, else `None`."""

    for index, item_id in enumerate(ranked_items[:top_k], start=1):
        if item_id == target_item:
            return index
    return None


def hit_rate_at_k(target_item: str, ranked_items: list[str], *, top_k: int) -> float:
    """Return 1.0 if the target is found in top-k else 0.0."""

    return 1.0 if rank_of_first_hit(target_item, ranked_items, top_k=top_k) is not None else 0.0


def mean_reciprocal_rank_at_k(target_item: str, ranked_items: list[str], *, top_k: int) -> float:
    """Return reciprocal rank within top-k else 0.0."""

    rank = rank_of_first_hit(target_item, ranked_items, top_k=top_k)
    if rank is None:
        return 0.0
    return 1.0 / rank


def ndcg_at_k(target_item: str, ranked_items: list[str], *, top_k: int) -> float:
    """Return binary-label nDCG@k for one target item."""

    rank = rank_of_first_hit(target_item, ranked_items, top_k=top_k)
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1.0)

