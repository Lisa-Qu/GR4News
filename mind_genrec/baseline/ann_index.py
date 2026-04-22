"""ANN index backends for two-tower retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IndexedCandidate:
    """One retrieved candidate from the vector index."""

    item_id: str
    score: float
    index: int


class FaissANNIndex:
    """FAISS IndexFlatIP backed ANN index (cosine via pre-normalized vectors).

    Vectors are L2-normalized before indexing so inner product equals cosine
    similarity, matching the ExactCosineANNIndex contract.
    """

    def __init__(
        self,
        *,
        item_ids: list[str],
        item_vectors: torch.Tensor,
    ) -> None:
        import faiss

        if item_vectors.ndim != 2:
            raise ValueError("item_vectors must have shape [item_count, dim]")
        if len(item_ids) != item_vectors.shape[0]:
            raise ValueError("item_ids length does not match item_vectors rows")

        self._item_ids = item_ids
        dim = item_vectors.shape[1]

        vectors_np = F.normalize(item_vectors.float().cpu(), dim=-1).numpy()
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors_np)

    def search(self, query_vector: torch.Tensor, *, top_k: int) -> list[IndexedCandidate]:
        """Return the top-k approximate cosine neighbors via FAISS."""

        if query_vector.ndim != 1:
            raise ValueError("query_vector must have shape [dim]")

        query_np = F.normalize(query_vector.float().cpu(), dim=-1).numpy()[np.newaxis, :]
        top_k = min(top_k, len(self._item_ids))
        scores, indices = self._index.search(query_np, top_k)
        return [
            IndexedCandidate(
                item_id=self._item_ids[int(idx)],
                score=float(score),
                index=int(idx),
            )
            for score, idx in zip(scores[0], indices[0], strict=True)
            if idx >= 0
        ]


class ExactCosineANNIndex:
    """Dependency-light exact cosine index (fallback when FAISS is unavailable)."""

    def __init__(
        self,
        *,
        item_ids: list[str],
        item_vectors: torch.Tensor,
        device: torch.device,
    ) -> None:
        if item_vectors.ndim != 2:
            raise ValueError("item_vectors must have shape [item_count, dim]")
        if len(item_ids) != item_vectors.shape[0]:
            raise ValueError("item_ids length does not match item_vectors rows")
        self._item_ids = item_ids
        self._device = device
        self._item_vectors = F.normalize(item_vectors.to(device), dim=-1)

    def search(self, query_vector: torch.Tensor, *, top_k: int) -> list[IndexedCandidate]:
        """Return the top-k exact cosine neighbors."""

        if query_vector.ndim != 1:
            raise ValueError("query_vector must have shape [dim]")
        query_vector = F.normalize(query_vector.to(self._device), dim=-1)
        scores = torch.matmul(self._item_vectors, query_vector)
        top_k = min(top_k, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=0)
        return [
            IndexedCandidate(
                item_id=self._item_ids[int(index)],
                score=float(score.item()),
                index=int(index.item()),
            )
            for score, index in zip(top_scores, top_indices, strict=True)
        ]


def build_ann_index(
    *,
    item_ids: list[str],
    item_vectors: torch.Tensor,
    device: torch.device,
) -> FaissANNIndex | ExactCosineANNIndex:
    """Build a FAISS index when available, otherwise fall back to exact search."""

    try:
        return FaissANNIndex(item_ids=item_ids, item_vectors=item_vectors)
    except ImportError:
        return ExactCosineANNIndex(
            item_ids=item_ids,
            item_vectors=item_vectors,
            device=device,
        )
