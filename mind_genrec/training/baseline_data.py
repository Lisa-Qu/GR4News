"""Shared data utilities for two-tower baseline training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mind_genrec.data import iter_jsonl


@dataclass(frozen=True)
class TwoTowerSample:
    """One baseline training or evaluation example."""

    sample_id: str
    history_indices: list[int]
    target_index: int
    target_news_id: str
    history_news_ids: list[str]
    candidate_news_ids: list[str]


class TwoTowerDataset(Dataset[TwoTowerSample]):
    """Dataset for two-tower next-click training."""

    def __init__(
        self,
        *,
        sample_path: str | Path,
        item_to_index: dict[str, int],
        max_history_length: int,
        max_samples: int | None = None,
    ) -> None:
        self._samples: list[TwoTowerSample] = []
        for payload in iter_jsonl(sample_path):
            target_news_id = payload["target_news_id"]
            if target_news_id not in item_to_index:
                continue
            history_news_ids = [
                item_id
                for item_id in payload["history"]
                if item_id in item_to_index
            ]
            history_news_ids = history_news_ids[-max_history_length:]
            if not history_news_ids:
                continue
            history_indices = [item_to_index[item_id] for item_id in history_news_ids]
            self._samples.append(
                TwoTowerSample(
                    sample_id=payload["sample_id"],
                    history_indices=history_indices,
                    target_index=item_to_index[target_news_id],
                    target_news_id=target_news_id,
                    history_news_ids=history_news_ids,
                    candidate_news_ids=list(payload.get("candidate_news_ids", [])),
                )
            )
            if max_samples is not None and len(self._samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> TwoTowerSample:
        return self._samples[index]


@dataclass(frozen=True)
class TwoTowerBatch:
    """Padded two-tower batch."""

    history_embeddings: torch.Tensor
    history_mask: torch.Tensor
    target_item_embeddings: torch.Tensor


class TwoTowerCollator:
    """Build padded batches from item-embedding lookups."""

    def __init__(self, item_embeddings: np.ndarray) -> None:
        self._embedding_table = torch.tensor(item_embeddings, dtype=torch.float32)
        self._embedding_dim = item_embeddings.shape[1]

    def __call__(self, samples: list[TwoTowerSample]) -> TwoTowerBatch:
        batch_size = len(samples)
        max_length = max(len(sample.history_indices) for sample in samples)
        history_embeddings = torch.zeros(
            batch_size,
            max_length,
            self._embedding_dim,
            dtype=torch.float32,
        )
        history_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
        target_item_embeddings = torch.zeros(
            batch_size,
            self._embedding_dim,
            dtype=torch.float32,
        )

        for row, sample in enumerate(samples):
            indices = torch.tensor(sample.history_indices, dtype=torch.long)
            seq_len = indices.numel()
            history_embeddings[row, :seq_len] = self._embedding_table[indices]
            history_mask[row, :seq_len] = True
            target_item_embeddings[row] = self._embedding_table[sample.target_index]

        return TwoTowerBatch(
            history_embeddings=history_embeddings,
            history_mask=history_mask,
            target_item_embeddings=target_item_embeddings,
        )


def move_batch(batch: TwoTowerBatch, device: torch.device) -> TwoTowerBatch:
    """Move one baseline batch to the target device."""

    return TwoTowerBatch(
        history_embeddings=batch.history_embeddings.to(device),
        history_mask=batch.history_mask.to(device),
        target_item_embeddings=batch.target_item_embeddings.to(device),
    )
