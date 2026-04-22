"""Shared data utilities for generator training and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mind_genrec.data import iter_jsonl
from mind_genrec.model import SemanticIDMapper


@dataclass(frozen=True)
class GeneratorSample:
    """One generator-training or evaluation example."""

    sample_id: str
    history_indices: list[int]
    target_code: tuple[int, ...]
    target_news_id: str
    history_news_ids: list[str]
    candidate_news_ids: list[str]


class GeneratorDataset(Dataset[GeneratorSample]):
    """Dataset for `history -> semantic code` training and evaluation."""

    def __init__(
        self,
        *,
        sample_path: str | Path,
        item_to_index: dict[str, int],
        mapper: SemanticIDMapper,
        max_history_length: int,
        max_samples: int | None = None,
    ) -> None:
        self._samples: list[GeneratorSample] = []
        for payload in iter_jsonl(sample_path):
            history_news_ids = [
                item_id
                for item_id in payload["history"]
                if item_id in item_to_index
            ]
            history_indices = [item_to_index[item_id] for item_id in history_news_ids]
            target_news_id = payload["target_news_id"]
            target_code = mapper.code_for_item(target_news_id)
            if target_code is None:
                continue
            history_news_ids = history_news_ids[-max_history_length:]
            history_indices = history_indices[-max_history_length:]
            if not history_indices:
                continue
            self._samples.append(
                GeneratorSample(
                    sample_id=payload["sample_id"],
                    history_indices=history_indices,
                    target_code=target_code,
                    target_news_id=target_news_id,
                    history_news_ids=history_news_ids,
                    candidate_news_ids=list(payload.get("candidate_news_ids", [])),
                )
            )
            if max_samples is not None and len(self._samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> GeneratorSample:
        return self._samples[index]


@dataclass(frozen=True)
class GeneratorBatch:
    """Padded generator batch."""

    history_embeddings: torch.Tensor
    history_mask: torch.Tensor
    target_codes: torch.Tensor


class GeneratorCollator:
    """Build padded batches from item-embedding lookups."""

    def __init__(self, item_embeddings: np.ndarray, code_length: int) -> None:
        self._embedding_table = torch.tensor(item_embeddings, dtype=torch.float32)
        self._embedding_dim = item_embeddings.shape[1]
        self._code_length = code_length

    def __call__(self, samples: list[GeneratorSample]) -> GeneratorBatch:
        batch_size = len(samples)
        max_length = max(len(sample.history_indices) for sample in samples)
        history_embeddings = torch.zeros(
            batch_size,
            max_length,
            self._embedding_dim,
            dtype=torch.float32,
        )
        history_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
        target_codes = torch.zeros(batch_size, self._code_length, dtype=torch.long)

        for row, sample in enumerate(samples):
            indices = torch.tensor(sample.history_indices, dtype=torch.long)
            seq_len = indices.numel()
            history_embeddings[row, :seq_len] = self._embedding_table[indices]
            history_mask[row, :seq_len] = True
            target_codes[row] = torch.tensor(sample.target_code, dtype=torch.long)

        return GeneratorBatch(
            history_embeddings=history_embeddings,
            history_mask=history_mask,
            target_codes=target_codes,
        )


def resolve_item_ids(artifact_dir: Path, mapper: SemanticIDMapper) -> list[str]:
    """Resolve item-id ordering for embedding lookup rows."""

    item_ids_path = artifact_dir / "item_ids.json"
    if item_ids_path.exists():
        return json.loads(item_ids_path.read_text(encoding="utf-8"))
    return sorted(mapper.item_to_code.keys())


def build_item_index(item_ids: list[str]) -> dict[str, int]:
    """Build `news_id -> row index` mapping."""

    return {item_id: index for index, item_id in enumerate(item_ids)}


def move_batch(batch: GeneratorBatch, device: torch.device) -> GeneratorBatch:
    """Move one batch to the target device."""

    return GeneratorBatch(
        history_embeddings=batch.history_embeddings.to(device),
        history_mask=batch.history_mask.to(device),
        target_codes=batch.target_codes.to(device),
    )
