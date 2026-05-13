"""PyTorch Dataset for GenRec-V2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class GenRecV2Dataset(Dataset):
    """Dataset: each sample = (history_embed_ids, target_code, target_emb_id)."""

    def __init__(
        self,
        samples: list[dict],
        item_to_index: dict[str, int],
        code_for_item: dict[str, tuple[int, ...]],
        item_embeddings: np.ndarray,
        max_history_len: int = 128,
    ) -> None:
        kept: list[dict] = []
        for s in samples:
            nid = s["target"]
            if nid not in code_for_item or nid not in item_to_index:
                continue
            kept.append(s)
        self._samples = kept
        self._item_to_index = item_to_index
        self._code_for_item = code_for_item
        self._emb_table = torch.tensor(item_embeddings, dtype=torch.float32)
        self._max_hist = max_history_len

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        hist_ids = s["history"][-self._max_hist :]
        hist_indices = [self._item_to_index[h] for h in hist_ids if h in self._item_to_index]
        target_code = list(self._code_for_item[s["target"]])
        target_emb_idx = self._item_to_index[s["target"]]
        return {
            "history_indices": hist_indices,
            "target_code": target_code,
            "target_emb_idx": target_emb_idx,
        }


_emb_dim: int = 0


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    max_len = max(len(s["history_indices"]) for s in batch)
    code_len = len(batch[0]["target_code"])
    B = len(batch)

    hist_emb = torch.zeros(B, max_len, _emb_dim)
    hist_mask = torch.zeros(B, max_len, dtype=torch.bool)
    tgt_code = torch.zeros(B, code_len, dtype=torch.long)
    tgt_emb_idx = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(batch):
        indices = s["history_indices"]
        L = len(indices)
        hist_emb[i, :L] = _embedding_table[torch.tensor(indices, dtype=torch.long)]
        hist_mask[i, :L] = True
        tgt_code[i] = torch.tensor(s["target_code"], dtype=torch.long)
        tgt_emb_idx[i] = s["target_emb_idx"]

    return {
        "history_emb": hist_emb,
        "history_mask": hist_mask,
        "target_code": tgt_code,
        "target_emb_idx": tgt_emb_idx,
    }


# Global embedding table set by the collator factory
_embedding_table: torch.Tensor


def make_collator(item_embeddings: np.ndarray):
    global _embedding_table, _emb_dim
    _embedding_table = torch.tensor(item_embeddings, dtype=torch.float32)
    _emb_dim = item_embeddings.shape[1]
    return collate_fn
