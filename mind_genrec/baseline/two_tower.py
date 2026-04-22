"""Two-tower baseline model and serving retriever."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mind_genrec.baseline.ann_index import build_ann_index
from mind_genrec.model.user_encoder import HistorySequenceEncoder, UserEncoderConfig


@dataclass(frozen=True)
class BaselineCandidate:
    """One baseline retrieval candidate."""

    news_id: str
    score: float


class TwoTowerRetriever(Protocol):
    """Serving-facing interface for the baseline retriever."""

    @property
    def model_name(self) -> str:
        """Human-readable active baseline name."""

    @property
    def is_placeholder(self) -> bool:
        """Whether the current implementation is only a placeholder."""

    def retrieve(self, history: list[str], top_k: int) -> list[BaselineCandidate]:
        """Return baseline top-k candidates."""


@dataclass(frozen=True)
class TwoTowerConfig:
    """Configuration for the first two-tower baseline."""

    input_embedding_dim: int
    hidden_dim: int = 256
    output_dim: int = 128
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    max_history_length: int = 50
    temperature: float = 0.07


class TwoTowerModel(nn.Module):
    """User tower plus item tower trained with in-batch contrastive loss."""

    def __init__(self, config: TwoTowerConfig) -> None:
        super().__init__()
        self.config = config
        self.user_encoder = HistorySequenceEncoder(
            UserEncoderConfig(
                input_dim=config.input_embedding_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                dropout=config.dropout,
                max_history_length=config.max_history_length,
            )
        )
        self.user_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        self.item_projection = nn.Sequential(
            nn.Linear(config.input_embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def encode_history(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one batch of histories into normalized user embeddings."""

        user_state, _ = self.user_encoder(history_embeddings, history_mask)
        return F.normalize(self.user_projection(user_state), dim=-1)

    def encode_items(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode raw item embeddings into normalized item embeddings."""

        return F.normalize(self.item_projection(item_embeddings), dim=-1)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        target_item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return in-batch similarity matrix."""

        user_vectors = self.encode_history(history_embeddings, history_mask)
        item_vectors = self.encode_items(target_item_embeddings)
        logits = torch.matmul(user_vectors, item_vectors.transpose(0, 1))
        return logits / self.config.temperature

    @staticmethod
    def compute_loss(logits: torch.Tensor) -> torch.Tensor:
        """Symmetric in-batch contrastive loss."""

        if logits.ndim != 2:
            raise ValueError("two-tower logits must have shape [batch, batch]")
        labels = torch.arange(logits.shape[0], device=logits.device)
        user_to_item = F.cross_entropy(logits, labels)
        item_to_user = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (user_to_item + item_to_user)


class CheckpointedTwoTowerRetriever:
    """Serving-time baseline retriever backed by a trained checkpoint."""

    def __init__(
        self,
        *,
        model: TwoTowerModel,
        item_embeddings: np.ndarray,
        item_ids: list[str],
        device: torch.device,
        model_name: str = "two-tower-baseline",
        encoding_batch_size: int = 4096,
    ) -> None:
        self._model = model.eval()
        self._device = device
        self._model_name = model_name
        self._item_embedding_table = torch.tensor(item_embeddings, dtype=torch.float32, device=device)
        self._item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
        self._index = self._build_index(
            item_ids=item_ids,
            batch_size=encoding_batch_size,
        )

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str | Path,
        semantic_artifact_dir: str | Path,
        device: str | torch.device = "cpu",
        model_name: str = "two-tower-baseline",
        encoding_batch_size: int = 4096,
    ) -> "CheckpointedTwoTowerRetriever":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = TwoTowerConfig(**checkpoint["model_config"])
        model = TwoTowerModel(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        artifact_dir = Path(semantic_artifact_dir)
        item_embeddings = np.load(artifact_dir / "item_embeddings.npy")
        item_ids = json.loads((artifact_dir / "item_ids.json").read_text(encoding="utf-8"))

        return cls(
            model=model,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            device=torch.device(device),
            model_name=model_name,
            encoding_batch_size=encoding_batch_size,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_placeholder(self) -> bool:
        return False

    def retrieve(self, history: list[str], top_k: int) -> list[BaselineCandidate]:
        history_indices = [
            self._item_to_index[item_id]
            for item_id in history
            if item_id in self._item_to_index
        ]
        history_indices = history_indices[-self._model.config.max_history_length :]
        if not history_indices:
            return []

        history_embeddings = self._item_embedding_table[history_indices].unsqueeze(0)
        history_mask = torch.ones(
            1,
            len(history_indices),
            dtype=torch.bool,
            device=self._device,
        )
        with torch.no_grad():
            user_vector = self._model.encode_history(history_embeddings, history_mask)[0]
        results = self._index.search(user_vector, top_k=top_k)
        return [
            BaselineCandidate(news_id=result.item_id, score=result.score)
            for result in results
        ]

    def _build_index(
        self,
        *,
        item_ids: list[str],
        batch_size: int,
    ) -> object:
        encoded_batches: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, self._item_embedding_table.shape[0], batch_size):
                stop = min(start + batch_size, self._item_embedding_table.shape[0])
                encoded_batches.append(self._model.encode_items(self._item_embedding_table[start:stop]))
        item_vectors = torch.cat(encoded_batches, dim=0)
        return build_ann_index(
            item_ids=item_ids,
            item_vectors=item_vectors,
            device=self._device,
        )


class StubTwoTowerRetriever:
    """Placeholder baseline retriever."""

    def __init__(self, model_name: str = "stub-two-tower") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_placeholder(self) -> bool:
        return True

    def retrieve(self, history: list[str], top_k: int) -> list[BaselineCandidate]:
        _ = history
        _ = top_k
        return []
