"""Item encoder used to build semantic IDs from news content."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

import numpy as np

from mind_genrec.data import NewsItem

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
EncoderType = Literal["hashing", "sbert"]


@dataclass(frozen=True)
class ItemEncoderConfig:
    """Configuration for item encoders."""

    embedding_dim: int = 256
    title_weight: float = 1.0
    abstract_weight: float = 0.5
    category_weight: float = 2.0
    subcategory_weight: float = 2.0
    use_bias_term: bool = True
    # sbert-specific
    sbert_model_name: str = "BAAI/bge-small-en-v1.5"
    sbert_batch_size: int = 256


class ItemEncoder(Protocol):
    """Interface for item-content encoders used by semantic ID training."""

    @property
    def config(self) -> ItemEncoderConfig:
        """Encoder configuration."""

    def encode_item(self, item: NewsItem) -> np.ndarray:
        """Encode one item."""

    def encode_items(self, items: Iterable[NewsItem]) -> np.ndarray:
        """Encode multiple items."""


class HashingItemEncoder:
    """Deterministic encoder for MIND news items.

    This is intentionally lightweight. It does not try to be the final
    production encoder; it provides a stable first-stage representation so the
    `semantic ID` pipeline can be trained and exported end-to-end.
    """

    def __init__(self, config: ItemEncoderConfig | None = None) -> None:
        self._config = config or ItemEncoderConfig()
        if self._config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

    @property
    def config(self) -> ItemEncoderConfig:
        return self._config

    def encode_item(self, item: NewsItem) -> np.ndarray:
        """Encode one `NewsItem` into a normalized dense vector."""

        vector = np.zeros(self._config.embedding_dim, dtype=np.float32)
        if self._config.use_bias_term:
            vector[0] = 1.0

        self._accumulate_text(
            vector,
            namespace="title",
            text=item.title,
            weight=self._config.title_weight,
        )
        self._accumulate_text(
            vector,
            namespace="abstract",
            text=item.abstract,
            weight=self._config.abstract_weight,
        )
        self._accumulate_token(
            vector,
            namespace="category",
            token=item.category.strip().lower(),
            weight=self._config.category_weight,
        )
        self._accumulate_token(
            vector,
            namespace="subcategory",
            token=item.subcategory.strip().lower(),
            weight=self._config.subcategory_weight,
        )

        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    def encode_items(self, items: Iterable[NewsItem]) -> np.ndarray:
        """Encode a sequence of items into one matrix."""

        encoded = [self.encode_item(item) for item in items]
        if not encoded:
            return np.zeros((0, self._config.embedding_dim), dtype=np.float32)
        return np.stack(encoded, axis=0)

    def _accumulate_text(
        self,
        vector: np.ndarray,
        *,
        namespace: str,
        text: str,
        weight: float,
    ) -> None:
        if not text or weight == 0.0:
            return
        for token in _TOKEN_PATTERN.findall(text.lower()):
            self._accumulate_token(vector, namespace=namespace, token=token, weight=weight)

    def _accumulate_token(
        self,
        vector: np.ndarray,
        *,
        namespace: str,
        token: str,
        weight: float,
    ) -> None:
        if not token or weight == 0.0:
            return
        payload = f"{namespace}:{token}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % self._config.embedding_dim
        sign = 1.0 if (digest[8] & 1) == 0 else -1.0
        vector[bucket] += float(weight) * sign


class SBERTItemEncoder:
    """Encode news items using a SentenceTransformer model (e.g. BGE)."""

    def __init__(self, config: ItemEncoderConfig | None = None) -> None:
        self._config = config or ItemEncoderConfig()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for sbert encoder. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self._config.sbert_model_name)
        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != self._config.embedding_dim:
            object.__setattr__(self._config, "embedding_dim", actual_dim)

    @property
    def config(self) -> ItemEncoderConfig:
        return self._config

    @staticmethod
    def _item_to_text(item: NewsItem) -> str:
        parts = []
        if item.category:
            parts.append(f"[{item.category}]")
        if item.subcategory:
            parts.append(f"[{item.subcategory}]")
        if item.title:
            parts.append(item.title)
        if item.abstract:
            parts.append(item.abstract)
        return " ".join(parts) if parts else "empty"

    def encode_item(self, item: NewsItem) -> np.ndarray:
        text = self._item_to_text(item)
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].astype(np.float32)

    def encode_items(self, items: Iterable[NewsItem]) -> np.ndarray:
        items_list = list(items)
        if not items_list:
            return np.zeros((0, self._config.embedding_dim), dtype=np.float32)
        texts = [self._item_to_text(item) for item in items_list]
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self._config.sbert_batch_size,
            show_progress_bar=True,
        )
        return embeddings.astype(np.float32)


def build_item_encoder(
    *,
    encoder_type: EncoderType,
    config: ItemEncoderConfig | None = None,
) -> ItemEncoder:
    """Build the configured item-content encoder."""

    if encoder_type == "hashing":
        return HashingItemEncoder(config)
    if encoder_type == "sbert":
        return SBERTItemEncoder(config)
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")
