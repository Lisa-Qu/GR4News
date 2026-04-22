"""Item encoder used to build the first version of semantic IDs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

import numpy as np

from mind_genrec.data import NewsItem

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
EncoderType = Literal["hashing"]


@dataclass(frozen=True)
class ItemEncoderConfig:
    """Configuration for the hashing-based item encoder."""

    embedding_dim: int = 256
    title_weight: float = 1.0
    abstract_weight: float = 0.5
    category_weight: float = 2.0
    subcategory_weight: float = 2.0
    use_bias_term: bool = True


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


def build_item_encoder(
    *,
    encoder_type: EncoderType,
    config: ItemEncoderConfig | None = None,
) -> ItemEncoder:
    """Build the configured item-content encoder.

    The first implementation only supports `hashing`. Keeping this factory now
    makes the future switch to stronger `embedding layer` backends explicit.
    """

    if encoder_type == "hashing":
        return HashingItemEncoder(config)
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")
