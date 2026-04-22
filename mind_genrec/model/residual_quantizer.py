"""Residual quantization used to assign semantic code sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualQuantizerConfig:
    """Configuration for residual quantization."""

    num_codebooks: int = 4
    codebook_size: int = 256
    max_iterations: int = 20
    batch_size: int = 2048
    sample_size: int = 20000
    seed: int = 7


class ResidualQuantizer:
    """A lightweight residual quantizer for semantic ID training."""

    def __init__(
        self,
        config: ResidualQuantizerConfig | None = None,
        *,
        codebooks: list[np.ndarray] | None = None,
    ) -> None:
        self._config = config or ResidualQuantizerConfig()
        self._codebooks = list(codebooks or [])

    @property
    def config(self) -> ResidualQuantizerConfig:
        return self._config

    @property
    def codebooks(self) -> list[np.ndarray]:
        return self._codebooks

    @property
    def is_fitted(self) -> bool:
        return len(self._codebooks) == self._config.num_codebooks

    def fit(self, vectors: np.ndarray) -> np.ndarray:
        """Fit codebooks stage by stage and return the assigned codes."""

        matrix = self._validate_vectors(vectors)
        if matrix.shape[0] == 0:
            raise ValueError("ResidualQuantizer.fit requires at least one vector")

        rng = np.random.default_rng(self._config.seed)
        residual = matrix.copy()
        reconstruction = np.zeros_like(matrix)
        codes = np.zeros((matrix.shape[0], self._config.num_codebooks), dtype=np.int32)
        self._codebooks = []

        for level in range(self._config.num_codebooks):
            fit_vectors = self._subsample(residual, rng)
            centers = self._run_kmeans(fit_vectors, rng=np.random.default_rng(self._config.seed + level))
            level_codes = self._assign_codes(residual, centers)
            self._codebooks.append(centers.astype(np.float32, copy=False))
            codes[:, level] = level_codes
            reconstruction += centers[level_codes]
            residual = matrix - reconstruction

        return codes

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Assign codes using already fitted codebooks."""

        if not self.is_fitted:
            raise RuntimeError("ResidualQuantizer.encode requires fitted codebooks")
        matrix = self._validate_vectors(vectors)
        residual = matrix.copy()
        reconstruction = np.zeros_like(matrix)
        codes = np.zeros((matrix.shape[0], self._config.num_codebooks), dtype=np.int32)

        for level, centers in enumerate(self._codebooks):
            level_codes = self._assign_codes(residual, centers)
            codes[:, level] = level_codes
            reconstruction += centers[level_codes]
            residual = matrix - reconstruction
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate vectors from code indices."""

        if not self.is_fitted:
            raise RuntimeError("ResidualQuantizer.decode requires fitted codebooks")

        indices = np.asarray(codes, dtype=np.int32)
        if indices.ndim != 2 or indices.shape[1] != self._config.num_codebooks:
            raise ValueError("codes must have shape [n_items, num_codebooks]")

        dim = self._codebooks[0].shape[1]
        reconstruction = np.zeros((indices.shape[0], dim), dtype=np.float32)
        for level, centers in enumerate(self._codebooks):
            reconstruction += centers[indices[:, level]]
        return reconstruction

    def _validate_vectors(self, vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must have shape [n_items, dim]")
        return matrix

    def _subsample(self, vectors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self._config.sample_size <= 0 or vectors.shape[0] <= self._config.sample_size:
            return vectors
        indices = rng.choice(vectors.shape[0], size=self._config.sample_size, replace=False)
        return vectors[indices]

    def _run_kmeans(self, vectors: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        if vectors.shape[0] == 0:
            raise ValueError("k-means requires at least one vector")

        cluster_count = min(self._config.codebook_size, vectors.shape[0])
        init_indices = rng.choice(vectors.shape[0], size=cluster_count, replace=False)
        centers = vectors[init_indices].astype(np.float32, copy=True)

        if cluster_count < self._config.codebook_size:
            pad_count = self._config.codebook_size - cluster_count
            pad = np.repeat(centers[:1], pad_count, axis=0)
            centers = np.concatenate([centers, pad], axis=0)

        for _ in range(self._config.max_iterations):
            assignments = self._assign_codes(vectors, centers)
            new_centers = np.zeros_like(centers)
            counts = np.zeros(centers.shape[0], dtype=np.int32)
            for index, cluster in enumerate(assignments):
                new_centers[cluster] += vectors[index]
                counts[cluster] += 1

            empty_clusters = np.where(counts == 0)[0]
            for cluster in empty_clusters:
                replacement = vectors[rng.integers(0, vectors.shape[0])]
                new_centers[cluster] = replacement
                counts[cluster] = 1

            new_centers /= counts[:, None]
            if np.allclose(centers, new_centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers

        return centers

    def _assign_codes(self, vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
        batch_size = max(1, self._config.batch_size)
        assignments = np.zeros(vectors.shape[0], dtype=np.int32)
        center_norms = np.sum(centers * centers, axis=1)
        for start in range(0, vectors.shape[0], batch_size):
            end = min(start + batch_size, vectors.shape[0])
            batch = vectors[start:end]
            batch_norms = np.sum(batch * batch, axis=1, keepdims=True)
            distances = batch_norms + center_norms[None, :] - 2.0 * (batch @ centers.T)
            assignments[start:end] = np.argmin(distances, axis=1)
        return assignments

