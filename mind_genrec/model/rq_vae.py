"""Residual Quantization VAE with learnable codebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class RQVAEConfig:
    """Configuration for RQ-VAE."""

    num_codebooks: int = 4
    codebook_size: int = 256
    embedding_dim: int = 384
    commitment_weight: float = 0.25
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 2048
    seed: int = 7


class ResidualVectorQuantizer(nn.Module):
    """Single-level vector quantizer with straight-through estimator."""

    def __init__(self, codebook_size: int, dim: int) -> None:
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input, return (quantized, codes, codebook_loss).

        Uses straight-through estimator: gradients pass through quantization.
        """
        distances = (
            x.pow(2).sum(dim=-1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=-1)
            - 2 * x @ self.codebook.weight.T
        )
        codes = distances.argmin(dim=-1)
        quantized = self.codebook(codes)

        codebook_loss = F.mse_loss(quantized.detach(), x) + F.mse_loss(
            quantized, x.detach()
        )

        # Straight-through estimator
        quantized_st = x + (quantized - x).detach()
        return quantized_st, codes, codebook_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Assign codes without gradient."""
        distances = (
            x.pow(2).sum(dim=-1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=-1)
            - 2 * x @ self.codebook.weight.T
        )
        return distances.argmin(dim=-1)


class RQVAE(nn.Module):
    """Residual Quantization VAE — multi-stage VQ with learnable codebooks."""

    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.quantizers = nn.ModuleList(
            [
                ResidualVectorQuantizer(config.codebook_size, config.embedding_dim)
                for _ in range(config.num_codebooks)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Multi-stage residual quantization.

        Returns (reconstructed, codes [B, num_codebooks], total_loss).
        """
        residual = x
        reconstructed = torch.zeros_like(x)
        all_codes = []
        total_loss = torch.tensor(0.0, device=x.device)

        for quantizer in self.quantizers:
            quantized, codes, cb_loss = quantizer(residual)
            reconstructed = reconstructed + quantized
            residual = x - reconstructed
            all_codes.append(codes)
            total_loss = total_loss + cb_loss

        codes_tensor = torch.stack(all_codes, dim=-1)  # [B, num_codebooks]
        return reconstructed, codes_tensor, total_loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Assign codes without gradient. Returns [B, num_codebooks]."""
        residual = x.clone()
        reconstructed = torch.zeros_like(x)
        all_codes = []

        for quantizer in self.quantizers:
            codes = quantizer.encode(residual)
            quantized = quantizer.codebook(codes)
            reconstructed = reconstructed + quantized
            residual = x - reconstructed
            all_codes.append(codes)

        return torch.stack(all_codes, dim=-1)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct from codes [B, num_codebooks]."""
        reconstructed = torch.zeros(
            codes.shape[0], self.config.embedding_dim, device=codes.device
        )
        for i, quantizer in enumerate(self.quantizers):
            reconstructed = reconstructed + quantizer.codebook(codes[:, i])
        return reconstructed


class RQVAEQuantizer:
    """Drop-in replacement for ResidualQuantizer with RQ-VAE backend.

    Matches the same interface: fit(), encode(), decode(), codebooks property.
    """

    def __init__(self, config: RQVAEConfig) -> None:
        self._config = config
        self._model: RQVAE | None = None
        self._device = torch.device("cpu")

    @property
    def config(self) -> RQVAEConfig:
        return self._config

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def codebooks(self) -> list[np.ndarray]:
        """Export codebooks as numpy arrays for downstream compatibility."""
        if self._model is None:
            raise RuntimeError("RQVAEQuantizer not fitted yet")
        return [
            q.codebook.weight.detach().cpu().numpy()
            for q in self._model.quantizers
        ]

    def fit(
        self, vectors: np.ndarray, device: str = "auto"
    ) -> np.ndarray:
        """Train RQ-VAE on embeddings and return assigned codes."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must have shape [n_items, dim]")

        if device == "auto":
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)

        torch.manual_seed(self._config.seed)
        self._model = RQVAE(self._config).to(self._device)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._config.learning_rate
        )

        tensor_data = torch.from_numpy(matrix)
        dataset = TensorDataset(tensor_data)
        loader = DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
        )

        self._model.train()
        for epoch in range(1, self._config.epochs + 1):
            total_recon = 0.0
            total_commit = 0.0
            n_batches = 0

            for (batch,) in loader:
                batch = batch.to(self._device)
                optimizer.zero_grad(set_to_none=True)

                reconstructed, _, codebook_loss = self._model(batch)
                recon_loss = F.mse_loss(reconstructed, batch)
                loss = recon_loss + self._config.commitment_weight * codebook_loss
                loss.backward()
                optimizer.step()

                total_recon += recon_loss.item()
                total_commit += codebook_loss.item()
                n_batches += 1

            if epoch % 10 == 0 or epoch == 1:
                avg_recon = total_recon / max(1, n_batches)
                avg_commit = total_commit / max(1, n_batches)
                print(
                    f"RQ-VAE epoch {epoch}/{self._config.epochs}: "
                    f"recon={avg_recon:.6f} commit={avg_commit:.6f}"
                )

        return self.encode(vectors)

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Assign codes using trained RQ-VAE."""
        if self._model is None:
            raise RuntimeError("RQVAEQuantizer.encode requires fitted model")

        self._model.eval()
        matrix = np.asarray(vectors, dtype=np.float32)
        tensor = torch.from_numpy(matrix).to(self._device)

        all_codes = []
        for start in range(0, tensor.shape[0], self._config.batch_size):
            end = min(start + self._config.batch_size, tensor.shape[0])
            codes = self._model.encode(tensor[start:end])
            all_codes.append(codes.cpu().numpy())

        return np.concatenate(all_codes, axis=0).astype(np.int32)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate vectors from code indices."""
        if self._model is None:
            raise RuntimeError("RQVAEQuantizer.decode requires fitted model")

        self._model.eval()
        indices = np.asarray(codes, dtype=np.int64)
        tensor = torch.from_numpy(indices).to(self._device)

        all_recon = []
        for start in range(0, tensor.shape[0], self._config.batch_size):
            end = min(start + self._config.batch_size, tensor.shape[0])
            recon = self._model.decode(tensor[start:end])
            all_recon.append(recon.cpu().numpy())

        return np.concatenate(all_recon, axis=0)
