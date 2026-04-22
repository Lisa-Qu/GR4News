"""User-history encoder for MIND generative recommendation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class UserEncoderConfig:
    """Configuration for the clicked-history encoder."""

    input_dim: int
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    max_history_length: int = 50


class HistorySequenceEncoder(nn.Module):
    """Encode clicked history into one user state."""

    def __init__(self, config: UserEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_history_length, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode padded history embeddings.

        Args:
            history_embeddings: `[batch, seq_len, input_dim]`
            history_mask: `[batch, seq_len]`, `True` on valid positions
        Returns:
            user_state: `[batch, hidden_dim]`
            encoded_history: `[batch, seq_len, hidden_dim]`
        """

        if history_embeddings.ndim != 3:
            raise ValueError("history_embeddings must have shape [batch, seq_len, input_dim]")
        if history_mask.ndim != 2:
            raise ValueError("history_mask must have shape [batch, seq_len]")
        if history_embeddings.shape[:2] != history_mask.shape:
            raise ValueError("history_embeddings and history_mask must agree on batch and seq_len")

        batch_size, seq_len, _ = history_embeddings.shape
        if seq_len == 0:
            raise ValueError("history_embeddings must contain at least one timestep")
        if seq_len > self.config.max_history_length:
            history_embeddings = history_embeddings[:, -self.config.max_history_length :, :]
            history_mask = history_mask[:, -self.config.max_history_length :]
            seq_len = self.config.max_history_length

        x = self.input_projection(history_embeddings)
        position_ids = torch.arange(seq_len, device=x.device)
        x = x + self.position_embedding(position_ids).unsqueeze(0)

        padding_mask = ~history_mask.bool()
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        encoded = self.output_norm(encoded)

        weights = history_mask.unsqueeze(-1).to(encoded.dtype)
        masked = encoded * weights
        denom = weights.sum(dim=1).clamp_min(1.0)
        user_state = masked.sum(dim=1) / denom

        return user_state, encoded
