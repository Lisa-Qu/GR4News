"""Autoregressive decoder over semantic code tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ARDecoderConfig:
    """Configuration for semantic-code autoregressive decoding."""

    hidden_dim: int = 256
    codebook_size: int = 256
    code_length: int = 4
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1


class CodeAutoregressiveDecoder(nn.Module):
    """Decode semantic codes token by token from a user state."""

    def __init__(self, config: ARDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.bos_token_id = config.codebook_size
        self.token_embedding = nn.Embedding(config.codebook_size + 1, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.code_length, config.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.codebook_size)

    def forward(self, user_state: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced decode.

        Args:
            user_state: `[batch, hidden_dim]`
            target_codes: `[batch, code_length]`
        Returns:
            logits: `[batch, code_length, codebook_size]`
        """

        if user_state.ndim != 2:
            raise ValueError("user_state must have shape [batch, hidden_dim]")
        if target_codes.ndim != 2:
            raise ValueError("target_codes must have shape [batch, code_length]")
        if target_codes.shape[1] != self.config.code_length:
            raise ValueError("target_codes second dimension must equal code_length")

        batch_size = target_codes.shape[0]
        bos = torch.full(
            (batch_size, 1),
            fill_value=self.bos_token_id,
            dtype=torch.long,
            device=target_codes.device,
        )
        decoder_input = torch.cat([bos, target_codes[:, :-1]], dim=1)
        x = self.token_embedding(decoder_input)
        positions = torch.arange(self.config.code_length, device=target_codes.device)
        x = x + self.position_embedding(positions).unsqueeze(0)
        memory = user_state.unsqueeze(1)
        decoded = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self._build_causal_mask(self.config.code_length, x.device),
        )
        decoded = self.output_norm(decoded)
        return self.output_projection(decoded)

    @torch.no_grad()
    def next_token_log_probs(
        self,
        *,
        user_state: torch.Tensor,
        prefix_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return log-probabilities for the next token.

        Args:
            user_state: `[batch, hidden_dim]`
            prefix_tokens: `[batch, prefix_len]`, including `BOS`
        Returns:
            log_probs: `[batch, codebook_size]`
        """

        if user_state.ndim != 2:
            raise ValueError("user_state must have shape [batch, hidden_dim]")
        if prefix_tokens.ndim != 2:
            raise ValueError("prefix_tokens must have shape [batch, prefix_len]")
        if prefix_tokens.shape[0] != user_state.shape[0]:
            raise ValueError("prefix_tokens and user_state must share batch size")

        x = self.token_embedding(prefix_tokens)
        positions = torch.arange(prefix_tokens.shape[1], device=prefix_tokens.device)
        x = x + self.position_embedding(positions).unsqueeze(0)
        memory = user_state.unsqueeze(1)
        decoded = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self._build_causal_mask(prefix_tokens.shape[1], prefix_tokens.device),
        )
        decoded = self.output_norm(decoded)
        step_logits = self.output_projection(decoded[:, -1, :])
        return torch.log_softmax(step_logits, dim=-1)

    @torch.no_grad()
    def greedy_decode(self, user_state: torch.Tensor) -> torch.Tensor:
        """Greedy decode semantic codes from the user state."""

        codes, _ = self.greedy_decode_with_scores(user_state)
        return codes

    @torch.no_grad()
    def greedy_decode_with_scores(self, user_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy decode semantic codes and return average log-probability scores."""

        if user_state.ndim != 2:
            raise ValueError("user_state must have shape [batch, hidden_dim]")

        batch_size = user_state.shape[0]
        memory = user_state.unsqueeze(1)
        prefix = torch.full(
            (batch_size, 1),
            fill_value=self.bos_token_id,
            dtype=torch.long,
            device=user_state.device,
        )
        outputs: list[torch.Tensor] = []
        log_scores: list[torch.Tensor] = []

        for step in range(self.config.code_length):
            _ = step
            step_log_probs = self.next_token_log_probs(
                user_state=user_state,
                prefix_tokens=prefix,
            )
            next_token = step_log_probs.argmax(dim=-1, keepdim=True)
            prefix = torch.cat([prefix, next_token], dim=1)
            outputs.append(next_token)
            log_scores.append(step_log_probs.gather(-1, next_token))

        stacked_codes = torch.cat(outputs, dim=1)
        stacked_log_scores = torch.cat(log_scores, dim=1).mean(dim=1)
        return stacked_codes, stacked_log_scores

    @staticmethod
    def _build_causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((length, length), float("-inf"), device=device),
            diagonal=1,
        )
