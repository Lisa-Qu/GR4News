"""Late-inject autoregressive decoder over semantic code tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LazyARDecoderConfig:
    """Configuration for late-inject autoregressive decoding."""

    hidden_dim: int = 256
    codebook_size: int = 256
    code_length: int = 4
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    parallel_layers: int | None = None


class LazyAutoregressiveDecoder(nn.Module):
    """LazyAR decoder with parallel shared layers and late token injection.

    The first `parallel_layers` process positional states conditioned only on the
    user memory. Previous semantic-token embeddings are injected afterwards, and
    the remaining layers decode autoregressively.
    """

    def __init__(self, config: LazyARDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.bos_token_id = config.codebook_size

        if config.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        default_parallel_layers = max(0, min(config.num_layers - 1, (2 * config.num_layers) // 3))
        parallel_layers = config.parallel_layers
        if parallel_layers is None:
            parallel_layers = default_parallel_layers
        parallel_layers = min(max(0, parallel_layers), max(0, config.num_layers - 1))
        self.parallel_layer_count = parallel_layers
        self.autoregressive_layer_count = config.num_layers - parallel_layers

        self.token_embedding = nn.Embedding(config.codebook_size + 1, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.code_length, config.hidden_dim)

        self.parallel_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    dropout=config.dropout,
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(self.parallel_layer_count)
            ]
        )
        self.autoregressive_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    dropout=config.dropout,
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(self.autoregressive_layer_count)
            ]
        )
        self.fusion_gate = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.fusion_value = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.codebook_size)

    def forward(self, user_state: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced decode with late token injection."""

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
        return self._decode(user_state=user_state, prefix_tokens=decoder_input)

    @torch.no_grad()
    def next_token_log_probs(
        self,
        *,
        user_state: torch.Tensor,
        prefix_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return log-probabilities for the next token."""

        if user_state.ndim != 2:
            raise ValueError("user_state must have shape [batch, hidden_dim]")
        if prefix_tokens.ndim != 2:
            raise ValueError("prefix_tokens must have shape [batch, prefix_len]")
        if prefix_tokens.shape[0] != user_state.shape[0]:
            raise ValueError("prefix_tokens and user_state must share batch size")

        decoded = self._decode(user_state=user_state, prefix_tokens=prefix_tokens)
        step_logits = decoded[:, -1, :]
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
        prefix = torch.full(
            (batch_size, 1),
            fill_value=self.bos_token_id,
            dtype=torch.long,
            device=user_state.device,
        )
        outputs: list[torch.Tensor] = []
        log_scores: list[torch.Tensor] = []

        for _ in range(self.config.code_length):
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

    def _decode(self, *, user_state: torch.Tensor, prefix_tokens: torch.Tensor) -> torch.Tensor:
        seq_len = prefix_tokens.shape[1]
        if seq_len <= 0 or seq_len > self.config.code_length:
            raise ValueError("prefix length must be in [1, code_length]")

        shared_states = self._build_parallel_states(user_state=user_state, seq_len=seq_len)
        token_states = self._embed_prefix_tokens(prefix_tokens)
        fused_states = self._late_inject(shared_states, token_states)
        autoregressive_states = self._run_autoregressive_layers(
            hidden_states=fused_states,
            user_state=user_state,
        )
        autoregressive_states = self.output_norm(autoregressive_states)
        return self.output_projection(autoregressive_states)

    def _build_parallel_states(self, *, user_state: torch.Tensor, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, device=user_state.device)
        hidden_states = self.position_embedding(positions).unsqueeze(0).expand(user_state.shape[0], -1, -1)
        memory = user_state.unsqueeze(1)
        for layer in self.parallel_layers:
            hidden_states = layer(tgt=hidden_states, memory=memory)
        return hidden_states

    def _embed_prefix_tokens(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        seq_len = prefix_tokens.shape[1]
        positions = torch.arange(seq_len, device=prefix_tokens.device)
        return self.token_embedding(prefix_tokens) + self.position_embedding(positions).unsqueeze(0)

    def _late_inject(self, shared_states: torch.Tensor, token_states: torch.Tensor) -> torch.Tensor:
        fused_features = torch.cat(
            [shared_states, token_states, shared_states * token_states],
            dim=-1,
        )
        gate = torch.sigmoid(self.fusion_gate(fused_features))
        value = torch.tanh(self.fusion_value(fused_features))
        return shared_states + gate * value

    def _run_autoregressive_layers(
        self,
        *,
        hidden_states: torch.Tensor,
        user_state: torch.Tensor,
    ) -> torch.Tensor:
        memory = user_state.unsqueeze(1)
        causal_mask = self._build_causal_mask(hidden_states.shape[1], hidden_states.device)
        for layer in self.autoregressive_layers:
            hidden_states = layer(
                tgt=hidden_states,
                memory=memory,
                tgt_mask=causal_mask,
            )
        return hidden_states

    @staticmethod
    def _build_causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((length, length), float("-inf"), device=device),
            diagonal=1,
        )
