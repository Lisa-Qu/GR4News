"""Beam search over semantic code tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class SemanticCodeDecoder(Protocol):
    """Decoder interface needed by beam search."""

    bos_token_id: int
    config: object

    def next_token_log_probs(
        self,
        *,
        user_state: torch.Tensor,
        prefix_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return log-probabilities for the next token."""


@dataclass(frozen=True)
class BeamSearchResult:
    """One complete semantic code candidate returned by beam search."""

    code: tuple[int, ...]
    score: float


class SemanticCodeBeamSearch:
    """Beam search for semantic-code decoders.

    The first implementation targets serving-time inference where batch sizes
    are small. It loops over samples explicitly to keep the logic simple and
    debuggable.
    """

    def __init__(self, decoder: SemanticCodeDecoder) -> None:
        self._decoder = decoder

    @torch.no_grad()
    def search(
        self,
        user_state: torch.Tensor,
        *,
        top_k: int,
        beam_width: int | None = None,
    ) -> list[list[BeamSearchResult]]:
        """Return top semantic-code candidates for each sample in the batch."""

        if user_state.ndim != 2:
            raise ValueError("user_state must have shape [batch, hidden_dim]")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        actual_beam_width = max(top_k, beam_width or top_k)
        results: list[list[BeamSearchResult]] = []
        for sample_index in range(user_state.shape[0]):
            sample_state = user_state[sample_index : sample_index + 1]
            results.append(
                self._search_single(
                    sample_state,
                    top_k=top_k,
                    beam_width=actual_beam_width,
                )
            )
        return results

    def _search_single(
        self,
        user_state: torch.Tensor,
        *,
        top_k: int,
        beam_width: int,
    ) -> list[BeamSearchResult]:
        device = user_state.device
        beams: list[tuple[list[int], float]] = [([self._decoder.bos_token_id], 0.0)]
        token_topk = min(beam_width, self._decoder.config.codebook_size)

        for _step in range(self._decoder.config.code_length):
            expanded: list[tuple[list[int], float]] = []
            for prefix, score in beams:
                prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
                step_log_probs = self._decoder.next_token_log_probs(
                    user_state=user_state,
                    prefix_tokens=prefix_tensor,
                )
                top_log_probs, top_tokens = torch.topk(step_log_probs[0], k=token_topk)
                for token_score, token_id in zip(top_log_probs.tolist(), top_tokens.tolist(), strict=True):
                    expanded.append((prefix + [int(token_id)], score + float(token_score)))

            expanded.sort(key=lambda item: item[1], reverse=True)
            beams = expanded[:beam_width]

        final_results: list[BeamSearchResult] = []
        for prefix, score in beams[:top_k]:
            final_results.append(
                BeamSearchResult(
                    code=tuple(prefix[1:]),
                    score=score / max(1, self._decoder.config.code_length),
                )
            )
        return final_results
