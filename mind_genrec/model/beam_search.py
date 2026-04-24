"""Beam search over semantic code tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from mind_genrec.model.code_trie import CodeTrie


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

    When a CodeTrie is provided, only valid code prefixes are expanded at
    each step — guaranteeing every result maps to at least one real item.
    Without a trie, falls back to unconstrained top-k expansion.
    """

    def __init__(
        self,
        decoder: SemanticCodeDecoder,
        trie: CodeTrie | None = None,
    ) -> None:
        self._decoder = decoder
        self._trie = trie

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
                # prefix[1:] strips the BOS token to get the real code prefix
                code_prefix = tuple(prefix[1:])
                valid_tokens = (
                    self._trie.valid_next_tokens(code_prefix)
                    if self._trie is not None
                    else None
                )
                # Skip dead-end prefixes (trie says no valid continuations)
                if valid_tokens is not None and len(valid_tokens) == 0:
                    continue

                prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
                step_log_probs = self._decoder.next_token_log_probs(
                    user_state=user_state,
                    prefix_tokens=prefix_tensor,
                )

                if valid_tokens is not None:
                    # Mask to trie-allowed tokens only
                    mask = torch.full(
                        step_log_probs.shape, float("-inf"), device=device
                    )
                    valid_idx = torch.tensor(valid_tokens, dtype=torch.long, device=device)
                    mask[0].scatter_(0, valid_idx, step_log_probs[0][valid_idx])
                    step_log_probs = mask

                k = min(token_topk, int((step_log_probs[0] > float("-inf")).sum()))
                if k == 0:
                    continue
                top_log_probs, top_tokens = torch.topk(step_log_probs[0], k=k)
                for token_score, token_id in zip(
                    top_log_probs.tolist(), top_tokens.tolist(), strict=True
                ):
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
