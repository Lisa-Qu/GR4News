"""GenRec-V2 model: HistoryEncoder → (optional HotNews) → ARDecoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec_v2.model.hot_news import HotNewsFusion


class GenRecV2Model(nn.Module):
    """End-to-end generative retrieval model with optional hot-news fusion."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        codebook: nn.ModuleList | None = None,
        hot_news_fusion: HotNewsFusion | None = None,
        embedding_table: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook = codebook  # list of nn.Embedding, one per level
        self.hot_news = hot_news_fusion
        self.register_buffer("_emb_table", embedding_table if embedding_table is not None else torch.empty(0))

    def forward(
        self,
        history_emb: torch.Tensor,   # [B, L, d]
        history_mask: torch.Tensor,  # [B, L]
        target_code: torch.Tensor,   # [B, code_len]
        target_emb_idx: torch.Tensor | None = None,  # [B]
    ) -> dict[str, torch.Tensor]:
        B = history_emb.shape[0]

        user_state, _ = self.encoder(history_emb, history_mask)  # [B, d]

        if self.hot_news is not None:
            user_state = self.hot_news(user_state)

        # Decoder: teacher-forced logits
        logits = self.decoder(user_state, target_code)  # [B, code_len, vocab_size]

        loss_gen = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_code.reshape(-1),
        )

        loss_code = torch.tensor(0.0, device=logits.device)
        if self.codebook is not None and target_emb_idx is not None:
            # Multi-level codebook decode: sum over code levels
            recon = torch.zeros(B, self.encoder.config.hidden_dim, device=logits.device)
            # Actually, decode to embedding dim
            recon = torch.zeros(B, self.codebook[0].weight.shape[1], device=logits.device)
            for level, cb in enumerate(self.codebook):
                # STE: use target codes to lookup, gradient flows through
                quantized = cb(target_code[:, level])  # [B, dim]
                recon = recon + quantized

            target_emb = self._emb_table[target_emb_idx]  # [B, dim]
            loss_code = F.mse_loss(recon, target_emb)

        return {
            "logits": logits,
            "loss_gen": loss_gen,
            "loss_code": loss_code,
        }

    @torch.no_grad()
    def greedy_decode(self, history_emb: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        user_state, _ = self.encoder(history_emb, history_mask)
        if self.hot_news is not None:
            user_state = self.hot_news(user_state)
        return self.decoder.greedy_decode(user_state)

    @torch.no_grad()
    def greedy_decode_with_hidden(
        self, history_emb: torch.Tensor, history_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Greedy decode + return per-step hidden states for calibration scorer.

        Returns:
            {"codes": [B, code_len], "hidden_states": [B, code_len, hidden_dim]}
        """
        user_state, _ = self.encoder(history_emb, history_mask)
        if self.hot_news is not None:
            user_state = self.hot_news(user_state)
        codes = self.decoder.greedy_decode(user_state)
        hidden = self.decoder.get_hidden_states(user_state, codes)
        return {"codes": codes, "hidden_states": hidden}

    @torch.no_grad()
    def beam_search(
        self,
        history_emb: torch.Tensor,
        history_mask: torch.Tensor,
        beam_width: int,
    ) -> list[list[tuple[list[int], float]]]:
        user_state, _ = self.encoder(history_emb, history_mask)
        if self.hot_news is not None:
            user_state = self.hot_news(user_state)

        # Simple beam search over codes
        batch_size = user_state.shape[0]
        all_results: list[list[tuple[list[int], float]]] = []

        for b in range(batch_size):
            us = user_state[b:b + 1]  # [1, d]
            beam = _beam_search_single(self.decoder, us, beam_width, self.decoder.config.code_length)
            all_results.append(beam)

        return all_results


def _beam_search_single(decoder, user_state, beam_width, code_length):
    """Generate top-k code sequences via beam search."""
    device = user_state.device
    vocab_size = decoder.config.codebook_size
    beams: list[tuple[list[int], float]] = [([], 0.0)]

    for step in range(code_length):
        candidates: list[tuple[list[int], float]] = []
        for code_seq, log_prob in beams:
            if step == 0:
                # First step: decoder takes user_state as input
                logits = decoder.predict_step(user_state, step_idx=0, prev_tokens=None)
            else:
                prev = torch.tensor([code_seq], dtype=torch.long, device=device)
                logits = decoder.predict_step(user_state, step_idx=step, prev_tokens=prev)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [vocab_size]
            topk = log_probs.topk(beam_width)
            for k in range(beam_width):
                candidates.append((code_seq + [int(topk.indices[k])], log_prob + float(topk.values[k])))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam_width]

    return beams
