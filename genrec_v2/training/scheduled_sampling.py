"""Scheduled Sampling training with prefix-confidence-gated loss.

Core idea: during training, probabilistically replace ground-truth prefix tokens
with the model's own predictions. When the prefix is likely wrong (low confidence
in the predicted token), the loss relaxes the CE constraint and adds entropy
regularization, teaching the model to express uncertainty rather than blindly
outputting high probability on wrong continuations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def scheduled_sampling_forward(
    model,
    history_emb: torch.Tensor,
    history_mask: torch.Tensor,
    target_code: torch.Tensor,
    *,
    ss_prob: float = 0.3,
    ce_floor: float = 0.3,
) -> dict[str, torch.Tensor]:
    """One training forward pass with scheduled sampling + gated loss.

    For each token position t:
      1. Decide whether to use GT token or model argmax as prefix.
      2. Compute CE and entropy for the current prediction.
      3. alpha = 1 - confidence_in_previous_prefix_token.
      4. loss[t] = (1-alpha)*CE + alpha*(ce_floor*CE + (1-ce_floor)*entropy).

    When alpha is large (prefix likely wrong): CE is suppressed, entropy is pushed.
    When alpha is small (prefix likely correct): standard CE training.
    ce_floor ensures self-attention gradient never fully vanishes.

    Args:
        model: GenRecV2Model instance.
        history_emb: [B, L, d] user history embeddings.
        history_mask: [B, L] attention mask.
        target_code: [B, code_len] ground-truth semantic codes.
        ss_prob: probability of using model prediction instead of GT.
        ce_floor: minimum CE weight in interpolated loss.

    Returns:
        dict with loss_gen, loss_code, logits, and diagnostic metrics.
    """
    B, code_len = target_code.shape
    device = history_emb.device
    hidden_dim = model.decoder.config.hidden_dim

    # Shared encoder forward (once)
    user_state, _ = model.encoder(history_emb, history_mask)
    if model.hot_news is not None:
        user_state = model.hot_news(user_state)
    memory = user_state.unsqueeze(1)

    # Accumulate step-by-step
    bos_id = model.decoder.bos_token_id
    prev_tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    prev_confidence = torch.ones(B, device=device)  # BOS is always "correct"

    all_ce: list[torch.Tensor] = []
    all_entropy: list[torch.Tensor] = []
    all_alpha: list[torch.Tensor] = []

    for t in range(code_len):
        # Embed current prefix
        x = model.decoder.token_embedding(prev_tokens)
        positions = torch.arange(t + 1, device=device)
        x = x + model.decoder.position_embedding(positions).unsqueeze(0)
        tgt_mask = model.decoder._build_causal_mask(t + 1, device)

        decoded = model.decoder.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)
        decoded = model.decoder.output_norm(decoded)
        logits_t = model.decoder.output_projection(decoded[:, -1, :])

        probs_t = F.softmax(logits_t, dim=-1)
        gt_t = target_code[:, t]
        loss_ce_t = F.cross_entropy(logits_t, gt_t, reduction="none")
        loss_ent_t = -(probs_t * F.log_softmax(logits_t, dim=-1)).sum(dim=-1)

        # alpha from PREVIOUS step's confidence in the token we used
        alpha_t = (1.0 - prev_confidence).detach()

        # Interpolated per-sample loss
        loss_t = (1.0 - alpha_t) * loss_ce_t + alpha_t * (
            ce_floor * loss_ce_t + (1.0 - ce_floor) * loss_ent_t
        )

        all_ce.append(loss_ce_t.mean())
        all_entropy.append(loss_ent_t.mean())
        all_alpha.append(alpha_t.mean())

        # Decide next prefix: coin flip per sample
        use_gt = torch.rand(B, device=device) > ss_prob
        gt_token_t = gt_t.unsqueeze(1)
        pred_token_t = probs_t.argmax(dim=-1, keepdim=True)
        next_token = torch.where(use_gt.unsqueeze(1), gt_token_t, pred_token_t)

        # Confidence in whatever token we used as prefix
        chosen_conf = torch.where(
            use_gt,
            probs_t[torch.arange(B), gt_t],
            probs_t.max(dim=-1).values,
        )
        prev_confidence = chosen_conf.detach()
        prev_tokens = torch.cat([prev_tokens, next_token], dim=1)

    # Aggregate
    total_ce = sum(all_ce) / code_len
    total_entropy = sum(all_entropy) / code_len
    avg_alpha = sum(all_alpha) / code_len

    # Also compute standard CE for reference
    with torch.no_grad():
        logits_full = model.decoder(user_state, target_code)
        loss_gen_std = F.cross_entropy(
            logits_full.reshape(-1, logits_full.shape[-1]),
            target_code.reshape(-1),
        )

    # Interpolated loss
    loss_gen_interp = sum(
        (1.0 - a.detach()) * ce + a.detach() * (ce_floor * ce + (1.0 - ce_floor) * ent)
        for a, ce, ent in zip(all_alpha, all_ce, all_entropy)
    ) / code_len

    # Codebook loss (unchanged)
    loss_code = torch.tensor(0.0, device=device)

    return {
        "logits": logits_full,
        "loss_gen": loss_gen_interp,
        "loss_code": loss_code,
        "ss_prob": ss_prob,
        "ss_avg_ce": total_ce.item(),
        "ss_avg_entropy": total_entropy.item(),
        "ss_avg_alpha": avg_alpha.item(),
        "ss_ref_ce": loss_gen_std.item(),
    }
