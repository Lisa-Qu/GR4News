"""Calibration scorer for generative retrieval.

Trains a lightweight MLP on frozen decoder hidden states to predict
whether a generated code sequence correctly retrieves the target item.

The scorer operates as a post-hoc re-ranker: it does not modify the
generator's training or decoding, only re-scores beam search candidates.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class CalibrationScorer(nn.Module):
    """Two-layer MLP: decoder hidden states → P(correct).

    Input: concatenated hidden states from all 4 decoding steps.
    Output: single logit; sigmoid → probability that the generated code hits.
    """

    def __init__(self, hidden_dim: int = 128, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Score a batch of generated sequences.

        Args:
            hidden_states: [B, 4, hidden_dim] from decoder.get_hidden_states()

        Returns:
            logits: [B, 1] — raw logit; sigmoid(logits) = P(correct)
        """
        flat = hidden_states.reshape(hidden_states.shape[0], -1)
        return self.net(flat)


def collect_calibration_data(
    model,
    val_loader: DataLoader,
    device: torch.device,
    item_ids: list[str],
    code_for_item: dict[str, tuple[int, ...]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy-decode every validation user; collect (hidden_states, label) pairs.

    Generator is frozen — no gradient computation.
    Returns (X, y) where X=[N, 4, hidden_dim], y=[N] in {0, 1}.
    """
    code_to_items: dict[tuple[int, ...], list[str]] = {}
    for nid, c in code_for_item.items():
        code_to_items.setdefault(tuple(c), []).append(nid)

    all_hidden: list[torch.Tensor] = []
    all_labels: list[float] = []

    model.eval()
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            result = model.greedy_decode_with_hidden(
                batch["history_emb"], batch["history_mask"]
            )
            # result: {"codes": [B, 4], "hidden_states": [B, 4, hidden_dim]}

        B = batch["history_emb"].shape[0]
        for b in range(B):
            pred_code = tuple(int(x) for x in result["codes"][b])
            tgt_nid = item_ids[int(batch["target_emb_idx"][b])]
            candidates = code_to_items.get(pred_code, [])
            label = 1.0 if tgt_nid in candidates else 0.0
            all_hidden.append(result["hidden_states"][b].cpu())
            all_labels.append(label)

    X = torch.stack(all_hidden)   # [N, 4, hidden_dim]
    y = torch.tensor(all_labels, dtype=torch.float32)  # [N]
    return X, y


def train_scorer(
    scorer: CalibrationScorer,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device | None = None,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    val_ratio: float = 0.3,
    mlflow_client=None,
):
    """Train calibration scorer with early stopping on a held-out split.

    Args:
        scorer: CalibrationScorer instance.
        X: [N, 4, hidden_dim] hidden states from greedy decode.
        y: [N] binary labels (1=hit, 0=miss).
        device: torch device.
        epochs: max training epochs.
        batch_size: scorer training batch size.
        lr: learning rate.
        patience: early-stop patience on val BCE.
        val_ratio: fraction of data held out for val.
        mlflow_client: optional mlflow module for logging.
    """
    if device is None:
        device = next(scorer.parameters()).device
    scorer = scorer.to(device)

    # Train/val split
    n = X.shape[0]
    perm = torch.randperm(n)
    val_n = int(n * val_ratio)
    train_idx = perm[val_n:]
    val_idx = perm[:val_n]

    X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
    X_val, y_val = X[val_idx].to(device), y[val_idx].to(device)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        scorer.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = scorer(Xb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))

        # Validation
        scorer.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = scorer(Xb).squeeze(-1)
                val_loss += F.binary_cross_entropy_with_logits(logits, yb).item()
        val_loss /= max(1, len(val_loader))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if mlflow_client:
            mlflow_client.log_metrics(
                {"scorer_train_bce": train_loss, "scorer_val_bce": val_loss},
                step=epoch,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return history


@torch.no_grad()
def beam_search_with_scorer(
    model,
    scorer: CalibrationScorer,
    history_emb: torch.Tensor,
    history_mask: torch.Tensor,
    *,
    beam_width: int = 50,
    scorer_lambda: float = 0.5,
) -> list[list[tuple[list[int], float]]]:
    """Beam search re-ranked by calibration scorer.

    final_score = avg_log_prob + scorer_lambda * log(P_correct)

    Args:
        model: GenRecV2Model (frozen).
        scorer: trained CalibrationScorer.
        history_emb: [B, L, d].
        history_mask: [B, L].
        beam_width: number of beams.
        scorer_lambda: weight of calibration score.

    Returns:
        Per-user list of (code_seq, final_score) sorted descending.
    """
    device = history_emb.device
    user_state, _ = model.encoder(history_emb, history_mask)
    if model.hot_news is not None:
        user_state = model.hot_news(user_state)
    decoder = model.decoder
    code_length = decoder.config.code_length

    B = user_state.shape[0]
    all_results: list[list[tuple[list[int], float]]] = []

    for b in range(B):
        us = user_state[b : b + 1]  # [1, hidden_dim]

        # ── Standard beam search ──
        beams: list[tuple[list[int], float]] = [([], 0.0)]
        for step in range(code_length):
            num_beams = len(beams)
            if step == 0:
                prefix_batch = torch.tensor(
                    [[decoder.bos_token_id]], dtype=torch.long, device=device
                )
                us_batch = us
            else:
                prefix_batch = torch.tensor(
                    [[decoder.bos_token_id] + seq for seq, _ in beams],
                    dtype=torch.long, device=device,
                )
                us_batch = us.expand(num_beams, -1)

            log_probs = decoder.next_token_log_probs(
                user_state=us_batch, prefix_tokens=prefix_batch
            )
            topk_values, topk_indices = log_probs.topk(beam_width, dim=-1)

            candidates: list[tuple[list[int], float]] = []
            for b_idx, (code_seq, cum_log_prob) in enumerate(beams):
                for k in range(beam_width):
                    token = int(topk_indices[b_idx, k])
                    score = cum_log_prob + float(topk_values[b_idx, k])
                    candidates.append((code_seq + [token], score))
            candidates.sort(key=lambda x: -x[1])
            beams = candidates[:beam_width]

        # ── Re-rank with scorer ──
        scored: list[tuple[list[int], float]] = []
        for code_seq, beam_score in beams:
            code_tensor = torch.tensor([code_seq], dtype=torch.long, device=device)
            hidden = decoder.get_hidden_states(us, code_tensor)  # [1, 4, hidden_dim]
            calib_logit = scorer(hidden).squeeze().item()
            P_correct = float(torch.sigmoid(torch.tensor(calib_logit)))
            avg_log_prob = beam_score / code_length
            final_score = avg_log_prob + scorer_lambda * float(
                torch.log(torch.tensor(max(P_correct, 1e-8)))
            )
            scored.append((code_seq, final_score))

        scored.sort(key=lambda x: -x[1])
        all_results.append(scored)

    return all_results
