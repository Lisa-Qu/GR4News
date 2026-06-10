"""Calibration scorer for generative retrieval.

Trains a lightweight MLP on frozen decoder hidden states to predict
whether a generated code sequence correctly retrieves the target item.

The scorer operates as a post-hoc re-ranker: it does not modify the
generator's training or decoding, only re-scores beam search candidates.

Includes both pointwise (CalibrationScorer) and listwise (ListwiseScorer)
variants for beam search candidate reranking.
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


class ListwiseScorer(nn.Module):
    """SetRank-style self-attention scorer for listwise beam reranking.

    Takes all K beam candidates at once; each candidate attends to the
    others (and optionally a user CLS token) via self-attention before
    producing a scalar score.  This captures *relative* quality among
    candidates, not just absolute.

    Architecture:
        1. Per-candidate projection: [K, code_length*hidden_dim+1] → [K, d_model]
        2. Prepend user CLS token: user_state → [1, d_model]
           → sequence becomes [K+1, d_model]
        3. Self-attention (2 layers): candidates + user token attend to each other
        4. Discard CLS output, score candidates: [K, d_model] → [K]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        code_length: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = hidden_dim * code_length + 1  # +1 for beam_score
        self.projection = nn.Linear(input_dim, d_model)
        self.proj_norm = nn.LayerNorm(d_model)
        self.user_proj = nn.Linear(hidden_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_head = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        beam_scores: torch.Tensor,
        user_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a list of beam candidates jointly.

        Args:
            hidden_states: [B, K, code_length, hidden_dim]
            beam_scores:   [B, K]
            user_state:    [B, hidden_dim] optional user representation

        Returns:
            scores: [B, K] — raw logits (higher = better).
        """
        B, K = beam_scores.shape
        flat = hidden_states.reshape(B, K, -1)  # [B, K, code_length*hidden_dim]
        features = torch.cat(
            [flat, beam_scores.unsqueeze(-1)], dim=-1
        )  # [B, K, code_length*hidden_dim + 1]
        x = self.proj_norm(self.projection(features))  # [B, K, d_model]

        # Prepend user CLS token so candidates can attend to user preference
        if user_state is not None:
            user_token = self.user_proj(user_state).unsqueeze(1)  # [B, 1, d_model]
            x = torch.cat([user_token, x], dim=1)  # [B, K+1, d_model]

        x = self.self_attn(x)  # [B, K+1, d_model] or [B, K, d_model]

        # Discard CLS token output, keep only candidate scores
        if user_state is not None:
            x = x[:, 1:, :]  # [B, K, d_model]

        return self.output_head(x).squeeze(-1)  # [B, K]


# ── Listwise loss functions ────────────────────────────────────────


def listmle_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ListMLE: likelihood of the ground-truth permutation.

    Args:
        scores: [B, K] predicted scores.
        labels: [B, K] relevance labels (higher = more relevant).

    Returns:
        Scalar loss (lower is better).
    """
    # Sort candidates by label descending → ground-truth permutation
    _, perm = labels.sort(dim=-1, descending=True)
    sorted_scores = scores.gather(1, perm)  # [B, K]
    # Top-k log-likelihood: log softmax over remaining candidates
    max_score = sorted_scores.max(dim=-1, keepdim=True).values
    shifted = sorted_scores - max_score  # numerical stability
    cumsums = shifted.flip(dims=[1]).logcumsumexp(dim=1).flip(dims=[1])
    loss = (cumsums - shifted).mean()
    return loss


def approx_ndcg_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Differentiable approximation to NDCG.

    Uses a sigmoid to approximate the rank indicator function.

    Args:
        scores: [B, K] predicted scores.
        labels: [B, K] relevance labels in [0, 1].
        temperature: controls sigmoid sharpness.

    Returns:
        Scalar loss (1 - approxNDCG, lower is better).
    """
    B, K = scores.shape
    # Approximate ranks via pairwise comparisons
    # approx_rank[i] = 1 + sum_{j != i} sigmoid((s_j - s_i) / temp)
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # [B, K, K]
    approx_rank = 1.0 + torch.sigmoid(-diff / temperature).sum(dim=-1) - 0.5  # [B, K]

    # DCG with approximate ranks
    gains = (2.0 ** labels) - 1.0  # [B, K]
    discounts = torch.log2(1.0 + approx_rank)  # [B, K]
    approx_dcg = (gains / discounts).sum(dim=-1)  # [B]

    # Ideal DCG
    sorted_labels, _ = labels.sort(dim=-1, descending=True)
    ideal_gains = (2.0 ** sorted_labels) - 1.0
    ideal_ranks = torch.arange(1, K + 1, dtype=scores.dtype, device=scores.device)
    ideal_discounts = torch.log2(1.0 + ideal_ranks).unsqueeze(0)  # [1, K]
    ideal_dcg = (ideal_gains / ideal_discounts).sum(dim=-1)  # [B]

    # Avoid division by zero for queries with all-zero labels
    ndcg = approx_dcg / ideal_dcg.clamp(min=1e-8)
    return 1.0 - ndcg.mean()


# ── Beam-level calibration data collection ─────────────────────────


def collect_beam_calibration_data(
    model,
    samples: list[dict],
    item_to_idx: dict[str, int],
    code_for_item: dict[str, tuple[int, ...]],
    item_embeddings,
    max_history_len: int,
    device: torch.device,
    item_ids: list[str],
    *,
    beam_width: int = 50,
    batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Per-sample beam search: each sample uses its own history for beam search.

    In mode B, different samples from the same user have different histories
    (accumulated clicks), so beam search must run independently per sample.

    Returns dict with keys:
        hidden:        [N, beam_width, 4, hidden_dim]
        beam_scores:   [N, beam_width]
        user_states:   [N, hidden_dim]
        labels_binary: [N, beam_width]  — per target_item hit/miss
        labels_soft:   [N, beam_width]  — prefix match fraction
    where N = number of samples.
    """
    from genrec_v2.data.dataset import GenRecV2Dataset, make_collator
    import gc

    code_to_items: dict[tuple[int, ...], list[str]] = {}
    for nid, c in code_for_item.items():
        code_to_items.setdefault(tuple(c), []).append(nid)

    model.eval()
    decoder = model.decoder
    code_length = decoder.config.code_length

    ds = GenRecV2Dataset(samples, item_to_idx, code_for_item, item_embeddings, max_history_len)
    coll = make_collator(item_embeddings)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    all_hidden: list[torch.Tensor] = []
    all_beam_scores: list[torch.Tensor] = []
    all_user_states: list[torch.Tensor] = []
    all_labels_binary: list[torch.Tensor] = []
    all_labels_soft: list[torch.Tensor] = []

    n_done = 0
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            user_state, _ = model.encoder(b["history_emb"], b["history_mask"])
            if model.hot_news is not None:
                user_state = model.hot_news(user_state)
        B = b["history_emb"].shape[0]

        for i in range(B):
            sample = samples[n_done + i]
            us = user_state[i : i + 1]

            # Beam search for this sample
            beams: list[tuple[list[int], float]] = [([], 0.0)]
            for step in range(code_length):
                if step == 0:
                    prefix = torch.tensor(
                        [[decoder.bos_token_id]], dtype=torch.long, device=device
                    )
                    us_b = us
                else:
                    prefix = torch.tensor(
                        [[decoder.bos_token_id] + seq for seq, _ in beams],
                        dtype=torch.long, device=device,
                    )
                    us_b = us.expand(len(beams), -1)
                lp = decoder.next_token_log_probs(user_state=us_b, prefix_tokens=prefix)
                tv, ti = lp.topk(beam_width, dim=-1)
                tv_c, ti_c = tv.cpu(), ti.cpu()
                cands: list[tuple[list[int], float]] = []
                for bi, (seq, sc) in enumerate(beams):
                    for k in range(beam_width):
                        cands.append((seq + [int(ti_c[bi, k])], sc + float(tv_c[bi, k])))
                cands.sort(key=lambda x: -x[1])
                beams = cands[:beam_width]

            while len(beams) < beam_width:
                beams.append(([0] * code_length, -1e9))

            beam_codes = torch.tensor(
                [seq for seq, _ in beams], dtype=torch.long, device=device
            )
            with torch.no_grad():
                hidden = decoder.get_hidden_states(
                    us.expand(beam_width, -1), beam_codes
                )

            # Compute labels for this sample's target
            tgt_nid = sample["target"]
            tgt_code = code_for_item.get(tgt_nid)
            labels_bin = torch.zeros(beam_width, dtype=torch.float32)
            labels_soft = torch.zeros(beam_width, dtype=torch.float32)
            if tgt_code is not None:
                tgt_tuple = tuple(tgt_code)
                for j, (seq, _) in enumerate(beams):
                    if tgt_nid in code_to_items.get(tuple(seq), []):
                        labels_bin[j] = 1.0
                    match = sum(1 for a, b_tok in zip(tgt_tuple, seq, strict=False) if a == b_tok)
                    labels_soft[j] = match / code_length

            all_hidden.append(hidden.cpu())
            all_beam_scores.append(torch.tensor([sc for _, sc in beams], dtype=torch.float32))
            all_user_states.append(us.detach().cpu().squeeze(0))
            all_labels_binary.append(labels_bin)
            all_labels_soft.append(labels_soft)

        del b, user_state
        n_done += B
        if n_done % 256 == 0:
            print(f"  beam collected: {n_done}/{len(samples)}")
            gc.collect()
            torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "hidden": torch.stack(all_hidden),  # [N, K, 4, hidden_dim]
        "beam_scores": torch.stack(all_beam_scores),  # [N, K]
        "user_states": torch.stack(all_user_states),  # [N, hidden_dim]
        "labels_binary": torch.stack(all_labels_binary),  # [N, K]
        "labels_soft": torch.stack(all_labels_soft),  # [N, K]
    }


def train_listwise_scorer(
    scorer: ListwiseScorer,
    data: dict[str, torch.Tensor],
    *,
    device: torch.device,
    loss_type: str = "approx_ndcg",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    val_ratio: float = 0.3,
) -> list[dict]:
    """Train listwise scorer with early stopping.

    Args:
        scorer: ListwiseScorer instance.
        data: dict from collect_beam_calibration_data.
        loss_type: 'bce', 'listmle', or 'approx_ndcg'.
        Others: standard training params.

    Returns:
        Training history (list of dicts).
    """
    scorer = scorer.to(device)
    N = data["hidden"].shape[0]
    perm = torch.randperm(N)
    val_n = int(N * val_ratio)
    tr_idx, va_idx = perm[val_n:], perm[:val_n]

    H_tr = data["hidden"][tr_idx].to(device)
    S_tr = data["beam_scores"][tr_idx].to(device)
    Y_tr = data["labels"][tr_idx].to(device)
    U_tr = data["user_states"][tr_idx].to(device)
    H_va = data["hidden"][va_idx].to(device)
    S_va = data["beam_scores"][va_idx].to(device)
    Y_va = data["labels"][va_idx].to(device)
    U_va = data["user_states"][va_idx].to(device)

    tr_ds = TensorDataset(H_tr, S_tr, Y_tr, U_tr)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=lr)
    best_val = float("inf")
    pat_count = 0
    best_state = None
    history: list[dict] = []

    def compute_loss(
        scores: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if loss_type == "bce":
            return F.binary_cross_entropy_with_logits(scores, labels)
        elif loss_type == "listmle":
            return listmle_loss(scores, labels)
        elif loss_type == "approx_ndcg":
            return approx_ndcg_loss(scores, labels)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    for epoch in range(1, epochs + 1):
        scorer.train()
        total_loss = 0.0
        for Hb, Sb, Yb, Ub in tr_loader:
            scores = scorer(Hb, Sb, user_state=Ub)
            loss = compute_loss(scores, Yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / max(1, len(tr_loader))

        scorer.eval()
        with torch.no_grad():
            val_scores = scorer(H_va, S_va, user_state=U_va)
            val_loss = compute_loss(val_scores, Y_va).item()

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val:
            best_val = val_loss
            pat_count = 0
            best_state = {k: v.clone() for k, v in scorer.state_dict().items()}
        else:
            pat_count += 1
            if pat_count >= patience:
                break

    if best_state is not None:
        scorer.load_state_dict(best_state)
    scorer.eval()
    return history


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
