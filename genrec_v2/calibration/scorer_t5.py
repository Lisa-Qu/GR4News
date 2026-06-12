"""Scorer adaptation for GRAM T5 model on Amazon-Beauty.

Teacher-forcing approach (Option A):
1. GRAM T5 beam search produces K=50 candidate token sequences
2. Feed these sequences back through T5 decoder to get hidden states
3. Hidden states [K, code_len, 512] → Scorer → [K] scores

Two scorer variants:
- PointwiseScorerT5: MLP on flattened hidden states per candidate
- ListwiseScorerT5: SetRank-style self-attention across all K candidates
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class PointwiseScorerT5(nn.Module):
    """Two-layer MLP: T5 decoder hidden states → P(correct).

    Input: hidden states from T5 decoder for one beam candidate.
    Output: single logit; sigmoid → probability that the generated code hits.

    Dimensions:
        input:  [B, code_length, hidden_dim]  e.g. [B, 7, 512]
        flatten: [B, code_length * hidden_dim]  e.g. [B, 3584]
        output: [B, 1]
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        code_length: int = 7,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        input_dim = hidden_dim * code_length  # 7 * 512 = 3584
        self.net = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, code_length, hidden_dim]

        Returns:
            logits: [B, 1]
        """
        flat = hidden_states.reshape(hidden_states.shape[0], -1)
        return self.net(flat)


class ListwiseScorerT5(nn.Module):
    """SetRank-style self-attention scorer adapted for T5 hidden dimensions.

    Architecture (identical structure to GR4AD ListwiseScorer):
        1. Per-candidate: flatten [K, code_length, hidden_dim] + beam_score
           → Linear(code_length*hidden_dim + 1, d_model) + LayerNorm
        2. Prepend user CLS token: encoder_output → mean pool → Linear → [1, d_model]
        3. Self-attention (2-layer TransformerEncoder)
        4. Discard CLS → Linear(d_model, 1) → [K] scores

    Dimensions for Beauty (T5-small):
        hidden_dim = 512 (T5 d_model)
        code_length = 7 (hierarchical lexical ID tokens)
        input_dim = 7 * 512 + 1 = 3585
        d_model = 128 (compressed, same as GR4AD — self-attention doesn't need full 512)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        code_length: int = 7,
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
            hidden_states: [B, K, code_length, hidden_dim]  e.g. [B, 50, 7, 512]
            beam_scores:   [B, K]                            e.g. [B, 50]
            user_state:    [B, hidden_dim]                   e.g. [B, 512]
                           (mean-pooled T5 encoder output)

        Returns:
            scores: [B, K]
        """
        B, K = beam_scores.shape
        flat = hidden_states.reshape(B, K, -1)  # [B, K, code_length * hidden_dim]
        features = torch.cat(
            [flat, beam_scores.unsqueeze(-1)], dim=-1
        )  # [B, K, code_length*hidden_dim + 1]
        x = self.proj_norm(self.projection(features))  # [B, K, d_model]

        if user_state is not None:
            user_token = self.user_proj(user_state).unsqueeze(1)  # [B, 1, d_model]
            x = torch.cat([user_token, x], dim=1)  # [B, K+1, d_model]

        x = self.self_attn(x)

        if user_state is not None:
            x = x[:, 1:, :]  # [B, K, d_model]

        return self.output_head(x).squeeze(-1)  # [B, K]


# ── Hidden state extraction from T5 ───────────────────────────────


@torch.no_grad()
def extract_decoder_hidden_states(
    model,
    beam_token_ids: torch.Tensor,
    encoder_output: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Teacher-force beam candidates through T5 decoder to get hidden states.

    Args:
        model: GRAM T5 model (or its decoder).
        beam_token_ids: [K, seq_len] — token IDs from beam search.
            These are the actual output tokens (not including decoder_start_token).
        encoder_output: [1, enc_seq_len, 512] — encoder hidden states.
        encoder_attention_mask: [1, enc_seq_len] — encoder attention mask.

    Returns:
        hidden_states: [K, seq_len, 512] — decoder last hidden state per position.
    """
    K = beam_token_ids.shape[0]
    device = beam_token_ids.device

    # Expand encoder output for all K beam candidates
    enc_expanded = encoder_output.expand(K, -1, -1)  # [K, enc_len, 512]
    mask_expanded = encoder_attention_mask.expand(K, -1)  # [K, enc_len]

    # T5 decoder expects decoder_input_ids to be shifted right
    # (prepend decoder_start_token_id, drop last token)
    pad_token_id = model.config.pad_token_id or 0
    decoder_start = torch.full(
        (K, 1), model.config.decoder_start_token_id, dtype=torch.long, device=device
    )
    # decoder_input_ids = [decoder_start, t0, t1, ..., t_{n-2}]
    decoder_input_ids = torch.cat(
        [decoder_start, beam_token_ids[:, :-1]], dim=1
    )  # [K, seq_len]

    # Forward through decoder
    decoder_outputs = model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=enc_expanded,
        encoder_attention_mask=mask_expanded,
        return_dict=True,
    )

    return decoder_outputs.last_hidden_state  # [K, seq_len, 512]


@torch.no_grad()
def get_user_state_from_encoder(
    encoder_output: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool T5 encoder output to get user state vector.

    Args:
        encoder_output: [1, enc_seq_len, 512]
        encoder_attention_mask: [1, enc_seq_len]

    Returns:
        user_state: [1, 512]
    """
    mask = encoder_attention_mask.unsqueeze(-1).float()  # [1, enc_len, 1]
    pooled = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled  # [1, 512]


# ── Beam calibration data collection for GRAM T5 ──────────────────


def collect_beam_data_t5(
    model,
    tokenizer,
    test_dataset,
    collate_fn,
    device: torch.device,
    *,
    beam_width: int = 50,
    code_length: int = 7,
    max_length: int = 10,
    length_penalty: float = 1.0,
    prefix_allowed_tokens_fn=None,
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    """Collect beam search data with hidden states for GRAM T5.

    For each test sample:
    1. Run beam search → K candidate sequences + beam scores
    2. Teacher-force candidates through decoder → hidden states
    3. Compute binary labels (does candidate match target?)

    Args:
        model: GRAM T5 model.
        tokenizer: T5 tokenizer.
        test_dataset: TestDatasetGRAM or similar.
        collate_fn: collation function for the dataset.
        device: torch device.
        beam_width: K candidates per sample.
        code_length: number of meaningful tokens in output (7 for Beauty).
        max_length: max generation length for beam search.
        length_penalty: beam search length penalty.
        prefix_allowed_tokens_fn: optional constrained decoding function.
        batch_size: must be 1 for per-sample beam search.

    Returns:
        dict with keys:
            hidden:        [N, K, code_length, 512]
            beam_scores:   [N, K]
            user_states:   [N, 512]
            labels_binary: [N, K]
    """
    import gc

    assert batch_size == 1, "Per-sample beam search requires batch_size=1"

    model.eval()
    loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    all_hidden = []
    all_beam_scores = []
    all_user_states = []
    all_labels_binary = []

    n_done = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        output_ids = batch["output_ids"]  # ground truth token ids
        gold_text = tokenizer.batch_decode(
            torch.where(output_ids == -100, 0, output_ids),
            skip_special_tokens=True,
        )

        # Step 1: Beam search
        prediction = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=beam_width,
            num_return_sequences=beam_width,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=length_penalty,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        beam_token_ids = prediction["sequences"]  # [K, seq_len] includes pad
        beam_scores_raw = prediction["sequences_scores"]  # [K]

        # Decode beam candidates to text for matching
        beam_texts = tokenizer.batch_decode(beam_token_ids, skip_special_tokens=True)

        # Step 2: Get encoder output for teacher forcing
        # Re-encode (or cache) — encoder output needed for decoder forward
        model.encoder.n_passages = input_ids.size(1) if input_ids.dim() == 3 else 1
        input_ids_flat = input_ids.view(1, -1) if input_ids.dim() == 3 else input_ids
        attn_mask_flat = (
            attention_mask.view(1, -1) if attention_mask.dim() == 3 else attention_mask
        )

        encoder_out = model.encoder(
            input_ids=input_ids_flat,
            attention_mask=attn_mask_flat,
            return_dict=True,
        )
        encoder_hidden = encoder_out[0]  # [1, enc_len, 512]

        # Step 3: Teacher-force to get decoder hidden states
        # Remove special tokens from beam sequences for clean hidden states
        # T5 generate output: [decoder_start_token, t0, t1, ..., pad, pad]
        # We need the meaningful tokens only
        # Strip leading decoder_start_token and trailing pads
        clean_seqs = []
        for seq in beam_token_ids:
            # Remove decoder_start_token (usually 0) and pad tokens
            tokens = [t.item() for t in seq if t.item() != tokenizer.pad_token_id]
            if tokens and tokens[0] == model.config.decoder_start_token_id:
                tokens = tokens[1:]
            # Truncate or pad to code_length
            tokens = tokens[:code_length]
            while len(tokens) < code_length:
                tokens.append(tokenizer.pad_token_id)
            clean_seqs.append(tokens)

        clean_tensor = torch.tensor(clean_seqs, dtype=torch.long, device=device)  # [K, code_length]

        hidden = extract_decoder_hidden_states(
            model, clean_tensor, encoder_hidden, attn_mask_flat
        )  # [K, code_length, 512]

        # Step 4: User state (mean-pooled encoder)
        user_state = get_user_state_from_encoder(
            encoder_hidden, attn_mask_flat
        )  # [1, 512]

        # Step 5: Compute labels — text match
        labels = torch.zeros(beam_width, dtype=torch.float32)
        for j, bt in enumerate(beam_texts):
            if bt.strip() == gold_text[0].strip():
                labels[j] = 1.0

        # Pad beam_scores if fewer than beam_width returned
        bs = beam_scores_raw.cpu()
        if len(bs) < beam_width:
            pad = torch.full((beam_width - len(bs),), -1e9)
            bs = torch.cat([bs, pad])

        all_hidden.append(hidden.cpu())
        all_beam_scores.append(bs[:beam_width])
        all_user_states.append(user_state.cpu().squeeze(0))
        all_labels_binary.append(labels)

        n_done += 1
        if n_done % 100 == 0:
            print(f"  beam collected: {n_done}/{len(test_dataset)}")
            gc.collect()
            torch.cuda.empty_cache()

    print(f"  {n_done} samples collected")

    return {
        "hidden": torch.stack(all_hidden),       # [N, K, code_length, 512]
        "beam_scores": torch.stack(all_beam_scores),  # [N, K]
        "user_states": torch.stack(all_user_states),  # [N, 512]
        "labels_binary": torch.stack(all_labels_binary),  # [N, K]
    }


# ── Training functions ─────────────────────────────────────────────


def train_listwise_scorer_t5(
    scorer: ListwiseScorerT5,
    data: dict[str, torch.Tensor],
    *,
    device: torch.device,
    loss_type: str = "bce",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    val_ratio: float = 0.3,
) -> list[dict]:
    """Train listwise scorer with early stopping."""
    scorer = scorer.to(device)
    N = data["hidden"].shape[0]
    perm = torch.randperm(N)
    val_n = int(N * val_ratio)
    tr_idx, va_idx = perm[val_n:], perm[:val_n]

    H_tr = data["hidden"][tr_idx].to(device)
    S_tr = data["beam_scores"][tr_idx].to(device)
    Y_tr = data["labels_binary"][tr_idx].to(device)
    U_tr = data["user_states"][tr_idx].to(device)
    H_va = data["hidden"][va_idx].to(device)
    S_va = data["beam_scores"][va_idx].to(device)
    Y_va = data["labels_binary"][va_idx].to(device)
    U_va = data["user_states"][va_idx].to(device)

    tr_ds = TensorDataset(H_tr, S_tr, Y_tr, U_tr)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=lr)
    best_val = float("inf")
    pat_count = 0
    best_state = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        scorer.train()
        total_loss = 0.0
        for Hb, Sb, Yb, Ub in tr_loader:
            scores = scorer(Hb, Sb, user_state=Ub)
            loss = F.binary_cross_entropy_with_logits(scores, Yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / max(1, len(tr_loader))

        scorer.eval()
        with torch.no_grad():
            val_scores = scorer(H_va, S_va, user_state=U_va)
            val_loss = F.binary_cross_entropy_with_logits(val_scores, Y_va).item()

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


def train_pointwise_scorer_t5(
    scorer: PointwiseScorerT5,
    data: dict[str, torch.Tensor],
    *,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    val_ratio: float = 0.3,
) -> list[dict]:
    """Train pointwise scorer with early stopping."""
    scorer = scorer.to(device)

    # Reshape: [N, K, code_len, hidden] → [N*K, code_len, hidden]
    N, K = data["hidden"].shape[:2]
    X = data["hidden"].reshape(N * K, *data["hidden"].shape[2:])
    y = data["labels_binary"].reshape(N * K)

    n = X.shape[0]
    perm = torch.randperm(n)
    val_n = int(n * val_ratio)
    tr_idx, va_idx = perm[val_n:], perm[:val_n]

    X_tr, y_tr = X[tr_idx].to(device), y[tr_idx].to(device)
    X_va, y_va = X[va_idx].to(device), y[va_idx].to(device)

    tr_ds = TensorDataset(X_tr, y_tr)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=lr)
    best_val = float("inf")
    pat_count = 0
    best_state = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        scorer.train()
        total_loss = 0.0
        for Xb, yb in tr_loader:
            logits = scorer(Xb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / max(1, len(tr_loader))

        scorer.eval()
        with torch.no_grad():
            va_logits = scorer(X_va).squeeze(-1)
            val_loss = F.binary_cross_entropy_with_logits(va_logits, y_va).item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

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


# ── Evaluation with scorer reranking ───────────────────────────────


@torch.no_grad()
def evaluate_with_scorer(
    data: dict[str, torch.Tensor],
    scorer: ListwiseScorerT5 | PointwiseScorerT5,
    *,
    device: torch.device,
    scorer_lambda: float = 0.5,
    code_length: int = 7,
) -> dict[str, float]:
    """Evaluate reranking performance.

    For listwise: final_score = beam_score/code_length + λ * scorer_score
    For pointwise: final_score = beam_score/code_length + λ * log(sigmoid(scorer_logit))

    Returns dict with R@1, R@5, R@10, R@50.
    """
    H = data["hidden"].to(device)
    S = data["beam_scores"].to(device)
    Y = data["labels_binary"]  # keep on CPU
    U = data["user_states"].to(device)

    N, K = S.shape
    is_listwise = isinstance(scorer, ListwiseScorerT5)

    if is_listwise:
        raw_scores = scorer(H, S, user_state=U)  # [N, K]
    else:
        # Pointwise: score each candidate independently
        H_flat = H.reshape(N * K, *H.shape[2:])
        logits = scorer(H_flat).squeeze(-1)  # [N*K]
        raw_scores = torch.log(torch.sigmoid(logits) + 1e-8).reshape(N, K)

    avg_beam = S / code_length
    final_scores = avg_beam + scorer_lambda * raw_scores

    # Rank by final score
    _, ranked_idx = final_scores.sort(dim=-1, descending=True)

    # Compute R@k
    results = {}
    for k in [1, 5, 10, 50]:
        hits = 0
        for i in range(N):
            top_k_idx = ranked_idx[i, :k]
            if Y[i][top_k_idx].sum() > 0:
                hits += 1
        results[f"R@{k}"] = hits / N

    return results
