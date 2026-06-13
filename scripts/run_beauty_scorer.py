#!/usr/bin/env python3
"""Run post-hoc scorer experiments on Amazon (Beauty/Sports/Toys) using frozen GRAM T5.

This is the Amazon-side (GRAM-T5) twin of MIND's ``genrec_v2/run_main_table.py``.
The two share IDENTICAL scorer-side conventions (control variables) so the Main
Table is comparable across the MIND and Amazon datasets:

    - LAMBDA_GRID = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)  (0.0 ⇒ vanilla)
    - λ selected on VALIDATION R@1 for BOTH Focal AND Listwise (no test leakage;
      5-seed listwise picks ONE λ = argmax of mean val-R@1 across seeds)
    - dropout = 0.1, focal γ = 2.0
    - within-beam ``_rank_metrics``: R@k via hit, MRR, nDCG (log2(1+rank), Oracle-bounded)
    - per_user_hits.npz persisted for ``genrec_v2/run_statistical_significance.py``

Usage (on gram-server):
    cd /home/lishazhai/workspace/GRAM/src
    python /home/lishazhai/workspace/GR4AD/scripts/run_beauty_scorer.py \
        --dataset Beauty --code-length 7 \
        2>&1 | tee /home/lishazhai/workspace/GR4AD/experiments/beauty_scorer.log

    # Toys uses 5-token codes — code-length MUST be 5 or codes silently truncate:
    python .../run_beauty_scorer.py --dataset Toys --code-length 5 \
        --checkpoint <toys_ckpt.pt> --item-id-path <...> \
        --hierarchical-id-type <...> --output-dir <.../toys_scorer>

Pipeline:
    1. Load frozen GRAM T5 model (dataset checkpoint)
    2. Beam search on val/test → collect hidden states
    3. Train Pointwise Focal + 5-seed Listwise BCE (early-stop on plain BCE val loss)
    4. Lambda sweep on VAL → pick best λ for Focal AND Listwise
    5. Evaluate on test (vanilla / focal / listwise) + persist results + per_user_hits
"""

import sys
import os
import gc
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── GRAM imports (run from /home/lishazhai/workspace/GRAM/src) ──
# Import order matters: model → utils → data (mirrors GRAM's main_generative_gram.py)
sys.path.insert(0, "/home/lishazhai/workspace/GRAM/src")

from model.gram import GRAM
from model.gram_t5_config import T5Config as T5Config_GRAM
import utils.generation_trie as gt
from utils import get_loader_gram_train  # triggers utils.__init__ which imports dataset_utils → data
from data import TestDatasetGRAM
from processor import CollatorGRAM

# ── Scorer imports ──
sys.path.insert(0, "/home/lishazhai/workspace/GR4AD")
from genrec_v2.calibration.scorer_t5 import (
    PointwiseScorerT5,
    ListwiseScorerT5,
    extract_decoder_hidden_states,
    get_user_state_from_encoder,
)

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GRAM model paths (roots fixed; per-dataset paths come from CLI args)
GRAM_ROOT = Path("/home/lishazhai/workspace/GRAM")
T5_BACKBONE = GRAM_ROOT / "t5-small"
DATA_PATH = GRAM_ROOT / "rec_datasets"

# ── Beauty defaults (reproduce the 2026-05-30 run) — overridable via CLI ──
DEFAULT_DATASET = "Beauty"
# AR checkpoint (exp 10): use_diffusion=0, pure T5 AR beam search
DEFAULT_CHECKPOINT = GRAM_ROOT / "log/Beauty/10_20260312_1403/id_0_rec_30/model_rec_phase_1_epoch_30.pt"
DEFAULT_CODE_LENGTH = 7   # Beauty/Sports = 7 hierarchical tokens; Toys = 5
DEFAULT_ITEM_ID_PATH = "item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt"
DEFAULT_HIERARCHICAL_ID_TYPE = "hierarchy_v1_c128_l7_len32768"
DEFAULT_ITEM_TO_CODE_PATH = GRAM_ROOT / "tokenizer_output/final2/item_to_code.json"
DEFAULT_CODE_TO_ITEMS_PATH = GRAM_ROOT / "tokenizer_output/final2/code_to_items.json"
DEFAULT_OUTPUT_DIR = GRAM_ROOT.parent / "GR4AD/experiments/beauty_scorer"

# Scorer hyperparams (control variables — IDENTICAL to MIND run_main_table.py)
HIDDEN_DIM = 512        # T5 d_model
BEAM_WIDTH = 50
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 256
DROPOUT = 0.1           # matches MIND scorer dropout=0.1 (control variable)
BOTTLENECK_DIM = 64    # match MIND CalibrationScorer bottleneck=64 (control variable —
                       # pointwise scorer internal dim held constant across settings, like
                       # listwise d_model=128; only input compression varies). Review-fix 2026-06-13.

# Training
LR = 1e-3
BATCH_SIZE_LISTWISE = 32
BATCH_SIZE_POINTWISE = 256
EPOCHS = 50
PATIENCE = 10
VAL_RATIO = 0.3
FOCAL_GAMMA = 2.0

# Eval metrics (shared with MIND _rank_metrics)
KS = (1, 5, 10, 50)
NDCG_KS = (5, 10)

# Locked λ grid + selection procedure — IDENTICAL to MIND LAMBDA_GRID (control variable).
# 0.0 ⇒ pure beam ranking (vanilla); selected on VALIDATION R@1, never test.
# Finer at the small end (0.02, 0.05): Beauty/high-SE optimum λ≈0.02-0.05 was unreachable
# on the old min-0.1 grid (Focal fell to -0.5%); superset keeps MIND λ*=0.5 unaffected.
LAMBDA_GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
SEEDS = [42, 123, 7, 999, 2024]


def parse_args() -> argparse.Namespace:
    """CLI for generalising across Beauty / Sports / Toys.

    Beauty defaults reproduce the 2026-05-30 run. Toys MUST pass
    ``--code-length 5`` (Toys uses 5-token codes) or candidate codes silently
    truncate to 7 and the whole pipeline is wrong.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   choices=["Beauty", "Sports", "Toys"],
                   help="Amazon subset (Beauty/Sports/Toys).")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="Path to the frozen GRAM AR checkpoint (.pt).")
    p.add_argument("--code-length", type=int, default=DEFAULT_CODE_LENGTH,
                   help="Hierarchical-ID token count (7 for Beauty/Sports, 5 for Toys).")
    p.add_argument("--item-id-path", default=DEFAULT_ITEM_ID_PATH,
                   help="Item-ID indexing filename (relative to GRAM dataset dir).")
    p.add_argument("--item-to-code-path", type=Path, default=DEFAULT_ITEM_TO_CODE_PATH,
                   help="tokenizer_output item_to_code.json (model init only, AR unused).")
    p.add_argument("--code-to-items-path", type=Path, default=DEFAULT_CODE_TO_ITEMS_PATH,
                   help="tokenizer_output code_to_items.json (model init only, AR unused).")
    p.add_argument("--hierarchical-id-type", default=DEFAULT_HIERARCHICAL_ID_TYPE,
                   help="GRAM hierarchical_id_type string for this dataset.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Where to write beam caches, results.json, per_user_hits.npz.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════

class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = targets * p + (1 - targets) * (1 - p)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ══════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════

def make_args(cli: argparse.Namespace) -> argparse.Namespace:
    """Create the args namespace GRAM expects, driven by the CLI config.

    ``num_semantic_digits`` tracks ``--code-length`` so VQ-init params stay
    consistent with the chosen dataset (Toys=5), even though AR mode ignores them.
    """
    args = argparse.Namespace(
        data_path=str(DATA_PATH),
        datasets=cli.dataset,
        max_his=20,
        his_sep=" ; ",
        skip_empty_his=1,
        reverse_history=1,
        user_id_without_target_item=1,
        item_id_path=cli.item_id_path,
        item_prompt="all_text",
        cf_model="sasrec",
        top_k_similar_item=10,  # match exp 10 config
        item_prompt_max_len=128,
        target_max_len=32,
        hierarchical_id_type=cli.hierarchical_id_type,
        item_id_type="split",
        id_linking=1,
        prompt_file=str(GRAM_ROOT / "prompt.txt"),
        sample_prompt=1,
        sample_num="1",
        tasks="sequential",
        backbone=str(T5_BACKBONE),
        metrics="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
        beam_size=BEAM_WIDTH,
        length_penalty=1.0,
        rank=0,
        distributed=0,
        verbose_input_output=0,
        debug_test_100=0,
        debug_test_small_set=0,
        save_predictions=0,
        # Semantic VQ params (needed for model init but not used in AR mode)
        use_diffusion=1,
        semantic_id_mode="parallel_vq",
        num_code_heads=4,
        codebook_size=256,
        item_to_code_path=str(cli.item_to_code_path),
        code_to_items_path=str(cli.code_to_items_path),
        semantic_decoder_hidden_dim=512,
        semantic_decoder_num_layers=4,
        semantic_decoder_num_heads=8,
        diffusion_warmup_epochs=3,
        diffusion_loss_weight=1.0,
        num_semantic_digits=cli.code_length,
        num_codebook_entries=128,
        diffusion_num_heads=8,
        diffusion_num_layers=4,
        diffusion_steps=100,
        use_position_embedding=1,
        valid_prompt="seen:0",
        valid_prompt_sample=1,
        valid_sample_num="1",
        test_prompt="seen:0",
    )
    return args


def load_gram_model(args, checkpoint: Path):
    """Load frozen GRAM T5 model from ``checkpoint``."""
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(str(T5_BACKBONE))

    config = T5Config_GRAM.from_pretrained(str(T5_BACKBONE))
    config.max_seq_len = args.item_prompt_max_len
    config.max_item_num = args.max_his  # pos_emb_size = max_item_num + 1 = 21
    config.use_position_embedding = 1

    model = GRAM(config)
    model.load_t5(torch.load(str(T5_BACKBONE / "pytorch_model.bin"), map_location="cpu"))

    # Load trained checkpoint
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded GRAM model from {checkpoint}")
    print(f"  d_model={config.d_model}, decoder_layers={config.num_decoder_layers}")

    return model, tokenizer, config


# ══════════════════════════════════════════════════════════════════
# Beam data collection
# ══════════════════════════════════════════════════════════════════

def _extract_user_id(batch, fallback: int):
    """Best-effort per-sample user id for the significance test.

    The GRAM batch's user-id key is not guaranteed (GRAM source not vendored
    here), so we probe the common keys and fall back to the running sample index
    (each sample its own "user"). McNemar (per-sample, the primary test) is exact
    regardless; only the Wilcoxon per-user grouping degrades to per-sample under
    the fallback. See report note.
    """
    for key in ("user_id", "user_idx", "users", "user", "user_name"):
        if key in batch:
            v = batch[key]
            if hasattr(v, "tolist"):
                v = v.tolist()
            if isinstance(v, (list, tuple)) and v:
                v = v[0]
            return str(v)
    return f"_idx{fallback}"


@torch.no_grad()
def collect_beam_data(model, tokenizer, dataset, collator, args, output_dir, code_length, tag="val"):
    """Collect beam search candidates + decoder hidden states.

    Returns dict with:
        hidden:        [N, K, code_length, 512]
        beam_scores:   [N, K]
        user_states:   [N, 512]
        labels_binary: [N, K]
        user_ids:      [N]  (str array, for the paired significance test)
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)

    # Build trie for constrained decoding
    all_items = dataset.all_items
    if args.item_id_type == "split":
        encoded_candidates = []
        for candidate in all_items:
            tmp_tokens = tokenizer.encode(candidate)
            encoded_candidate = [0]
            for tok in tmp_tokens:
                if tok in [1820, 9175]:  # '|', '_|'
                    continue
                encoded_candidate.append(tok)
            encoded_candidates.append(encoded_candidate)
    else:
        encoded_candidates = [
            [0] + tokenizer.encode(f"{candidate}") for candidate in all_items
        ]
    candidate_trie = gt.Trie(encoded_candidates)
    max_length = max(len(c) for c in encoded_candidates)

    # Fixed prefix_allowed_tokens_fn: handle finished beams that already
    # contain EOS. Newer transformers versions still call pfn on finished
    # beams, which causes empty-result crash if EOS is not in the trie.
    EOS_ID = tokenizer.eos_token_id  # 1
    PAD_ID = tokenizer.pad_token_id  # 0
    def prefix_allowed_tokens(batch_id, sentence):
        s = sentence.tolist()
        if EOS_ID in s[1:]:  # already finished (skip decoder_start at pos 0)
            return [PAD_ID]
        r = candidate_trie.get(s)
        if len(r) == 0:
            return [EOS_ID]  # force stop if stuck
        return r

    all_hidden = []
    all_beam_scores = []
    all_user_states = []
    all_labels_binary = []
    all_user_ids = []

    n_done = 0
    t0 = time.time()

    for batch in loader:
        input_ids = batch["item_text_ids"].to(DEVICE)
        attention_mask = batch["item_text_masks"].to(DEVICE)
        output_ids = batch["target_ids"]
        all_user_ids.append(_extract_user_id(batch, n_done))

        # Ground truth text
        gt_ids = torch.where(output_ids == -100, 0, output_ids)
        gold_text = tokenizer.batch_decode(gt_ids, skip_special_tokens=True)[0].strip()

        # Step 1: Beam search
        prediction = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            num_beams=BEAM_WIDTH,
            num_return_sequences=BEAM_WIDTH,
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=args.length_penalty,
        )

        beam_ids = prediction["sequences"]       # [K, seq_len]
        beam_scores = prediction["sequences_scores"]  # [K]
        beam_texts = tokenizer.batch_decode(beam_ids, skip_special_tokens=True)

        K_actual = len(beam_texts)

        # Step 2: Get encoder hidden states (re-encode)
        model.encoder.n_passages = input_ids.size(1) if input_ids.dim() == 3 else 1
        inp_flat = input_ids.view(1, -1) if input_ids.dim() == 3 else input_ids
        mask_flat = attention_mask.view(1, -1) if attention_mask.dim() == 3 else attention_mask

        enc_out = model.encoder(
            input_ids=inp_flat, attention_mask=mask_flat, return_dict=True
        )
        encoder_hidden = enc_out[0]  # [1, enc_len, 512]

        # Step 3: Clean beam token IDs for teacher forcing
        # Remove separator tokens and pad to code_length
        clean_seqs = []
        for seq in beam_ids:
            tokens = [t.item() for t in seq if t.item() != tokenizer.pad_token_id]
            # Remove decoder_start_token (0) and EOS (1)
            if tokens and tokens[0] == model.config.decoder_start_token_id:
                tokens = tokens[1:]
            if tokens and tokens[-1] == tokenizer.eos_token_id:
                tokens = tokens[:-1]
            # Remove separator tokens (|, _|)
            tokens = [t for t in tokens if t not in [1820, 9175]]
            # Truncate/pad to code_length (5 for Toys, 7 for Beauty/Sports)
            tokens = tokens[:code_length]
            while len(tokens) < code_length:
                tokens.append(tokenizer.pad_token_id)
            clean_seqs.append(tokens)

        clean_tensor = torch.tensor(clean_seqs, dtype=torch.long, device=DEVICE)

        # Step 4: Teacher-force through decoder
        hidden = extract_decoder_hidden_states(
            model, clean_tensor, encoder_hidden, mask_flat
        )  # [K, code_length, 512]

        # Step 5: User state
        user_state = get_user_state_from_encoder(encoder_hidden, mask_flat)  # [1, 512]

        # Step 6: Labels
        labels = torch.zeros(BEAM_WIDTH, dtype=torch.float32)
        for j, bt in enumerate(beam_texts[:BEAM_WIDTH]):
            if bt.strip() == gold_text:
                labels[j] = 1.0

        # Pad if fewer candidates
        h = hidden.cpu()
        bs = beam_scores.cpu()
        if K_actual < BEAM_WIDTH:
            pad_h = torch.zeros(BEAM_WIDTH - K_actual, code_length, HIDDEN_DIM)
            h = torch.cat([h, pad_h], dim=0)
            pad_s = torch.full((BEAM_WIDTH - K_actual,), -1e9)
            bs = torch.cat([bs, pad_s])

        all_hidden.append(h[:BEAM_WIDTH])
        all_beam_scores.append(bs[:BEAM_WIDTH])
        all_user_states.append(user_state.cpu().squeeze(0))
        all_labels_binary.append(labels)

        n_done += 1
        if n_done % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {tag} beam collected: {n_done}/{len(dataset)} [{elapsed:.0f}s]")
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  {n_done} {tag} samples [{elapsed:.0f}s]")

    result = {
        "hidden": torch.stack(all_hidden),
        "beam_scores": torch.stack(all_beam_scores),
        "user_states": torch.stack(all_user_states),
        "labels_binary": torch.stack(all_labels_binary),
        "user_ids": np.array(all_user_ids),
    }

    # Save to disk
    torch.save(result, output_dir / f"beam_data_{tag}.pt")
    print(f"  Saved to {output_dir / f'beam_data_{tag}.pt'}")

    # Stats
    hit_rate = result["labels_binary"].sum(dim=1).clamp(max=1).mean().item()
    print(f"  R@{BEAM_WIDTH} (hit in beam): {hit_rate:.4f} ({hit_rate*100:.2f}%)")

    return result


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_pointwise(X_tr, y_tr, X_va, y_va, loss_fn, code_length, seed=42):
    """Train pointwise scorer.

    Trains with ``loss_fn`` (Focal) but EARLY-STOPS on plain BCE val loss —
    mirrors MIND ``train_pointwise`` so the same scorer stops consistently
    across loss settings (review finding: pointwise early-stop on BCE).
    """
    torch.manual_seed(seed)
    scorer = PointwiseScorerT5(
        hidden_dim=HIDDEN_DIM, code_length=code_length,
        bottleneck_dim=BOTTLENECK_DIM,
    ).to(DEVICE)
    opt = torch.optim.AdamW(scorer.parameters(), lr=LR)
    # X_tr/X_va stay on CPU; move per-batch to GPU (T5 candidates are large → moving the
    # whole split at once OOMs). (Review-OOM fix 2026-06-12.)
    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=BATCH_SIZE_POINTWISE, shuffle=True, pin_memory=True)
    va_ds = TensorDataset(X_va, y_va)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE_POINTWISE)
    best_vl = float("inf")
    pat = 0
    best_state = None
    for ep in range(EPOCHS):
        scorer.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            lo = scorer(Xb).squeeze(-1)
            loss = loss_fn(lo, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        scorer.eval()
        with torch.no_grad():
            # Early-stop on plain BCE (selection criterion), not the focal loss.
            tot, n = 0.0, 0
            for Xb, yb in va_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                tot += F.binary_cross_entropy_with_logits(
                    scorer(Xb).squeeze(-1), yb, reduction="sum").item()
                n += yb.numel()
            vl = tot / max(1, n)
        if vl < best_vl:
            best_vl = vl
            pat = 0
            best_state = {k: v.clone() for k, v in scorer.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    scorer.load_state_dict(best_state)
    scorer.eval()
    return scorer


def train_listwise(val_data, code_length, seed=42):
    """Train listwise BCE scorer (early-stops on plain BCE val loss)."""
    torch.manual_seed(seed)
    scorer = ListwiseScorerT5(
        hidden_dim=HIDDEN_DIM, code_length=code_length,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF, dropout=DROPOUT,
    ).to(DEVICE)

    N = val_data["hidden"].shape[0]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    val_n = int(N * VAL_RATIO)
    tr_idx, va_idx = perm[val_n:], perm[:val_n]

    # Keep master tensors on CPU; move per-batch to GPU. T5 hidden states are large
    # (512-dim × 7 tokens) → moving the full split at once OOMs. (Review-OOM fix 2026-06-12.)
    H_tr = val_data["hidden"][tr_idx]
    S_tr = val_data["beam_scores"][tr_idx]
    Y_tr = val_data["labels_binary"][tr_idx]
    U_tr = val_data["user_states"][tr_idx]
    H_va = val_data["hidden"][va_idx]
    S_va = val_data["beam_scores"][va_idx]
    Y_va = val_data["labels_binary"][va_idx]
    U_va = val_data["user_states"][va_idx]

    tr_ds = TensorDataset(H_tr, S_tr, Y_tr, U_tr)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE_LISTWISE, shuffle=True, pin_memory=True)
    va_ds = TensorDataset(H_va, S_va, Y_va, U_va)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE_LISTWISE)
    opt = torch.optim.AdamW(scorer.parameters(), lr=LR)
    best_vl = float("inf")
    pat = 0
    best_state = None

    for ep in range(EPOCHS):
        scorer.train()
        for Hb, Sb, Yb, Ub in tr_loader:
            Hb, Sb, Yb, Ub = Hb.to(DEVICE), Sb.to(DEVICE), Yb.to(DEVICE), Ub.to(DEVICE)
            scores = scorer(Hb, Sb, user_state=Ub)
            loss = F.binary_cross_entropy_with_logits(scores, Yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        scorer.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for Hb, Sb, Yb, Ub in va_loader:
                Hb, Sb, Yb, Ub = Hb.to(DEVICE), Sb.to(DEVICE), Yb.to(DEVICE), Ub.to(DEVICE)
                vs = scorer(Hb, Sb, user_state=Ub)
                tot += F.binary_cross_entropy_with_logits(vs, Yb, reduction="sum").item()
                n += Yb.numel()
            vl = tot / max(1, n)
        if vl < best_vl:
            best_vl = vl
            pat = 0
            best_state = {k: v.clone() for k, v in scorer.state_dict().items()}
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    scorer.load_state_dict(best_state)
    scorer.eval()
    print(f"  {ep+1} ep, val_loss={best_vl:.4f}")
    return scorer


# ══════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════

def _rank_metrics(rel_ordered: np.ndarray) -> tuple[dict, float, dict]:
    """From a binary relevance vector in ranked order → hit@k, MRR, nDCG@k.

    Computed within the beam (Oracle-bounded). Shared formula used identically
    across every setting/baseline AND identical to MIND ``run_main_table._rank_metrics``
    so the MIND vs Amazon Main Table is comparable.
    """
    hit = {k: bool(rel_ordered[:k].max() > 0) for k in KS}
    nz = np.flatnonzero(rel_ordered)
    mrr = float(1.0 / (nz[0] + 1)) if nz.size else 0.0
    n_rel = int(rel_ordered.sum())
    ndcg: dict[int, float] = {}
    for k in NDCG_KS:
        topk = rel_ordered[:k]
        dcg = float(np.sum(topk / np.log2(np.arange(2, topk.size + 2))))
        idcg = float(np.sum(1.0 / np.log2(np.arange(2, min(k, n_rel) + 2)))) if n_rel else 0.0
        ndcg[k] = dcg / idcg if idcg > 0 else 0.0
    return hit, mrr, ndcg


@torch.no_grad()
def eval_peruser(scorer, is_listwise, lam, H, S, U, Y, code_length):
    """Evaluate ``H/S/U/Y`` with reranking → (aggregate metrics, per-sample hit@k).

    Reranking formula is UNIFORM (no λ>0 / else special-case):
        final = beam_score/code_length + λ * scorer_term
    so ``λ=0`` ⇒ pure beam ranking (vanilla). ``scorer=None`` also reproduces the
    vanilla order. A "scorer-only" config (beam off) is ``lam=None`` — a
    separately-named config, NOT the grid's 0.0.

    Aggregate dict keys: R@k (k in KS), 'MRR', 'nDCG@5', 'nDCG@10' via the shared
    ``_rank_metrics``. Per-sample hit@k arrays feed the significance test.
    Scorer forward is GPU-batched; ranking metrics are computed per-sample on CPU.
    """
    N, K = S.shape
    hits = {k: np.zeros(N, dtype=bool) for k in KS}
    mrr_arr = np.zeros(N, dtype=np.float64)
    ndcg_arr = {k: np.zeros(N, dtype=np.float64) for k in NDCG_KS}
    if scorer is not None:
        scorer = scorer.to(DEVICE)
    EVAL_BATCH = 512 if is_listwise else 2048

    for start in range(0, N, EVAL_BATCH):
        end = min(start + EVAL_BATCH, N)
        S_b = S[start:end]
        Y_b = Y[start:end]
        n_b = S_b.shape[0]

        if scorer is None:
            # Vanilla: rank by beam score (== λ=0 pure beam ranking).
            order_b = (S_b / code_length).argsort(dim=-1, descending=True)
        else:
            H_b = H[start:end].to(DEVICE)
            S_gpu = S_b.to(DEVICE)
            if is_listwise:
                U_b = U[start:end].to(DEVICE)
                raw = scorer(H_b, S_gpu, user_state=U_b).cpu()
            else:
                H_flat = H_b.reshape(n_b * K, code_length, HIDDEN_DIM)
                logits = scorer(H_flat).squeeze(-1)
                raw = torch.log(torch.sigmoid(logits) + 1e-8).reshape(n_b, K).cpu()
            if lam is None:
                final = raw  # "scorer-only" config (beam off) — explicitly named.
            else:
                final = S_b / code_length + lam * raw
            order_b = final.argsort(dim=-1, descending=True)
            del H_b
            torch.cuda.empty_cache()

        for ci in range(n_b):
            rel_ordered = Y_b[ci][order_b[ci]].numpy()
            hit, mrr, ndcg = _rank_metrics(rel_ordered)
            idx = start + ci
            for k in KS:
                hits[k][idx] = hit[k]
            mrr_arr[idx] = mrr
            for k in NDCG_KS:
                ndcg_arr[k][idx] = ndcg[k]

    if scorer is not None:
        scorer = scorer.cpu()
        torch.cuda.empty_cache()
    agg: dict = {f"R@{k}": float(hits[k].mean()) for k in KS}
    agg["MRR"] = float(mrr_arr.mean())
    for k in NDCG_KS:
        agg[f"nDCG@{k}"] = float(ndcg_arr[k].mean())
    return agg, hits


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    cli = parse_args()
    code_length = cli.code_length
    output_dir = cli.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset={cli.dataset}  code_length={code_length}  output_dir={output_dir}")
    print(f"Checkpoint={cli.checkpoint}")

    args = make_args(cli)
    print("Loading GRAM model...")
    model, tokenizer, config = load_gram_model(args, cli.checkpoint)

    # ── Build datasets ──
    print("\nBuilding val/test datasets...")
    val_dataset = TestDatasetGRAM(
        args, cli.dataset, "sequential", model, tokenizer,
        regenerate=False, phase=0, mode="validation",
    )
    test_dataset = TestDatasetGRAM(
        args, cli.dataset, "sequential", model, tokenizer,
        regenerate=False, phase=0, mode="test",
    )
    collator = CollatorGRAM(tokenizer, args, mode="test")
    print(f"  Val: {len(val_dataset)} samples, Test: {len(test_dataset)} samples")

    # ── Collect beam data (fingerprint-guarded: never reuse a cache from a different
    # dataset / checkpoint / code_length / K — e.g. Toys code_length=5 must NOT load
    # Beauty's code_length=7 cache). Review-fix 2026-06-13. ──
    def _beam_fp(dataset, n):
        return (f"ds={cli.dataset}|ckpt={cli.checkpoint}|code_len={code_length}|"
                f"K={BEAM_WIDTH}|n={n}")

    def _load_or_collect(tag, dataset):
        cache = output_dir / f"beam_data_{tag}.pt"
        fp = _beam_fp(cli.dataset, len(dataset))
        if cache.exists():
            data = torch.load(cache, map_location="cpu")
            if data.get("_fingerprint") == fp:
                print(f"\nLoading cached {tag} beam data from {cache}")
                return data
            print(f"\n{tag} cache fingerprint mismatch -> recollecting\n"
                  f"  cached: {data.get('_fingerprint')}\n  want:   {fp}")
        print(f"\nCollecting {tag} beam data...")
        data = collect_beam_data(model, tokenizer, dataset, collator, args,
                                 output_dir, code_length, tag)
        data["_fingerprint"] = fp
        torch.save(data, cache)  # re-save with fingerprint stamped
        return data

    val_data = _load_or_collect("val", val_dataset)
    test_data = _load_or_collect("test", test_dataset)

    # Free model memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Stats (vanilla == eval_peruser with scorer=None, λ=0) ──
    val_r50 = val_data["labels_binary"].sum(dim=1).clamp(max=1).mean().item()
    test_r50 = test_data["labels_binary"].sum(dim=1).clamp(max=1).mean().item()
    val_van_agg, _ = eval_peruser(None, False, 0.0, val_data["hidden"],
                                  val_data["beam_scores"], val_data["user_states"],
                                  val_data["labels_binary"], code_length)
    print(f"\nVal:  R@1={val_van_agg['R@1']:.4f}, R@5={val_van_agg['R@5']:.4f}, "
          f"R@10={val_van_agg['R@10']:.4f}, R@50={val_r50:.4f}")

    # ══════════════════════════════════════════════════════════════
    # PART 1: Pointwise Focal — val-selected λ
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 1: Pointwise Focal BCE")
    print("=" * 60)

    # Prepare pointwise data
    N_val, K = val_data["hidden"].shape[:2]
    pw_X = val_data["hidden"].reshape(N_val * K, code_length, HIDDEN_DIM)
    pw_y = val_data["labels_binary"].reshape(N_val * K)

    n_pw = pw_X.shape[0]
    perm = torch.randperm(n_pw)
    val_n = int(n_pw * VAL_RATIO)
    # Keep on CPU; train_pointwise moves batches to GPU (T5 candidates too big for one move).
    X_tr = pw_X[perm[val_n:]]
    y_tr = pw_y[perm[val_n:]]
    X_va = pw_X[perm[:val_n]]
    y_va = pw_y[perm[:val_n]]

    print("Training Pointwise Focal BCE (train=focal, early-stop=BCE)...")
    focal_scorer = train_pointwise(X_tr, y_tr, X_va, y_va, FocalBCELoss(FOCAL_GAMMA), code_length)
    print("  Done")

    del X_tr, y_tr, X_va, y_va, pw_X, pw_y
    gc.collect()
    torch.cuda.empty_cache()

    # Select Focal λ on VAL R@1 over the locked grid (incl. 0.0 ⇒ vanilla).
    print("Selecting Focal lambda on val (val-R@1 argmax over locked grid)...")
    best_focal_lam, best_focal_r1 = LAMBDA_GRID[0], -1.0
    for lam in LAMBDA_GRID:
        agg, _ = eval_peruser(focal_scorer, False, lam, val_data["hidden"],
                              val_data["beam_scores"], val_data["user_states"],
                              val_data["labels_binary"], code_length)
        print(f"  lambda={lam:<5} val R@1={agg['R@1']:.4f}")
        if agg["R@1"] > best_focal_r1:
            best_focal_r1, best_focal_lam = agg["R@1"], lam
    print(f"  -> best Focal lambda={best_focal_lam} (val R@1={best_focal_r1:.4f})")

    # ══════════════════════════════════════════════════════════════
    # PART 2: 5-seed Listwise BCE — val-selected λ (ONE λ across seeds)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 2: 5-seed Listwise BCE")
    print("=" * 60)

    seed_scorers = {}
    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        scorer = train_listwise(val_data, code_length, seed=seed)
        seed_scorers[seed] = scorer.cpu()

    print("Selecting Listwise lambda on val (mean val-R@1 across seeds, same grid)...")
    best_lw_lam, best_lw_r1 = LAMBDA_GRID[0], -1.0
    for lam in LAMBDA_GRID:
        r1s = [
            eval_peruser(seed_scorers[seed], True, lam, val_data["hidden"],
                         val_data["beam_scores"], val_data["user_states"],
                         val_data["labels_binary"], code_length)[0]["R@1"]
            for seed in SEEDS
        ]
        m = float(np.mean(r1s))
        print(f"  lambda={lam:<5} val mean R@1={m:.4f}")
        if m > best_lw_r1:
            best_lw_r1, best_lw_lam = m, lam
    print(f"  -> best Listwise lambda={best_lw_lam} (val mean R@1={best_lw_r1:.4f})")

    # ══════════════════════════════════════════════════════════════
    # PART 3: Test evaluation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 3: Test evaluation")
    print("=" * 60)

    test_h = test_data["hidden"]
    test_s = test_data["beam_scores"]
    test_u = test_data["user_states"]
    test_l = test_data["labels_binary"]
    test_user_ids = test_data.get("user_ids")
    if test_user_ids is None:  # cache from an older run without user_ids
        test_user_ids = np.array([f"_idx{i}" for i in range(test_l.shape[0])])

    # Vanilla (scorer=None ⇒ pure beam ranking).
    van_agg, van_hits = eval_peruser(None, False, 0.0, test_h, test_s, test_u, test_l, code_length)
    # Self-consistency: listwise λ=0 must reproduce vanilla R@1 (scorer off).
    lw0_agg, _ = eval_peruser(seed_scorers[SEEDS[0]], True, 0.0, test_h, test_s, test_u, test_l, code_length)
    assert abs(lw0_agg["R@1"] - van_agg["R@1"]) < 1e-9, (
        f"λ=0 != vanilla (λ0 R@1={lw0_agg['R@1']}, vanilla={van_agg['R@1']}) "
        "— λ=0 special-case present in eval")

    def vs_van(x: float) -> float:
        return (x - van_agg["R@1"]) / max(1e-8, van_agg["R@1"]) * 100

    # Focal at val-selected λ — distinct variable names so focal metrics are NOT
    # clobbered by the listwise seed loop below (review finding: `r`-reuse bug).
    focal_agg, focal_hits = eval_peruser(focal_scorer, False, best_focal_lam,
                                         test_h, test_s, test_u, test_l, code_length)

    print(f"\n{'Method':<40} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@50':>8} {'vs Van':>8}")
    print("-" * 88)
    print(f"{'Vanilla BS-50':<40} {van_agg['R@1']:>8.4f} {van_agg['R@5']:>8.4f} "
          f"{van_agg['R@10']:>8.4f} {van_agg['R@50']:>8.4f} {'base':>8}")
    print(f"{'Pointwise Focal λ=' + str(best_focal_lam):<40} {focal_agg['R@1']:>8.4f} "
          f"{focal_agg['R@5']:>8.4f} {focal_agg['R@10']:>8.4f} {focal_agg['R@50']:>8.4f} "
          f"{vs_van(focal_agg['R@1']):>+7.1f}%")

    # 5-seed listwise at val-selected λ.
    print(f"\n{'Seed':<8} {'R@1':>8} {'vs Van':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 50)
    seed_aggs, seed_r1s, seed_hits = {}, [], {}
    for seed in SEEDS:
        lw_agg, lw_hh = eval_peruser(seed_scorers[seed], True, best_lw_lam,
                                     test_h, test_s, test_u, test_l, code_length)
        seed_aggs[str(seed)] = lw_agg
        seed_r1s.append(lw_agg["R@1"])
        seed_hits[seed] = lw_hh
        print(f"{seed:<8} {lw_agg['R@1']:>8.4f} {vs_van(lw_agg['R@1']):>+7.1f}% "
              f"{lw_agg['R@5']:>8.4f} {lw_agg['R@10']:>8.4f}")

    mean_r1 = float(np.mean(seed_r1s))
    std_r1 = float(np.std(seed_r1s))
    print(f"{'Mean±Std':<8} {mean_r1:>7.4f}±{std_r1:.4f} {vs_van(mean_r1):>+7.1f}%")
    # Per-sample listwise hit = majority vote across the 5 seeds, so significance
    # tests the REPORTED 5-seed row (not a single seed).
    lw_hits = {k: (np.mean([seed_hits[s][k] for s in SEEDS], axis=0) >= 0.5)
               for k in (1, 10)}

    # Oracle
    oracle = van_agg["R@50"]  # perfect ranking → Oracle R@k == R@50
    print(f"\n--- Oracle (perfect ranking) ---")
    print(f"Oracle R@1 = R@50 = {oracle:.4f}")
    print(f"Scoring Efficiency: Vanilla={van_agg['R@1'] / max(oracle, 1e-8) * 100:.1f}%, "
          f"Best Scorer={mean_r1 / max(oracle, 1e-8) * 100:.1f}%")

    # ── Persist per-user hits (for genrec_v2/run_statistical_significance.py) ──
    # Per-seed listwise hits (5×N) → significance tests the per-seed effect the reported
    # 5-seed-mean R@1 describes, not an ad-hoc ensemble (review-fix 2026-06-13).
    np.savez(
        output_dir / "per_user_hits.npz",
        user_ids=np.array(test_user_ids),
        vanilla_hit1=van_hits[1], vanilla_hit10=van_hits[10],
        focal_hit1=focal_hits[1], focal_hit10=focal_hits[10],
        listwise_hit1=lw_hits[1], listwise_hit10=lw_hits[10],  # voted ensemble (kept)
        listwise_hit1_seeds=np.stack([seed_hits[s][1] for s in SEEDS]),
        listwise_hit10_seeds=np.stack([seed_hits[s][10] for s in SEEDS]),
    )

    # ── Persist results (R@k + MRR + nDCG@5/10) ──
    metric_keys = [f"R@{k}" for k in KS] + ["MRR", "nDCG@5", "nDCG@10"]
    lw_mean = {m: float(np.mean([seed_aggs[str(s)][m] for s in SEEDS])) for m in metric_keys}
    results = {
        "dataset": cli.dataset,
        "checkpoint": str(cli.checkpoint),
        "eval": {"beam_width": BEAM_WIDTH, "code_length": code_length,
                 "mode": "per_sample_beam", "n_test_samples": int(test_l.shape[0])},
        "lambda_grid": list(LAMBDA_GRID),
        "rows": {
            "vanilla": dict(van_agg),
            "pointwise_focal": {"lambda": best_focal_lam, **dict(focal_agg)},
            "listwise_bce_5seed": {
                "lambda": best_lw_lam, "mean_R@1": mean_r1, "std_R@1": std_r1,
                "vs_vanilla_pct": vs_van(mean_r1), "mean": lw_mean, "per_seed": seed_aggs,
            },
            "oracle": {f"R@{k}": oracle for k in KS},
        },
        "scoring_efficiency": {
            "vanilla": van_agg["R@1"] / max(oracle, 1e-8),
            "best_scorer": mean_r1 / max(oracle, 1e-8),
        },
        "runtime_sec": time.time() - t_start,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {output_dir / 'results.json'} and per_user_hits.npz")
    print(f"\nDone [{(time.time() - t_start) / 60:.0f} min].")


if __name__ == "__main__":
    main()
