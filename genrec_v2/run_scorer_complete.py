"""Complete scorer evaluation: nnPU pointwise + Focal pointwise + 5-seed Listwise BCE.

Uses per-sample beam search (bug-fixed) on baseline_retrain checkpoint.
Reuses beam data from the fixed run if available, otherwise re-collects.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u genrec_v2/run_scorer_complete.py
"""
from __future__ import annotations

import gc
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from genrec_v2.calibration.scorer import (
    CalibrationScorer,
    ListwiseScorer,
    collect_beam_calibration_data,
    train_listwise_scorer,
)
from genrec_v2.config import GenRecV2Config
from genrec_v2.data.build_samples import build_samples
from genrec_v2.data.dataset import GenRecV2Dataset, make_collator
from genrec_v2.model.model import GenRecV2Model
from mind_genrec.model.ar_decoder import ARDecoderConfig, CodeAutoregressiveDecoder
from mind_genrec.model.user_encoder import HistorySequenceEncoder, UserEncoderConfig

BASE_DIR = Path("/home/lishazhai/workspace/GR4AD")
SEMANTIC_DIR = BASE_DIR / "output/sbert_baseline_20260508_153306/semantic_ids"
CKPT_PATH = BASE_DIR / "experiments/genrec_v2_exposure_bias/baseline_retrain/best_model.pt"
BEAM_WIDTH = 50
HIDDEN_DIM = 128
CODE_LENGTH = 4
MAX_HIST = 128
DEVICE = torch.device("cuda")
LAMBDAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
SEEDS = [42, 123, 7, 999, 2024]


class NNPULoss(nn.Module):
    def __init__(self, pi: float = 0.02):
        super().__init__()
        self.pi = pi

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        p = torch.sigmoid(logits)
        pos_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
        neg_loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")
        n_pos = pos_mask.sum().clamp(min=1)
        n_neg = neg_mask.sum().clamp(min=1)
        r_pos_pos = (pos_loss * pos_mask).sum() / n_pos
        r_pos_neg = (neg_loss * pos_mask).sum() / n_pos
        r_u_neg = (neg_loss * neg_mask).sum() / n_neg
        neg_risk = torch.clamp(r_u_neg - self.pi * r_pos_neg, min=0.0)
        return self.pi * r_pos_pos + neg_risk


class FocalBCELoss(nn.Module):
    def __init__(self, g: float = 2.0):
        super().__init__()
        self.g = g

    def forward(self, lo: torch.Tensor, tg: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(lo)
        ce = F.binary_cross_entropy_with_logits(lo, tg, reduction="none")
        pt = p * tg + (1 - p) * (1 - tg)
        return ((1 - pt) ** self.g * ce).mean()


def build_model() -> GenRecV2Model:
    enc = HistorySequenceEncoder(UserEncoderConfig(
        input_dim=384, hidden_dim=HIDDEN_DIM, num_heads=4,
        num_layers=2, dropout=0.1, max_history_length=MAX_HIST,
    ))
    dc = ARDecoderConfig(
        hidden_dim=HIDDEN_DIM, codebook_size=256, code_length=CODE_LENGTH,
        num_heads=4, num_layers=2, dropout=0.1,
    )
    dec = CodeAutoregressiveDecoder(dc)
    item_embeddings = np.load(SEMANTIC_DIR / "item_embeddings.npy")
    cb_data = np.load(SEMANTIC_DIR / "codebooks.npz")
    cbs = nn.ModuleList([nn.Embedding(256, 384) for _ in range(4)])
    for i in range(4):
        cbs[i].weight.data.copy_(torch.tensor(cb_data[f"codebook_{i}"]))
    model = GenRecV2Model(
        encoder=enc, decoder=dec, codebook=cbs, hot_news_fusion=None,
        embedding_table=torch.tensor(item_embeddings, dtype=torch.float32),
    )
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def prepare_data():
    all_samples = build_samples(
        str(BASE_DIR / "data/mind_small_raw/train/MINDsmall_train/behaviors.tsv"), mode="B",
    )
    item_embeddings = np.load(SEMANTIC_DIR / "item_embeddings.npy")
    item_ids = json.loads((SEMANTIC_DIR / "item_ids.json").read_text())
    mapper = json.loads((SEMANTIC_DIR / "item_to_code.json").read_text())
    cfi = {k: tuple(v) for k, v in mapper.items()}
    iti = {nid: i for i, nid in enumerate(item_ids)}
    vs = [s for s in all_samples if s["target"] in cfi and any(h in iti for h in s["history"][:50])]
    user_groups: dict[str, list] = {}
    for s in vs:
        user_groups.setdefault(s["user_id"], []).append(s)
    uids = sorted(user_groups.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(uids)
    n = len(uids)
    tn, vn = int(n * 0.7), int(n * 0.15)
    val_uids = set(uids[tn: tn + vn])
    test_uids = set(uids[tn + vn:])
    vsl = [s for uid in val_uids for s in user_groups[uid]]
    tsl = [s for uid in test_uids for s in user_groups[uid]]
    return item_ids, cfi, iti, vsl, tsl, item_embeddings


def train_pointwise(X_tr, y_tr, X_va, y_va, loss_fn, seed=42):
    torch.manual_seed(seed)
    scorer = CalibrationScorer(hidden_dim=HIDDEN_DIM).to(DEVICE)
    opt = torch.optim.AdamW(scorer.parameters(), lr=1e-3)
    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    best_vl = float("inf")
    pat = 0
    best_state = None
    for ep in range(50):
        scorer.train()
        for Xb, yb in loader:
            lo = scorer(Xb).squeeze(-1)
            loss = loss_fn(lo, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        scorer.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(scorer(X_va).squeeze(-1), y_va).item()
        if vl < best_vl:
            best_vl = vl; pat = 0
            best_state = {k: v.clone() for k, v in scorer.state_dict().items()}
        else:
            pat += 1
            if pat >= 10:
                break
    if best_state:
        scorer.load_state_dict(best_state)
    return scorer.cpu()


def eval_scorer(scorer, is_listwise, lam, hidden, bscores, ustates, labels, code_length):
    chunk_size = 128
    N = hidden.shape[0]
    stats = {k: [0, 0] for k in [1, 5, 10, 50]}
    if scorer is not None:
        scorer = scorer.to(DEVICE)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        h = hidden[start:end].to(DEVICE)
        s = bscores[start:end]
        u = ustates[start:end].to(DEVICE) if is_listwise else None
        l = labels[start:end]
        C = h.shape[0]
        if scorer is None:
            for ci in range(C):
                for kv in [1, 5, 10, 50]:
                    stats[kv][0] += int(l[ci][:kv].max().item() > 0)
                    stats[kv][1] += 1
        elif is_listwise:
            s_dev = s.to(DEVICE)
            with torch.no_grad():
                lw = scorer(h, s_dev, user_state=u).cpu()
            for ci in range(C):
                final = s[ci] / code_length + lam * lw[ci] if lam > 0 else lw[ci]
                order = final.argsort(descending=True)
                for kv in [1, 5, 10, 50]:
                    stats[kv][0] += int(l[ci][order[:kv]].max().item() > 0)
                    stats[kv][1] += 1
        else:
            K = h.shape[1]
            with torch.no_grad():
                lo = scorer(h.reshape(C * K, CODE_LENGTH, HIDDEN_DIM)).squeeze(-1)
                p = torch.sigmoid(lo).cpu().reshape(C, K)
            for ci in range(C):
                final = s[ci] / code_length + lam * torch.log(p[ci].clamp(min=1e-8))
                order = final.argsort(descending=True)
                for kv in [1, 5, 10, 50]:
                    stats[kv][0] += int(l[ci][order[:kv]].max().item() > 0)
                    stats[kv][1] += 1
    if scorer is not None:
        scorer = scorer.cpu()
    return {k: stats[k][0] / max(1, stats[k][1]) for k in stats}


def main():
    t_start = time.time()
    print("Loading model and data...")
    model = build_model()
    item_ids, cfi, iti, vsl, tsl, item_embeddings = prepare_data()

    # ── Collect beam data (per-sample, fixed) ──
    print("\n" + "=" * 60)
    print("Collecting val beam data (per-sample)...")
    t0 = time.time()
    val_data = collect_beam_calibration_data(
        model, vsl, iti, cfi, item_embeddings, MAX_HIST,
        DEVICE, item_ids, beam_width=BEAM_WIDTH, batch_size=32,
    )
    print(f"  {val_data['hidden'].shape[0]} val samples [{time.time()-t0:.0f}s]")

    print("Collecting test beam data (per-sample)...")
    t0 = time.time()
    test_data = collect_beam_calibration_data(
        model, tsl, iti, cfi, item_embeddings, MAX_HIST,
        DEVICE, item_ids, beam_width=BEAM_WIDTH, batch_size=32,
    )
    print(f"  {test_data['hidden'].shape[0]} test samples [{time.time()-t0:.0f}s]")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Prepare pointwise data ──
    pw_X = val_data["hidden"].reshape(-1, CODE_LENGTH, HIDDEN_DIM)
    pw_y = val_data["labels_binary"].reshape(-1)
    N_pw = pw_X.shape[0]
    perm = torch.randperm(N_pw)
    vn_pw = int(N_pw * 0.3)
    X_tr = pw_X[perm[vn_pw:]].to(DEVICE)
    y_tr = pw_y[perm[vn_pw:]].to(DEVICE)
    X_va = pw_X[perm[:vn_pw]].to(DEVICE)
    y_va = pw_y[perm[:vn_pw]].to(DEVICE)

    # ══════════════════════════════════════════════
    # PART 1: Pointwise scorers (Focal + nnPU sweep)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 1: Pointwise scorers")
    print("=" * 60)

    # Focal
    print("\nTraining Pointwise Focal BCE...")
    focal_scorer = train_pointwise(X_tr, y_tr, X_va, y_va, FocalBCELoss(g=2.0))
    print("  Done")

    # nnPU sweep
    PI_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]
    nnpu_scorers = {}
    for pi in PI_VALUES:
        print(f"Training nnPU pi={pi}...")
        nnpu_scorers[pi] = train_pointwise(X_tr, y_tr, X_va, y_va, NNPULoss(pi=pi))
        print(f"  Done")

    del X_tr, y_tr, X_va, y_va, pw_X, pw_y
    gc.collect()
    torch.cuda.empty_cache()

    # Val sweep for all pointwise
    print("\nPointwise val sweep:")
    print(f"{'Method':<28} {'Lambda':<8} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 62)

    pw_results = []
    for lam in [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]:
        r = eval_scorer(focal_scorer, False, lam,
                        val_data["hidden"], val_data["beam_scores"], val_data["user_states"],
                        val_data["labels_binary"], CODE_LENGTH)
        print(f"{'focal':<28} {lam:<8.2f} {r[1]:>8.4f} {r[5]:>8.4f} {r[10]:>8.4f}")
        pw_results.append(("focal", lam, r))

    for pi, scorer in nnpu_scorers.items():
        for lam in [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]:
            r = eval_scorer(scorer, False, lam,
                            val_data["hidden"], val_data["beam_scores"], val_data["user_states"],
                            val_data["labels_binary"], CODE_LENGTH)
            name = f"nnpu_pi{pi}"
            print(f"{name:<28} {lam:<8.2f} {r[1]:>8.4f} {r[5]:>8.4f} {r[10]:>8.4f}")
            pw_results.append((name, lam, r))

    # Best pointwise configs
    pw_results.sort(key=lambda x: -x[2][1])
    best_focal = max([x for x in pw_results if x[0] == "focal"], key=lambda x: x[2][1])
    best_nnpu = max([x for x in pw_results if x[0] != "focal"], key=lambda x: x[2][1])
    print(f"\nBest Focal: {best_focal[0]} λ={best_focal[1]} R@1={best_focal[2][1]:.4f}")
    print(f"Best nnPU:  {best_nnpu[0]} λ={best_nnpu[1]} R@1={best_nnpu[2][1]:.4f}")

    # ══════════════════════════════════════════════
    # PART 2: 5-seed Listwise BCE binary
    # ══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 2: 5-seed Listwise BCE binary (λ=0.5)")
    print("=" * 60)

    lw_data = {
        "hidden": val_data["hidden"],
        "beam_scores": val_data["beam_scores"],
        "user_states": val_data["user_states"],
        "labels": val_data["labels_binary"],
    }

    seed_scorers = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        print(f"\nSeed {seed}:")
        scorer = ListwiseScorer(
            hidden_dim=HIDDEN_DIM, code_length=CODE_LENGTH,
            d_model=128, nhead=4, num_layers=2,
            dim_feedforward=256, dropout=0.1,
        )
        history = train_listwise_scorer(
            scorer, lw_data, device=DEVICE,
            loss_type="bce", epochs=50, batch_size=32, lr=1e-3, patience=10,
        )
        print(f"  {len(history)} ep, val_loss={history[-1]['val_loss']:.4f}")
        seed_scorers[seed] = scorer.cpu()

    # ══════════════════════════════════════════════
    # PART 3: Test evaluation
    # ══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 3: Test evaluation")
    print("=" * 60)

    test_h = test_data["hidden"]
    test_s = test_data["beam_scores"]
    test_u = test_data["user_states"]
    test_l = test_data["labels_binary"]

    # Vanilla
    van = eval_scorer(None, False, 0, test_h, test_s, test_u, test_l, CODE_LENGTH)

    print(f"\n{'Method':<36} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@50':>8} {'vs Van':>8}")
    print("-" * 88)
    print(f"{'Vanilla BS-50':<36} {van[1]:>8.4f} {van[5]:>8.4f} {van[10]:>8.4f} {van[50]:>8.4f} {'base':>8}")

    # Focal
    r = eval_scorer(focal_scorer, False, best_focal[1], test_h, test_s, test_u, test_l, CODE_LENGTH)
    d = (r[1] - van[1]) / max(1e-8, van[1]) * 100
    print(f"{'Pointwise Focal λ=' + str(best_focal[1]):<36} {r[1]:>8.4f} {r[5]:>8.4f} {r[10]:>8.4f} {r[50]:>8.4f} {d:>+7.1f}%")

    # Best nnPU
    pi = float(best_nnpu[0].split("pi")[1])
    nnpu_sc = nnpu_scorers[pi]
    r = eval_scorer(nnpu_sc, False, best_nnpu[1], test_h, test_s, test_u, test_l, CODE_LENGTH)
    d = (r[1] - van[1]) / max(1e-8, van[1]) * 100
    print(f"{'Pointwise nnPU ' + best_nnpu[0] + ' λ=' + str(best_nnpu[1]):<36} {r[1]:>8.4f} {r[5]:>8.4f} {r[10]:>8.4f} {r[50]:>8.4f} {d:>+7.1f}%")

    # 5-seed listwise
    print(f"\n{'Seed':<8} {'R@1':>8} {'vs Van':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 50)
    seed_r1s = []
    for seed in SEEDS:
        r = eval_scorer(seed_scorers[seed], True, 0.5, test_h, test_s, test_u, test_l, CODE_LENGTH)
        d = (r[1] - van[1]) / max(1e-8, van[1]) * 100
        print(f"{seed:<8} {r[1]:>8.4f} {d:>+7.1f}% {r[5]:>8.4f} {r[10]:>8.4f}")
        seed_r1s.append(r[1])

    mean_r1 = np.mean(seed_r1s)
    std_r1 = np.std(seed_r1s)
    mean_d = (mean_r1 - van[1]) / max(1e-8, van[1]) * 100
    print(f"{'Mean±Std':<8} {mean_r1:>7.4f}±{std_r1:.4f} {mean_d:>+7.1f}%")

    # Oracle
    print(f"\n--- Oracle (perfect ranking) ---")
    oracle = eval_scorer(None, False, 0, test_h, test_s, test_u, test_l, CODE_LENGTH)
    # Oracle = R@50 since perfect ranking puts all hits at top
    print(f"Oracle R@1 = R@50 = {van[50]:.4f}")
    print(f"Scoring Efficiency: Vanilla={van[1]/van[50]*100:.1f}%, Best Scorer={mean_r1/van[50]*100:.1f}%")

    print(f"\nTotal: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.0f}min)")


if __name__ == "__main__":
    main()
