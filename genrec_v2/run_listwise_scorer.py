"""EXP-014: Listwise Calibration Scorer for beam search reranking.

Tune on val, evaluate top-3 on test.
Compares: vanilla BS / pointwise scorer / listwise scorer.

Usage:
    CUDA_VISIBLE_DEVICES=3 python -u genrec_v2/run_listwise_scorer.py
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

# ── Config ─────────────────────────────────────────────────────────

BASE_DIR = Path("/home/lishazhai/workspace/GR4AD")
SEMANTIC_DIR = BASE_DIR / "output/sbert_baseline_20260508_153306/semantic_ids"
CKPT_PATH = BASE_DIR / "experiments/genrec_v2_exposure_bias/baseline_retrain/best_model.pt"  # match headline (was B_nohot — review-fix 2026-06-16)
BEAM_WIDTH = 50
HIDDEN_DIM = 128
CODE_LENGTH = 4
MAX_HIST = 128
DEVICE = torch.device("cuda")

LOSS_TYPES = ["bce", "listmle", "approx_ndcg"]
LABEL_TYPES = ["binary", "soft"]
LAMBDAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]


def build_model() -> GenRecV2Model:
    """Build and load the frozen generator."""
    enc = HistorySequenceEncoder(
        UserEncoderConfig(
            input_dim=384, hidden_dim=HIDDEN_DIM, num_heads=4,
            num_layers=2, dropout=0.1, max_history_length=MAX_HIST,
        )
    )
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
    """Load MIND data and split into val/test."""
    all_samples = build_samples(
        str(BASE_DIR / "data/mind_small_raw/train/MINDsmall_train/behaviors.tsv"),
        mode="B",
    )
    item_embeddings = np.load(SEMANTIC_DIR / "item_embeddings.npy")
    item_ids = json.loads((SEMANTIC_DIR / "item_ids.json").read_text())
    mapper = json.loads((SEMANTIC_DIR / "item_to_code.json").read_text())
    cfi = {k: tuple(v) for k, v in mapper.items()}
    iti = {nid: i for i, nid in enumerate(item_ids)}

    vs = [
        s for s in all_samples
        if s["target"] in cfi and any(h in iti for h in s["history"][:50])
    ]
    user_groups: dict[str, list] = {}
    for s in vs:
        user_groups.setdefault(s["user_id"], []).append(s)
    uids = sorted(user_groups.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(uids)
    n = len(uids)
    tn, vn = int(n * 0.7), int(n * 0.15)
    val_uids = set(uids[tn : tn + vn])
    test_uids = set(uids[tn + vn :])
    vsl = [s for uid in val_uids for s in user_groups[uid]]
    tsl = [s for uid in test_uids for s in user_groups[uid]]

    cti: dict[tuple[int, ...], list[str]] = {}
    for nid, c in cfi.items():
        cti.setdefault(c, []).append(nid)

    return item_ids, cfi, iti, cti, vsl, tsl, item_embeddings


def main() -> None:
    t_start = time.time()

    print("Loading model and data...")
    model = build_model()
    item_ids, cfi, iti, cti, vsl, tsl, item_embeddings = prepare_data()
    dec = model.decoder

    # ============================================================
    # STEP 1: Collect beam data (deduped by user, single pass)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1: Collect beam calibration data (val, deduped by user)")
    print("=" * 60)

    t0 = time.time()
    val_data = collect_beam_calibration_data(
        model, vsl, iti, cfi, item_embeddings, MAX_HIST,
        DEVICE, item_ids, beam_width=BEAM_WIDTH, batch_size=32,
    )
    n_samples = val_data["hidden"].shape[0]
    n_hits_bin = (val_data["labels_binary"].max(dim=1).values > 0).sum().item()
    n_hits_soft = (val_data["labels_soft"].max(dim=1).values > 0).sum().item()
    print(f"  {n_samples} samples, {n_hits_bin} with binary hit ({n_hits_bin/n_samples*100:.1f}%)")
    print(f"  {n_hits_soft} with soft match > 0 ({n_hits_soft/n_samples*100:.1f}%)")
    print(f"  [{time.time()-t0:.0f}s]")

    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # STEP 2: Train scorers (6 listwise + 1 pointwise)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Train scorers")
    print("=" * 60)

    trained_scorers: dict[str, ListwiseScorer] = {}
    for loss_type in LOSS_TYPES:
        for label_type in LABEL_TYPES:
            name = f"{loss_type}_{label_type}"
            labels_key = f"labels_{label_type}"
            data = {
                "hidden": val_data["hidden"],
                "beam_scores": val_data["beam_scores"],
                "user_states": val_data["user_states"],
                "labels": val_data[labels_key],
            }
            scorer = ListwiseScorer(
                hidden_dim=HIDDEN_DIM, code_length=CODE_LENGTH,
                d_model=128, nhead=4, num_layers=2,
                dim_feedforward=256, dropout=0.1,
            )
            t0 = time.time()
            history = train_listwise_scorer(
                scorer, data, device=DEVICE,
                loss_type=loss_type, epochs=50,
                batch_size=32, lr=1e-3, patience=10,
            )
            ep = len(history)
            vl = history[-1]["val_loss"]
            print(f"  {name:24s}: {ep} ep, val_loss={vl:.4f} [{time.time()-t0:.0f}s]")
            trained_scorers[name] = scorer.cpu()

    # Pointwise baseline (Focal BCE)
    print("  Training pointwise baseline (Focal BCE)...")

    class FocalBCELoss(nn.Module):
        def __init__(self, g: float = 2.0):
            super().__init__()
            self.g = g
        def forward(self, lo: torch.Tensor, tg: torch.Tensor) -> torch.Tensor:
            p = torch.sigmoid(lo)
            ce = F.binary_cross_entropy_with_logits(lo, tg, reduction="none")
            pt = p * tg + (1 - p) * (1 - tg)
            return ((1 - pt) ** self.g * ce).mean()

    pw_scorer = CalibrationScorer(hidden_dim=HIDDEN_DIM).to(DEVICE)
    pw_opt = torch.optim.AdamW(pw_scorer.parameters(), lr=1e-3)
    focal = FocalBCELoss(g=2.0)
    pw_X = val_data["hidden"].reshape(-1, CODE_LENGTH, HIDDEN_DIM)
    pw_y = val_data["labels_binary"].reshape(-1)
    N_pw = pw_X.shape[0]
    perm_pw = torch.randperm(N_pw)
    vn_pw = int(N_pw * 0.3)
    pw_Xtr = pw_X[perm_pw[vn_pw:]].to(DEVICE)
    pw_ytr = pw_y[perm_pw[vn_pw:]].to(DEVICE)
    pw_Xva = pw_X[perm_pw[:vn_pw]].to(DEVICE)
    pw_yva = pw_y[perm_pw[:vn_pw]].to(DEVICE)
    pw_ds = TensorDataset(pw_Xtr, pw_ytr)
    pw_loader = DataLoader(pw_ds, batch_size=256, shuffle=True)
    best_pw = float("inf")
    pw_pat = 0
    pw_best_state = None
    for ep in range(50):
        pw_scorer.train()
        for Xb, yb in pw_loader:
            lo = pw_scorer(Xb).squeeze(-1)
            loss = focal(lo, yb)
            pw_opt.zero_grad(set_to_none=True)
            loss.backward()
            pw_opt.step()
        pw_scorer.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(
                pw_scorer(pw_Xva).squeeze(-1), pw_yva
            ).item()
        if vl < best_pw:
            best_pw = vl; pw_pat = 0
            pw_best_state = {k: v.clone() for k, v in pw_scorer.state_dict().items()}
        else:
            pw_pat += 1
            if pw_pat >= 10:
                break
    if pw_best_state:
        pw_scorer.load_state_dict(pw_best_state)
    pw_scorer = pw_scorer.cpu()
    print(f"  pointwise_focal:        {ep+1} ep")

    del pw_X, pw_y, pw_Xtr, pw_ytr, pw_Xva, pw_yva
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # STEP 3: Lambda sweep on val
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Lambda sweep on val")
    print("=" * 60)

    val_hidden = val_data["hidden"]
    val_bscores = val_data["beam_scores"]
    val_ustates = val_data["user_states"]
    val_labels = val_data["labels_binary"]  # eval always on binary hit

    chunk_size = 128
    all_results: list[tuple[str, float, float, float, float, float]] = []

    print(f"\n{'Method':<28} {'Lambda':<8} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 62)

    # Vanilla (beams already sorted by score)
    van_stats: dict[int, list[int]] = {k: [0, 0] for k in [1, 5, 10, 50]}
    for i in range(val_hidden.shape[0]):
        for k in [1, 5, 10, 50]:
            hit = val_labels[i][:k].max().item() > 0
            van_stats[k][0] += int(hit)
            van_stats[k][1] += 1
    van_r = {k: van_stats[k][0] / max(1, van_stats[k][1]) for k in van_stats}
    print(f"{'vanilla':<28} {'—':<8} {van_r[1]:>8.4f} {van_r[5]:>8.4f} {van_r[10]:>8.4f}")
    all_results.append(("vanilla", 0.0, van_r[1], van_r[5], van_r[10], van_r[50]))

    # Pointwise
    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        pw_dev = pw_scorer.to(DEVICE)
        stats: dict[int, list[int]] = {k: [0, 0] for k in [1, 5, 10, 50]}
        for start in range(0, val_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, val_hidden.shape[0])
            h = val_hidden[start:end].to(DEVICE)
            s = val_bscores[start:end]
            l = val_labels[start:end]
            C, K = s.shape
            with torch.no_grad():
                pw_lo = pw_dev(h.reshape(C * K, CODE_LENGTH, HIDDEN_DIM)).squeeze(-1)
                pw_p = torch.sigmoid(pw_lo).cpu().reshape(C, K)
            for ci in range(C):
                final = s[ci] / CODE_LENGTH + lam * torch.log(pw_p[ci].clamp(min=1e-8))
                order = final.argsort(descending=True)
                for kv in [1, 5, 10, 50]:
                    stats[kv][0] += int(l[ci][order[:kv]].max().item() > 0)
                    stats[kv][1] += 1
        r = {k: stats[k][0] / max(1, stats[k][1]) for k in stats}
        print(f"{'pointwise_focal':<28} {lam:<8.2f} {r[1]:>8.4f} {r[5]:>8.4f} {r[10]:>8.4f}")
        all_results.append(("pointwise_focal", lam, r[1], r[5], r[10], r[50]))
        pw_dev = pw_dev.cpu()

    # Listwise scorers
    for name, scorer in trained_scorers.items():
        sc_dev = scorer.to(DEVICE)
        for lam in LAMBDAS:
            stats = {k: [0, 0] for k in [1, 5, 10, 50]}
            for start in range(0, val_hidden.shape[0], chunk_size):
                end = min(start + chunk_size, val_hidden.shape[0])
                h = val_hidden[start:end].to(DEVICE)
                s = val_bscores[start:end].to(DEVICE)
                u = val_ustates[start:end].to(DEVICE)
                l = val_labels[start:end]
                with torch.no_grad():
                    lw_sc = sc_dev(h, s, user_state=u).cpu()
                for ci in range(lw_sc.shape[0]):
                    if lam == 0.0:
                        final = lw_sc[ci]
                    else:
                        final = val_bscores[start + ci] / CODE_LENGTH + lam * lw_sc[ci]
                    order = final.argsort(descending=True)
                    for kv in [1, 5, 10, 50]:
                        stats[kv][0] += int(l[ci][order[:kv]].max().item() > 0)
                        stats[kv][1] += 1
            r = {k: stats[k][0] / max(1, stats[k][1]) for k in stats}
            lam_s = f"{lam:.2f}" if lam > 0 else "pure"
            print(f"{name:<28} {lam_s:<8} {r[1]:>8.4f} {r[5]:>8.4f} {r[10]:>8.4f}")
            all_results.append((name, lam, r[1], r[5], r[10], r[50]))
        sc_dev = sc_dev.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    non_van = [r for r in all_results if r[0] != "vanilla"]
    non_van.sort(key=lambda x: -x[2])
    top3 = non_van[:3]

    print("\n--- VAL Top 3 ---")
    for i, (sn, lam, r1, r5, r10, r50) in enumerate(top3):
        print(f"  #{i+1}: {sn} λ={lam} → R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f}")
    print(f"  Baseline: vanilla → R@1={van_r[1]:.4f}")

    # ============================================================
    # STEP 4: Test evaluation
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: Test evaluation (top-3 + vanilla + pointwise)")
    print("=" * 60)

    t0 = time.time()
    print("Collecting test beam data (deduped by user)...")
    test_data = collect_beam_calibration_data(
        model, tsl, iti, cfi, item_embeddings, MAX_HIST,
        DEVICE, item_ids, beam_width=BEAM_WIDTH, batch_size=32,
    )
    n_test = test_data["hidden"].shape[0]
    print(f"  {n_test} test samples [{time.time()-t0:.0f}s]")

    test_hidden = test_data["hidden"]
    test_bscores = test_data["beam_scores"]
    test_ustates = test_data["user_states"]
    test_labels = test_data["labels_binary"]

    def test_eval(scorer_obj, lam: float, is_listwise: bool) -> dict[int, float]:
        stats: dict[int, list[int]] = {k: [0, 0] for k in [1, 5, 10, 50]}
        if scorer_obj is not None:
            scorer_obj = scorer_obj.to(DEVICE)
        for start in range(0, n_test, chunk_size):
            end = min(start + chunk_size, n_test)
            h = test_hidden[start:end].to(DEVICE)
            s = test_bscores[start:end]
            u = test_ustates[start:end].to(DEVICE) if is_listwise else None
            l = test_labels[start:end]
            C = h.shape[0]
            if scorer_obj is None:
                for ci in range(C):
                    for kv in [1, 5, 10, 50]:
                        stats[kv][0] += int(l[ci][:kv].max().item() > 0)
                        stats[kv][1] += 1
            elif is_listwise:
                s_dev = s.to(DEVICE)
                with torch.no_grad():
                    lw = scorer_obj(h, s_dev, user_state=u).cpu()
                for ci in range(C):
                    final = s[ci] / CODE_LENGTH + lam * lw[ci] if lam > 0 else lw[ci]
                    order = final.argsort(descending=True)
                    for kv in [1, 5, 10, 50]:
                        stats[kv][0] += int(l[ci][order[:kv]].max().item() > 0)
                        stats[kv][1] += 1
            else:
                K = h.shape[1]
                with torch.no_grad():
                    pw_lo = scorer_obj(h.reshape(C * K, CODE_LENGTH, HIDDEN_DIM)).squeeze(-1)
                    pw_p = torch.sigmoid(pw_lo).cpu().reshape(C, K)
                for ci in range(C):
                    final = s[ci] / CODE_LENGTH + lam * torch.log(pw_p[ci].clamp(min=1e-8))
                    order = final.argsort(descending=True)
                    for kv in [1, 5, 10, 50]:
                        stats[kv][0] += int(l[ci][order[:kv]].max().item() > 0)
                        stats[kv][1] += 1
        if scorer_obj is not None:
            scorer_obj = scorer_obj.cpu()
        return {k: stats[k][0] / max(1, stats[k][1]) for k in stats}

    van_test = test_eval(None, 0.0, False)
    pw_test = test_eval(pw_scorer, 0.35, False)

    print(f"\n{'Method':<36} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@50':>8} {'vs Van':>8}")
    print("-" * 88)
    print(f"{'Vanilla BS-50':<36} {van_test[1]:>8.4f} {van_test[5]:>8.4f} {van_test[10]:>8.4f} {van_test[50]:>8.4f} {'base':>8}")
    d_pw = (pw_test[1] - van_test[1]) / max(1e-8, van_test[1]) * 100
    print(f"{'Pointwise Focal λ=0.35':<36} {pw_test[1]:>8.4f} {pw_test[5]:>8.4f} {pw_test[10]:>8.4f} {pw_test[50]:>8.4f} {d_pw:>+7.1f}%")

    for i, (sn, lam, _, _, _, _) in enumerate(top3):
        is_lw = sn != "pointwise_focal"
        scorer_obj = trained_scorers[sn] if is_lw else pw_scorer
        r = test_eval(scorer_obj, lam, is_lw)
        delta = (r[1] - van_test[1]) / max(1e-8, van_test[1]) * 100
        label = f"#{i+1} {sn} λ={lam}"
        print(f"{label:<36} {r[1]:>8.4f} {r[5]:>8.4f} {r[10]:>8.4f} {r[50]:>8.4f} {delta:>+7.1f}%")

    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.0f}min)")


if __name__ == "__main__":
    main()
