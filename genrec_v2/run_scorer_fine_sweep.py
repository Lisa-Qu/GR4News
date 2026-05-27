"""Fine sweep: Focal Loss gamma + nnPU pi for calibration scorer."""
from __future__ import annotations

import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from genrec_v2.calibration.scorer import CalibrationScorer, collect_calibration_data
from genrec_v2.config import GenRecV2Config
from genrec_v2.data.build_samples import build_samples
from genrec_v2.data.dataset import GenRecV2Dataset, make_collator
from genrec_v2.model.model import GenRecV2Model
from mind_genrec.model.ar_decoder import ARDecoderConfig, CodeAutoregressiveDecoder
from mind_genrec.model.user_encoder import HistorySequenceEncoder, UserEncoderConfig


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** self.gamma * ce).mean()


def make_nnpu(pi: float):
    def fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = targets == 1
        unlab = targets == 0
        np_ = max(1, pos.sum().item())
        nu = max(1, unlab.sum().item())
        lp = F.binary_cross_entropy_with_logits(
            logits[pos], targets[pos], reduction="sum"
        ) / np_
        lu = F.binary_cross_entropy_with_logits(
            logits[unlab], torch.zeros_like(logits[unlab]), reduction="sum"
        ) / nu
        lpan = F.binary_cross_entropy_with_logits(
            logits[pos], torch.zeros_like(logits[pos]), reduction="sum"
        ) / np_
        return lp + torch.clamp(lu - pi * lpan, min=0.0)
    return fn


def train_one(name, loss_fn, X_tr, y_tr, X_va, y_va, device, epochs=50, lr=1e-3):
    scorer = CalibrationScorer(hidden_dim=128).to(device)
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr)
    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    best_val = float("inf")
    patience = 0
    for _ in range(epochs):
        scorer.train()
        for Xb, yb in train_loader:
            logits = scorer(Xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        scorer.eval()
        with torch.no_grad():
            vloss = F.binary_cross_entropy_with_logits(
                scorer(X_va).squeeze(-1), y_va
            ).item()
        if vloss < best_val:
            best_val = vloss
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    scorer.eval()
    with torch.no_grad():
        vp = torch.sigmoid(scorer(X_va).squeeze(-1))
        ph = vp[y_va == 1].mean().item()
        pm = vp[y_va == 0].mean().item()
    return {"name": name, "val_bce": best_val, "P_hit": ph, "P_miss": pm, "gap": ph - pm}


def main():
    base_dir = Path("/home/lishazhai/workspace/GR4AD")
    semantic_dir = base_dir / "output/sbert_baseline_20260508_153306/semantic_ids"
    ckpt_path = base_dir / "experiments/genrec_v2_exposure_bias/baseline_retrain/best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GenRecV2Config.proxy(
        train_tsv=str(base_dir / "data/mind_small_raw/train/MINDsmall_train/behaviors.tsv"),
        news_jsonl=str(base_dir / "data/mind_small/news.jsonl"),
        semantic_dir=str(semantic_dir), seq_mode="B", use_hot_news=False,
        hidden_dim=128, num_layers=2, num_heads=4, max_history_len=128, seed=42,
    )
    all_samples = build_samples(config.train_tsv, mode="B")
    item_embeddings = np.load(semantic_dir / "item_embeddings.npy")
    item_ids = json.loads((semantic_dir / "item_ids.json").read_text())
    mapper_data = json.loads((semantic_dir / "item_to_code.json").read_text())
    code_for_item = {k: tuple(v) for k, v in mapper_data.items()}
    item_to_index = {nid: i for i, nid in enumerate(item_ids)}
    cb_data = np.load(semantic_dir / "codebooks.npz")

    valid_samples = [
        s for s in all_samples
        if s["target"] in code_for_item
        and any(h in item_to_index for h in s["history"][:50])
    ]
    user_samples = {}
    for s in valid_samples:
        user_samples.setdefault(s["user_id"], []).append(s)
    uids = sorted(user_samples.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(uids)
    n = len(uids)
    train_n, val_n = int(n * 0.7), int(n * 0.15)
    val_uids = set(uids[train_n : train_n + val_n])
    val_samples_list = [s for uid in val_uids for s in user_samples[uid]]

    encoder = HistorySequenceEncoder(UserEncoderConfig(
        input_dim=384, hidden_dim=128, num_heads=4, num_layers=2,
        dropout=0.1, max_history_length=128,
    ))
    dec_config = ARDecoderConfig(
        hidden_dim=128, codebook_size=256, code_length=4,
        num_heads=4, num_layers=2, dropout=0.1,
    )
    decoder = CodeAutoregressiveDecoder(dec_config)
    codebooks = nn.ModuleList([nn.Embedding(256, 384) for _ in range(4)])
    for i in range(4):
        codebooks[i].weight.data.copy_(torch.tensor(cb_data[f"codebook_{i}"]))
    model = GenRecV2Model(
        encoder=encoder, decoder=decoder, codebook=codebooks,
        hot_news_fusion=None,
        embedding_table=torch.tensor(item_embeddings, dtype=torch.float32),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    collator = make_collator(item_embeddings)
    val_ds = GenRecV2Dataset(val_samples_list, item_to_index, code_for_item, item_embeddings, 128)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collator)

    print("Collecting calibration data...")
    X, y = collect_calibration_data(model, val_loader, device, item_ids, code_for_item)
    X, y = X.to(device), y.to(device)
    n_total = X.shape[0]
    perm = torch.randperm(n_total)
    v_n = int(n_total * 0.3)
    X_tr, y_tr = X[perm[v_n:]], y[perm[v_n:]]
    X_va, y_va = X[perm[:v_n]], y[perm[:v_n]]
    print(f"Train scorer: {len(X_tr)} ({int(y_tr.sum())} pos), Val: {len(X_va)} ({int(y_va.sum())} pos)")

    results = []

    # BCE baseline
    results.append(train_one(
        "BCE baseline",
        lambda logits, targets: F.binary_cross_entropy_with_logits(logits, targets),
        X_tr, y_tr, X_va, y_va, device,
    ))

    # Focal: fine sweep
    for g in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]:
        focal = FocalBCELoss(gamma=g)
        results.append(train_one(f"Focal g={g}", focal, X_tr, y_tr, X_va, y_va, device))

    # nnPU: fine sweep
    for pi in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10]:
        results.append(train_one(f"nnPU pi={pi}", make_nnpu(pi), X_tr, y_tr, X_va, y_va, device))

    print()
    print("=" * 65)
    print(f"{'Method':<20} {'Val BCE':>8} {'P_hit':>8} {'P_miss':>8} {'Gap':>8}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: -x["gap"]):
        print(f"{r['name']:<20} {r['val_bce']:>8.4f} {r['P_hit']:>8.4f} {r['P_miss']:>8.4f} {r['gap']:>8.4f}")

    # Pick best from each
    focal_best = max([r for r in results if "Focal" in r["name"]], key=lambda x: x["gap"])
    nnpu_best = max([r for r in results if "nnPU" in r["name"]], key=lambda x: x["gap"])
    print(f"\nBest Focal:  {focal_best['name']} (gap={focal_best['gap']:.4f})")
    print(f"Best nnPU:   {nnpu_best['name']} (gap={nnpu_best['gap']:.4f})")


if __name__ == "__main__":
    main()
