"""Multi-seed stability test for nnPU + Focal scorer variants."""
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


SEEDS = [1, 42, 123, 456, 789]


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** self.gamma * ce).mean()


def train_nnpu(pi, seed, X, y, device):
    torch.manual_seed(seed)
    scorer = CalibrationScorer(hidden_dim=128).to(device)
    opt = torch.optim.AdamW(scorer.parameters(), lr=1e-3)

    n_total = len(X)
    perm = torch.randperm(n_total)
    v_n = int(n_total * 0.3)
    X_tr, y_tr = X[perm[v_n:]], y[perm[v_n:]]
    X_va, y_va = X[perm[:v_n]], y[perm[:v_n]]

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    best_val = float("inf")
    patience = 0

    for ep in range(50):
        scorer.train()
        for Xb, yb in train_loader:
            logits = scorer(Xb).squeeze(-1)
            pm = yb == 1
            um = yb == 0
            np_ = max(1, pm.sum().item())
            nu = max(1, um.sum().item())
            lp = F.binary_cross_entropy_with_logits(
                logits[pm], yb[pm], reduction="sum"
            ) / np_
            lu = F.binary_cross_entropy_with_logits(
                logits[um], torch.zeros_like(logits[um]), reduction="sum"
            ) / nu
            lpan = F.binary_cross_entropy_with_logits(
                logits[pm], torch.zeros_like(logits[pm]), reduction="sum"
            ) / np_
            loss = lp + torch.clamp(lu - pi * lpan, min=0.0)
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
    return {"P_hit": ph, "P_miss": pm, "gap": ph - pm, "epochs": ep + 1}


def train_focal(gamma, seed, X, y, device):
    torch.manual_seed(seed)
    scorer = CalibrationScorer(hidden_dim=128).to(device)
    opt = torch.optim.AdamW(scorer.parameters(), lr=1e-3)
    loss_fn = FocalBCELoss(gamma=gamma)

    n_total = len(X)
    perm = torch.randperm(n_total)
    v_n = int(n_total * 0.3)
    X_tr, y_tr = X[perm[v_n:]], y[perm[v_n:]]
    X_va, y_va = X[perm[:v_n]], y[perm[:v_n]]

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    best_val = float("inf")
    patience = 0

    for ep in range(50):
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
    return {"P_hit": ph, "P_miss": pm, "gap": ph - pm, "epochs": ep + 1}


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
    print(f"{len(X)} pairs, {int(y.sum())} hits")

    all_results = {}

    configs = [
        ("nnPU pi=0.02", "nnpu", 0.02),
        ("nnPU pi=0.08", "nnpu", 0.08),
        ("nnPU pi=0.1", "nnpu", 0.1),
        ("Focal g=2.0", "focal", 2.0),
        ("Focal g=2.5", "focal", 2.5),
    ]

    for name, method, param in configs:
        runs = []
        for s in SEEDS:
            if method == "nnpu":
                r = train_nnpu(param, s, X, y, device)
            else:
                r = train_focal(param, s, X, y, device)
            runs.append(r)
        gaps = [r["gap"] for r in runs]
        phs = [r["P_hit"] for r in runs]
        pms = [r["P_miss"] for r in runs]
        eps = [r["epochs"] for r in runs]
        all_results[name] = {
            "P_hit_mean": np.mean(phs), "P_hit_std": np.std(phs),
            "P_miss_mean": np.mean(pms), "P_miss_std": np.std(pms),
            "gap_mean": np.mean(gaps), "gap_std": np.std(gaps),
            "raw_gaps": gaps, "epochs": eps,
        }

    print()
    header = f"{'Method':<20} {'P_hit':>18} {'P_miss':>18} {'Gap':>18} {'Epochs':>10}"
    print(header)
    print("-" * 88)
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["gap_mean"]):
        print(
            f"{name:<20} "
            f'{r["P_hit_mean"]:.4f}+/-{r["P_hit_std"]:.4f}   '
            f'{r["P_miss_mean"]:.4f}+/-{r["P_miss_std"]:.4f}   '
            f'{r["gap_mean"]:.4f}+/-{r["gap_std"]:.4f}   '
            f'{int(np.mean(r["epochs"]))}'
        )
        print(f"  raw gaps: {[f'{g:.4f}' for g in r['raw_gaps']]}")


if __name__ == "__main__":
    main()
