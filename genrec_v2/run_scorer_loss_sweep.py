"""Quick sweep: Focal Loss / Class-Balanced / nnPU for calibration scorer."""
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


def class_balanced_bce(
    logits: torch.Tensor, targets: torch.Tensor,
    beta: float = 0.999, n_pos: int = 24, n_neg: int = 776,
) -> torch.Tensor:
    samples_per_class = [n_neg, n_pos]
    effective_num = (1.0 - beta ** np.array(samples_per_class)) / (1.0 - beta)
    weights = 1.0 / effective_num
    weights = weights / weights.sum() * len(samples_per_class)
    sample_weights = torch.where(targets == 1, weights[1], weights[0])
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (sample_weights * ce).mean()


def nnpu_loss(
    logits: torch.Tensor, targets: torch.Tensor, pi: float = 0.02,
) -> torch.Tensor:
    pos_mask = targets == 1
    unlab_mask = targets == 0
    n_pos = max(1, pos_mask.sum().item())
    n_unlab = max(1, unlab_mask.sum().item())

    loss_pos = F.binary_cross_entropy_with_logits(
        logits[pos_mask], targets[pos_mask], reduction="sum"
    ) / n_pos

    loss_unlab = F.binary_cross_entropy_with_logits(
        logits[unlab_mask], torch.zeros_like(logits[unlab_mask]), reduction="sum"
    ) / n_unlab

    loss_pos_as_neg = F.binary_cross_entropy_with_logits(
        logits[pos_mask], torch.zeros_like(logits[pos_mask]), reduction="sum"
    ) / n_pos

    return loss_pos + torch.clamp(loss_unlab - pi * loss_pos_as_neg, min=0.0)


def train_and_eval(name: str, loss_fn, X_train, y_train, X_val, y_val, device, epochs=50, lr=1e-3):
    scorer = CalibrationScorer(hidden_dim=128).to(device)
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for _ in range(epochs):
        scorer.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = scorer(Xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        scorer.eval()
        with torch.no_grad():
            val_logits = scorer(X_val).squeeze(-1)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    scorer.eval()
    with torch.no_grad():
        val_logits = scorer(X_val).squeeze(-1)
        val_probs = torch.sigmoid(val_logits)
        prob_hit = val_probs[y_val == 1].mean().item()
        prob_miss = val_probs[y_val == 0].mean().item()
        gap = prob_hit - prob_miss

    return {
        "name": name, "val_bce": best_val_loss,
        "prob_hit": prob_hit, "prob_miss": prob_miss,
        "gap": gap, "ratio": prob_hit / max(1e-8, prob_miss),
    }


def main():
    base_dir = Path("/home/lishazhai/workspace/GR4AD")
    semantic_dir = base_dir / "output/sbert_baseline_20260508_153306/semantic_ids"
    ckpt_path = base_dir / "experiments/genrec_v2_exposure_bias/baseline_retrain/best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GenRecV2Config.proxy(
        train_tsv=str(base_dir / "data/mind_small_raw/train/MINDsmall_train/behaviors.tsv"),
        news_jsonl=str(base_dir / "data/mind_small/news.jsonl"),
        semantic_dir=str(semantic_dir), seq_mode="B", use_hot_news=False,
        hidden_dim=128, num_layers=2, num_heads=4, max_history_len=128, seed=42, batch_size=128,
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
    user_samples: dict[str, list] = {}
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
    print(f"Collected {X.shape[0]} pairs, {int(y.sum())} hits ({y.sum()/len(y)*100:.1f}%)")

    n_total = X.shape[0]
    perm = torch.randperm(n_total)
    val_n = int(n_total * 0.3)
    train_idx, val_idx = perm[val_n:], perm[:val_n]
    X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
    X_val, y_val = X[val_idx].to(device), y[val_idx].to(device)

    results: list[dict] = []

    # BCE baseline
    results.append(train_and_eval(
        "BCE (baseline)",
        lambda logits, targets: F.binary_cross_entropy_with_logits(logits, targets),
        X_train, y_train, X_val, y_val, device,
    ))

    # Focal variants
    for gamma in [1, 2, 3, 5]:
        focal = FocalBCELoss(gamma=gamma)
        results.append(train_and_eval(
            f"Focal gamma={gamma}", focal,
            X_train, y_train, X_val, y_val, device,
        ))

    # Class-Balanced variants
    n_pos = int(y_train.sum().item())
    n_neg = len(y_train) - n_pos
    for beta in [0.9, 0.99, 0.999]:
        results.append(train_and_eval(
            f"CB beta={beta}",
            lambda logits, targets, b=beta, np_=n_pos, nn_=n_neg:
                class_balanced_bce(logits, targets, beta=b, n_pos=np_, n_neg=nn_),
            X_train, y_train, X_val, y_val, device,
        ))

    # nnPU variants
    for pi in [0.01, 0.02, 0.05]:
        results.append(train_and_eval(
            f"nnPU pi={pi}",
            lambda logits, targets, p=pi: nnpu_loss(logits, targets, pi=p),
            X_train, y_train, X_val, y_val, device,
        ))

    # Print table
    print()
    print("=" * 75)
    header = f"{'Method':<22} {'Val BCE':>9} {'P(hit)':>8} {'P(miss)':>8} {'Gap':>8} {'Ratio':>8}"
    print(header)
    print("-" * 75)
    for r in sorted(results, key=lambda x: -x["gap"]):
        print(f"{r['name']:<22} {r['val_bce']:>9.4f} {r['prob_hit']:>8.4f} {r['prob_miss']:>8.4f} {r['gap']:>8.4f} {r['ratio']:>7.1f}x")

    best = max(results, key=lambda x: x["gap"])
    print(f"\nBest: {best['name']} (gap={best['gap']:.4f})")


if __name__ == "__main__":
    main()
