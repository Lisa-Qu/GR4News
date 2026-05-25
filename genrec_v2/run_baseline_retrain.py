"""Baseline retrain: B_nohot with epochs=50, patience=10.

Re-trains the baseline to serve as fair comparison for EXP-008 and EXP-009.
Identical to EXP-002 B_nohot except for extended training budget.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mind_genrec.model.ar_decoder import ARDecoderConfig, CodeAutoregressiveDecoder
from mind_genrec.model.user_encoder import HistorySequenceEncoder, UserEncoderConfig

from genrec_v2.config import GenRecV2Config
from genrec_v2.data.build_samples import build_samples
from genrec_v2.model.model import GenRecV2Model
from genrec_v2.train import train_experiment


def main() -> None:
    base_dir = Path("/home/lishazhai/workspace/GR4AD")

    config = GenRecV2Config.proxy(
        train_tsv=str(base_dir / "data/mind_small_raw/train/MINDsmall_train/behaviors.tsv"),
        news_jsonl=str(base_dir / "data/mind_small/news.jsonl"),
        semantic_dir=str(base_dir / "output/sbert_baseline_20260508_153306/semantic_ids"),
        experiment_name="genrec_v2_exposure_bias",
        seq_mode="B",
        use_hot_news=False,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        epochs=50,
        patience=10,
        eval_every=2,
        batch_size=128,
        lr=1e-3,
        codebook_lr_ratio=0.1,
        freeze_codebook_epochs=0,
        lambda_code=0.25,
        max_history_len=128,
        embedding_dim=384,
        codebook_size=256,
        code_length=4,
        seed=42,
        sample_pct=0.10,
        use_scheduled_sampling=False,
    )

    run_dir = base_dir / "experiments/genrec_v2_exposure_bias/baseline_retrain"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(run_dir)

    print(f"\n{'='*60}")
    print("BASELINE RETRAIN — epochs=50, patience=10")
    print(f"Output: {run_dir}")

    # ── Data ──
    all_samples = build_samples(config.train_tsv, mode=config.seq_mode)
    semantic_dir = Path(config.semantic_dir)
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
    rng = np.random.default_rng(config.seed)
    rng.shuffle(uids)
    n = len(uids)
    train_n, val_n = int(n * 0.7), int(n * 0.15)
    train_uids = set(uids[:train_n])
    val_uids = set(uids[train_n:train_n + val_n])
    test_uids = set(uids[train_n + val_n:])

    train_samples = [s for uid in train_uids for s in user_samples[uid]]
    val_samples = [s for uid in val_uids for s in user_samples[uid]]
    test_samples = [s for uid in test_uids for s in user_samples[uid]]

    print(f"users: train={len(train_uids)}, val={len(val_uids)}, test={len(test_uids)}")

    # ── Model ──
    encoder = HistorySequenceEncoder(UserEncoderConfig(
        input_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
        num_heads=config.num_heads, num_layers=config.num_layers,
        dropout=config.dropout, max_history_length=config.max_history_len,
    ))
    dec_config = ARDecoderConfig(
        hidden_dim=config.hidden_dim, codebook_size=config.codebook_size,
        code_length=config.code_length, num_heads=config.num_heads,
        num_layers=config.num_layers, dropout=config.dropout,
    )
    decoder = CodeAutoregressiveDecoder(dec_config)

    codebooks = nn.ModuleList([
        nn.Embedding(config.codebook_size, config.embedding_dim)
        for _ in range(config.code_length)
    ])
    for i in range(config.code_length):
        codebooks[i].weight.data.copy_(torch.tensor(cb_data[f"codebook_{i}"]))

    emb_table = torch.tensor(item_embeddings, dtype=torch.float32)
    model = GenRecV2Model(
        encoder=encoder, decoder=decoder, codebook=codebooks,
        hot_news_fusion=None, embedding_table=emb_table,
    )

    # ── Train ──
    results = train_experiment(
        config, train_samples, val_samples, test_samples,
        model, item_to_index, code_for_item, item_embeddings, item_ids,
    )

    print(f"\nBaseline retrain results:")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
