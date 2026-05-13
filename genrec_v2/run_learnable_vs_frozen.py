"""Single-variable comparison: learnable vs frozen codebook (mode B, no hot)."""
from __future__ import annotations

import json, sys
from collections import Counter
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

base_dir = Path("/home/lishazhai/workspace/GR4AD")
sem_dir = base_dir / "output/sbert_baseline_20260508_153306/semantic_ids"
tsv_path = str(base_dir / "data/mind_small_raw/train/MINDsmall_train/behaviors.tsv")
news_path = str(base_dir / "data/mind_small/news.jsonl")
exp_dir = base_dir / "experiments/genrec_v2_learnable"

# Build samples once (mode B)
samples = build_samples(tsv_path, mode="B")
item_ids = json.loads((sem_dir / "item_ids.json").read_text())
item_emb = np.load(sem_dir / "item_embeddings.npy")
mapper_data = json.loads((sem_dir / "item_to_code.json").read_text())
code_for_item = {k: tuple(v) for k, v in mapper_data.items()}
item_to_index = {nid: i for i, nid in enumerate(item_ids)}
cb_data = np.load(sem_dir / "codebooks.npz")

# Filter valid
valid_samples = [s for s in samples if s["target"] in code_for_item and any(h in item_to_index for h in s["history"][:50])]

# User split
user_samples: dict[str, list] = {}
for s in valid_samples:
    user_samples.setdefault(s["user_id"], []).append(s)
uids = sorted(user_samples.keys())
rng = np.random.default_rng(42)
rng.shuffle(uids)
n = len(uids)
train_uids = set(uids[:int(n * 0.7)])
val_uids = set(uids[int(n * 0.7):int(n * 0.85)])
test_uids = set(uids[int(n * 0.85):])

def filter_by_uids(uid_set):
    return [s for uid in uid_set for s in user_samples[uid]]

train_samples = filter_by_uids(train_uids)
val_samples = filter_by_uids(val_uids)
test_samples = filter_by_uids(test_uids)

def build_model(config):
    enc = HistorySequenceEncoder(UserEncoderConfig(
        input_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
        num_heads=config.num_heads, num_layers=config.num_layers,
        dropout=config.dropout, max_history_length=config.max_history_len))
    dec = CodeAutoregressiveDecoder(ARDecoderConfig(
        hidden_dim=config.hidden_dim, codebook_size=config.codebook_size,
        code_length=config.code_length, num_heads=config.num_heads,
        num_layers=config.num_layers, dropout=config.dropout))
    cbs = nn.ModuleList([nn.Embedding(256, 384) for _ in range(4)])
    for i in range(4):
        cbs[i].weight.data.copy_(torch.tensor(cb_data[f"codebook_{i}"]))
    return GenRecV2Model(encoder=enc, decoder=dec, codebook=cbs, embedding_table=torch.tensor(item_emb))

results = {}
for name, freeze_eps in [("frozen", 3), ("learnable", 0)]:
    cfg = GenRecV2Config.proxy(
        seq_mode="B", use_hot_news=False,
        freeze_codebook_epochs=freeze_eps,
        train_tsv=tsv_path, news_jsonl=news_path,
        semantic_dir=str(sem_dir),
        experiment_name="genrec_v2_codebook",
    )
    model = build_model(cfg)
    run_dir = exp_dir / name
    print(f"\n{'='*60}\n{name}: freeze={freeze_eps}\n{'='*60}")
    results[name] = train_experiment(cfg, train_samples, val_samples, test_samples, model, item_to_index, code_for_item, item_emb, item_ids)

print("\n=== COMPARISON ===")
for name, m in results.items():
    print(f"\n{name}:")
    for k, v in sorted(m.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

(exp_dir / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
