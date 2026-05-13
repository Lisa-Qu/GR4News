"""GenRec-V2 ablation v2: hidden=128, epochs=20, λ=0.25, freeze=0, cb_lr=0.1."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mind_genrec.model.ar_decoder import ARDecoderConfig, CodeAutoregressiveDecoder
from mind_genrec.model.user_encoder import HistorySequenceEncoder, UserEncoderConfig

from genrec_v2.config import GenRecV2Config
from genrec_v2.data.build_samples import build_samples
from genrec_v2.model.hot_news import HotNewsFusion
from genrec_v2.model.model import GenRecV2Model
from genrec_v2.train import train_experiment


def compute_click_counts(tsv_path: str) -> Counter:
    c = Counter()
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            impressions = parts[4].split()
            clicked = [imp.split("-")[0] for imp in impressions if imp.endswith("-1")]
            c.update(clicked)
    return c


def build_hot_embeddings_from_clicks(
    click_counts: Counter,
    news_jsonl: str,
    item_embeddings: np.ndarray,
    item_to_index: dict[str, int],
    topk: int = 5,
    min_cat_clicks: int = 100,
) -> torch.Tensor:
    news_cat: dict[str, str] = {}
    with open(news_jsonl) as f:
        for line in f:
            d = json.loads(line)
            news_cat[d["news_id"]] = d.get("category", "")

    cat_items: dict[str, list[tuple[str, int]]] = {}
    for nid, cat in news_cat.items():
        cnt = click_counts.get(nid, 0)
        cat_items.setdefault(cat, []).append((nid, cnt))

    hot_indices = []
    for cat, items in sorted(cat_items.items()):
        total = sum(cnt for _, cnt in items)
        if total < min_cat_clicks:
            continue
        top = sorted(items, key=lambda x: -x[1])[:topk]
        for nid, _ in top:
            if nid in item_to_index:
                hot_indices.append(item_to_index[nid])

    return torch.tensor(item_embeddings[hot_indices], dtype=torch.float32)


def run_single(config: GenRecV2Config, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(run_dir)

    print(f"\n{'='*60}")
    print(f"mode={config.seq_mode}, hot={config.use_hot_news}, freeze={config.freeze_codebook_epochs}")

    all_samples = build_samples(config.train_tsv, mode=config.seq_mode)
    semantic_dir = Path(config.semantic_dir)
    item_embeddings = np.load(semantic_dir / "item_embeddings.npy")
    item_ids = json.loads((semantic_dir / "item_ids.json").read_text())
    mapper_data = json.loads((semantic_dir / "item_to_code.json").read_text())
    code_for_item = {k: tuple(v) for k, v in mapper_data.items()}
    item_to_index = {nid: i for i, nid in enumerate(item_ids)}
    cb_data = np.load(semantic_dir / "codebooks.npz")

    valid_samples = [s for s in all_samples if s["target"] in code_for_item and any(h in item_to_index for h in s["history"][:50])]

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
    print(f"samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    encoder = HistorySequenceEncoder(UserEncoderConfig(
        input_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
        num_heads=config.num_heads, num_layers=config.num_layers,
        dropout=config.dropout, max_history_length=config.max_history_len))

    dec_config = ARDecoderConfig(
        hidden_dim=config.hidden_dim, codebook_size=config.codebook_size,
        code_length=config.code_length, num_heads=config.num_heads,
        num_layers=config.num_layers, dropout=config.dropout)
    decoder = CodeAutoregressiveDecoder(dec_config)

    codebooks = nn.ModuleList([nn.Embedding(config.codebook_size, config.embedding_dim) for _ in range(config.code_length)])
    for i in range(config.code_length):
        codebooks[i].weight.data.copy_(torch.tensor(cb_data[f"codebook_{i}"]))

    hot_news = None
    if config.use_hot_news:
        click_cnts = compute_click_counts(config.train_tsv)
        hot_embs = build_hot_embeddings_from_clicks(
            click_cnts, config.news_jsonl, item_embeddings, item_to_index,
            topk=config.hot_news_topk, min_cat_clicks=config.hot_news_min_cat_clicks)
        hot_news = HotNewsFusion(config.hidden_dim, hot_embs)
        print(f"hot_news: {hot_embs.shape[0]} items")

    emb_table = torch.tensor(item_embeddings, dtype=torch.float32)
    model = GenRecV2Model(encoder=encoder, decoder=decoder, codebook=codebooks,
                          hot_news_fusion=hot_news, embedding_table=emb_table)

    return train_experiment(config, train_samples, val_samples, test_samples,
                            model, item_to_index, code_for_item, item_embeddings, item_ids)


def main() -> None:
    base_dir = Path("/home/lishazhai/workspace/GR4AD")

    common = dict(
        train_tsv=str(base_dir / "data/mind_small_raw/train/MINDsmall_train/behaviors.tsv"),
        news_jsonl=str(base_dir / "data/mind_small/news.jsonl"),
        semantic_dir=str(base_dir / "output/sbert_baseline_20260508_153306/semantic_ids"),
        experiment_name="genrec_v2_ablation_v2",
    )

    configs = [
        ("A_nohot",  GenRecV2Config.proxy(seq_mode="A", use_hot_news=False, **common)),
        ("A_hot",    GenRecV2Config.proxy(seq_mode="A", use_hot_news=True,  **common)),
        ("B_nohot",  GenRecV2Config.proxy(seq_mode="B", use_hot_news=False, **common)),
        ("B_hot",    GenRecV2Config.proxy(seq_mode="B", use_hot_news=True,  **common)),
        # frozen baseline
        ("B_nohot_frozen", GenRecV2Config.proxy(seq_mode="B", use_hot_news=False, freeze_codebook_epochs=3, codebook_lr_ratio=0.05, **common)),
    ]

    exp_dir = base_dir / "experiments/genrec_v2_ablation_v2"
    results = {}
    for name, cfg in configs:
        run_dir = exp_dir / name
        try:
            results[name] = run_single(cfg, run_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    for name, m in sorted(results.items()):
        print(f"\n{name}:")
        for k, v in sorted(m.items()):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    (exp_dir / "all_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
