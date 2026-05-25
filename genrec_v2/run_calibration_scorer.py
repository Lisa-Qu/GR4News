"""EXP-008: Calibration Scorer for beam search re-ranking.

Phase 1: Train baseline generator (or use retrained checkpoint).
Phase 2: Greedy-decode val set → collect (hidden_states, correctness_label) pairs.
Phase 3: Train CalibrationScorer on collected data.
Phase 4: Evaluate beam_search_with_scorer on test set, log to MLflow.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mind_genrec.model.ar_decoder import ARDecoderConfig, CodeAutoregressiveDecoder
from mind_genrec.model.user_encoder import HistorySequenceEncoder, UserEncoderConfig

from genrec_v2.calibration.scorer import (
    CalibrationScorer,
    beam_search_with_scorer,
    collect_calibration_data,
    train_scorer,
)
from genrec_v2.config import GenRecV2Config
from genrec_v2.data.build_samples import build_samples
from genrec_v2.data.dataset import GenRecV2Dataset, make_collator
from genrec_v2.model.model import GenRecV2Model


def evaluate_scorer_beam_search(
    model,
    scorer: CalibrationScorer,
    loader: DataLoader,
    device: torch.device,
    code_for_item: dict[str, tuple[int, ...]],
    item_ids: list[str],
    beam_width: int = 50,
    scorer_lambda: float = 0.5,
    max_users: int = 1000,
) -> dict[str, float]:
    """Evaluate beam search with scorer re-ranking."""
    model.eval()
    scorer.eval()

    code_to_items: dict[tuple[int, ...], list[str]] = {}
    for nid, c in code_for_item.items():
        code_to_items.setdefault(tuple(c), []).append(nid)

    total_users = 0
    stats: dict[int, tuple[int, int]] = {}  # k → (correct, total)
    greedy_total = 0
    greedy_hit = 0
    t_start = time.time()

    for batch in loader:
        device_batch = {k: v.to(device) for k, v in batch.items()}
        B = device_batch["history_emb"].shape[0]

        # Greedy baseline
        pred_codes = model.greedy_decode(
            device_batch["history_emb"], device_batch["history_mask"]
        )
        for b in range(B):
            greedy_total += 1
            tgt_nid = item_ids[int(device_batch["target_emb_idx"][b])]
            pred_code = tuple(int(x) for x in pred_codes[b])
            if tgt_nid in code_to_items.get(pred_code, []):
                greedy_hit += 1

        # Beam search with scorer
        scored_beams = beam_search_with_scorer(
            model, scorer,
            device_batch["history_emb"], device_batch["history_mask"],
            beam_width=beam_width, scorer_lambda=scorer_lambda,
        )

        for b in range(B):
            total_users += 1
            if max_users > 0 and total_users > max_users:
                break

            tgt_nid = item_ids[int(device_batch["target_emb_idx"][b])]
            tgt_code_str = tuple(int(x) for x in device_batch["target_code"][b])
            beams = scored_beams[b]

            for k in [1, 5, 10, 50]:
                k_eff = min(k, len(beams))
                hit = False
                for rank in range(k_eff):
                    pred_code = tuple(beams[rank][0])
                    if tgt_nid in code_to_items.get(pred_code, []):
                        hit = True
                        break
                correct, total = stats.setdefault(k, (0, 0))
                if hit:
                    correct += 1
                total += 1
                stats[k] = (correct, total)

        if max_users > 0 and total_users >= max_users:
            break

    elapsed = time.time() - t_start
    print(f"Scorer eval: {total_users} users in {elapsed:.0f}s")

    results: dict[str, float] = {}
    results["greedy_recall@1"] = greedy_hit / max(1, greedy_total)
    for k in [1, 5, 10, 50]:
        correct, total = stats.get(k, (0, 1))
        results[f"scorer_beam{beam_width}_recall@{k}"] = correct / max(1, total)
    # Scoring efficiency
    if results.get("scorer_beam50_recall@50", 0) > 0:
        results["scorer_scoring_efficiency"] = (
            results.get("scorer_beam50_recall@1", 0)
            / results["scorer_beam50_recall@50"]
        )
    return results


def main() -> None:
    base_dir = Path("/home/lishazhai/workspace/GR4AD")
    semantic_dir = base_dir / "output/sbert_baseline_20260508_153306/semantic_ids"

    config = GenRecV2Config.proxy(
        train_tsv=str(base_dir / "data/mind_small_raw/train/MINDsmall_train/behaviors.tsv"),
        news_jsonl=str(base_dir / "data/mind_small/news.jsonl"),
        semantic_dir=str(semantic_dir),
        experiment_name="genrec_v2_exposure_bias",
        seq_mode="B",
        use_hot_news=False,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_history_len=128,
        embedding_dim=384,
        codebook_size=256,
        code_length=4,
        seed=42,
        batch_size=128,
        sample_pct=0.10,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    print("Loading data...")
    all_samples = build_samples(config.train_tsv, mode=config.seq_mode)
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

    # ── Build model ──
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
    model = model.to(device)

    # ── Load checkpoint ──
    # Use baseline_retrain checkpoint if available, otherwise EXP-002 B_nohot
    ckpt_path = base_dir / "experiments/genrec_v2_exposure_bias/baseline_retrain/best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = base_dir / "experiments/genrec_v2_ablation_v2/B_nohot/best_model.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── Phase 2: Collect calibration data ──
    collator = make_collator(item_embeddings)
    val_ds = GenRecV2Dataset(
        val_samples, item_to_index, code_for_item, item_embeddings, config.max_history_len,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collator)

    print(f"Collecting calibration data from {len(val_ds)} val samples...")
    t0 = time.time()
    X, y = collect_calibration_data(model, val_loader, device, item_ids, code_for_item)
    n_pos = int(y.sum().item())
    print(f"  Collected {X.shape[0]} pairs ({n_pos} hits, {n_pos/max(1,X.shape[0])*100:.1f}%) in {time.time()-t0:.1f}s")

    # ── Phase 3: Train scorer ──
    print("Training calibration scorer...")
    scorer = CalibrationScorer(hidden_dim=config.hidden_dim)
    scorer = scorer.to(device)

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("genrec_v2_exposure_bias")
        run_name = "calibration_scorer"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "base_checkpoint": str(ckpt_path),
            "calibration_samples": X.shape[0],
            "calibration_positive_rate": n_pos / max(1, X.shape[0]),
            "scorer_hidden_dim": config.hidden_dim,
            "scorer_epochs": 50,
            "scorer_patience": 10,
            "scorer_lr": 1e-3,
        })
        _mlflow = mlflow
    except Exception:
        _mlflow = None

    train_scorer(
        scorer, X, y, device=device, epochs=50, patience=10, lr=1e-3,
        mlflow_client=_mlflow,
    )

    # ── Phase 4: Evaluate ──
    test_ds = GenRecV2Dataset(
        test_samples, item_to_index, code_for_item, item_embeddings, config.max_history_len,
    )
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collator)

    print("Evaluating beam search with scorer...")
    results = evaluate_scorer_beam_search(
        model, scorer, test_loader, device, code_for_item, item_ids,
        beam_width=50, scorer_lambda=0.5, max_users=1000,
    )

    print("\n" + "=" * 60)
    print("CALIBRATION SCORER RESULTS")
    print("=" * 60)
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")

    if _mlflow:
        _mlflow.log_metrics(results)
        _mlflow.end_run()
        print("\nMLflow: logged")

    # Save
    out_dir = base_dir / "experiments/genrec_v2_exposure_bias/calibration_scorer"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    torch.save(scorer.state_dict(), out_dir / "scorer.pt")


if __name__ == "__main__":
    main()
