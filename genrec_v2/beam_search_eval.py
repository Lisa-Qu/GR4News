"""Beam search evaluation — no training, just inference on best checkpoint.

Loads B_nohot checkpoint, runs beam search with multiple widths,
computes recall@k, logs to MLflow.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mind_genrec.model.ar_decoder import ARDecoderConfig, CodeAutoregressiveDecoder
from mind_genrec.model.user_encoder import HistorySequenceEncoder, UserEncoderConfig

from genrec_v2.config import GenRecV2Config
from genrec_v2.data.build_samples import build_samples
from genrec_v2.data.dataset import GenRecV2Dataset, make_collator
from genrec_v2.model.model import GenRecV2Model


def beam_search_single(
    decoder: CodeAutoregressiveDecoder,
    user_state: torch.Tensor,  # [1, hidden_dim]
    beam_width: int,
) -> list[tuple[list[int], float]]:
    """Beam search over code tokens for a single user.

    Batches all beams into one forward pass per step.
    Returns list of (code_seq, avg_log_prob) sorted by score descending.
    """
    device = user_state.device
    code_length = decoder.config.code_length

    # Each beam: (token_list, cumulative_log_prob)
    beams: list[tuple[list[int], float]] = [([], 0.0)]

    for step in range(code_length):
        num_beams = len(beams)
        if step == 0:
            prefix_batch = torch.tensor(
                [[decoder.bos_token_id]], dtype=torch.long, device=device
            )
            us_batch = user_state  # [1, d]
        else:
            prefix_batch = torch.tensor(
                [[decoder.bos_token_id] + seq for seq, _ in beams],
                dtype=torch.long, device=device,
            )  # [num_beams, step+1]
            us_batch = user_state.expand(num_beams, -1)  # [num_beams, d]

        log_probs = decoder.next_token_log_probs(
            user_state=us_batch, prefix_tokens=prefix_batch
        )  # [num_beams, vocab_size]

        topk_values, topk_indices = log_probs.topk(beam_width, dim=-1)

        candidates: list[tuple[list[int], float]] = []
        for b_idx, (code_seq, cum_log_prob) in enumerate(beams):
            for k in range(beam_width):
                token = int(topk_indices[b_idx, k])
                score = cum_log_prob + float(topk_values[b_idx, k])
                candidates.append((code_seq + [token], score))

        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam_width]

    return [(seq, score / code_length) for seq, score in beams]


@torch.no_grad()
def evaluate_beam_search(
    model: GenRecV2Model,
    loader: DataLoader,
    device: torch.device,
    code_for_item: dict[str, tuple[int, ...]],
    item_ids: list[str],
    beam_widths: list[int],
    max_users: int = 0,
) -> dict[str, float]:
    """Run beam search evaluation for multiple beam widths."""
    model.eval()
    decoder = model.decoder

    # Build code -> items reverse index
    code_to_items: dict[tuple[int, ...], list[str]] = {}
    for nid, c in code_for_item.items():
        code_to_items.setdefault(c, []).append(nid)

    stats: dict[int, dict[int, tuple[int, int]]] = {}
    exact_match: dict[int, int] = {}
    total_users = 0
    t_start = time.time()

    for batch_idx, batch in enumerate(loader):
        device_batch = {k: v.to(device) for k, v in batch.items()}
        user_state, _ = model.encoder(
            device_batch["history_emb"], device_batch["history_mask"]
        )

        B = user_state.shape[0]
        for b in range(B):
            total_users += 1
            if max_users > 0 and total_users > max_users:
                break

            us = user_state[b : b + 1]  # [1, d]
            tgt_code = tuple(int(x) for x in device_batch["target_code"][b])
            tgt_nid = item_ids[int(device_batch["target_emb_idx"][b])]

            for bw in beam_widths:
                if bw not in stats:
                    stats[bw] = {k: (0, 0) for k in [1, 5, 10, 50, 100]}
                    exact_match[bw] = 0

                beams = beam_search_single(decoder, us, bw)

                if beams and tuple(beams[0][0]) == tgt_code:
                    exact_match[bw] += 1

                for k in [1, 5, 10, 50, 100]:
                    k_eff = min(k, len(beams))
                    hit = False
                    for rank in range(k_eff):
                        pred_code = tuple(beams[rank][0])
                        if tgt_nid in code_to_items.get(pred_code, []):
                            hit = True
                            break
                    correct, total = stats[bw][k]
                    if hit:
                        correct += 1
                    total += 1
                    stats[bw][k] = (correct, total)

            if total_users % 100 == 0:
                elapsed = time.time() - t_start
                rate = total_users / max(1, elapsed)
                print(f"  [{total_users} users] {elapsed:.0f}s ({rate:.1f} users/s)")

        if max_users > 0 and total_users > max_users:
            break

    elapsed = time.time() - t_start
    print(f"Beam search done: {total_users} users in {elapsed:.0f}s ({total_users/max(1,elapsed):.1f} users/s)")

    results: dict[str, float] = {}
    for bw in sorted(beam_widths):
        results[f"beam{bw}_exact_match"] = exact_match[bw] / max(1, total_users)
        for k in [1, 5, 10, 50, 100]:
            if k <= bw:
                correct, total = stats[bw][k]
                results[f"beam{bw}_recall_at{k}"] = correct / max(1, total)

    return results


def main() -> None:
    base_dir = Path("/home/lishazhai/workspace/GR4AD")
    checkpoint_dir = base_dir / "experiments/genrec_v2_ablation_v2/B_nohot"
    semantic_dir = base_dir / "output/sbert_baseline_20260508_153306/semantic_ids"

    config = GenRecV2Config.proxy(
        train_tsv=str(base_dir / "data/mind_small_raw/train/MINDsmall_train/behaviors.tsv"),
        news_jsonl=str(base_dir / "data/mind_small/news.jsonl"),
        semantic_dir=str(semantic_dir),
        max_history_len=128,
        seq_mode="B",
        seed=42,
        embedding_dim=384,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        codebook_size=256,
        code_length=4,
        use_hot_news=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    print("Loading data...")
    t0 = time.time()
    all_samples = build_samples(config.train_tsv, mode=config.seq_mode)
    item_embeddings = np.load(semantic_dir / "item_embeddings.npy")
    item_ids = json.loads((semantic_dir / "item_ids.json").read_text())
    mapper_data = json.loads((semantic_dir / "item_to_code.json").read_text())
    code_for_item = {k: tuple(v) for k, v in mapper_data.items()}
    item_to_index = {nid: i for i, nid in enumerate(item_ids)}
    cb_data = np.load(semantic_dir / "codebooks.npz")
    print(f"  Data loaded in {time.time()-t0:.1f}s")

    # Filter valid samples
    valid_samples = [
        s for s in all_samples
        if s["target"] in code_for_item
        and any(h in item_to_index for h in s["history"][:50])
    ]

    # User-level split (same seed as training)
    user_samples: dict[str, list] = {}
    for s in valid_samples:
        user_samples.setdefault(s["user_id"], []).append(s)
    uids = sorted(user_samples.keys())
    rng = np.random.default_rng(config.seed)
    rng.shuffle(uids)
    n = len(uids)
    train_n, val_n = int(n * 0.7), int(n * 0.15)
    test_uids = set(uids[train_n + val_n :])
    test_samples = [s for uid in test_uids for s in user_samples[uid]]
    print(f"Test users: {len(test_uids)}, test samples: {len(test_samples)}")

    # ── Build model ──
    print("Building model...")
    encoder = HistorySequenceEncoder(
        UserEncoderConfig(
            input_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
            num_heads=config.num_heads, num_layers=config.num_layers,
            dropout=config.dropout, max_history_length=config.max_history_len,
        )
    )
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
    ckpt_path = checkpoint_dir / "best_model.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("  Model loaded OK")

    # ── DataLoader ──
    collator = make_collator(item_embeddings)
    test_ds = GenRecV2Dataset(
        test_samples, item_to_index, code_for_item, item_embeddings, config.max_history_len
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collator
    )

    # ── Run greedy first (fast baseline) ──
    print("\nRunning greedy decode...")
    t0 = time.time()
    greedy_correct = 0
    greedy_recall_at1 = 0
    greedy_total = 0

    code_to_items: dict[tuple[int, ...], list[str]] = {}
    for nid, c in code_for_item.items():
        code_to_items.setdefault(c, []).append(nid)

    for batch in test_loader:
        device_batch = {k: v.to(device) for k, v in batch.items()}
        user_state, _ = model.encoder(
            device_batch["history_emb"], device_batch["history_mask"]
        )
        pred_codes = model.decoder.greedy_decode(user_state)
        B = pred_codes.shape[0]
        for b in range(B):
            greedy_total += 1
            tgt_code = tuple(int(x) for x in device_batch["target_code"][b])
            pred_code = tuple(int(x) for x in pred_codes[b])
            if pred_code == tgt_code:
                greedy_correct += 1
            tgt_nid = item_ids[int(device_batch["target_emb_idx"][b])]
            if tgt_nid in code_to_items.get(pred_code, []):
                greedy_recall_at1 += 1

    results = {
        "greedy_code_exact": greedy_correct / max(1, greedy_total),
        "greedy_recall_at1": greedy_recall_at1 / max(1, greedy_total),
    }
    print(f"  Greedy: recall@1={results['greedy_recall_at1']:.4f} ({results['greedy_recall_at1']*100:.2f}%), "
          f"exact={results['greedy_code_exact']:.4f}, "
          f"time={time.time()-t0:.1f}s")

    # ── Beam search (use max 1000 users for speed) ──
    beam_widths = [5, 10, 50]
    max_eval_users = 1000
    print(f"\nRunning beam search: widths={beam_widths}, max_users={max_eval_users}")
    beam_results = evaluate_beam_search(
        model, test_loader, device, code_for_item, item_ids, beam_widths,
        max_users=max_eval_users,
    )
    results.update(beam_results)

    # Beam-1 = greedy (from beam search)
    beam_widths_for_1 = [1]
    beam1_results = evaluate_beam_search(
        model, test_loader, device, code_for_item, item_ids, beam_widths_for_1,
        max_users=max_eval_users,
    )
    results.update({k.replace("beam1_", "beam1_"): v for k, v in beam1_results.items()})

    # ── Print results ──
    print("\n" + "=" * 60)
    print("BEAM SEARCH RESULTS")
    print("=" * 60)
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f} ({v*100:.2f}%)")

    # ── MLflow ──
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("genrec_v2_beam_search")
        run_name = f"beam_search_B_nohot_{max_eval_users}users"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "checkpoint": str(ckpt_path),
            "beam_widths": str([1] + beam_widths),
            "max_eval_users": max_eval_users,
            "test_users": len(test_uids),
            "test_samples": len(test_samples),
        })
        mlflow.log_metrics(results)
        mlflow.log_dict(results, "beam_search_results.json")
        mlflow.end_run()
        print(f"\nMLflow: {run_name} — logged")
    except Exception as e:
        print(f"\nMLflow logging failed: {e}")

    # ── Save locally ──
    out_path = checkpoint_dir / "beam_search_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
