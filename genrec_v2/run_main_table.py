"""Main-Table P0 baselines for the paper (T18 Oracle + T3 TIGER-equivalent + scorers).

Produces, on the canonical ``baseline_retrain`` checkpoint with per-sample beam
search (K=50), the rows the CCF-B Main Table needs:

    - Vanilla BS-50          → reported as the **TIGER-equivalent** row (T3, our
      codebase with all enhancements removed: k-means tokenizer, no hot-news, no
      scheduled sampling).
    - Oracle (perfect rank)  → upper bound (T18); Oracle R@1 == R@50 since perfect
      ranking lifts every in-beam hit to the top.
    - Pointwise Focal        → best lambda picked on val.
    - Listwise BCE (5 seeds)  → main scorer result, mean +/- std.

Unlike ``run_scorer_complete.py`` (which only prints), this script PERSISTS:
    - ``experiments/main_table/results.json``       — Main-Table rows
    - ``experiments/main_table/per_user_hits.npz``  — per-sample hit@k for the
      paired significance test (consumed by ``run_statistical_significance.py``)
    - MLflow run (localhost:5000, experiment ``mind_genrec``)
and CACHES collected beam data to ``beam_{val,test}.pt`` so reruns are cheap.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u genrec_v2/run_main_table.py
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from genrec_v2.calibration.scorer import (
    ListwiseScorer,
    collect_beam_calibration_data,
    train_listwise_scorer,
)
from genrec_v2.run_scorer_complete import (
    BASE_DIR,
    BEAM_WIDTH,
    CKPT_PATH,
    CODE_LENGTH,
    DEVICE,
    HIDDEN_DIM,
    MAX_HIST,
    SEEDS,
    FocalBCELoss,
    build_model,
    prepare_data,
    train_pointwise,
)
from mind_genrec.tracking import MlflowRunLogger

OUT_DIR = BASE_DIR / "experiments/main_table"
KS = (1, 5, 10, 50)
NDCG_KS = (5, 10)
# Locked λ grid + selection procedure — IDENTICAL across all settings (control variable).
LAMBDA_GRID = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)


def get_beam_data(split: str, samples, model, item_ids, cfi, iti, item_embeddings) -> dict:
    """Collect (or load cached) per-sample beam calibration data for one split."""
    cache = OUT_DIR / f"beam_{split}.pt"
    if cache.exists():
        print(f"  loading cached beam data: {cache}")
        return torch.load(cache)
    t0 = time.time()
    data = collect_beam_calibration_data(
        model, samples, iti, cfi, item_embeddings, MAX_HIST,
        DEVICE, item_ids, beam_width=BEAM_WIDTH, batch_size=32,
    )
    torch.save(data, cache)
    print(f"  {data['hidden'].shape[0]} {split} samples [{time.time() - t0:.0f}s] -> cached")
    return data


def _rank_metrics(rel_ordered: np.ndarray) -> tuple[dict, float, dict]:
    """From a binary relevance vector in ranked order → hit@k, MRR, nDCG@k.

    Computed within the beam (Oracle-bounded). Shared formula used identically
    across every setting/baseline so the Main Table is comparable.
    """
    hit = {k: bool(rel_ordered[:k].max() > 0) for k in KS}
    nz = np.flatnonzero(rel_ordered)
    mrr = float(1.0 / (nz[0] + 1)) if nz.size else 0.0
    n_rel = int(rel_ordered.sum())
    ndcg: dict[int, float] = {}
    for k in NDCG_KS:
        topk = rel_ordered[:k]
        dcg = float(np.sum(topk / np.log2(np.arange(2, topk.size + 2))))
        idcg = float(np.sum(1.0 / np.log2(np.arange(2, min(k, n_rel) + 2)))) if n_rel else 0.0
        ndcg[k] = dcg / idcg if idcg > 0 else 0.0
    return hit, mrr, ndcg


def eval_peruser(scorer, is_listwise: bool, lam: float, data: dict):
    """Evaluate on ``data``; return (aggregate metrics dict, per-sample hit@k dict).

    Aggregate dict keys: R@k (k in KS), 'MRR', 'nDCG@5', 'nDCG@10', computed via
    the shared ``_rank_metrics``. ``scorer=None`` reproduces the vanilla beam
    ranking (== TIGER-equivalent). Per-sample hit@k arrays feed significance.
    """
    hidden, bscores = data["hidden"], data["beam_scores"]
    ustates, labels = data["user_states"], data["labels_binary"]
    n = hidden.shape[0]
    hits = {k: np.zeros(n, dtype=bool) for k in KS}
    mrr_arr = np.zeros(n, dtype=np.float64)
    ndcg_arr = {k: np.zeros(n, dtype=np.float64) for k in NDCG_KS}
    if scorer is not None:
        scorer = scorer.to(DEVICE)

    chunk = 128
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        h = hidden[start:end].to(DEVICE)
        s = bscores[start:end]
        l = labels[start:end]
        c = h.shape[0]
        lw = pw = None
        if scorer is not None and is_listwise:
            u = ustates[start:end].to(DEVICE)
            with torch.no_grad():
                lw = scorer(h, s.to(DEVICE), user_state=u).cpu()
        elif scorer is not None:
            kbeam = h.shape[1]
            with torch.no_grad():
                lo = scorer(h.reshape(c * kbeam, CODE_LENGTH, HIDDEN_DIM)).squeeze(-1)
                pw = torch.sigmoid(lo).cpu().reshape(c, kbeam)
        for ci in range(c):
            if scorer is None:
                order = torch.arange(l.shape[1])  # vanilla = beam order
            elif is_listwise:
                # λ=0 ⇒ vanilla (scorer off). "scorer-only" (beam off) is lam=None, a
                # separately-named config — NOT the grid's 0.0 (review finding #1).
                final = lw[ci] if lam is None else s[ci] / CODE_LENGTH + lam * lw[ci]
                order = final.argsort(descending=True)
            else:
                final = s[ci] / CODE_LENGTH + lam * torch.log(pw[ci].clamp(min=1e-8))
                order = final.argsort(descending=True)
            rel_ordered = l[ci][order].numpy()
            hit, mrr, ndcg = _rank_metrics(rel_ordered)
            idx = start + ci
            for k in KS:
                hits[k][idx] = hit[k]
            mrr_arr[idx] = mrr
            for k in NDCG_KS:
                ndcg_arr[k][idx] = ndcg[k]

    if scorer is not None:
        scorer.cpu()
    agg: dict = {f"R@{k}": float(hits[k].mean()) for k in KS}
    agg["MRR"] = float(mrr_arr.mean())
    for k in NDCG_KS:
        agg[f"nDCG@{k}"] = float(ndcg_arr[k].mean())
    return agg, hits


def train_focal(val_data: dict):
    """Train one Pointwise Focal scorer on a 70/30 split of val candidates."""
    pw_x = val_data["hidden"].reshape(-1, CODE_LENGTH, HIDDEN_DIM)
    pw_y = val_data["labels_binary"].reshape(-1)
    perm = torch.randperm(pw_x.shape[0])
    vn = int(pw_x.shape[0] * 0.3)
    x_tr, y_tr = pw_x[perm[vn:]].to(DEVICE), pw_y[perm[vn:]].to(DEVICE)
    x_va, y_va = pw_x[perm[:vn]].to(DEVICE), pw_y[perm[:vn]].to(DEVICE)
    scorer = train_pointwise(x_tr, y_tr, x_va, y_va, FocalBCELoss(g=2.0))
    del x_tr, y_tr, x_va, y_va, pw_x, pw_y
    gc.collect()
    torch.cuda.empty_cache()
    return scorer


def train_listwise_seeds(val_data: dict) -> dict:
    """Train one Listwise BCE scorer per seed."""
    lw_data = {
        "hidden": val_data["hidden"],
        "beam_scores": val_data["beam_scores"],
        "user_states": val_data["user_states"],
        "labels": val_data["labels_binary"],
    }
    scorers = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        scorer = ListwiseScorer(
            hidden_dim=HIDDEN_DIM, code_length=CODE_LENGTH,
            d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1,
        )
        history = train_listwise_scorer(
            scorer, lw_data, device=DEVICE,
            loss_type="bce", epochs=50, batch_size=32, lr=1e-3, patience=10,
        )
        print(f"  seed {seed}: {len(history)} ep, val_loss={history[-1]['val_loss']:.4f}")
        scorers[seed] = scorer.cpu()
    return scorers


def main() -> None:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model + data...")
    model = build_model()
    item_ids, cfi, iti, vsl, tsl, item_embeddings = prepare_data()
    test_user_ids = [s["user_id"] for s in tsl]

    print("\nCollecting beam data (per-sample)...")
    val_data = get_beam_data("val", vsl, model, item_ids, cfi, iti, item_embeddings)
    test_data = get_beam_data("test", tsl, model, item_ids, cfi, iti, item_embeddings)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\nTraining Pointwise Focal...")
    focal = train_focal(val_data)
    print("Selecting Focal lambda on val (val-R@1 argmax over locked grid)...")
    best_lam, best_r1 = LAMBDA_GRID[0], -1.0
    for lam in LAMBDA_GRID:
        agg, _ = eval_peruser(focal, False, lam, val_data)
        print(f"  lambda={lam:<5} val R@1={agg['R@1']:.4f}")
        if agg["R@1"] > best_r1:
            best_r1, best_lam = agg["R@1"], lam
    print(f"  -> best Focal lambda={best_lam} (val R@1={best_r1:.4f})")

    print("\nTraining 5-seed Listwise BCE...")
    seed_scorers = train_listwise_seeds(val_data)
    print("Selecting Listwise lambda on val (mean val-R@1 across seeds, same grid)...")
    best_lw_lam, best_lw_r1 = LAMBDA_GRID[0], -1.0
    for lam in LAMBDA_GRID:
        r1s = [eval_peruser(seed_scorers[seed], True, lam, val_data)[0]["R@1"] for seed in SEEDS]
        m = float(np.mean(r1s))
        print(f"  lambda={lam:<5} val mean R@1={m:.4f}")
        if m > best_lw_r1:
            best_lw_r1, best_lw_lam = m, lam
    print(f"  -> best Listwise lambda={best_lw_lam} (val mean R@1={best_lw_r1:.4f})")

    # ── Test evaluation ──
    print("\nTest evaluation...")
    van_agg, van_hits = eval_peruser(None, False, 0.0, test_data)
    focal_agg, focal_hits = eval_peruser(focal, False, best_lam, test_data)
    seed_aggs, seed_r1s, seed_hits = {}, [], {}
    for seed in SEEDS:
        agg, hh = eval_peruser(seed_scorers[seed], True, best_lw_lam, test_data)
        seed_aggs[str(seed)] = agg
        seed_r1s.append(agg["R@1"])
        seed_hits[seed] = hh
    mean_r1, std_r1 = float(np.mean(seed_r1s)), float(np.std(seed_r1s))
    # Per-sample listwise hit = majority vote across the 5 seeds (binary), so significance
    # tests the REPORTED 5-seed row, not a single seed (review finding #5).
    lw_hits = {k: (np.mean([seed_hits[s][k] for s in SEEDS], axis=0) >= 0.5)
               for k in (1, 10)}
    oracle = van_agg["R@50"]  # perfect ranking → Oracle R@k == R@50

    def vs_van(x: float) -> float:
        return (x - van_agg["R@1"]) / max(1e-8, van_agg["R@1"]) * 100

    print(f"\n{'Method':<34}{'R@1':>9}{'R@5':>9}{'R@10':>9}{'R@50':>9}{'vsVan':>8}")
    print("-" * 78)
    print(f"{'TIGER-equiv (Vanilla BS-50)':<34}{van_agg['R@1']:>9.4f}{van_agg['R@5']:>9.4f}"
          f"{van_agg['R@10']:>9.4f}{van_agg['R@50']:>9.4f}{'base':>8}")
    print(f"{'Pointwise Focal lam=' + str(best_lam):<34}{focal_agg['R@1']:>9.4f}{focal_agg['R@5']:>9.4f}"
          f"{focal_agg['R@10']:>9.4f}{focal_agg['R@50']:>9.4f}{vs_van(focal_agg['R@1']):>+7.1f}%")
    print(f"{'Listwise BCE (5-seed mean)':<34}{mean_r1:>9.4f}{'':>9}{'':>9}{'':>9}{vs_van(mean_r1):>+7.1f}%")
    print(f"{'Oracle (perfect rank)':<34}{oracle:>9.4f}{oracle:>9.4f}{oracle:>9.4f}{oracle:>9.4f}")
    print(f"\nScoring Efficiency: vanilla={van_agg["R@1"] / oracle * 100:.1f}%  "
          f"best_scorer={mean_r1 / oracle * 100:.1f}%")

    # ── Persist per-user hits (for significance test) ──
    np.savez(
        OUT_DIR / "per_user_hits.npz",
        user_ids=np.array(test_user_ids),
        vanilla_hit1=van_hits[1], vanilla_hit10=van_hits[10],
        focal_hit1=focal_hits[1], focal_hit10=focal_hits[10],
        listwise_hit1=lw_hits[1], listwise_hit10=lw_hits[10],
    )

    # ── Persist Main-Table results ──
    metric_keys = [f"R@{k}" for k in KS] + ["MRR", "nDCG@5", "nDCG@10"]
    lw_mean = {m: float(np.mean([seed_aggs[str(s)][m] for s in SEEDS])) for m in metric_keys}
    results = {
        "checkpoint": str(CKPT_PATH),
        "eval": {"beam_width": BEAM_WIDTH, "code_length": CODE_LENGTH,
                 "mode": "per_sample_beam", "n_test_samples": len(tsl)},
        "rows": {
            "tiger_equivalent_vanilla": dict(van_agg),
            "pointwise_focal": {"lambda": best_lam, **dict(focal_agg)},
            "listwise_bce_5seed": {
                "lambda": best_lw_lam, "mean_R@1": mean_r1, "std_R@1": std_r1,
                "vs_vanilla_pct": vs_van(mean_r1), "mean": lw_mean, "per_seed": seed_aggs,
            },
            "oracle": {f"R@{k}": oracle for k in KS},
        },
        "scoring_efficiency": {
            "vanilla": van_agg["R@1"] / oracle, "best_scorer": mean_r1 / oracle,
        },
        "runtime_sec": time.time() - t_start,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_DIR / 'results.json'} and per_user_hits.npz")

    # ── MLflow ──
    logger = MlflowRunLogger(
        enabled=True, tracking_uri="http://localhost:5000",
        experiment_name="mind_genrec", run_name="main_table_baselines",
        tags={"dataset": "MIND-small", "checkpoint": "baseline_retrain",
              "eval": "per_sample_beam_K50", "task": "T18+T3+scorers"},
    )
    with logger:
        logger.log_params({"beam_width": BEAM_WIDTH, "code_length": CODE_LENGTH,
                           "focal_lambda": best_lam, "listwise_lambda": best_lw_lam,
                           "lambda_grid": str(LAMBDA_GRID),
                           "n_test_samples": len(tsl), "checkpoint": str(CKPT_PATH)})
        metrics = {f"tiger_equiv.{m}": van_agg[m] for m in metric_keys}
        metrics.update({f"focal.{m}": focal_agg[m] for m in metric_keys})
        metrics.update({f"listwise.mean_{m}": lw_mean[m] for m in metric_keys})
        metrics.update({"listwise.std_R@1": std_r1, "oracle.R@1": oracle,
                        "se.vanilla": van_agg["R@1"] / oracle, "se.best_scorer": mean_r1 / oracle})
        logger.log_metrics(metrics)
        logger.log_dict(results, "main_table_results.json")

    print(f"\nTotal: {(time.time() - t_start) / 60:.0f} min")


if __name__ == "__main__":
    main()
