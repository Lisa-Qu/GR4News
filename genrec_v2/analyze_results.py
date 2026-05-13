"""Deep analysis of proxy ablation results — PhD-level diagnostics."""
from __future__ import annotations

import mlflow
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.MlflowClient()
exp = client.get_experiment_by_name("genrec_v2_ablation")

runs_data = {}
for r in client.search_runs(experiment_ids=[exp.experiment_id]):
    name = r.info.run_name
    runs_data[name] = {
        "run_id": r.info.run_id,
        "metrics": r.data.metrics,
        "train_loss": client.get_metric_history(r.info.run_id, "train_loss"),
        "val_loss": client.get_metric_history(r.info.run_id, "val_loss"),
        "val_acc": client.get_metric_history(r.info.run_id, "val_token_acc"),
        "val_recall": client.get_metric_history(r.info.run_id, "val_recall_at1"),
    }

# ─── 1. Per-epoch table ───
print("=" * 80)
print("1. FULL PER-EPOCH TRAJECTORIES")
print("=" * 80)
for name in sorted(runs_data):
    d = runs_data[name]
    train = {h.step: h.value for h in d["train_loss"]}
    val_l = {h.step: h.value for h in d["val_loss"]}
    val_a = {h.step: h.value for h in d["val_acc"]}
    val_r = {h.step: h.value for h in d["val_recall"]}

    print(f"\n--- {name} ---")
    print(f"{'Ep':>4} {'TrLoss':>8} {'VaLoss':>8} {'VaAcc':>8} {'VaRec':>8} {'Gap':>8} {'dLoss':>8}")
    prev = None
    for ep in sorted(train):
        tl = train[ep]
        vl = val_l.get(ep)
        va = val_a.get(ep)
        vr = val_r.get(ep)
        gap = (vl - tl) if vl is not None else float("nan")
        dloss = (tl - prev) if prev is not None else float("nan")
        prev = tl
        vl_s = f"{vl:.4f}" if vl is not None else "   -   "
        va_s = f"{va:.4f}" if va is not None else "   -   "
        vr_s = f"{vr:.4f}" if vr is not None else "   -   "
        print(f"{ep:>4} {tl:>8.4f} {vl_s:>8} {va_s:>8} {vr_s:>8} {gap:>7.4f} {dloss:>7.4f}")

    # Peak
    raw_recall = d["val_recall"]
    if raw_recall:
        best = max(raw_recall, key=lambda h: h.value)
        print(f"  >> PEAK Val Rec@1: {best.value:.4f} @ ep {best.step}")

for name in sorted(runs_data):
    m = runs_data[name]["metrics"]
    for k in ["test_token_acc", "test_recall_at1", "test_loss"]:
        if k in m:
            print(f"  FINAL {k}: {m[k]:.4f}")

# ─── 2. Convergence speed ───
print("\n" + "=" * 80)
print("2. CONVERGENCE SPEED (epoch to first reach val_loss < X)")
print("=" * 80)
for name in sorted(runs_data):
    d = runs_data[name]
    for thresh in [2.1, 2.0, 1.90, 1.85, 1.82]:
        for h in d["val_loss"]:
            if h.value < thresh:
                print(f"  {name}: <{thresh} @ ep {h.step}")
                break
        else:
            print(f"  {name}: NEVER <{thresh}")

# ─── 3. Training stability (loss variance) ───
print("\n" + "=" * 80)
print("3. TRAINING STABILITY (epoch-to-epoch loss variance)")
print("=" * 80)
for name in sorted(runs_data):
    losses = [h.value for h in runs_data[name]["train_loss"]]
    if len(losses) >= 3:
        diffs = np.diff(losses)
        var = np.var(diffs)
        mean_drop = np.mean(diffs)
        print(f"  {name}: mean_dLoss={abs(mean_drop):.4f}, var(dLoss)={var:.6f}, last3_std={np.std(losses[-3:]):.6f}")

# ─── 4. Overfitting signal (train-val gap) ───
print("\n" + "=" * 80)
print("4. OVERFITTING SIGNAL (Train-Val Gap @ last epoch)")
print("=" * 80)
for name in sorted(runs_data):
    d = runs_data[name]
    train_last = d["train_loss"][-1].value
    val_match = [h for h in d["val_loss"] if h.step == d["train_loss"][-1].step]
    if val_match:
        gap = val_match[0].value - train_last
        print(f"  {name}: train={train_last:.4f}, val={val_match[0].value:.4f}, gap={gap:.4f}")

# ─── 5. Rank normalization (statistical significance proxy) ───
print("\n" + "=" * 80)
print("5. FINAL RANKING (average percentile across all val metrics)")
print("=" * 80)
scores = {}
for name in sorted(runs_data):
    d = runs_data[name]
    last = d["train_loss"][-1].step
    va = next((h.value for h in d["val_acc"] if h.step == last), 0)
    vr = next((h.value for h in d["val_recall"] if h.step == last), 0)
    vl = next((h.value for h in d["val_loss"] if h.step == last), 10)
    mt = d["metrics"].get("test_token_acc", 0)
    mr = d["metrics"].get("test_recall_at1", 0)
    scores[name] = {"val_acc": va, "val_recall": vr, "val_loss": vl, "test_acc": mt, "test_recall": mr}

# Rank each metric (1=best, higher rank score = better)
ranks = {name: 0 for name in scores}
for metric in ["val_acc", "test_acc", "test_recall"]:
    descending = True  # higher is better
    ordered = sorted(scores.items(), key=lambda x: -x[1][metric] if descending else x[1][metric])
    for rank, (name, _) in enumerate(ordered):
        ranks[name] += rank

for metric in ["val_loss"]:
    descending = False  # lower is better
    ordered = sorted(scores.items(), key=lambda x: x[1][metric])
    for rank, (name, _) in enumerate(ordered):
        ranks[name] += rank

print(f"{'Run':>20} {'ValAcc':>8} {'TestAcc':>8} {'TestRec':>8} {'ValLoss':>8} {'AvgRank':>8} {'Sort':>8}")
for name in sorted(ranks, key=lambda n: ranks[n]):
    s = scores[name]
    print(f"{name:>20} {s['val_acc']:>7.4f} {s['test_acc']:>7.4f} {s['test_recall']:>7.4f} {s['val_loss']:>7.4f} {ranks[name]/4:>7.2f} {'<<<' if ranks[name] == min(ranks.values()) else ''}")

# ─── 6. Key insight summary ───
print("\n" + "=" * 80)
print("6. INTERPRETATION")
print("=" * 80)

# Get the raw values for conclusions
names_ordered = sorted(ranks, key=lambda n: ranks[n])
best = names_ordered[0]
worst = names_ordered[-1]
delta = scores[best]["test_acc"] - scores[worst]["test_acc"]

print(f"  Best run: {best}")
print(f"  Worst run: {worst}")
print(f"  Delta (test_token_acc): {delta:.4f} ({delta*100:.2f} percentage points)")

# Check if delta is within convergence noise
noise_levels = []
for name in names_ordered:
    losses = [h.value for h in runs_data[name]["train_loss"]]
    if len(losses) >= 3:
        noise_levels.append(np.std(losses[-3:]))
avg_noise = np.mean(noise_levels) if noise_levels else 0
print(f"  Average last-3-epoch noise (std): {avg_noise:.6f}")
print(f"  Delta / noise ratio: {delta / max(1e-8, avg_noise):.2f}x")
if delta < avg_noise:
    print("  ⚠️ Delta is within noise level → conclusions are TENTATIVE")
else:
    print("  ✅ Delta exceeds noise level → conclusions are RELIABLE")

# Per-variable analysis
print("\n  --- Variable: Data Mode (A vs B) ---")
for hot in [False, True]:
    a_name = f"modeA_{'hot' if hot else 'nohot'}"
    b_name = f"modeB_{'hot' if hot else 'nohot'}"
    if a_name in scores and b_name in scores:
        delta_acc = scores[b_name]["test_acc"] - scores[a_name]["test_acc"]
        delta_rec = scores[b_name]["test_recall"] - scores[a_name]["test_recall"]
        print(f"  Hot={'ON' if hot else 'OFF'}: B - A = test_acc: {delta_acc:+.4f}, test_recall: {delta_rec:+.5f}")

print("\n  --- Variable: Hot News (ON vs OFF) ---")
for mode in ["A", "B"]:
    off_name = f"mode{mode}_nohot"
    on_name = f"mode{mode}_hot"
    if off_name in scores and on_name in scores:
        delta_acc = scores[on_name]["test_acc"] - scores[off_name]["test_acc"]
        delta_rec = scores[on_name]["test_recall"] - scores[off_name]["test_recall"]
        print(f"  Mode {mode}: HOT - NOHOT = test_acc: {delta_acc:+.4f}, test_recall: {delta_rec:+.5f}")
