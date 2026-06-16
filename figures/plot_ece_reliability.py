"""Per-candidate reliability diagram on baseline_retrain (reuses the cached per-sample beam).
Two curves: (a) vanilla = softmax over each sample's 50-beam of avg_logprob; (b) scorer = sigmoid of
the listwise scorer logit. Empirical = candidate is the target (labels_binary). Lower ECE = better
calibrated → shows the scorer's calibration gain."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
BASE = Path("/data/lishazhai/workspace/GR4AD")
sys.path.insert(0, str(BASE))
from genrec_v2.run_main_table import train_listwise_seeds, CODE_LENGTH, HIDDEN_DIM, SEEDS  # reuse
OUT = BASE / "figures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _reliability(pred, hit, n_bins=10):
    """Return (bin_centers, bin_acc, bin_conf, ece) over predicted prob `pred` vs binary `hit`."""
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs, ece = [], [], [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred >= lo) & (pred < hi if hi < 1.0 else pred <= hi)
        centers.append((lo + hi) / 2)
        if m.sum() == 0:
            accs.append(np.nan); confs.append(np.nan); continue
        a, c = float(hit[m].mean()), float(pred[m].mean())
        accs.append(a); confs.append(c); ece += (m.mean()) * abs(a - c)
    return np.array(centers), np.array(accs), np.array(confs), ece


def main():
    val = torch.load(BASE / "experiments/main_table/beam_val.pt", map_location="cpu")
    test = torch.load(BASE / "experiments/main_table/beam_test.pt", map_location="cpu")
    seed_scorers = train_listwise_seeds(val)            # reuse headline training (cached beam)
    scorer = seed_scorers[SEEDS[0]].to(DEVICE).eval()

    S = test["beam_scores"]; H = test["hidden"]; U = test["user_states"]; Y = test["labels_binary"]
    van_p, sc_p, hit = [], [], []
    with torch.no_grad():
        for i in range(0, S.shape[0], 256):
            s = S[i:i+256]; y = Y[i:i+256]
            # Vanilla P = the generator's ABSOLUTE per-candidate confidence exp(avg_logprob) — NOT a
            # softmax over the 50-beam (which sums to 1 → per-candidate mean forced to 1/K, making the
            # ECE a pure normalization artifact rather than a calibration measure; review-fix 2026-06-16).
            # exp(avg_logprob) is the sequence probability the generator assigns; the cascade diagnosis
            # shows it is OVER-confident, which the scorer's BCE-calibrated sigmoid recalibrates.
            van_p.append(torch.exp(s / CODE_LENGTH).clamp(0, 1).reshape(-1).numpy())
            lw = scorer(H[i:i+256].to(DEVICE), s.to(DEVICE), user_state=U[i:i+256].to(DEVICE)).cpu()
            sc_p.append(torch.sigmoid(lw).reshape(-1).numpy())
            hit.append(y.reshape(-1).numpy())
    van_p, sc_p, hit = np.concatenate(van_p), np.concatenate(sc_p), np.concatenate(hit)
    _, va, vc, ece_v = _reliability(van_p, hit)
    _, sa, sc, ece_s = _reliability(sc_p, hit)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k:", label="perfect")
    ax.plot(vc, va, "s--", color="#d62728", label=f"vanilla beam (ECE={ece_v:.3f})")
    ax.plot(sc, sa, "o-", color="#2ca02c", label=f"listwise scorer (ECE={ece_s:.3f})")
    ax.set_xlabel("predicted P(relevant)"); ax.set_ylabel("empirical hit rate")
    ax.set_title("Per-candidate reliability (baseline_retrain)"); ax.legend()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ece_reliability.{ext}", dpi=150, bbox_inches="tight")
    (OUT / "ece_reliability.json").write_text(json.dumps({"ece_vanilla": ece_v, "ece_scorer": ece_s}, indent=2))
    print(f"ECE vanilla={ece_v:.4f} scorer={ece_s:.4f} -> {'scorer better' if ece_s<ece_v else 'NOT better'}")


if __name__ == "__main__":
    main()
