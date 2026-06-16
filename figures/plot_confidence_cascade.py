"""Confidence-cascade figure (contribution 1) from cascade_data.json (baseline_retrain).
Left: recall as the decode chain extends pos0→pos3 (the cascade collapse). Right: per-position ECE,
oracle vs greedy (greedy ECE explodes at later positions = the over-confidence the scorer fixes)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/data/lishazhai/workspace/GR4AD")
DATA = BASE / "experiments/confidence_cascade_diagnosis/cascade_data.json"
OUT = BASE / "figures"


def main():
    d = json.loads(DATA.read_text())
    chain = d["chain"]
    xs = ["t0", "t0t1", "t0t1t2", "all4"]
    chain_vals = [chain[k] for k in xs]
    assert chain_vals == sorted(chain_vals, reverse=True), "cascade recall not monotone non-increasing"
    eo, eg = d["ece_oracle"], d["ece_greedy"]
    assert len(eo) == len(eg) == 4, "expected 4 positions of ECE"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(4), chain_vals, "o-", color="#1f77b4")
    ax1.set_xticks(range(4)); ax1.set_xticklabels(["pos0", "0-1", "0-2", "0-3"])
    ax1.set_ylabel("recall (chain consistent)"); ax1.set_title("Confidence cascade (chain recall)")
    ax2.plot(range(4), eo, "o-", label="oracle decode", color="#2ca02c")
    ax2.plot(range(4), eg, "s--", label="greedy decode", color="#d62728")
    ax2.set_xticks(range(4)); ax2.set_xticklabels([f"pos{i}" for i in range(4)])
    ax2.set_ylabel("ECE"); ax2.set_title("Per-position ECE: oracle vs greedy"); ax2.legend()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"confidence_cascade.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}/confidence_cascade.png/.pdf ; greedy ECE pos3={eg[3]:.3f} vs oracle {eo[3]:.3f}")


if __name__ == "__main__":
    main()
