"""λ-sensitivity: held-out (Listwise) R@1 vs λ for all 4 settings + λ*↔SE annotation."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figures.lambda_log_parser import parse_heldout_log

BASE = Path("/data/lishazhai/workspace/GR4AD")
LOGS = BASE / "experiments/lambda_logs"
OUT = BASE / "figures"
SETTINGS = [("MIND", "mind_heldout_lambda.txt"), ("Beauty", "beauty_heldout_lambda.txt"),
            ("Toys", "toys_full_heldout.txt"), ("Sports", "sports_full_heldout.txt")]


def main():
    fig, ax = plt.subplots(figsize=(7, 5))
    rows = []
    for name, fn in SETTINGS:
        r = parse_heldout_log((LOGS / fn).read_text())
        curve = r["listwise"]
        xs = sorted(curve); ys = [curve[x] for x in xs]
        van = curve.get(0.0)
        rel = [y / van for y in ys] if van else ys      # normalize to λ=0 (vanilla) for cross-setting
        sv, ss = r.get("se_vanilla"), r.get("se_scorer")
        se_lbl = f"SE {sv:.0%}→{ss:.0%}" if (sv is not None and ss is not None) else "SE ?"
        ax.plot(xs, rel, "o-", label=f"{name} (Listwise λ*={r.get('best_listwise_lambda')}, {se_lbl})")
        rows.append((name, r.get("best_listwise_lambda"), sv, ss))
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("λ (fusion weight)"); ax.set_ylabel("held-out R@1 / vanilla R@1")
    ax.set_title("λ-sensitivity (held-out, Listwise) across settings"); ax.legend(fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"lambda_sensitivity.{ext}", dpi=150, bbox_inches="tight")
    print("λ*↔SE:", rows)


if __name__ == "__main__":
    main()
