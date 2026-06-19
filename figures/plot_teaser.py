"""Fig. 1 teaser: the post-hoc reranking pipeline + the problem it fixes.
Frozen generative retriever emits an overconfident beam (high ECE, poor ranking despite high recall);
a small post-hoc scorer rescores the SAME beam by fusing avg_logprob with a calibrated relevance term,
selected by held-out λ. Renders a self-contained schematic to figures/teaser.{pdf,png}."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent

def _box(ax, x, y, w, h, text, fc, ec="#333333"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=1.4, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, zorder=5)

def _arrow(ax, x0, y0, x1, y1, text=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color="#333333"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.045, text, ha="center", va="bottom", fontsize=8)

def main():
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")

    _box(ax, 0.1, 1.2, 1.5, 0.8, "user\nhistory", "#eef2f7")
    _box(ax, 2.2, 1.0, 2.0, 1.2, "FROZEN\ngenerative\nretriever\n(beam $K{=}50$)", "#dbe9f6")
    _box(ax, 4.9, 1.2, 2.0, 0.8, "post-hoc\nSCORER $f$\n(trained)", "#dff0df")
    _box(ax, 7.6, 1.2, 2.2, 0.8, "reranked list\n$s+\\lambda\\,f(\\cdot)$", "#fde9d9")

    _arrow(ax, 1.6, 1.6, 2.2, 1.6)
    _arrow(ax, 4.2, 1.6, 4.9, 1.6, "beam + avg\\_logprob")
    _arrow(ax, 6.9, 1.6, 7.6, 1.6, "fuse")

    # Problem annotation under the generator; fix annotation under the scorer.
    ax.text(3.2, 0.55, "overconfident:\nECE $0.28$, SE $9$–$19\\%$", ha="center", va="top",
            fontsize=8, color="#b22222")
    ax.text(5.9, 0.55, "recalibrated:\nECE $0.001$", ha="center", va="top",
            fontsize=8, color="#1a7a1a")
    ax.text(8.7, 0.55, "$+3.2$–$6.3\\%$ R@1\n(model-agnostic)", ha="center", va="top",
            fontsize=8, color="#1a7a1a")
    # held-out lambda callout above the scorer.
    ax.text(5.9, 2.25, "$\\lambda$ via held-out val", ha="center", va="bottom", fontsize=8,
            style="italic", color="#555555")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"teaser.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}/teaser.pdf/.png")

if __name__ == "__main__":
    main()
