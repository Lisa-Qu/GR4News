"""Assemble the two final main tables (News + E-commerce) from results/significance
jsons, plus the loss×label diagnostic table from the run_listwise_scorer grid log.

All numbers trace to already-computed artifacts under experiments/. Markdown (and
optional .tex) emitted to experiments/main_tables/."""
from __future__ import annotations
import json
import re
from pathlib import Path

BASE = Path("/data/lishazhai/workspace/GR4AD")


# ---------------------------------------------------------------------------
# Task 3: loss×label diagnostic table
# ---------------------------------------------------------------------------
def loss_label_table_from_log(log_text: str) -> dict:
    """Parse run_listwise_scorer stdout → {loss: {label: best R@1}} (max R@1 across the λ sweep).

    Handles two row formats:
      (a) explicit  ``bce_binary  ...  R@1=0.0361``  (the documented/test form), and
      (b) the actual grid table  ``bce_binary  0.50   0.0361   0.1134   0.1720``
          (``run_listwise_scorer`` prints space-separated columns: name, λ, R@1, R@5, R@10).
    Whichever matches a line wins; (a) is tried first so an explicit ``R@1=`` is never
    mis-read as a column value."""
    best: dict = {}
    pat_explicit = re.compile(r"(bce|listmle|approx_ndcg)_(binary|soft)\b.*?R@1=([0-9.]+)")
    # Grid row: name then λ (a float, or the literal "pure") then R@1 as the next float.
    pat_grid = re.compile(
        r"^(bce|listmle|approx_ndcg)_(binary|soft)\s+(?:[0-9.]+|pure)\s+([0-9.]+)"
    )

    def _record(loss: str, label: str, r1: float) -> None:
        best.setdefault(loss, {}).setdefault(label, r1)
        best[loss][label] = max(best[loss][label], r1)

    for line in log_text.splitlines():
        m = pat_explicit.search(line)
        if m:
            _record(m.group(1), m.group(2), float(m.group(3)))
            continue
        g = pat_grid.match(line.strip())
        if g:
            _record(g.group(1), g.group(2), float(g.group(3)))
    return best


# ---------------------------------------------------------------------------
# Task 5: significance-star + cell-format helpers
# ---------------------------------------------------------------------------
def stars_for(p: float) -> str:
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def fmt_cell(r1: float, pct: float | None = None) -> str:
    return f"{r1:.4f}" + (f" ({pct:+.1f}%)" if pct is not None else "")


def _load(path: str) -> dict:
    return json.loads((BASE / path).read_text())


# ---------------------------------------------------------------------------
# Task 5: main-table assemblers
# ---------------------------------------------------------------------------
def build_news_table() -> str:
    sc = _load("experiments/main_table/results.json")["rows"]
    nrms = _load("experiments/nrms/results.json")["rows"]["nrms"]
    sig = {c["comparison"]: c for c in _load("experiments/main_table/significance.json")["comparisons"]}
    van = sc["tiger_equivalent_vanilla"]; foc = sc["pointwise_focal"]; lw = sc["listwise_bce_5seed"]
    def pct(x): return (x - van["R@1"]) / van["R@1"] * 100
    lw_star = stars_for(sig["listwise_vs_vanilla_hit@1"]["mcnemar_persample"]["p_value"])
    rows = [
        ("NRMS (full-catalog)", fmt_cell(nrms["R@1"]), f"{nrms['R@10']:.4f}", f"{nrms['R@50']:.4f}"),
        ("TIGER-equiv (vanilla)", fmt_cell(van["R@1"]), f"{van['R@10']:.4f}", f"{van['R@50']:.4f}"),
        ("+Pointwise Focal", fmt_cell(foc["R@1"], pct(foc["R@1"])), f"{foc['R@10']:.4f}", "—"),
        (f"+Listwise (5-seed){lw_star}", fmt_cell(lw["mean_R@1"], lw["vs_vanilla_pct"]),
         f"{lw['mean']['R@10']:.4f}", "—"),
        ("Oracle", f"{van['R@50']:.4f}", f"{van['R@50']:.4f}", f"{van['R@50']:.4f}"),
    ]
    return _render("News (MIND-small, user-split, full-catalog)", ["Method", "R@1", "R@10", "R@50"], rows)


def build_ecom_table() -> str:
    out_rows = []
    sasrec = {ds: _load(f"experiments/sasrec_{ds}/results.json")["rows"]["sasrec"] for ds in ("Beauty", "Sports", "Toys")}
    scorers = {ds: _load(f"experiments/{ds.lower()}_scorer/results.json")["rows"] for ds in ("Beauty", "Sports", "Toys")}
    header = ["Method", "Beauty R@10", "Sports R@10", "Toys R@10"]
    out_rows.append(("SASRec (full-catalog)", *[f"{sasrec[ds]['R@10']:.4f}" for ds in ("Beauty", "Sports", "Toys")]))
    out_rows.append(("TIGER-equiv (vanilla)", *[f"{scorers[ds]['vanilla']['R@10']:.4f}" for ds in ("Beauty", "Sports", "Toys")]))
    out_rows.append(("+Listwise (5-seed)", *[f"{scorers[ds]['listwise_bce_5seed']['mean']['R@10']:.4f}" for ds in ("Beauty", "Sports", "Toys")]))
    out_rows.append(("Oracle", *[f"{scorers[ds]['vanilla']['R@50']:.4f}" for ds in ("Beauty", "Sports", "Toys")]))
    return _render("E-commerce (Beauty/Sports/Toys, LOO, full-catalog)", header, out_rows)


CAVEAT = ("Footnote: generative rows (TIGER-equiv/Focal/Listwise/Oracle) are beam-recall-bounded "
          "(rank within the K=50 beam); NRMS/SASRec rank the full catalog. Absolute R@K is not "
          "directly comparable across the two; ** p<0.01, * p<0.05 (McNemar vs vanilla).")


def _render(title: str, header: list, rows: list) -> str:
    md = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    md += ["", CAVEAT, ""]
    return "\n".join(md)


LOSS_LABEL_CAPTION = (
    "Caption: loss×label diagnostic on MIND (this script's own beam collection + 7-pt λ grid; "
    "relative loss×label comparison only — the headline uses held-out λ + a dense grid). "
    "Each cell = best held-out R@1 across the λ sweep; **bold** = grid max. BCE-binary is the "
    "locked headline choice."
)


def _render_loss_label(best: dict, fmt: str = "md") -> str:
    """Render the loss×label grid (rows=loss, cols=label), bolding the global max cell."""
    losses = [l for l in ("bce", "listmle", "approx_ndcg") if l in best]
    labels = ("binary", "soft")
    flat = [best[l][lb] for l in losses for lb in labels if lb in best.get(l, {})]
    top = max(flat) if flat else None

    def cell(loss: str, label: str) -> str:
        v = best.get(loss, {}).get(label)
        if v is None:
            return "—"
        s = f"{v:.4f}"
        if top is not None and abs(v - top) < 1e-12:
            return (f"**{s}**" if fmt == "md" else r"\textbf{" + s + "}")
        return s

    if fmt == "md":
        header = ["loss \\ label", "binary", "soft"]
        out = ["### Loss × Label diagnostic (MIND, best held-out R@1)", "",
               "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for l in losses:
            out.append("| " + " | ".join([l] + [cell(l, lb) for lb in labels]) + " |")
        out += ["", LOSS_LABEL_CAPTION, ""]
        return "\n".join(out)
    # tex
    out = [r"\begin{tabular}{lcc}", r"\toprule", r"loss / label & binary & soft \\", r"\midrule"]
    for l in losses:
        out.append(f"{l} & " + " & ".join(cell(l, lb) for lb in labels) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def emit_loss_label_table():
    """Task 3 Step 5: read the grid log → write loss_label_table.md (+ .tex). Report BCE-binary."""
    out = BASE / "experiments/main_tables"; out.mkdir(parents=True, exist_ok=True)
    log_text = (out / "loss_label_raw.log").read_text()
    best = loss_label_table_from_log(log_text)
    md = _render_loss_label(best, "md")
    (out / "loss_label_table.md").write_text(md)
    (out / "loss_label_table.tex").write_text(_render_loss_label(best, "tex"))
    print(md)
    flat = {(l, lb): best[l][lb] for l in best for lb in best[l]}
    if flat:
        bm = max(flat.values()); bb = best.get("bce", {}).get("binary")
        if bb is not None:
            gap = (bb - bm) / bm * 100
            verdict = "best" if abs(bb - bm) < 1e-12 else f"within {abs(gap):.1f}% of best"
            print(f"\nBCE-binary={bb:.4f}  grid-max={bm:.4f}  -> BCE-binary is {verdict}")


def main():
    out = BASE / "experiments/main_tables"; out.mkdir(parents=True, exist_ok=True)
    news, ecom = build_news_table(), build_ecom_table()
    (out / "news_table.md").write_text(news)
    (out / "ecom_table.md").write_text(ecom)
    print(news); print(); print(ecom)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "loss_label":
        emit_loss_label_table()
    else:
        main()
