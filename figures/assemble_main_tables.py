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
    """Parse run_listwise_scorer stdout → {loss: {label: best R@1}} (max R@1 across the λ sweep)."""
    best: dict = {}
    pat = re.compile(r"(bce|listmle|approx_ndcg)_(binary|soft)\b.*?R@1=([0-9.]+)")
    for m in pat.finditer(log_text):
        loss, label, r1 = m.group(1), m.group(2), float(m.group(3))
        best.setdefault(loss, {}).setdefault(label, r1)
        best[loss][label] = max(best[loss][label], r1)
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


def main():
    out = BASE / "experiments/main_tables"; out.mkdir(parents=True, exist_ok=True)
    news, ecom = build_news_table(), build_ecom_table()
    (out / "news_table.md").write_text(news)
    (out / "ecom_table.md").write_text(ecom)
    print(news); print(); print(ecom)


if __name__ == "__main__":
    main()
