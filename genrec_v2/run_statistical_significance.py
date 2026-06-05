"""Statistical significance for scorer vs vanilla (T13).

Consumes ``experiments/main_table/per_user_hits.npz`` (written by
``run_main_table.py``) and tests whether the post-hoc scorer's improvement over
the vanilla beam ranking is significant.

Two complementary tests per comparison:
    1. Paired Wilcoxon signed-rank on **per-user hit-rate** (samples grouped by
       user, rate = mean hit over that user's samples). This is the conventional
       RecSys significance test and respects user-level pairing.
    2. McNemar's exact test on **per-sample hit@1** (paired binary outcomes),
       which is the textbook test for paired 0/1 data and is robust when most
       users contribute a single sample (Wilcoxon then drops many ties).

Output: ``experiments/main_table/significance.json`` + console table with
* (p<0.05) / ** (p<0.01) markers.

Usage:
    python -u genrec_v2/run_statistical_significance.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2, wilcoxon

OUT_DIR = Path("/home/lishazhai/workspace/GR4AD/experiments/main_table")


def stars(p: float) -> str:
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def per_user_rate(user_ids: np.ndarray, hits: np.ndarray) -> np.ndarray:
    """Aggregate per-sample hits into per-user hit-rates (paired user order)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for uid, hit in zip(user_ids, hits, strict=True):
        groups[uid].append(float(hit))
    return np.array([np.mean(groups[u]) for u in sorted(groups)])


def wilcoxon_test(base: np.ndarray, scorer: np.ndarray) -> dict:
    """Paired Wilcoxon on per-user rates (scorer - base)."""
    diff = scorer - base
    nonzero = int(np.count_nonzero(diff))
    if nonzero == 0:
        return {"p_value": 1.0, "n_users": len(base), "n_nonzero_pairs": 0,
                "base_mean": float(base.mean()), "scorer_mean": float(scorer.mean())}
    _, p = wilcoxon(scorer, base, zero_method="wilcox", alternative="two-sided")
    return {"p_value": float(p), "n_users": len(base), "n_nonzero_pairs": nonzero,
            "base_mean": float(base.mean()), "scorer_mean": float(scorer.mean())}


def mcnemar_test(base: np.ndarray, scorer: np.ndarray) -> dict:
    """McNemar exact-ish test on paired per-sample binary hits (continuity-corrected)."""
    b = base.astype(bool)
    s = scorer.astype(bool)
    n01 = int(np.sum(~b & s))   # vanilla miss, scorer hit  (scorer wins)
    n10 = int(np.sum(b & ~s))   # vanilla hit,  scorer miss (scorer loses)
    discordant = n01 + n10
    if discordant == 0:
        p = 1.0
        stat = 0.0
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / discordant
        p = float(chi2.sf(stat, df=1))
    return {"p_value": p, "chi2": float(stat), "scorer_wins": n01,
            "scorer_losses": n10, "discordant": discordant}


def run_comparison(name: str, uids: np.ndarray, base: np.ndarray, scorer: np.ndarray) -> dict:
    wil = wilcoxon_test(per_user_rate(uids, base), per_user_rate(uids, scorer))
    mc = mcnemar_test(base, scorer)
    return {
        "comparison": name,
        "wilcoxon_peruser": {**wil, "stars": stars(wil["p_value"])},
        "mcnemar_persample": {**mc, "stars": stars(mc["p_value"])},
    }


def main() -> None:
    npz = OUT_DIR / "per_user_hits.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} not found — run run_main_table.py first.")
    d = np.load(npz, allow_pickle=True)
    uids = d["user_ids"]

    comparisons = [
        ("listwise_vs_vanilla_hit@1", d["vanilla_hit1"], d["listwise_hit1"]),
        ("listwise_vs_vanilla_hit@10", d["vanilla_hit10"], d["listwise_hit10"]),
        ("focal_vs_vanilla_hit@1", d["vanilla_hit1"], d["focal_hit1"]),
        ("focal_vs_vanilla_hit@10", d["vanilla_hit10"], d["focal_hit10"]),
    ]
    results = [run_comparison(name, uids, base, scorer) for name, base, scorer in comparisons]

    print(f"{'Comparison':<32}{'Wilcoxon p':>14}{'McNemar p':>14}{'win/lose':>14}")
    print("-" * 74)
    for r in results:
        w, m = r["wilcoxon_peruser"], r["mcnemar_persample"]
        print(f"{r['comparison']:<32}"
              f"{w['p_value']:>11.2e}{w['stars']:<3}"
              f"{m['p_value']:>11.2e}{m['stars']:<3}"
              f"{str(m['scorer_wins']) + '/' + str(m['scorer_losses']):>14}")

    out = {"source": str(npz), "n_test_samples": int(len(uids)),
           "n_test_users": int(len(set(uids.tolist()))), "comparisons": results}
    (OUT_DIR / "significance.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_DIR / 'significance.json'}")


if __name__ == "__main__":
    main()
