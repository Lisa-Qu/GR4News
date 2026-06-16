"""Parse a held-out λ-selection log into per-λ R@1 curves + chosen λ + SE."""
from __future__ import annotations
import re


def parse_heldout_log(text: str) -> dict:
    out: dict = {"focal": {}, "listwise": {}}
    section = None
    for line in text.splitlines():
        if "Focal lambda on held-out" in line:
            section = "focal"
            continue
        if "Listwise lambda on held-out" in line:
            section = "listwise"
            continue
        m = re.search(r"lambda=([0-9.]+)\s+held-out (?:mean )?R@1=([0-9.]+)", line)
        if m and section:
            out[section][float(m.group(1))] = float(m.group(2))
        b = re.search(r"best Focal lambda=([0-9.]+)", line)
        if b:
            out["best_focal_lambda"] = float(b.group(1))
        b = re.search(r"best Listwise lambda=([0-9.]+)", line)
        if b:
            out["best_listwise_lambda"] = float(b.group(1))
        # SE line has two known formats across settings:
        #   MIND:   "Scoring Efficiency: vanilla=9.0%  best_scorer=9.6%"
        #   Amazon: "Scoring Efficiency: Vanilla=15.2%, Best Scorer=15.9%"
        # Tolerate case + the comma/space separator + optional space in "Best Scorer".
        se = re.search(
            r"Scoring Efficiency:\s*vanilla=([0-9.]+)%\s*,?\s*best[ _]scorer=([0-9.]+)%",
            line,
            flags=re.IGNORECASE,
        )
        if se:
            out["se_vanilla"] = float(se.group(1)) / 100
            out["se_scorer"] = float(se.group(2)) / 100
    assert out["focal"] or out["listwise"], "no λ curve parsed — log format changed"
    return out
