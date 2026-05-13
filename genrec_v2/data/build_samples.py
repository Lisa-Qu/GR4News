"""Build (history, target) pairs from MIND behaviors.tsv.

A-mode: Input = that impression's History field (static snapshot)
B-mode: Input = History + all previous impression clickeds (cumulative)
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterator


def _parse_tsv(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            history = parts[3].split() if parts[3] else []
            impressions = parts[4].split()
            clicked = [imp.split("-")[0] for imp in impressions if imp.endswith("-1")]
            yield {
                "user_id": parts[1],
                "time": datetime.strptime(parts[2], "%m/%d/%Y %I:%M:%S %p"),
                "history": history,
                "clicked": clicked,
            }


def build_samples(
    tsv_path: str,
    mode: str = "A",
) -> list[dict]:
    """Return list of (user_id, history_ids, target_news_id) dicts."""
    # Group impressions by user, sorted by time
    user_impressions = defaultdict(list)
    for imp in _parse_tsv(tsv_path):
        user_impressions[imp["user_id"]].append(imp)

    samples: list[dict] = []
    for uid, impressions in user_impressions.items():
        impressions.sort(key=lambda x: x["time"])

        accumulated_clicks: list[str] = []

        for imp in impressions:
            if mode == "A":
                history = list(imp["history"])
            elif mode == "B":
                history = list(imp["history"]) + list(accumulated_clicks)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if not history:
                # Can't train without any history
                for c in imp["clicked"]:
                    accumulated_clicks.append(c)
                continue

            for target in imp["clicked"]:
                samples.append({
                    "user_id": uid,
                    "history": history,
                    "target": target,
                })

            # After processing, add this impression's clicks to accumulated
            accumulated_clicks.extend(imp["clicked"])

    return samples
