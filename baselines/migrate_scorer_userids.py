#!/usr/bin/env python3
"""Migrate scorer ``per_user_hits.npz`` ``user_ids`` from synthetic ``_idx`` to REAL ASINs.

The scorer's ``experiments/{ds}_scorer/per_user_hits.npz`` ``user_ids`` were written as
``_idx0,_idx1,...`` because the OLD ``_extract_user_id`` in ``scripts/run_beauty_scorer.py``
fell back (it never probed the real ``"user_ids"`` collator key). Without real ASINs the
SASRec / NRMS ``baseline_vs_vanilla`` significance test cannot pair baseline users against the
generative vanilla anchor.

Fix (mirrors the Toys migration that already succeeded): row ``i`` of ``per_user_hits.npz``
corresponds to sample ``i`` of ``TestDatasetGRAM(..., mode="test")`` in ``DataLoader(shuffle=False)``
order — the SAME order the beam cache was collected. ``TestDatasetGRAM`` builds ``data_samples``
by iterating ``for user in self.user_seq_dict`` (insertion order = ``user_sequence.txt`` line
order) and stores ``one_sample["user_id"] = user`` = the REAL, un-reindexed ASIN
(== ``user_sequence.txt`` ``parts[0]``). So ``ds.data_samples[i]["user_id"]`` is sample ``i``'s
real ASIN, positionally aligned to ``per_user_hits.npz`` row ``i`` and to SASRec's per-user
``uid`` (``data_amazon.load_amazon_loo`` also keys on ``parts[0]``).

This OVERWRITES only the ``user_ids`` field of ``per_user_hits.npz`` (all hit arrays kept). It
validates ``len(real) == N`` (the per_user_hits row count) BEFORE writing, and re-loads + verifies
afterwards (no ``_idx`` remain; samples exist in ``user_sequence.txt`` first column).

Usage (on gram-server, cwd=/data/lishazhai/workspace/GR4AD, PYTHONPATH=that):
    <gram python> baselines/migrate_scorer_userids.py --dataset Beauty \
        --checkpoint <beauty_ckpt.pt> --code-length 7 \
        --item-id-path item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt \
        --hierarchical-id-type hierarchy_v1_c128_l7_len32768 --top-k-similar-item 10 \
        --npz experiments/beauty_scorer/per_user_hits.npz \
        --user-sequence /data/lishazhai/workspace/GRAM/rec_datasets/Beauty/user_sequence.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── Reuse the scorer's exact GRAM model / args construction (single source of truth). ──
# scripts/run_beauty_scorer.py inserts the GRAM src + GR4AD paths and exposes make_args /
# load_gram_model; importing it gives us the identical dataset-build configuration.
sys.path.insert(0, "/data/lishazhai/workspace/GR4AD")
sys.path.insert(0, "/data/lishazhai/workspace/GRAM/src")

from scripts.run_beauty_scorer import (  # noqa: E402
    make_args,
    load_gram_model,
    DEFAULT_ITEM_TO_CODE_PATH,
    DEFAULT_CODE_TO_ITEMS_PATH,
)


def build_real_user_ids(args, dataset_name: str, model, tokenizer, mode: str = "test") -> np.ndarray:
    """Ordered REAL ASIN user_id per sample of ``TestDatasetGRAM(mode)``.

    Mirrors ``collect_beam_data``'s DataLoader(shuffle=False) order: ``data_samples`` is built by
    ``for user in self.user_seq_dict`` (insertion order) and ``one_sample["user_id"] = user`` is
    the real ASIN. We read ``ds.data_samples[i]["user_id"]`` directly (no beam pass needed).
    """
    from data import TestDatasetGRAM

    ds = TestDatasetGRAM(
        args, dataset_name, "sequential", model, tokenizer,
        regenerate=False, phase=0, mode=mode,
    )
    return np.array([str(s["user_id"]) for s in ds.data_samples])


def _load_seq_userids(path: Path) -> set[str]:
    """First column (real ASIN user_id) of every ``user_sequence.txt`` line."""
    users: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                users.add(parts[0])
    return users


def migrate(npz_path: Path, real_uids: np.ndarray, seq_path: Path | None) -> None:
    """Overwrite ONLY ``user_ids`` in ``npz_path`` with ``real_uids`` (all hit arrays kept).

    Validates ``len(real_uids) == N`` (per_user_hits row count) BEFORE writing, then reloads and
    verifies the written file (no ``_idx`` remain; sampled ids exist in ``user_sequence.txt``).
    """
    d = np.load(npz_path, allow_pickle=True)
    arrays = {k: d[k] for k in d.files}
    old = arrays.get("user_ids")

    # Row count anchor: use a hit array (definitely N rows) so we never trust the (possibly stale)
    # old user_ids length blindly — though for these files they coincide.
    hit_key = next((k for k in d.files if k.startswith(("vanilla_hit", "baseline_hit"))), None)
    assert hit_key is not None, f"{npz_path}: no *_hit* array to anchor N"
    n = len(arrays[hit_key])

    assert len(real_uids) == n, (
        f"{npz_path}: real n={len(real_uids)} != per_user_hits n={n} (anchor '{hit_key}'); "
        f"TestDatasetGRAM(test) order does NOT match the cached per_user_hits rows — DO NOT write."
    )

    old_n = None if old is None else len(old)
    print(f"  per_user_hits n={n} (anchor '{hit_key}'), real uids n={len(real_uids)}, "
          f"old user_ids n={old_n}")
    print(f"  old sample={None if old is None else list(old[:3])} -> new sample={list(real_uids[:3])}")

    arrays["user_ids"] = real_uids
    np.savez(npz_path, **arrays)
    print(f"  OVERWROTE user_ids in {npz_path}")

    # ── Verify the written file ──
    chk = np.load(npz_path, allow_pickle=True)
    wrote = chk["user_ids"]
    assert len(wrote) == n, f"verify: wrote n={len(wrote)} != {n}"
    n_idx = sum(1 for u in wrote if str(u).startswith("_idx"))
    assert n_idx == 0, f"verify: {n_idx} synthetic _idx ids remain in {npz_path}"
    # Hit arrays must be byte-identical (we only touched user_ids).
    for k in d.files:
        if k == "user_ids":
            continue
        assert np.array_equal(chk[k], arrays[k]), f"verify: hit array '{k}' changed!"
    print(f"  VERIFY ok: N={n}, no _idx, hit arrays intact. samples={list(wrote[:5])}")

    if seq_path is not None:
        seq_users = _load_seq_userids(seq_path)
        sample = list(wrote[:5]) + list(wrote[-5:])
        missing = [u for u in sample if u not in seq_users]
        assert not missing, f"verify: sampled user_ids not in {seq_path} first column: {missing}"
        print(f"  VERIFY ok: sampled ids present in {seq_path.name} first column "
              f"({len(seq_users)} users).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["Beauty", "Sports", "Toys"])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--code-length", type=int, required=True)
    p.add_argument("--item-id-path", required=True)
    p.add_argument("--hierarchical-id-type", required=True)
    p.add_argument("--top-k-similar-item", type=int, default=10)
    # make_args() reads these for GRAM model init; AR mode ignores them, so the
    # run_beauty_scorer module defaults are fine (same as the Toys migration).
    p.add_argument("--item-to-code-path", type=Path, default=DEFAULT_ITEM_TO_CODE_PATH)
    p.add_argument("--code-to-items-path", type=Path, default=DEFAULT_CODE_TO_ITEMS_PATH)
    p.add_argument("--npz", type=Path, required=True,
                   help="experiments/{ds}_scorer/per_user_hits.npz to patch.")
    p.add_argument("--user-sequence", type=Path, default=None,
                   help="rec_datasets/{ds}/user_sequence.txt for the post-write membership check.")
    p.add_argument("--mode", default="test", choices=["test", "validation"],
                   help="Which TestDatasetGRAM split the per_user_hits rows came from (test).")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    assert cli.npz.exists(), f"--npz not found: {cli.npz}"
    args = make_args(cli)
    print(f"[{cli.dataset}] Loading GRAM model from {cli.checkpoint} ...")
    model, tokenizer, _ = load_gram_model(args, cli.checkpoint)
    print(f"[{cli.dataset}] Building TestDatasetGRAM(mode={cli.mode}) for real user_ids ...")
    real = build_real_user_ids(args, cli.dataset, model, tokenizer, cli.mode)
    print(f"[{cli.dataset}] migrating {cli.npz} ...")
    migrate(cli.npz, real, cli.user_sequence)
    print(f"[{cli.dataset}] DONE.")


if __name__ == "__main__":
    main()
