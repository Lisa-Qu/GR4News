"""Build normalized next-click training samples from raw MIND behaviors."""

from __future__ import annotations

import argparse
from pathlib import Path

from mind_genrec.data import (
    TrainingSample,
    build_training_samples,
    iter_behavior_tsv,
    write_jsonl,
)


def build_samples(
    behaviors_path: Path,
    *,
    split: str,
    max_history_length: int,
    skip_empty_history: bool,
    skip_unlabeled: bool,
) -> list[TrainingSample]:
    """Convert one raw `behaviors.tsv` into in-memory normalized samples."""

    return list(
        build_training_samples(
            iter_behavior_tsv(behaviors_path, split=split),
            split=split,
            max_history_length=max_history_length,
            skip_empty_history=skip_empty_history,
            skip_unlabeled=skip_unlabeled,
        )
    )


def export_samples(
    behaviors_path: Path,
    output_path: Path,
    *,
    split: str,
    max_history_length: int,
    skip_empty_history: bool,
    skip_unlabeled: bool,
) -> int:
    """Convert one raw `behaviors.tsv` into normalized JSONL training samples."""

    samples = build_samples(
        behaviors_path,
        split=split,
        max_history_length=max_history_length,
        skip_empty_history=skip_empty_history,
        skip_unlabeled=skip_unlabeled,
    )
    write_jsonl(output_path, samples)
    return len(samples)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build normalized MIND training samples.")
    parser.add_argument("--behaviors-path", required=True, help="Path to raw `behaviors.tsv`.")
    parser.add_argument("--output-path", required=True, help="Target JSONL path.")
    parser.add_argument("--split", required=True, help="Split name, for example `train` or `valid`.")
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--keep-empty-history", action="store_true")
    parser.add_argument("--keep-unlabeled", action="store_true")
    return parser


def main() -> None:
    """Entry point."""

    args = build_parser().parse_args()
    count = export_samples(
        Path(args.behaviors_path),
        Path(args.output_path),
        split=args.split,
        max_history_length=args.max_history_length,
        skip_empty_history=not args.keep_empty_history,
        skip_unlabeled=not args.keep_unlabeled,
    )
    print(f"Wrote {count} normalized samples to {args.output_path}")


if __name__ == "__main__":
    main()
