"""Orchestrate end-to-end normalization for MIND-small or MIND-large."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from mind_genrec.data.build_item_corpus import build_item_corpus
from mind_genrec.data.build_train_samples import build_samples, export_samples
from mind_genrec.data.dataset import TrainingSample, write_jsonl


def _stable_impression_sort_key(impression_id: str) -> str:
    digest = hashlib.blake2b(impression_id.encode("utf-8"), digest_size=8).hexdigest()
    return f"{digest}:{impression_id}"


def _split_validation_and_test(
    samples: list[TrainingSample],
    *,
    validation_ratio: float,
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    """Split evaluation samples by impression id so one row never leaks across splits."""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be strictly between 0 and 1")

    grouped: dict[str, list[TrainingSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.impression_id, []).append(sample)

    ordered_impressions = sorted(grouped.keys(), key=_stable_impression_sort_key)
    if not ordered_impressions:
        return [], []
    if len(ordered_impressions) == 1:
        only = grouped[ordered_impressions[0]]
        return list(only), []

    validation_impression_count = int(round(len(ordered_impressions) * validation_ratio))
    validation_impression_count = min(
        max(1, validation_impression_count),
        len(ordered_impressions) - 1,
    )
    validation_ids = set(ordered_impressions[:validation_impression_count])

    validation_samples: list[TrainingSample] = []
    test_samples: list[TrainingSample] = []
    for impression_id in ordered_impressions:
        target = validation_samples if impression_id in validation_ids else test_samples
        target.extend(grouped[impression_id])
    return validation_samples, test_samples


def preprocess_dataset(
    *,
    train_dir: Path,
    valid_dir: Path,
    output_dir: Path,
    max_history_length: int,
    validation_ratio: float = 0.5,
) -> dict[str, int]:
    """Build normalized news/train/validation/test JSONL files from raw split directories."""

    output_dir.mkdir(parents=True, exist_ok=True)

    news_count = build_item_corpus(
        [train_dir / "news.tsv", valid_dir / "news.tsv"],
        output_dir / "news.jsonl",
    )
    train_count = export_samples(
        train_dir / "behaviors.tsv",
        output_dir / "train.jsonl",
        split="train",
        max_history_length=max_history_length,
        skip_empty_history=True,
        skip_unlabeled=True,
    )
    raw_eval_samples = build_samples(
        valid_dir / "behaviors.tsv",
        split="valid",
        max_history_length=max_history_length,
        skip_empty_history=True,
        skip_unlabeled=True,
    )
    validation_samples, test_samples = _split_validation_and_test(
        raw_eval_samples,
        validation_ratio=validation_ratio,
    )
    write_jsonl(output_dir / "validation.jsonl", validation_samples)
    write_jsonl(output_dir / "valid.jsonl", validation_samples)
    write_jsonl(output_dir / "test.jsonl", test_samples)

    return {
        "news_count": news_count,
        "train_sample_count": train_count,
        "validation_sample_count": len(validation_samples),
        "test_sample_count": len(test_samples),
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Normalize raw MIND data into JSONL files.")
    parser.add_argument("--train-dir", required=True, help="Directory containing raw train split files.")
    parser.add_argument("--valid-dir", required=True, help="Directory containing raw valid split files.")
    parser.add_argument("--output-dir", required=True, help="Target directory for normalized data.")
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--validation-ratio", type=float, default=0.5)
    return parser


def main() -> None:
    """Entry point."""

    args = build_parser().parse_args()
    summary = preprocess_dataset(
        train_dir=Path(args.train_dir),
        valid_dir=Path(args.valid_dir),
        output_dir=Path(args.output_dir),
        max_history_length=args.max_history_length,
        validation_ratio=args.validation_ratio,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
