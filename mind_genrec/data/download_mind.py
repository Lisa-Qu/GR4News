"""Stage raw MIND files into the local project tree.

This script does not force a specific download source. Instead it accepts a
local archive or a directory that already contains `news.tsv` and
`behaviors.tsv`, then copies or extracts it into a structured raw-data layout.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def _copy_tree(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for candidate in ("news.tsv", "behaviors.tsv"):
        source_file = source_dir / candidate
        if source_file.exists():
            shutil.copy2(source_file, destination_dir / candidate)


def stage_source(source: Path, output_dir: Path, split_name: str) -> Path:
    """Copy or extract one raw split into the requested destination."""

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        _copy_tree(source, split_dir)
        return split_dir

    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source, "r") as archive:
            archive.extractall(split_dir)
        return split_dir

    raise ValueError(f"Unsupported source type: {source}")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Stage raw MIND files locally.")
    parser.add_argument("--train-source", required=True, help="Directory or zip for train split.")
    parser.add_argument("--valid-source", required=True, help="Directory or zip for valid split.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where `train/` and `valid/` raw MIND files will be staged.",
    )
    return parser


def main() -> None:
    """Entry point."""

    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    train_dir = stage_source(Path(args.train_source), output_dir, "train")
    valid_dir = stage_source(Path(args.valid_source), output_dir, "valid")

    print(f"Staged train split at: {train_dir}")
    print(f"Staged valid split at: {valid_dir}")


if __name__ == "__main__":
    main()
