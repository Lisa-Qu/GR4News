"""Build a normalized item corpus from raw MIND `news.tsv` files."""

from __future__ import annotations

import argparse
from pathlib import Path

from mind_genrec.data import iter_news_tsv, write_jsonl


def build_item_corpus(news_paths: list[Path], output_path: Path) -> int:
    """Merge one or more raw `news.tsv` files into one deduplicated JSONL corpus."""

    merged: dict[str, object] = {}
    for news_path in news_paths:
        for record in iter_news_tsv(news_path):
            merged[record.news_id] = record
    ordered_records = [merged[news_id] for news_id in sorted(merged)]
    write_jsonl(output_path, ordered_records)
    return len(ordered_records)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Build normalized MIND news corpus JSONL.")
    parser.add_argument(
        "--news-path",
        action="append",
        required=True,
        help="Path to one raw `news.tsv`. Repeat for multiple splits.",
    )
    parser.add_argument("--output-path", required=True, help="Target `news.jsonl` path.")
    return parser


def main() -> None:
    """Entry point."""

    args = build_parser().parse_args()
    count = build_item_corpus(
        [Path(path) for path in args.news_path],
        Path(args.output_path),
    )
    print(f"Wrote {count} normalized news items to {args.output_path}")


if __name__ == "__main__":
    main()
