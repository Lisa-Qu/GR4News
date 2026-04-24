"""Train and export the first-stage semantic ID mapping."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mind_genrec.data import NewsItem, iter_jsonl
from mind_genrec.model.item_encoder import EncoderType, ItemEncoderConfig, build_item_encoder
from mind_genrec.model.residual_quantizer import ResidualQuantizer, ResidualQuantizerConfig
from mind_genrec.model.semantic_id_mapper import SemanticIDMapper


def load_news_items(path: str | Path) -> list[NewsItem]:
    """Load normalized news JSONL as `NewsItem` objects."""

    records: list[NewsItem] = []
    for payload in iter_jsonl(path):
        records.append(
            NewsItem(
                news_id=payload["news_id"],
                category=payload.get("category", ""),
                subcategory=payload.get("subcategory", ""),
                title=payload.get("title", ""),
                abstract=payload.get("abstract", ""),
                url=payload.get("url", ""),
                title_entities=payload.get("title_entities", ""),
                abstract_entities=payload.get("abstract_entities", ""),
            )
        )
    return records


def train_quantizer(
    items: list[NewsItem],
    *,
    encoder_type: EncoderType,
    encoder_config: ItemEncoderConfig,
    quantizer_config: ResidualQuantizerConfig,
) -> tuple[np.ndarray, ResidualQuantizer, SemanticIDMapper]:
    """Encode items, fit a residual quantizer, and build the mapping."""

    if not items:
        raise ValueError("train_quantizer requires at least one item")

    encoder = build_item_encoder(encoder_type=encoder_type, config=encoder_config)
    embeddings = encoder.encode_items(items)
    quantizer = ResidualQuantizer(quantizer_config)
    codes = quantizer.fit(embeddings)
    mapper = SemanticIDMapper.from_codes(
        [item.news_id for item in items],
        codes,
    )
    return embeddings, quantizer, mapper


def export_quantizer_artifacts(
    *,
    items: list[NewsItem],
    embeddings: np.ndarray,
    quantizer: ResidualQuantizer,
    mapper: SemanticIDMapper,
    encoder_type: EncoderType,
    encoder_config: ItemEncoderConfig,
    quantizer_config: ResidualQuantizerConfig,
    output_dir: str | Path,
) -> dict[str, object]:
    """Write quantizer artifacts to disk."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    np.save(target_dir / "item_embeddings.npy", embeddings)
    (target_dir / "item_ids.json").write_text(
        json.dumps([item.news_id for item in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        target_dir / "codebooks.npz",
        **{f"codebook_{index}": codebook for index, codebook in enumerate(quantizer.codebooks)},
    )
    code_matrix = np.asarray(
        [mapper.item_to_code[item.news_id] for item in items],
        dtype=np.int32,
    )
    np.save(target_dir / "semantic_codes.npy", code_matrix)

    mapper.save(target_dir)
    summary = mapper.summary()
    export_metadata = {
        "item_count": len(items),
        "embedding_dim": int(embeddings.shape[1]),
        "code_length": quantizer.config.num_codebooks,
        "codebook_size": quantizer.config.codebook_size,
        "collided_code_count": summary.collided_code_count,
        "max_collision_size": summary.max_collision_size,
        "encoder_type": encoder_type,
        "encoder_config": asdict(encoder_config),
        "quantizer_config": asdict(quantizer_config),
    }
    (target_dir / "quantizer_metadata.json").write_text(
        json.dumps(export_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return export_metadata


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Train and export semantic IDs for MIND news items.")
    parser.add_argument("--news-jsonl", required=True, help="Path to normalized `news.jsonl`.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported semantic ID artifacts.")
    parser.add_argument("--encoder-type", default="hashing", choices=["hashing", "sbert"])
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--code-length", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    """Entry point."""

    args = build_parser().parse_args()
    items = load_news_items(args.news_jsonl)
    encoder_config = ItemEncoderConfig(embedding_dim=args.embedding_dim)
    quantizer_config = ResidualQuantizerConfig(
        num_codebooks=args.code_length,
        codebook_size=args.codebook_size,
        max_iterations=args.max_iterations,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    embeddings, quantizer, mapper = train_quantizer(
        items,
        encoder_type=args.encoder_type,
        encoder_config=encoder_config,
        quantizer_config=quantizer_config,
    )
    summary = export_quantizer_artifacts(
        items=items,
        embeddings=embeddings,
        quantizer=quantizer,
        mapper=mapper,
        encoder_type=args.encoder_type,
        encoder_config=encoder_config,
        quantizer_config=quantizer_config,
        output_dir=args.output_dir,
    )
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
