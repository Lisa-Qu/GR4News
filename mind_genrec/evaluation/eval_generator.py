"""Offline evaluation for the AR semantic-code generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mind_genrec.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank_at_k,
    ndcg_at_k,
)
from mind_genrec.model import ARSemanticIdGenerator, SemanticIDMapper, SemanticIdBeamSearchRetriever
from mind_genrec.model import GeneratorConfig
from mind_genrec.training.generator_data import (
    GeneratorCollator,
    GeneratorDataset,
    build_item_index,
    move_batch,
    resolve_item_ids,
)
from mind_genrec.training.train_generator import evaluate


def _parse_top_ks(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("top-k list must not be empty")
    return sorted(set(values))


def _build_code_metrics(
    *,
    checkpoint_path: Path,
    semantic_artifact_dir: Path,
    eval_jsonl: Path,
    max_history_length: int,
    batch_size: int,
    max_eval_samples: int | None,
    device: torch.device,
) -> dict[str, float]:
    mapper = SemanticIDMapper.load(semantic_artifact_dir)
    item_embeddings = np.load(semantic_artifact_dir / "item_embeddings.npy")
    item_ids = resolve_item_ids(semantic_artifact_dir, mapper)
    item_to_index = build_item_index(item_ids)
    dataset = GeneratorDataset(
        sample_path=eval_jsonl,
        item_to_index=item_to_index,
        mapper=mapper,
        max_history_length=max_history_length,
        max_samples=max_eval_samples,
    )
    if len(dataset) == 0:
        return {"loss": 0.0, "token_accuracy": 0.0, "sequence_accuracy": 0.0}

    collator = GeneratorCollator(item_embeddings=item_embeddings, code_length=len(next(iter(mapper.item_to_code.values()))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ARSemanticIdGenerator(GeneratorConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return evaluate(model, loader, device)


def _build_ranking_metrics(
    *,
    checkpoint_path: Path,
    semantic_artifact_dir: Path,
    eval_jsonl: Path,
    top_ks: list[int],
    max_eval_samples: int | None,
    device: torch.device,
) -> dict[str, float]:
    mapper = SemanticIDMapper.load(semantic_artifact_dir)
    retriever = SemanticIdBeamSearchRetriever.from_checkpoint(
        checkpoint_path=checkpoint_path,
        semantic_artifact_dir=semantic_artifact_dir,
        mapper=mapper,
        device=device,
    )

    totals: dict[str, float] = {}
    sample_count = 0
    max_top_k = max(top_ks)

    for payload in iter_eval_jsonl(eval_jsonl, max_eval_samples=max_eval_samples):
        results = retriever.recommend(payload["history"], top_k=max_top_k)
        ranked_items = [candidate.news_id for candidate in results]
        target_item = payload["target_news_id"]

        for k in top_ks:
            totals.setdefault(f"hit_rate@{k}", 0.0)
            totals.setdefault(f"mrr@{k}", 0.0)
            totals.setdefault(f"ndcg@{k}", 0.0)
            totals[f"hit_rate@{k}"] += hit_rate_at_k(target_item, ranked_items, top_k=k)
            totals[f"mrr@{k}"] += mean_reciprocal_rank_at_k(target_item, ranked_items, top_k=k)
            totals[f"ndcg@{k}"] += ndcg_at_k(target_item, ranked_items, top_k=k)

        sample_count += 1

    if sample_count == 0:
        return {f"{metric}@{k}": 0.0 for metric in ("hit_rate", "mrr", "ndcg") for k in top_ks}

    return {key: value / sample_count for key, value in sorted(totals.items())}


def iter_eval_jsonl(path: Path, *, max_eval_samples: int | None) -> list[dict[str, object]]:
    """Load normalized eval samples with an optional limit."""

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_eval_samples is not None and len(records) >= max_eval_samples:
                break
    return records


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Evaluate the MIND AR semantic-code generator.")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--semantic-artifact-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-ks", default="5,10")
    parser.add_argument("--eval-mode", default="full_corpus_retrieval")
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def evaluate_generator_model(
    *,
    eval_jsonl: str | Path,
    semantic_artifact_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    top_ks: str | list[int] = "5,10",
    eval_mode: str = "full_corpus_retrieval",
    max_history_length: int = 50,
    batch_size: int = 64,
    max_eval_samples: int | None = None,
    device: str = "auto",
) -> dict[str, object]:
    """Run offline evaluation for a trained generator checkpoint."""

    parsed_top_ks = _parse_top_ks(top_ks) if isinstance(top_ks, str) else sorted(set(top_ks))
    if eval_mode != "full_corpus_retrieval":
        raise NotImplementedError(
            "Only eval_mode='full_corpus_retrieval' is implemented. "
            "This matches the current GR4AD-style open retrieval pipeline."
        )
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    code_metrics = _build_code_metrics(
        checkpoint_path=Path(checkpoint_path),
        semantic_artifact_dir=Path(semantic_artifact_dir),
        eval_jsonl=Path(eval_jsonl),
        max_history_length=max_history_length,
        batch_size=batch_size,
        max_eval_samples=max_eval_samples,
        device=device,
    )
    ranking_metrics = _build_ranking_metrics(
        checkpoint_path=Path(checkpoint_path),
        semantic_artifact_dir=Path(semantic_artifact_dir),
        eval_jsonl=Path(eval_jsonl),
        top_ks=parsed_top_ks,
        max_eval_samples=max_eval_samples,
        device=device,
    )
    summary = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "semantic_artifact_dir": str(Path(semantic_artifact_dir).resolve()),
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "eval_mode": eval_mode,
        "device": str(device),
        "code_metrics": code_metrics,
        "ranking_metrics": ranking_metrics,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Entry point."""

    args = build_parser().parse_args()
    summary = evaluate_generator_model(
        eval_jsonl=args.eval_jsonl,
        semantic_artifact_dir=args.semantic_artifact_dir,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        top_ks=args.top_ks,
        eval_mode=args.eval_mode,
        max_history_length=args.max_history_length,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
