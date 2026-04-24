"""Run the full MIND generator pipeline from one config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mind_genrec.data.preprocess_mind import preprocess_dataset
from mind_genrec.evaluation.compare_models import compare_evaluation_summaries
from mind_genrec.evaluation.eval_baseline import evaluate_baseline_model
from mind_genrec.evaluation.eval_generator import evaluate_generator_model
from mind_genrec.tracking.mlflow_logger import MlflowRunLogger
from mind_genrec.training.train_baseline import train_baseline_model
from mind_genrec.training.train_generator import train_generator_model
from mind_genrec.training.train_quantizer import (
    ItemEncoderConfig,
    ResidualQuantizerConfig,
    export_quantizer_artifacts,
    load_news_items,
    train_quantizer,
)


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def run_pipeline(
    *,
    config_path: str | Path,
    train_dir: str | Path,
    valid_dir: str | Path,
    work_dir: str | Path,
    max_train_samples: int | None = None,
    max_valid_samples: int | None = None,
    max_test_samples: int | None = None,
    device: str = "auto",
) -> dict[str, object]:
    """Run preprocess -> semantic ID -> generator/baseline train -> eval."""

    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    mlflow_logger = MlflowRunLogger.from_config(
        tracking_config=config.get("tracking"),
        default_experiment_name=str(config.get("project_name", "mind_genrec")),
        default_run_name=str(config.get("dataset_name", "run")),
        tags={"dataset": str(config.get("dataset_name", "")), "device": device},
    )

    normalized_dir = work_dir / "normalized"
    semantic_dir = work_dir / "semantic_ids"
    generator_dir = work_dir / "generator"
    baseline_dir = work_dir / "baseline"
    evaluation_path = work_dir / "evaluation_summary.json"
    baseline_evaluation_path = work_dir / "baseline_evaluation_summary.json"
    comparison_summary_path = work_dir / "comparison_summary.json"
    comparison_markdown_path = work_dir / "comparison_report.md"

    preprocess_summary = preprocess_dataset(
        train_dir=Path(train_dir),
        valid_dir=Path(valid_dir),
        output_dir=normalized_dir,
        max_history_length=int(config["data"]["max_history_length"]),
        validation_ratio=float(config["evaluation"].get("validation_ratio", 0.5)),
    )

    news_items = load_news_items(normalized_dir / "news.jsonl")
    encoder_config = ItemEncoderConfig(
        embedding_dim=int(config["semantic_id"]["embedding_dim"]),
        sbert_model_name=str(config["semantic_id"].get("sbert_model_name", "BAAI/bge-small-en-v1.5")),
        sbert_batch_size=int(config["semantic_id"].get("sbert_batch_size", 256)),
    )
    quantizer_config = ResidualQuantizerConfig(
        num_codebooks=int(config["semantic_id"]["code_length"]),
        codebook_size=int(config["semantic_id"]["codebook_size"]),
        max_iterations=int(config["semantic_id"]["max_iterations"]),
        sample_size=int(config["semantic_id"]["sample_size"]),
        batch_size=int(config["semantic_id"]["batch_size"]),
        seed=int(config["semantic_id"].get("seed", 7)),
    )
    embeddings, quantizer, mapper = train_quantizer(
        news_items,
        encoder_type=config["semantic_id"]["encoder_type"],
        encoder_config=encoder_config,
        quantizer_config=quantizer_config,
    )
    semantic_summary = export_quantizer_artifacts(
        items=news_items,
        embeddings=embeddings,
        quantizer=quantizer,
        mapper=mapper,
        encoder_type=config["semantic_id"]["encoder_type"],
        encoder_config=encoder_config,
        quantizer_config=quantizer_config,
        output_dir=semantic_dir,
    )

    with mlflow_logger:
        mlflow_logger.log_params(config.get("data", {}), prefix="data")
        mlflow_logger.log_params(config.get("semantic_id", {}), prefix="semantic_id")
        mlflow_logger.log_params(config.get("model", {}), prefix="model")
        mlflow_logger.log_params(config.get("training", {}), prefix="training")
        mlflow_logger.log_params(config.get("baseline", {}), prefix="baseline_config")
        mlflow_logger.log_metrics(
            {"news_count": preprocess_summary["news_count"]},
            prefix="data",
        )

        generator_metadata = train_generator_model(
            train_jsonl=normalized_dir / "train.jsonl",
            valid_jsonl=normalized_dir / "validation.jsonl",
            semantic_artifact_dir=semantic_dir,
            output_dir=generator_dir,
            max_history_length=int(config["data"]["max_history_length"]),
            decoder_type=str(config["model"].get("decoder_type", "ar")),
            hidden_dim=int(config["model"]["hidden_dim"]),
            num_heads=int(config["model"]["num_heads"]),
            num_layers=int(config["model"]["num_layers"]),
            dropout=float(config["model"]["dropout"]),
            lazy_parallel_layers=config["model"].get("lazy_parallel_layers"),
            batch_size=int(config["training"]["batch_size"]),
            learning_rate=float(config["training"]["learning_rate"]),
            epochs=int(config["training"]["epochs"]),
            warmup_steps=int(config["training"].get("warmup_steps", 500)),
            max_train_samples=max_train_samples,
            max_valid_samples=max_valid_samples,
            device=device,
            mlflow_logger=mlflow_logger,
        )

        baseline_metadata = train_baseline_model(
            train_jsonl=normalized_dir / "train.jsonl",
            valid_jsonl=normalized_dir / "validation.jsonl",
            semantic_artifact_dir=semantic_dir,
            output_dir=baseline_dir,
            max_history_length=int(config["data"]["max_history_length"]),
            hidden_dim=int(config["baseline"]["hidden_dim"]),
            output_dim=int(config["baseline"]["output_dim"]),
            num_heads=int(config["baseline"]["num_heads"]),
            num_layers=int(config["baseline"]["num_layers"]),
            dropout=float(config["baseline"]["dropout"]),
            temperature=float(config["baseline"]["temperature"]),
            batch_size=int(config["baseline"]["batch_size"]),
            learning_rate=float(config["baseline"]["learning_rate"]),
            epochs=int(config["baseline"]["epochs"]),
            warmup_steps=int(config["baseline"].get("warmup_steps", 500)),
            max_train_samples=max_train_samples,
            max_valid_samples=max_valid_samples,
            device=device,
            mlflow_logger=mlflow_logger,
        )

        best_checkpoint = generator_dir / "best_generator.pt"
        if not best_checkpoint.exists():
            best_checkpoint = generator_dir / "last_generator.pt"

        evaluation_summary = evaluate_generator_model(
            eval_jsonl=normalized_dir / "test.jsonl",
            semantic_artifact_dir=semantic_dir,
            checkpoint_path=best_checkpoint,
            output_path=evaluation_path,
            top_ks=config["evaluation"]["top_ks"],
            max_history_length=int(config["data"]["max_history_length"]),
            batch_size=int(config["evaluation"]["batch_size"]),
            max_eval_samples=max_test_samples,
            eval_mode=str(config["evaluation"].get("mode", "full_corpus_retrieval")),
            device=device,
        )
        mlflow_logger.log_metrics(
            dict(evaluation_summary.get("ranking_metrics", {})),
            prefix="generator.eval",
        )
        mlflow_logger.log_metrics(
            dict(evaluation_summary.get("code_metrics", {})),
            prefix="generator.code",
        )

        best_baseline_checkpoint = baseline_dir / "best_baseline.pt"
        if not best_baseline_checkpoint.exists():
            best_baseline_checkpoint = baseline_dir / "last_baseline.pt"
        baseline_evaluation_summary = evaluate_baseline_model(
            eval_jsonl=normalized_dir / "test.jsonl",
            semantic_artifact_dir=semantic_dir,
            checkpoint_path=best_baseline_checkpoint,
            output_path=baseline_evaluation_path,
            top_ks=config["evaluation"]["top_ks"],
            max_history_length=int(config["data"]["max_history_length"]),
            batch_size=int(config["evaluation"]["batch_size"]),
            max_eval_samples=max_test_samples,
            eval_mode=str(config["evaluation"].get("mode", "full_corpus_retrieval")),
            device=device,
        )
        mlflow_logger.log_metrics(
            dict(baseline_evaluation_summary.get("ranking_metrics", {})),
            prefix="baseline.eval",
        )

        comparison_summary = compare_evaluation_summaries(
            generator_summary_path=evaluation_path,
            baseline_summary_path=baseline_evaluation_path,
            output_path=comparison_summary_path,
            markdown_path=comparison_markdown_path,
        )
        mlflow_logger.log_artifact(comparison_markdown_path)

    summary = {
        "config_path": str(config_path.resolve()),
        "work_dir": str(work_dir.resolve()),
        "normalized_dir": str(normalized_dir.resolve()),
        "semantic_dir": str(semantic_dir.resolve()),
        "generator_dir": str(generator_dir.resolve()),
        "baseline_dir": str(baseline_dir.resolve()),
        "evaluation_path": str(evaluation_path.resolve()),
        "baseline_evaluation_path": str(baseline_evaluation_path.resolve()),
        "comparison_summary_path": str(comparison_summary_path.resolve()),
        "comparison_markdown_path": str(comparison_markdown_path.resolve()),
        "preprocess_summary": preprocess_summary,
        "semantic_summary": semantic_summary,
        "generator_metadata": generator_metadata,
        "baseline_metadata": baseline_metadata,
        "evaluation_summary": evaluation_summary,
        "baseline_evaluation_summary": baseline_evaluation_summary,
        "comparison_summary": comparison_summary,
    }
    (work_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Run the full MIND generator pipeline.")
    parser.add_argument("--config", required=True, help="Path to pipeline config YAML.")
    parser.add_argument("--train-dir", required=True, help="Raw MIND train directory containing news.tsv and behaviors.tsv.")
    parser.add_argument("--valid-dir", required=True, help="Raw MIND valid directory containing news.tsv and behaviors.tsv.")
    parser.add_argument("--work-dir", required=True, help="Output directory for all pipeline artifacts.")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main() -> None:
    """Entry point."""

    args = build_parser().parse_args()
    summary = run_pipeline(
        config_path=args.config,
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        work_dir=args.work_dir,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        max_test_samples=args.max_test_samples,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
