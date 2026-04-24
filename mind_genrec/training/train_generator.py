"""Train the first-stage AR generator on MIND semantic IDs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import math

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mind_genrec.model import ARSemanticIdGenerator, GeneratorConfig, SemanticIDMapper
from mind_genrec.tracking.mlflow_logger import MlflowRunLogger
from mind_genrec.training.generator_data import (
    GeneratorBatch,
    GeneratorCollator,
    GeneratorDataset,
    build_item_index,
    move_batch,
    resolve_item_ids,
)


@torch.no_grad()
def evaluate(
    model: ARSemanticIdGenerator,
    loader: DataLoader[GeneratorBatch],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate token and full-sequence accuracy."""

    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0
    total_sequences = 0

    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch.history_embeddings, batch.history_mask, batch.target_codes)
        loss = model.compute_loss(logits, batch.target_codes)
        predictions = logits.argmax(dim=-1)

        total_loss += float(loss.item())
        total_batches += 1
        correct_tokens += int((predictions == batch.target_codes).sum().item())
        total_tokens += int(batch.target_codes.numel())
        correct_sequences += int((predictions == batch.target_codes).all(dim=1).sum().item())
        total_sequences += int(batch.target_codes.shape[0])

    if total_batches == 0:
        return {
            "loss": 0.0,
            "token_accuracy": 0.0,
            "sequence_accuracy": 0.0,
        }

    return {
        "loss": total_loss / total_batches,
        "token_accuracy": correct_tokens / max(1, total_tokens),
        "sequence_accuracy": correct_sequences / max(1, total_sequences),
    }


def train_one_epoch(
    model: ARSemanticIdGenerator,
    loader: DataLoader[GeneratorBatch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    """Run one training epoch."""

    model.train()
    total_loss = 0.0
    total_batches = 0
    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0
    total_sequences = 0

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.history_embeddings, batch.history_mask, batch.target_codes)
        loss = model.compute_loss(logits, batch.target_codes)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        predictions = logits.argmax(dim=-1)
        total_loss += float(loss.item())
        total_batches += 1
        correct_tokens += int((predictions == batch.target_codes).sum().item())
        total_tokens += int(batch.target_codes.numel())
        correct_sequences += int((predictions == batch.target_codes).all(dim=1).sum().item())
        total_sequences += int(batch.target_codes.shape[0])

    return {
        "loss": total_loss / max(1, total_batches),
        "token_accuracy": correct_tokens / max(1, total_tokens),
        "sequence_accuracy": correct_sequences / max(1, total_sequences),
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Train the MIND AR semantic-code generator.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--valid-jsonl")
    parser.add_argument("--semantic-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--decoder-type", default="ar")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lazy-parallel-layers", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def train_generator_model(
    *,
    train_jsonl: str | Path,
    semantic_artifact_dir: str | Path,
    output_dir: str | Path,
    valid_jsonl: str | Path | None = None,
    max_history_length: int = 50,
    decoder_type: str = "ar",
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    dropout: float = 0.1,
    lazy_parallel_layers: int | None = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    epochs: int = 3,
    warmup_steps: int = 500,
    max_train_samples: int | None = None,
    max_valid_samples: int | None = None,
    device: str = "auto",
    mlflow_logger: MlflowRunLogger | None = None,
) -> dict[str, object]:
    """Train the AR generator and save checkpoints/metadata."""

    artifact_dir = Path(semantic_artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapper = SemanticIDMapper.load(artifact_dir)
    item_embeddings = np.load(artifact_dir / "item_embeddings.npy")
    quantizer_metadata = json.loads(
        (artifact_dir / "quantizer_metadata.json").read_text(encoding="utf-8")
    )
    item_ids = resolve_item_ids(artifact_dir, mapper)
    if item_embeddings.shape[0] != len(item_ids):
        raise ValueError("item_embeddings row count does not match item_ids length")
    item_to_index = build_item_index(item_ids)

    code_length = len(next(iter(mapper.item_to_code.values())))
    train_dataset = GeneratorDataset(
        sample_path=train_jsonl,
        item_to_index=item_to_index,
        mapper=mapper,
        max_history_length=max_history_length,
        max_samples=max_train_samples,
    )
    if len(train_dataset) == 0:
        raise ValueError("GeneratorDataset is empty after filtering; cannot start training")
    valid_dataset = None
    if valid_jsonl:
        valid_dataset = GeneratorDataset(
            sample_path=valid_jsonl,
            item_to_index=item_to_index,
            mapper=mapper,
            max_history_length=max_history_length,
            max_samples=max_valid_samples,
        )
        if len(valid_dataset) == 0:
            valid_dataset = None

    collator = GeneratorCollator(item_embeddings=item_embeddings, code_length=code_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model_config = GeneratorConfig(
        input_embedding_dim=int(item_embeddings.shape[1]),
        decoder_type=decoder_type,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        code_length=code_length,
        codebook_size=int(quantizer_metadata["codebook_size"]),
        max_history_length=max_history_length,
        lazy_parallel_layers=lazy_parallel_layers,
    )
    model = ARSemanticIdGenerator(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader) * epochs

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    logger = mlflow_logger or MlflowRunLogger()
    logger.log_params(asdict(model_config), prefix="generator")
    logger.log_params(
        {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "warmup_steps": warmup_steps,
            "scheduler": "cosine_with_warmup",
        },
        prefix="generator.training",
    )

    history: list[dict[str, object]] = []
    best_valid = None
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scheduler=scheduler)
        epoch_summary: dict[str, object] = {
            "epoch": epoch,
            "train": train_metrics,
        }
        logger.log_metrics(train_metrics, prefix="generator.train", step=epoch)
        if valid_loader is not None:
            valid_metrics = evaluate(model, valid_loader, device)
            epoch_summary["valid"] = valid_metrics
            logger.log_metrics(valid_metrics, prefix="generator.valid", step=epoch)
            if best_valid is None or valid_metrics["loss"] < best_valid:
                best_valid = valid_metrics["loss"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": asdict(model_config),
                    },
                    output_dir / "best_generator.pt",
                )
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
        },
        output_dir / "last_generator.pt",
    )
    metadata = {
        "train_sample_count": len(train_dataset),
        "valid_sample_count": len(valid_dataset) if valid_dataset is not None else 0,
        "device": str(device),
        "model_config": asdict(model_config),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "scheduler": "cosine_with_warmup",
        "history": history,
        "semantic_artifact_dir": str(artifact_dir.resolve()),
    }
    (output_dir / "generator_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    """Entry point."""

    args = build_parser().parse_args()
    metadata = train_generator_model(
        train_jsonl=args.train_jsonl,
        valid_jsonl=args.valid_jsonl,
        semantic_artifact_dir=args.semantic_artifact_dir,
        output_dir=args.output_dir,
        max_history_length=args.max_history_length,
        decoder_type=args.decoder_type,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lazy_parallel_layers=args.lazy_parallel_layers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        device=args.device,
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
