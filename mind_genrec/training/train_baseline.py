"""Train the first two-tower baseline on MIND next-click samples."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mind_genrec.baseline import TwoTowerConfig, TwoTowerModel
from mind_genrec.model import SemanticIDMapper
from mind_genrec.tracking.mlflow_logger import MlflowRunLogger
from mind_genrec.training.baseline_data import (
    TwoTowerBatch,
    TwoTowerCollator,
    TwoTowerDataset,
    move_batch,
)
from mind_genrec.training.generator_data import build_item_index, resolve_item_ids


@torch.no_grad()
def evaluate(
    model: TwoTowerModel,
    loader: DataLoader[TwoTowerBatch],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate in-batch loss and accuracy."""

    model.eval()
    total_loss = 0.0
    total_batches = 0
    correct_top1 = 0
    total_rows = 0

    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(
            batch.history_embeddings,
            batch.history_mask,
            batch.target_item_embeddings,
        )
        loss = model.compute_loss(logits)
        predictions = logits.argmax(dim=1)
        labels = torch.arange(logits.shape[0], device=logits.device)

        total_loss += float(loss.item())
        total_batches += 1
        correct_top1 += int((predictions == labels).sum().item())
        total_rows += int(logits.shape[0])

    if total_batches == 0:
        return {"loss": 0.0, "in_batch_accuracy": 0.0}
    return {
        "loss": total_loss / total_batches,
        "in_batch_accuracy": correct_top1 / max(1, total_rows),
    }


def train_one_epoch(
    model: TwoTowerModel,
    loader: DataLoader[TwoTowerBatch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Run one baseline training epoch."""

    model.train()
    total_loss = 0.0
    total_batches = 0
    correct_top1 = 0
    total_rows = 0

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch.history_embeddings,
            batch.history_mask,
            batch.target_item_embeddings,
        )
        loss = model.compute_loss(logits)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        labels = torch.arange(logits.shape[0], device=logits.device)

        total_loss += float(loss.item())
        total_batches += 1
        correct_top1 += int((predictions == labels).sum().item())
        total_rows += int(logits.shape[0])

    return {
        "loss": total_loss / max(1, total_batches),
        "in_batch_accuracy": correct_top1 / max(1, total_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Train the MIND two-tower baseline.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--valid-jsonl")
    parser.add_argument("--semantic-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-history-length", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def train_baseline_model(
    *,
    train_jsonl: str | Path,
    semantic_artifact_dir: str | Path,
    output_dir: str | Path,
    valid_jsonl: str | Path | None = None,
    max_history_length: int = 50,
    hidden_dim: int = 256,
    output_dim: int = 128,
    num_heads: int = 8,
    num_layers: int = 2,
    dropout: float = 0.1,
    temperature: float = 0.07,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    epochs: int = 3,
    max_train_samples: int | None = None,
    max_valid_samples: int | None = None,
    device: str = "auto",
    mlflow_logger: MlflowRunLogger | None = None,
) -> dict[str, object]:
    """Train the two-tower baseline and save checkpoints/metadata."""

    artifact_dir = Path(semantic_artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapper = SemanticIDMapper.load(artifact_dir)
    item_embeddings = np.load(artifact_dir / "item_embeddings.npy")
    item_ids = resolve_item_ids(artifact_dir, mapper)
    if item_embeddings.shape[0] != len(item_ids):
        raise ValueError("item_embeddings row count does not match item_ids length")
    item_to_index = build_item_index(item_ids)

    train_dataset = TwoTowerDataset(
        sample_path=train_jsonl,
        item_to_index=item_to_index,
        max_history_length=max_history_length,
        max_samples=max_train_samples,
    )
    if len(train_dataset) == 0:
        raise ValueError("TwoTowerDataset is empty after filtering; cannot start baseline training")

    valid_dataset = None
    if valid_jsonl:
        valid_dataset = TwoTowerDataset(
            sample_path=valid_jsonl,
            item_to_index=item_to_index,
            max_history_length=max_history_length,
            max_samples=max_valid_samples,
        )
        if len(valid_dataset) == 0:
            valid_dataset = None

    collator = TwoTowerCollator(item_embeddings=item_embeddings)
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

    model_config = TwoTowerConfig(
        input_embedding_dim=int(item_embeddings.shape[1]),
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_history_length=max_history_length,
        temperature=temperature,
    )
    model = TwoTowerModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    logger = mlflow_logger or MlflowRunLogger()
    logger.log_params(asdict(model_config), prefix="baseline")
    logger.log_params(
        {"batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs},
        prefix="baseline.training",
    )

    history: list[dict[str, object]] = []
    best_valid = None
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        epoch_summary: dict[str, object] = {
            "epoch": epoch,
            "train": train_metrics,
        }
        logger.log_metrics(train_metrics, prefix="baseline.train", step=epoch)
        if valid_loader is not None:
            valid_metrics = evaluate(model, valid_loader, device)
            epoch_summary["valid"] = valid_metrics
            logger.log_metrics(valid_metrics, prefix="baseline.valid", step=epoch)
            if best_valid is None or valid_metrics["loss"] < best_valid:
                best_valid = valid_metrics["loss"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": asdict(model_config),
                    },
                    output_dir / "best_baseline.pt",
                )
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
        },
        output_dir / "last_baseline.pt",
    )
    metadata = {
        "train_sample_count": len(train_dataset),
        "valid_sample_count": len(valid_dataset) if valid_dataset is not None else 0,
        "device": str(device),
        "model_config": asdict(model_config),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "history": history,
        "semantic_artifact_dir": str(artifact_dir.resolve()),
    }
    (output_dir / "baseline_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    """Entry point."""

    args = build_parser().parse_args()
    metadata = train_baseline_model(
        train_jsonl=args.train_jsonl,
        valid_jsonl=args.valid_jsonl,
        semantic_artifact_dir=args.semantic_artifact_dir,
        output_dir=args.output_dir,
        max_history_length=args.max_history_length,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        temperature=args.temperature,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        device=args.device,
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
