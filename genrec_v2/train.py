"""Single-experiment training loop with MLflow logging."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from genrec_v2.config import GenRecV2Config
from genrec_v2.data.dataset import GenRecV2Dataset, make_collator
from genrec_v2.eval import evaluate


def train_experiment(
    config: GenRecV2Config,
    train_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
    model: nn.Module,
    item_to_index: dict[str, int],
    code_for_item: dict[str, tuple[int, ...]],
    item_embeddings: np.ndarray,
    item_ids: list[str],
    codebooks: list[nn.Module] | None = None,
) -> dict:
    """Run one experiment. Returns metrics dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    collator = make_collator(item_embeddings)
    train_ds = GenRecV2Dataset(train_samples, item_to_index, code_for_item, item_embeddings, config.max_history_len)
    val_ds = GenRecV2Dataset(val_samples, item_to_index, code_for_item, item_embeddings, config.max_history_len)
    test_ds = GenRecV2Dataset(test_samples, item_to_index, code_for_item, item_embeddings, config.max_history_len)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collator)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Separate LR: codebook params get lower LR to prevent drift
    cb_param_ids = set()
    if hasattr(model, "codebook") and model.codebook is not None:
        for cb in model.codebook:
            for p in cb.parameters():
                cb_param_ids.add(id(p))
    gen_params = [p for p in model.parameters() if id(p) not in cb_param_ids]
    cb_params = [p for p in model.parameters() if id(p) in cb_param_ids]

    optimizer = torch.optim.AdamW([
        {"params": gen_params, "lr": config.lr},
        {"params": cb_params, "lr": config.lr * config.codebook_lr_ratio},
    ])

    def warmup_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        return 1.0

    scheduler = LambdaLR(optimizer, warmup_lambda)

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(config.experiment_name)
        mlflow.start_run(run_name=Path(config.output_dir).name)
        mlflow.log_params(asdict(config))
        _mlflow = mlflow
    except Exception:
        _mlflow = None

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        # Unfreeze codebooks after freeze period
        if epoch == config.freeze_codebook_epochs + 1 and codebooks:
            for cb in codebooks:
                for p in cb.parameters():
                    p.requires_grad = True

        model.train()
        total_loss = 0.0
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            optimizer.zero_grad(set_to_none=True)

            out = model(
                batch["history_emb"], batch["history_mask"],
                batch["target_code"], batch["target_emb_idx"],
            )
            loss = out["loss_gen"] + config.lambda_code * out["loss_code"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))

        epoch_summary: dict = {"epoch": epoch, "train_loss": train_loss}

        if epoch % config.eval_every == 0:
            val_metrics = evaluate(model, val_loader, device, item_ids, code_for_item)
            epoch_summary.update({f"val_{k}": v for k, v in val_metrics.items()})

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                torch.save(model.state_dict(), output / "best_model.pt")
            else:
                patience_counter += 1

        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False, indent=None))

        if _mlflow:
            _mlflow.log_metrics(epoch_summary, step=epoch)

        if patience_counter >= config.patience:
            print(f"Early stop at epoch {epoch}")
            break

    # Final test
    model.load_state_dict(torch.load(output / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device, item_ids, code_for_item)
    print(f"Test: {json.dumps(test_metrics, ensure_ascii=False)}")

    if _mlflow:
        _mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        _mlflow.log_dict({"config": asdict(config), "history": history}, "summary.json")
        _mlflow.end_run()

    (output / "metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    (output / "config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2))

    return test_metrics
