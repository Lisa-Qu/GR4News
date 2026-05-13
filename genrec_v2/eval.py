"""Generative retrieval evaluation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    item_ids: list[str],
    code_for_item: dict[str, tuple[int, ...]],
) -> dict[str, float]:
    model.eval()

    # Build code → items reverse index
    code_to_items: dict[tuple[int, ...], list[str]] = {}
    for nid, code_vals in code_for_item.items():
        c = tuple(code_vals)
        code_to_items.setdefault(c, []).append(nid)

    total_loss = 0.0
    n_batches = 0
    correct_tokens = 0
    total_tokens = 0
    correct_code = 0
    total_targets = 0
    recall_at1 = 0
    recall_code_exact = 0

    for batch in loader:
        device_batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            device_batch["history_emb"],
            device_batch["history_mask"],
            device_batch["target_code"],
            device_batch["target_emb_idx"],
        )
        total_loss += (out["loss_gen"] + out.get("loss_code", 0)).item()
        n_batches += 1

        logits = out["logits"]  # [B, code_len, vocab]
        preds = logits.argmax(dim=-1)  # [B, code_len]
        target_code = device_batch["target_code"]

        correct_tokens += (preds == target_code).sum().item()
        total_tokens += target_code.numel()

        # Per-sample recall
        B = target_code.shape[0]
        for b in range(B):
            total_targets += 1
            tgt_code = tuple(int(x) for x in target_code[b])
            pred_code = tuple(int(x) for x in preds[b])

            if pred_code == tgt_code:
                correct_code += 1
                recall_code_exact += 1

            # recall@1: check if target item is behind the predicted code
            tgt_nid = item_ids[int(device_batch["target_emb_idx"][b])]
            candidates = code_to_items.get(pred_code, [])
            if tgt_nid in candidates:
                recall_at1 += 1

    def div(a, b):
        return a / max(1, b)

    return {
        "loss": div(total_loss, n_batches),
        "token_acc": div(correct_tokens, total_tokens),
        "code_exact_acc": div(correct_code, total_targets),
        "recall_at1": div(recall_at1, total_targets),
        "recall_code_exact": div(recall_code_exact, total_targets),
    }
