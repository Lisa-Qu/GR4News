# baselines/data_amazon.py
"""Amazon GRAM leave-one-out sequences for SASRec. Reads the SAME per-user ASIN sequences the GRAM
generator used (rec_datasets/{ds}/user_sequence.txt). Item vocab from item_plain_text.txt (ASIN→int,
0=PAD). LOO: test target = last item, val target = 2nd-last; training = the full train sub-sequence
items[:-2] supervised at EVERY position (all-position SASRec). Full catalog = all vocab items.

Confirmed format (server spike 2026-06-14):
  user_sequence.txt   line = "<user_id> <ASIN> <ASIN> ..." (ASIN strings, space-sep)
  item_plain_text.txt line = "<ASIN> title: ..."  → catalog (Beauty 12101 items, 22363 users)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

GRAM_ROOT = Path("/data/lishazhai/workspace/GRAM")

@dataclass(frozen=True)
class AmazonSeq:
    train_seqs: list      # list[list[int]] — full train sub-sequence (items[:-2]) per user,
                          # for all-position SASRec supervision (predict next item at every position)
    val: list             # list[(history_ids, target_id, user_id)]
    test: list            # list[(history_ids, target_id, user_id)]
    n_items: int          # catalog size (ids 1..n_items; 0 = PAD)
    item2id: dict         # ASIN -> int

def _build_item2id(ds_dir: Path) -> dict:
    item2id: dict[str, int] = {}
    with open(ds_dir / "item_plain_text.txt", encoding="utf-8") as f:
        for line in f:
            asin = line.split(maxsplit=1)[0]
            if asin and asin not in item2id:
                item2id[asin] = len(item2id) + 1  # 0 = PAD
    return item2id

def load_amazon_loo(dataset: str, max_history: int = 20) -> AmazonSeq:
    ds_dir = GRAM_ROOT / "rec_datasets" / dataset
    item2id = _build_item2id(ds_dir)
    def clip(h): return h[-max_history:]
    train_seqs, val, test = [], [], []
    oov = 0
    with open(ds_dir / "user_sequence.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue  # need >=2 items for LOO (history + target)
            uid = parts[0]
            oov += sum(a not in item2id for a in parts[1:])
            items = [item2id[a] for a in parts[1:] if a in item2id]
            if len(items) < 2:
                continue
            test.append((clip(items[:-1]), items[-1], uid))
            if len(items) >= 3:
                val.append((clip(items[:-2]), items[-2], uid))
            # All-position SASRec training: the train sub-sequence is items[:-2] (val+test
            # targets held out). Need >=2 items (>=1 input position + >=1 next-item target).
            train_sub = clip(items[:-2])
            if len(train_sub) >= 2:
                train_seqs.append(train_sub)
    # Tripwire (review #2): user_sequence items MUST all be in the item_plain_text catalog, else
    # dropping an OOV item could silently shift a user's LOO target away from the one the generator
    # scored, breaking the paired comparison. Verified 0 on Beauty/Toys/Sports (2026-06-14).
    assert oov == 0, (f"{oov} user_sequence items are OOV vs item_plain_text.txt for {dataset}; "
                      "LOO targets may shift — rebuild item2id from the union of both files")
    return AmazonSeq(train_seqs, val, test, len(item2id), item2id)
