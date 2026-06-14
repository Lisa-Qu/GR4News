# baselines/data_mind.py
"""MIND data for NRMS, mirroring genrec_v2.run_scorer_complete.prepare_data's user 70/15/15
seed42 split (same users/targets as the generative setting), joined with news titles.

news.tsv columns (tab): news_id, category, subcategory, title, abstract, url, t_ent, a_ent.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import re
import numpy as np

from genrec_v2.data.build_samples import build_samples  # reuse the scorer's sample builder

BASE_DIR = Path("/data/lishazhai/workspace/GR4AD")
BEHAVIORS = BASE_DIR / "data/mind_small_raw/train/MINDsmall_train/behaviors.tsv"
NEWS_TSV = BASE_DIR / "data/mind_small_raw/train/MINDsmall_train/news.tsv"
SEMANTIC_DIR = BASE_DIR / "output/sbert_baseline_20260508_153306/semantic_ids"

@dataclass(frozen=True)
class MindData:
    train_samples: list
    val_samples: list
    test_samples: list
    news2idx: dict
    title_tokens: np.ndarray   # (n_news, max_title_len) int word ids; 0 = PAD
    vocab_size: int

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def load_mind_for_nrms(max_history: int = 50, max_title_len: int = 30) -> MindData:
    item_ids = json.loads((SEMANTIC_DIR / "item_ids.json").read_text())
    mapper = json.loads((SEMANTIC_DIR / "item_to_code.json").read_text())
    cfi = {k: tuple(v) for k, v in mapper.items()}
    iti = {nid: i for i, nid in enumerate(item_ids)}
    all_samples = build_samples(str(BEHAVIORS), mode="B")
    vs = [s for s in all_samples if s["target"] in cfi
          and any(h in iti for h in s["history"][:max_history])]
    # SAME split as prepare_data: group by user, seed42 shuffle, 70/15/15.
    groups: dict[str, list] = {}
    for s in vs:
        groups.setdefault(s["user_id"], []).append(s)
    uids = sorted(groups)
    np.random.default_rng(42).shuffle(uids)
    n = len(uids); tn, vn = int(n * 0.7), int(n * 0.15)
    val_uids, test_uids = set(uids[tn:tn + vn]), set(uids[tn + vn:])
    train = [s for u in uids[:tn] for s in groups[u]]
    val = [s for u in val_uids for s in groups[u]]
    test = [s for u in test_uids for s in groups[u]]
    # Catalog = all news that ever appear as a target (the retrieval candidate set).
    catalog = sorted({s["target"] for s in vs})
    news2idx = {nid: i for i, nid in enumerate(catalog)}
    # Titles → word-id tensor (vocab built on catalog titles only).
    titles = {}
    with open(NEWS_TSV, encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) > 3:
                titles[c[0]] = c[3]
    word2idx: dict[str, int] = {}
    tok = np.zeros((len(catalog), max_title_len), dtype=np.int64)
    for nid, idx in news2idx.items():
        for j, w in enumerate(_tokenize(titles.get(nid, ""))[:max_title_len]):
            tok[idx, j] = word2idx.setdefault(w, len(word2idx) + 1)  # 0 = PAD
    return MindData(train, val, test, news2idx, tok, len(word2idx) + 1)
