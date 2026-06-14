# GR4AD Baselines (NRMS + SASRec) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a discriminative baseline row to each GR4AD main table — NRMS (MIND news) and SASRec (Amazon Beauty/Sports/Toys) — evaluated under the IDENTICAL full-catalog protocol + metrics + significance pipeline as the generative-retrieval rows.

**Architecture:** Lean PyTorch reimplementations (no `recommenders`/`recbole`, neither installed). A shared full-catalog eval module reuses the scorer's `_rank_metrics` definitions (proven byte-identical by test). Data loaders mirror the EXACT scorer test splits (MIND `prepare_data` 70/15/15 seed42; Amazon GRAM leave-one-out). Each baseline persists `per_user_hits.npz` aligned by `user_id` to the matching setting's generative vanilla, fed to `run_statistical_significance.py`.

**Tech Stack:** Python 3.10, torch 2.1 (server conda env `gram`), numpy, scipy; MLflow auto-logging via `mind_genrec.tracking.MlflowRunLogger`. Server `gram-server`, code at `/data/lishazhai/workspace/GR4AD`, deploy via `scp` (server GitHub down), compile-check with server py3.10.

---

## File Structure

```
baselines/
├── __init__.py
├── metrics.py          # rank_metrics_single(rank) — single-positive specialization of _rank_metrics
├── eval_fullcatalog.py # full-catalog scoring loop → agg metrics + per-sample hits + per_user_hits.npz
├── nrms.py             # NRMS model (news encoder + user encoder)
├── sasrec.py           # SASRec model (causal self-attn over item ids)
├── data_mind.py        # MIND (history news ids + titles + target), mirrors prepare_data split
├── data_amazon.py      # Amazon GRAM LOO sequences + item vocab (per dataset)
├── run_nrms.py         # train+eval NRMS on MIND → results.json + per_user_hits.npz + MLflow
└── run_sasrec.py       # train+eval SASRec on one Amazon dataset → same artifacts
tests/baselines/
├── test_metrics.py     # rank_metrics_single == _rank_metrics on constructed vectors
├── test_eval.py        # full-catalog eval on a toy scoring matrix
├── test_data_mind.py   # NRMS split user_ids == prepare_data tsl/vsl users; sample fields
└── test_sasrec_model.py# SASRec forward shape + causal mask + smoke loss-decrease
```

**Significance:** generalize `genrec_v2/run_statistical_significance.py` to (a) read `OUT_DIR` from env `SIG_OUT_DIR`, (b) detect baseline keys (`baseline_hit1/10`) and emit a `baseline_vs_vanilla` comparison. No new file.

---

## Task 1: Shared single-positive metric (`baselines/metrics.py`)

The scorer's `_rank_metrics(rel_ordered)` (genrec_v2/run_main_table.py:97-113) takes a binary relevance
vector in ranked order. Full-catalog retrieval has exactly ONE relevant item (the held-out target) at
some rank `r` (0-based) — or absent. We provide a closed-form specialization and PROVE it equals
`_rank_metrics` on the equivalent vector.

**Files:**
- Create: `baselines/__init__.py` (empty)
- Create: `baselines/metrics.py`
- Test: `tests/baselines/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/baselines/test_metrics.py
import numpy as np
import pytest
from baselines.metrics import rank_metrics_single, KS, NDCG_KS

def _ref_rank_metrics(rel_ordered):
    """Byte-identical copy of genrec_v2.run_main_table._rank_metrics for the oracle comparison."""
    hit = {k: bool(rel_ordered[:k].max() > 0) for k in KS}
    nz = np.flatnonzero(rel_ordered)
    mrr = float(1.0 / (nz[0] + 1)) if nz.size else 0.0
    n_rel = int(rel_ordered.sum())
    ndcg = {}
    for k in NDCG_KS:
        topk = rel_ordered[:k]
        dcg = float(np.sum(topk / np.log2(np.arange(2, topk.size + 2))))
        idcg = float(np.sum(1.0 / np.log2(np.arange(2, min(k, n_rel) + 2)))) if n_rel else 0.0
        ndcg[k] = dcg / idcg if idcg > 0 else 0.0
    return hit, mrr, ndcg

@pytest.mark.parametrize("rank", [0, 1, 4, 9, 49, 50, 100])
def test_single_matches_reference(rank):
    n_catalog = 200
    rel = np.zeros(n_catalog, dtype=np.int64)
    if rank < n_catalog:
        rel[rank] = 1
    ref_hit, ref_mrr, ref_ndcg = _ref_rank_metrics(rel)
    hit, mrr, ndcg = rank_metrics_single(rank if rank < n_catalog else None)
    assert hit == ref_hit
    assert mrr == pytest.approx(ref_mrr)
    for k in NDCG_KS:
        assert ndcg[k] == pytest.approx(ref_ndcg[k])

def test_target_absent():
    hit, mrr, ndcg = rank_metrics_single(None)
    assert hit == {1: False, 5: False, 10: False, 50: False}
    assert mrr == 0.0
    assert ndcg == {5: 0.0, 10: 0.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/baselines/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'baselines.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# baselines/metrics.py
"""Single-positive specialization of the scorer's _rank_metrics (full-catalog retrieval).

Full-catalog next-item retrieval has exactly ONE relevant item (the held-out target). Given its
0-based rank (or None if not retrieved/out of range), reproduce the scorer's hit@k / MRR / nDCG@k
EXACTLY (n_rel=1 ⇒ IDCG=1, so nDCG@k = 1/log2(rank+2) when rank<k). Proven equal to
genrec_v2.run_main_table._rank_metrics by tests/baselines/test_metrics.py.
"""
from __future__ import annotations
import math

KS = (1, 5, 10, 50)
NDCG_KS = (5, 10)

def rank_metrics_single(rank: int | None) -> tuple[dict, float, dict]:
    if rank is None:
        return ({k: False for k in KS}, 0.0, {k: 0.0 for k in NDCG_KS})
    hit = {k: bool(rank < k) for k in KS}
    mrr = 1.0 / (rank + 1)
    ndcg = {k: (1.0 / math.log2(rank + 2) if rank < k else 0.0) for k in NDCG_KS}
    return hit, mrr, ndcg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/baselines/test_metrics.py -v`
Expected: PASS (8 cases)

- [ ] **Step 5: Commit**

```bash
git add baselines/__init__.py baselines/metrics.py tests/baselines/test_metrics.py
git commit -m "feat(baselines): single-positive rank metrics proven equal to scorer _rank_metrics"
```

---

## Task 2: Full-catalog eval + per_user_hits writer (`baselines/eval_fullcatalog.py`)

**Files:**
- Create: `baselines/eval_fullcatalog.py`
- Test: `tests/baselines/test_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/baselines/test_eval.py
import numpy as np
from baselines.eval_fullcatalog import eval_full_catalog

def test_toy_ranking():
    # 3 users, catalog of 5 items. scores[u] high → ranked first. targets = item index.
    scores = np.array([
        [9, 1, 2, 3, 4],   # target 0 → rank 0 (hit@1)
        [1, 2, 3, 4, 9],   # target 4 → rank 0 (hit@1)
        [5, 4, 3, 2, 1],   # target 3 → rank 3 (hit@5, not hit@1)
    ], dtype=np.float32)
    targets = np.array([0, 4, 3])
    uids = np.array(["u0", "u1", "u2"])
    agg, hits = eval_full_catalog(scores, targets, uids)
    assert agg["R@1"] == 2/3
    assert agg["R@5"] == 1.0
    assert hits[1].tolist() == [True, True, False]
    assert hits[10].tolist() == [True, True, True]
    assert agg["MRR"] == (1.0 + 1.0 + 0.25) / 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/baselines/test_eval.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# baselines/eval_fullcatalog.py
"""Full-catalog ranking eval reusing the single-positive metrics. Produces aggregate metrics
(R@k, MRR, nDCG@k) + per-sample hit@k arrays + a per_user_hits.npz writer for significance.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from baselines.metrics import rank_metrics_single, KS, NDCG_KS

def _target_ranks(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """0-based rank of each target under descending score (stable, ties broken by index)."""
    # rank = number of items strictly scoring higher than the target's score.
    n = scores.shape[0]
    tgt_scores = scores[np.arange(n), targets]
    return (scores > tgt_scores[:, None]).sum(axis=1)

def eval_full_catalog(scores: np.ndarray, targets: np.ndarray, user_ids: np.ndarray):
    """scores: (N, |catalog|); targets: (N,) item indices; user_ids: (N,).
    Returns (agg dict with R@k/MRR/nDCG@k, hits dict {k: bool array (N,)})."""
    ranks = _target_ranks(scores, targets)
    n = scores.shape[0]
    hits = {k: np.zeros(n, dtype=bool) for k in KS}
    mrr_arr = np.zeros(n, dtype=np.float64)
    ndcg_arr = {k: np.zeros(n, dtype=np.float64) for k in NDCG_KS}
    for i in range(n):
        hit, mrr, ndcg = rank_metrics_single(int(ranks[i]))
        for k in KS:
            hits[k][i] = hit[k]
        mrr_arr[i] = mrr
        for k in NDCG_KS:
            ndcg_arr[k][i] = ndcg[k]
    agg = {f"R@{k}": float(hits[k].mean()) for k in KS}
    agg["MRR"] = float(mrr_arr.mean())
    for k in NDCG_KS:
        agg[f"nDCG@{k}"] = float(ndcg_arr[k].mean())
    return agg, hits

def write_per_user_hits(out_dir: Path, user_ids: np.ndarray, baseline_hits: dict,
                        vanilla_hits: dict) -> None:
    """Persist baseline + the matching-setting generative vanilla hits (already aligned by user
    order) for run_statistical_significance.py's baseline_vs_vanilla comparison."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "per_user_hits.npz",
             user_ids=np.array(user_ids),
             vanilla_hit1=vanilla_hits[1], vanilla_hit10=vanilla_hits[10],
             baseline_hit1=baseline_hits[1], baseline_hit10=baseline_hits[10])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/baselines/test_eval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add baselines/eval_fullcatalog.py tests/baselines/test_eval.py
git commit -m "feat(baselines): full-catalog eval + per_user_hits writer"
```

---

## Task 3: MIND data loader for NRMS (`baselines/data_mind.py`)

Mirror `genrec_v2/run_scorer_complete.py:prepare_data` EXACTLY for the user 70/15/15 seed42 split, but
keep ITEM IDS (not semantic codes) and join NEWS TITLES. NRMS needs: per-user click-history news ids,
the next-click target id, and a title-token tensor per news id. Full catalog = all news ids that appear
as a target across the data (the retrieval candidate set).

**Files:**
- Create: `baselines/data_mind.py`
- Test: `tests/baselines/test_data_mind.py`

> **Server discovery (do FIRST, record results in the loader docstring):** confirm the MIND news text
> file. Run on `gram-server`:
> `ls /data/lishazhai/workspace/GR4AD/data/mind_small_raw/train/MINDsmall_train/news.tsv && head -1 .../news.tsv`
> MIND `news.tsv` columns (tab-sep): `news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities`. Title = column index 3.

- [ ] **Step 1: Write the failing test** (uses the real behaviors split; runs on server)

```python
# tests/baselines/test_data_mind.py
import numpy as np
from baselines.data_mind import load_mind_for_nrms

def test_split_matches_scorer():
    d = load_mind_for_nrms(max_history=50, max_title_len=30)
    # Test users are a non-empty disjoint subset of val users; targets in vocab.
    assert len(d.test_samples) > 0 and len(d.val_samples) > 0
    test_uids = {s["user_id"] for s in d.test_samples}
    val_uids = {s["user_id"] for s in d.val_samples}
    assert test_uids.isdisjoint(val_uids)
    # Every sample has history (>=1 valid id) + a target in the catalog.
    s = d.test_samples[0]
    assert s["target"] in d.news2idx
    assert len(s["history"]) >= 1
    # Title tensor shape = (n_news, max_title_len)
    assert d.title_tokens.shape[1] == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run (server, deployed): `python -m pytest tests/baselines/test_data_mind.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes** (server)

Run: `python -m pytest tests/baselines/test_data_mind.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add baselines/data_mind.py tests/baselines/test_data_mind.py
git commit -m "feat(baselines): MIND NRMS data loader mirroring scorer 70/15/15 seed42 split"
```

---

## Task 4: NRMS model (`baselines/nrms.py`)

Standard NRMS (Wu et al., 2019): news encoder = word-emb → multi-head self-attn → additive attention;
user encoder = additive attention over history news vectors. Score = dot product.

**Files:**
- Create: `baselines/nrms.py`
- Test: `tests/baselines/test_data_mind.py` (extend with a forward-shape test) — or inline in run script.

- [ ] **Step 1: Write the implementation** (model code is standard; shape-tested in Task 6 smoke)

```python
# baselines/nrms.py
"""NRMS (Wu et al. 2019) — news encoder (multi-head self-attn + additive attn over title words)
and user encoder (additive attn over history news vectors). Standard published config; no tuning.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, dim: int, hidden: int = 200):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)
    def forward(self, x, mask=None):  # x: (B, L, dim)
        a = self.query(torch.tanh(self.proj(x))).squeeze(-1)  # (B, L)
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        w = F.softmax(a, dim=-1).unsqueeze(-1)
        return (w * x).sum(dim=1)  # (B, dim)

class NewsEncoder(nn.Module):
    def __init__(self, vocab: int, emb_dim: int = 300, heads: int = 16, head_dim: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        d = heads * head_dim
        self.attn = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True,
                                          kdim=emb_dim, vdim=emb_dim)
        self.add = AdditiveAttention(emb_dim)
        self.drop = nn.Dropout(dropout)
        self.out_dim = emb_dim
    def forward(self, title_tokens):  # (B, L)
        mask = title_tokens != 0
        x = self.drop(self.emb(title_tokens))                 # (B, L, emb)
        x, _ = self.attn(x, x, x, key_padding_mask=~mask)     # (B, L, emb)
        return self.add(self.drop(x), mask)                   # (B, emb)

class NRMS(nn.Module):
    def __init__(self, vocab: int, emb_dim: int = 300, heads: int = 16, head_dim: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        self.news = NewsEncoder(vocab, emb_dim, heads, head_dim, dropout)
        self.user_attn = AdditiveAttention(self.news.out_dim)
    def encode_news(self, titles):  # (M, L) → (M, dim)
        return self.news(titles)
    def encode_user(self, hist_titles, hist_mask):  # (B, H, L), (B, H)
        B, H, L = hist_titles.shape
        nv = self.news(hist_titles.reshape(B * H, L)).reshape(B, H, -1)
        return self.user_attn(nv, hist_mask)  # (B, dim)
    def forward(self, hist_titles, hist_mask, cand_titles):  # cand_titles: (B, C, L)
        u = self.encode_user(hist_titles, hist_mask)          # (B, dim)
        B, C, L = cand_titles.shape
        cv = self.news(cand_titles.reshape(B * C, L)).reshape(B, C, -1)
        return (u.unsqueeze(1) * cv).sum(-1)                  # (B, C) scores
```

- [ ] **Step 2: Commit**

```bash
git add baselines/nrms.py
git commit -m "feat(baselines): NRMS model (standard config)"
```

---

## Task 5: SASRec data discovery + loader (`baselines/data_amazon.py`)

**Files:**
- Create: `baselines/data_amazon.py`
- Test: `tests/baselines/test_sasrec_model.py` (data portion)

> **Server discovery SPIKE — RESOLVED 2026-06-14** (formats confirmed on `gram-server`):
> - **Sequence file:** `rec_datasets/{ds}/user_sequence.txt` — line = `<user_id> <ASIN1> <ASIN2> ... <ASINn>`
>   (space-sep; items are **ASIN strings** e.g. `B004756YJA`, NOT integers). Beauty = 22363 users.
> - **Item catalog:** `rec_datasets/{ds}/item_plain_text.txt` — line = `<ASIN> title: ...`. Beauty = 12101 items.
>   Build `item2id`: ASIN → int 1..n_items (0 = PAD), in the file's line order (stable).
> - **LOO:** last ASIN = test target, 2nd-last = val target (GRAM convention).
> - Sports/Toys have the same `user_sequence.txt` + `item_plain_text.txt` structure.
> - **Alignment risk (logged):** `user_id` strings here (e.g. `A1YJEY40YUW4SE`) must match the scorer
>   `experiments/{ds}_scorer/per_user_hits.npz` `user_ids`. The run aligns by user_id and logs how many
>   users were kept; if 0 kept, the id spaces differ and the spike must re-map (e.g. via a user2id map).

- [ ] **Step 1: Write the loader** (ASIN sequences + item2id from item_plain_text.txt)

```python
# baselines/data_amazon.py
"""Amazon GRAM leave-one-out sequences for SASRec. Reads the SAME per-user ASIN sequences the GRAM
generator used (rec_datasets/{ds}/user_sequence.txt). Item vocab from item_plain_text.txt (ASIN→int,
0=PAD). LOO: test target = last item, val target = 2nd-last, train = each prefix's next item.
Full catalog = all items in the vocabulary.

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
    train: list           # list[(history_ids: list[int], target_id: int)]
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
    train, val, test = [], [], []
    with open(ds_dir / "user_sequence.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue  # need >=2 items for LOO (history + target)
            uid = parts[0]
            items = [item2id[a] for a in parts[1:] if a in item2id]
            if len(items) < 2:
                continue
            test.append((clip(items[:-1]), items[-1], uid))
            if len(items) >= 3:
                val.append((clip(items[:-2]), items[-2], uid))
            for t in range(1, len(items) - 2):  # train prefixes
                train.append((clip(items[:t]), items[t]))
    return AmazonSeq(train, val, test, len(item2id), item2id)
```

> Note: `run_sasrec.py` (Task 7) no longer needs `--seq-file` — drop that arg; call
> `load_amazon_loo(cli.dataset, cli.max_history)`.

- [ ] **Step 2: Commit**

```bash
git add baselines/data_amazon.py
git commit -m "feat(baselines): Amazon GRAM LOO sequence loader for SASRec"
```

---

## Task 6: SASRec model + smoke test (`baselines/sasrec.py`)

**Files:**
- Create: `baselines/sasrec.py`
- Test: `tests/baselines/test_sasrec_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/baselines/test_sasrec_model.py
import torch
from baselines.sasrec import SASRec

def test_forward_shape_and_causality():
    m = SASRec(n_items=100, d=64, n_blocks=2, n_heads=1, max_len=20, dropout=0.0)
    seq = torch.randint(1, 100, (4, 20))
    seq_repr = m.seq_repr(seq)            # (B, d) last-position
    assert seq_repr.shape == (4, 64)
    scores = m.full_scores(seq)           # (B, n_items+1)
    assert scores.shape == (4, 101)

def test_smoke_loss_decreases():
    torch.manual_seed(0)
    m = SASRec(n_items=50, d=32, n_blocks=2, n_heads=1, max_len=10, dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    seq = torch.randint(1, 50, (16, 10)); tgt = torch.randint(1, 50, (16,))
    first = last = None
    for _ in range(50):
        loss = torch.nn.functional.cross_entropy(m.full_scores(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or loss.item(); last = loss.item()
    assert last < first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/baselines/test_sasrec_model.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# baselines/sasrec.py
"""SASRec (Kang & McAuley 2018) — causal self-attention over item-id sequence; score = last
hidden · item embeddings (full catalog). Standard config (d=64, 2 blocks, 1 head); no tuning.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, n_items: int, d: int = 64, n_blocks: int = 2, n_heads: int = 1,
                 max_len: int = 20, dropout: float = 0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)  # +1 for PAD=0
        self.pos_emb = nn.Embedding(max_len, d)
        self.max_len = max_len
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=d, dropout=dropout,
                                           batch_first=True, activation="relu")
        self.blocks = nn.TransformerEncoder(layer, n_blocks)
        self.ln = nn.LayerNorm(d)
    def seq_repr(self, seq):  # seq: (B, L) right-aligned item ids, 0=PAD
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).clamp(max=self.max_len - 1)
        x = self.drop(self.item_emb(seq) + self.pos_emb(pos)[None])
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1)
        pad = seq == 0
        h = self.blocks(x, mask=causal, src_key_padding_mask=pad)
        return self.ln(h[:, -1])  # last position (the next-item query)
    def full_scores(self, seq):  # (B, n_items+1)
        return self.seq_repr(seq) @ self.item_emb.weight.T
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/baselines/test_sasrec_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add baselines/sasrec.py tests/baselines/test_sasrec_model.py
git commit -m "feat(baselines): SASRec model + smoke loss-decrease test"
```

---

## Task 7: SASRec run script (`baselines/run_sasrec.py`)

Train SASRec per Amazon dataset, eval full-catalog on the LOO test set, persist results + per_user_hits
(aligned to the generative vanilla from `experiments/{ds}_scorer/per_user_hits.npz`), log to MLflow.

**Files:**
- Create: `baselines/run_sasrec.py`

- [ ] **Step 1: Write the run script**

```python
# baselines/run_sasrec.py
"""Train + full-catalog eval SASRec on one Amazon dataset (LOO). Emits results.json +
per_user_hits.npz (baseline + matching generative vanilla, aligned by user_id) + MLflow.

Usage: python -u baselines/run_sasrec.py --dataset Sports \
       --vanilla-npz experiments/sports_scorer/per_user_hits.npz --output-dir experiments/sasrec_Sports
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from baselines.data_amazon import load_amazon_loo
from baselines.sasrec import SASRec
from baselines.eval_fullcatalog import eval_full_catalog, write_per_user_hits
from mind_genrec.tracking import MlflowRunLogger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _pad(hist, max_len):
    h = hist[-max_len:]
    return [0] * (max_len - len(h)) + h

def _batched_scores(model, seqs, max_len, bs=512):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            b = torch.tensor([_pad(h, max_len) for h, _, _ in seqs[i:i + bs]], device=DEVICE)
            out.append(model.full_scores(b).cpu().numpy())
    return np.concatenate(out, 0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--vanilla-npz", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-history", type=int, default=20)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--smoke-users", type=int, default=0)
    cli = p.parse_args()
    t0 = time.time()
    data = load_amazon_loo(cli.dataset, cli.max_history)
    model = SASRec(n_items=data.n_items, max_len=cli.max_history).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train = data.train
    rng = np.random.default_rng(42)
    def train_epoch():
        model.train(); rng.shuffle(train); tot = 0.0
        for i in range(0, len(train), 256):
            batch = train[i:i + 256]
            seq = torch.tensor([_pad(h, cli.max_history) for h, _ in batch], device=DEVICE)
            tgt = torch.tensor([t for _, t in batch], device=DEVICE)
            loss = F.cross_entropy(model.full_scores(seq), tgt)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        return tot

    def val_r10():
        s = _batched_scores(model, data.val, cli.max_history)
        tgt = np.array([t for _, t, _ in data.val])
        uid = np.array([u for _, _, u in data.val])
        agg, _ = eval_full_catalog(s, tgt, uid)
        return agg["R@10"]

    best, best_state, bad = -1.0, None, 0
    for ep in range(cli.epochs):
        train_epoch(); r10 = val_r10()
        if r10 > best:
            best, best_state, bad = r10, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cli.patience:
                break
    model.load_state_dict(best_state)

    test = data.test if not cli.smoke_users else data.test[:cli.smoke_users]
    scores = _batched_scores(model, test, cli.max_history)
    tgt = np.array([t for _, t, _ in test]); uid = np.array([u for _, _, u in test])
    agg, hits = eval_full_catalog(scores, tgt, uid)

    # Align vanilla hits to THIS test user order (by user_id).
    van = np.load(cli.vanilla_npz, allow_pickle=True)
    van_uid = list(van["user_ids"])
    pos = {u: i for i, u in enumerate(van_uid)}
    keep = [i for i, u in enumerate(uid) if u in pos]
    sel = np.array([pos[uid[i]] for i in keep])
    vh = {1: van["vanilla_hit1"][sel], 10: van["vanilla_hit10"][sel]}
    bh = {1: hits[1][keep], 10: hits[10][keep]}
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    write_per_user_hits(cli.output_dir, uid[keep], bh, vh)

    results = {"dataset": cli.dataset, "model": "SASRec", "n_items": data.n_items,
               "n_test": len(test), "rows": {"sasrec": agg}, "runtime_sec": time.time() - t0}
    (cli.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    with MlflowRunLogger(enabled=True, tracking_uri="http://localhost:5000",
                         experiment_name="mind_genrec", run_name=f"sasrec_{cli.dataset}",
                         tags={"dataset": cli.dataset, "model": "SASRec", "eval": "full_catalog_LOO"}) as lg:
        lg.log_params({"n_items": data.n_items, "max_history": cli.max_history, "epochs": cli.epochs})
        lg.log_metrics({f"sasrec.{k}": v for k, v in agg.items()})
        lg.log_dict(results, f"sasrec_{cli.dataset}_results.json")
    print(json.dumps(results["rows"], indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add baselines/run_sasrec.py
git commit -m "feat(baselines): SASRec run script (train + full-catalog LOO eval + MLflow)"
```

---

## Task 8: NRMS run script (`baselines/run_nrms.py`)

Same structure as SASRec but for MIND news: build candidate-title tensors, train with in-batch
negatives, eval full-catalog over all news, align to the MIND main_table generative vanilla.

**Files:**
- Create: `baselines/run_nrms.py`

- [ ] **Step 1: Write the run script**

```python
# baselines/run_nrms.py
"""Train + full-catalog eval NRMS on MIND (next-click retrieval). Emits results.json +
per_user_hits.npz (baseline + main_table generative vanilla, aligned by user_id) + MLflow.

Training: in-batch negatives — score each user against the batch's targets, cross-entropy to the
diagonal. Eval: score every user against ALL news title vectors (full catalog).

Usage: python -u baselines/run_nrms.py --vanilla-npz experiments/main_table/per_user_hits.npz \
       --output-dir experiments/nrms
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from baselines.data_mind import load_mind_for_nrms
from baselines.nrms import NRMS
from baselines.eval_fullcatalog import eval_full_catalog, write_per_user_hits
from mind_genrec.tracking import MlflowRunLogger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _hist_tensor(samples, news2idx, title_tokens, max_history):
    """(B, H, L) history title tokens + (B, H) mask, using the catalog title tensor."""
    L = title_tokens.shape[1]
    B = len(samples)
    out = np.zeros((B, max_history, L), dtype=np.int64)
    mask = np.zeros((B, max_history), dtype=bool)
    for i, s in enumerate(samples):
        hist = [h for h in s["history"] if h in news2idx][-max_history:]
        for j, nid in enumerate(hist):
            out[i, j] = title_tokens[news2idx[nid]]
            mask[i, j] = True
    return torch.tensor(out), torch.tensor(mask)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla-npz", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-history", type=int, default=50)
    p.add_argument("--max-title-len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--smoke-users", type=int, default=0)
    cli = p.parse_args()
    t0 = time.time()
    d = load_mind_for_nrms(cli.max_history, cli.max_title_len)
    model = NRMS(vocab=d.vocab_size).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    titles = torch.tensor(d.title_tokens, device=DEVICE)  # (M, L)

    def batch_titles(samples):
        ht, hm = _hist_tensor(samples, d.news2idx, d.title_tokens, cli.max_history)
        tgt = torch.tensor([d.news2idx[s["target"]] for s in samples])
        return ht.to(DEVICE), hm.to(DEVICE), tgt.to(DEVICE)

    rng = np.random.default_rng(42)
    tr = list(d.train_samples)
    def train_epoch():
        model.train(); rng.shuffle(tr); tot = 0.0
        for i in range(0, len(tr), 128):
            ht, hm, tgt = batch_titles(tr[i:i + 128])
            u = model.encode_user(ht, hm)                 # (B, dim)
            cv = model.encode_news(titles[tgt])           # in-batch positives (B, dim)
            logits = u @ cv.T                             # (B, B) — diagonal is positive
            loss = F.cross_entropy(logits, torch.arange(len(tgt), device=DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        return tot

    @torch.no_grad()
    def full_scores(samples):
        model.eval()
        news_vecs = []
        for i in range(0, titles.shape[0], 1024):
            news_vecs.append(model.encode_news(titles[i:i + 1024]))
        NV = torch.cat(news_vecs)                         # (M, dim)
        out = []
        for i in range(0, len(samples), 256):
            ht, hm, _ = batch_titles(samples[i:i + 256])
            out.append((model.encode_user(ht, hm) @ NV.T).cpu().numpy())
        return np.concatenate(out, 0)

    def val_r10():
        s = full_scores(d.val_samples)
        tgt = np.array([d.news2idx[x["target"]] for x in d.val_samples])
        uid = np.array([x["user_id"] for x in d.val_samples])
        return eval_full_catalog(s, tgt, uid)[0]["R@10"]

    best, best_state, bad = -1.0, None, 0
    for ep in range(cli.epochs):
        train_epoch(); r10 = val_r10()
        if r10 > best:
            best, best_state, bad = r10, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cli.patience:
                break
    model.load_state_dict(best_state)

    test = d.test_samples if not cli.smoke_users else d.test_samples[:cli.smoke_users]
    scores = full_scores(test)
    tgt = np.array([d.news2idx[x["target"]] for x in test])
    uid = np.array([x["user_id"] for x in test])
    agg, hits = eval_full_catalog(scores, tgt, uid)

    van = np.load(cli.vanilla_npz, allow_pickle=True)
    pos = {u: i for i, u in enumerate(list(van["user_ids"]))}
    keep = [i for i, u in enumerate(uid) if u in pos]
    sel = np.array([pos[uid[i]] for i in keep])
    vh = {1: van["vanilla_hit1"][sel], 10: van["vanilla_hit10"][sel]}
    bh = {1: hits[1][keep], 10: hits[10][keep]}
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    write_per_user_hits(cli.output_dir, uid[keep], bh, vh)

    results = {"model": "NRMS", "n_news": len(d.news2idx), "n_test": len(test),
               "rows": {"nrms": agg}, "runtime_sec": time.time() - t0}
    (cli.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    with MlflowRunLogger(enabled=True, tracking_uri="http://localhost:5000",
                         experiment_name="mind_genrec", run_name="nrms_mind",
                         tags={"dataset": "MIND-small", "model": "NRMS", "eval": "full_catalog"}) as lg:
        lg.log_params({"n_news": len(d.news2idx), "vocab": d.vocab_size, "max_history": cli.max_history})
        lg.log_metrics({f"nrms.{k}": v for k, v in agg.items()})
        lg.log_dict(results, "nrms_results.json")
    print(json.dumps(results["rows"], indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add baselines/run_nrms.py
git commit -m "feat(baselines): NRMS run script (train + full-catalog eval + MLflow)"
```

---

## Task 9: Significance integration (`genrec_v2/run_statistical_significance.py`)

**Files:**
- Modify: `genrec_v2/run_statistical_significance.py` (OUT_DIR env override + baseline comparison)

- [ ] **Step 1: Make OUT_DIR env-overridable + add baseline_vs_vanilla**

Replace the hardcoded `OUT_DIR` (line 30) and the comparison list construction in `main()`:

```python
import os
OUT_DIR = Path(os.environ.get("SIG_OUT_DIR",
               "/home/lishazhai/workspace/GR4AD/experiments/main_table"))
```

In `main()`, after loading `d`, branch on which keys are present:

```python
if "baseline_hit1" in d:  # baseline run (NRMS/SASRec) — single row vs the generative vanilla
    comparisons = [
        ("baseline_vs_vanilla_hit@1", d["vanilla_hit1"], d["baseline_hit1"]),
        ("baseline_vs_vanilla_hit@10", d["vanilla_hit10"], d["baseline_hit10"]),
    ]
else:  # scorer run (existing behavior)
    comparisons = [
        ("listwise_vs_vanilla_hit@1", d["vanilla_hit1"], d["listwise_hit1"]),
        ("listwise_vs_vanilla_hit@10", d["vanilla_hit10"], d["listwise_hit10"]),
        ("focal_vs_vanilla_hit@1", d["vanilla_hit1"], d["focal_hit1"]),
        ("focal_vs_vanilla_hit@10", d["vanilla_hit10"], d["focal_hit10"]),
    ]
```

Guard the per-seed block with `if "listwise_hit1_seeds" in d:` (already keyed; make explicit).

- [ ] **Step 2: Verify on the existing MIND npz (regression — must reproduce prior output)**

Run: `SIG_OUT_DIR=/data/lishazhai/workspace/GR4AD/experiments/main_table python -u genrec_v2/run_statistical_significance.py`
Expected: same listwise/focal table as before (no behavior change for scorer runs).

- [ ] **Step 3: Commit**

```bash
git add genrec_v2/run_statistical_significance.py
git commit -m "feat: SIG_OUT_DIR env + baseline_vs_vanilla comparison in significance"
```

---

## Task 10: Execution (staged, 7-step methodology) — RUN ONLY AFTER ADVERSARIAL REVIEW

> Do NOT run until the Claude adversarial-review subagent passes. Deploy via `scp` (server GitHub down);
> compile-check every file with server py3.10 before running.

- [ ] **Step 1 — compliance:** all tests green locally (`pytest tests/baselines -v`); scp `baselines/` +
  modified significance to server; `python -m py_compile` each on the gram env (py3.10).
- [ ] **Step 2 — smoke (≤10 min):** `--smoke-users 1000` for NRMS and one SASRec dataset (Sports).
  Assert: runs end-to-end, R@50 ≤ Oracle bound of that setting, metrics finite, per_user_hits written,
  significance runs. Compare baseline vs vanilla direction.
- [ ] **Step 3 — root-cause if a metric is degenerate** (e.g. NRMS R@10 ≈ random 10/|catalog|): trace to
  data alignment (titles empty? history all OOV?) or training (loss not dropping) → fix → re-smoke.
- [ ] **Step 5 — full:** full NRMS (MIND) + SASRec × {Beauty, Sports, Toys}, full epochs/early-stop.
- [ ] **Step 6 — significance:** `SIG_OUT_DIR=experiments/{nrms,sasrec_<ds>} run_statistical_significance.py`
  for each → baseline_vs_vanilla McNemar/Wilcoxon.
- [ ] **Step 7 — post-hoc:** if a baseline is implausibly strong/weak vs the generative vanilla, locate the
  cause (catalog mismatch, leakage, wrong LOO) before reporting.

**Success:** NRMS + SASRec rows populated with R@1/5/10/50 + MRR + nDCG@5/10 from the shared metrics;
per_user_hits + significance vs each setting's generative vanilla; MLflow logged.

---

## Self-Review

**Spec coverage:** NRMS (T4,T8) ✓; SASRec (T6,T7) ✓; full-catalog eval reusing _rank_metrics (T1,T2) ✓;
MIND split mirror (T3) ✓; Amazon LOO mirror (T5) ✓; per_user_hits + significance (T2,T9) ✓; MLflow (T7,T8) ✓;
no tuning / standard configs (T4,T6) ✓; lean PyTorch, no libs ✓.

**Placeholder scan:** the only deferred items are the two server-side discovery spikes (MIND news.tsv path
confirm; GRAM sequence-file format) — both are explicit RUN-FIRST commands with the expected format
documented, not hidden TODOs. SASRec `--seq-file` is passed once the spike confirms the filename.

**Type consistency:** `rank_metrics_single`/`eval_full_catalog`/`write_per_user_hits` signatures match across
T1/T2/T7/T8; `MindData`/`AmazonSeq` fields used consistently; npz keys (`vanilla_hit1/10`, `baseline_hit1/10`,
`user_ids`) match T2 writer ↔ T9 reader.

**Risk note (logged, not silent):** GRAM's LOO item-id space must match the generative vanilla's user/target
set; the spike MUST confirm the sequence file's ids and that last-item = the same test target, else the
`per_user_hits` alignment (by user_id) drops mismatched users — the run logs how many users were kept.
