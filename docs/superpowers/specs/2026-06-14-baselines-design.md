# GR4AD Baselines — NRMS (News) + SASRec (E-commerce)

**Date:** 2026-06-14
**Status:** Approved (design)

## Intent

Add a strong discriminative baseline ROW to each GR4AD main table, comparable to the
generative-retrieval rows under an IDENTICAL evaluation protocol:

- **News table** (MIND-small): NRMS
- **E-commerce table** (Beauty / Sports / Toys): SASRec

The baseline is a *different ranked-list producer*, NOT the scorer method. Control variables:
same user split / same held-out target / same full-catalog ranking / same `_rank_metrics`
helper / same `per_user_hits.npz` significance pipeline as the scorer settings.

## Non-Goals (YAGNI)

- No hyperparameter tuning — baselines use STANDARD published configs (fair-baseline convention).
- No impression-list CTR ranking (NRMS native task) — we use full-catalog next-item retrieval.
- No reuse of `similar_item_sasrec.txt` (verified: it is a precomputed item-kNN feature table,
  NOT a rankable SASRec checkpoint).
- No library dependency — `recommenders` (TF) and `recbole` are NOT installed; lean PyTorch
  reimplementation avoids install + protocol-mismatch fighting and keeps the control clean.

## Architecture (3 focused modules)

### `baselines/nrms.py` (~200 lines)
- **News encoder:** title token embedding → multi-head self-attention → additive attention → news vector.
- **User encoder:** history news vectors → additive attention → user vector.
- **Score:** `user · news` (dot product).
- Config (NRMS paper standard): title max-len 30; word-emb dim 300 (random-init, trainable —
  GloVe optional, deferred); self-attn 16 heads × 16 dim; additive-attn dim 200; dropout 0.2.

### `baselines/sasrec.py` (~180 lines)
- item-embedding + learnable position-embedding → 2× (causal self-attn block + point-wise FFN)
  → last-position hidden.
- **Score:** `hidden · item_emb` over the full catalog.
- Config (SASRec paper standard): d=64; 2 blocks; 1 head; max-history 20 (Amazon) / 50 (MIND);
  dropout 0.2. ID-only (no CF/text features — SASRec is the pure-ID sequential baseline).

### `baselines/eval_fullcatalog.py` (~120 lines)
- Reuses the scorer's metric DEFINITIONS (`hit@k`, `MRR = 1/rank`, `nDCG@k = 1/log2(1+rank)`),
  applied to the **full-catalog rank of the single held-out target**.
- Emits R@1/5/10/50 + MRR + nDCG@5/10 + `per_user_hits.npz` (vanilla + baseline hit@1/@10 + user_ids).
- Self-consistency asserts: `0 ≤ R@k ≤ 1`, metrics finite, R@k monotone non-decreasing in k.

## Data Alignment (1:1 mirror of the scorer test set)

- **NRMS:** reuse the MIND GenRec data pipeline to obtain the EXACT `(history, next-click target)`
  per user under user 70/15/15 seed42 split (same `vsl`/`tsl` the scorer used); news title text
  from `mind_small/news.tsv`. Full catalog = all news items.
- **SASRec:** GRAM `rec_datasets/{Beauty,Sports,Toys}` sequential interactions (same sequences the
  generator used); leave-one-out target = last item; full catalog = all items of that dataset.

## Training

- Task = next-item / next-click full-catalog retrieval.
- Loss = sampled-softmax over the catalog (in-batch positives + uniform-sampled negatives);
  exact full-catalog scoring at eval time. Optimizer AdamW.
- Early-stop on validation R@10; fixed seed; standard epochs/patience.

## Evaluation + Significance

- For each test user: score the full catalog → target rank → hit@k / MRR / nDCG@k.
- Significance vs the SAME-setting vanilla via `run_statistical_significance.py`
  (McNemar per-sample primary for LOO; Wilcoxon per-user). Baseline compared to vanilla in the
  same table, NOT to the scorer.

## Persistence + MLflow

- Each baseline writes `results.json` + `per_user_hits.npz` to `experiments/{nrms, sasrec_<dataset>}/`.
- MLflow auto-logging (hyperparams, per-epoch metrics, dataset hash) — URL-only monitoring.

## Process (user-mandated)

design (approved) → spec → writing-plans → subagent implement → subagent **adversarial review
(Claude agent)** → satisfy review → run. GPU: free after Sports; baseline training is lightweight.

## Verification

- NRMS / SASRec produce all shared-metric cells under the matched protocol.
- Numbers plausible (NRMS news; SASRec on LOO above random); R@50 ≤ Oracle bound.
- per_user_hits persisted; significance runs against the reported row.

Related: [[project-heldout-lambda-selection]] (scorer rows use the leakage-free held-out λ procedure).
