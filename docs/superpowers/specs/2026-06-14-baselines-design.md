# GR4AD Baselines — NRMS (News) + SASRec (E-commerce)

**Date:** 2026-06-14
**Status:** Approved (design)

## Intent

Add a strong discriminative baseline ROW to each GR4AD main table, comparable to the
generative-retrieval rows:

- **News table** (MIND-small): NRMS
- **E-commerce table** (Beauty / Sports / Toys): SASRec

The baseline is a *different ranked-list producer*, NOT the scorer method. Control variables:
same user split / same held-out target / same `_rank_metrics` helper / same `per_user_hits.npz`
significance pipeline as the scorer settings.

**Metric framing (corrected after adversarial review 2026-06-14):** each system reports
**Recall@K / MRR / nDCG@K of its own ranked list** — the field-standard cross-method comparison
(TIGER/GRAM tables put generative and sequential rows side by side this way). The generative
rows are **beam-recall-bounded**: the generator emits a K=50 beam, so its R@50 equals the Oracle
upper bound (target-in-beam) and its metrics are computed within that beam. The baselines rank the
**full catalog** directly. This is NOT "identical full-catalog protocol" — it is the same metric
(Recall@K is a per-user hit/miss, so the paired McNemar/Wilcoxon vs the generative vanilla is valid)
applied to each method's natural candidate list. The beam-bound MUST be stated as a table footnote.

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
  applied to the **full-catalog rank of the single held-out target**. Rank derivation matches the
  scorer's stable `argsort(descending)` tie semantics (NOT optimistic strict-greater); the PAD
  column (item id 0) is excluded from the candidate set.
- Emits R@1/5/10/50 + MRR + nDCG@5/10 + `per_user_hits.npz` with **baseline + generative-vanilla
  hit@{1,5,10,50}** + user_ids (all KS persisted so @5/@50 significance is possible).
- Self-consistency asserts (ENFORCED, per review): `0 ≤ R@k ≤ 1`, metrics finite, R@k monotone
  non-decreasing in k, and `len(kept user_ids) > 0` after alignment (fail loudly, never empty-pair).

**User-id alignment (review-critical):** the Amazon scorer's `_extract_user_id` previously fell back
to `_idx{n}` (GRAM batch key unavailable), so `experiments/{Toys,Sports}_scorer/per_user_hits.npz`
stored synthetic ids — SASRec's real ASIN user-ids could not pair. FIX: capture the real user id in
`run_beauty_scorer.py` and re-run the Amazon scorers (beam cached → fast) so both sides key by the
same real user id. NRMS is unaffected (MIND user_ids are already real, e.g. `U25421`).

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
