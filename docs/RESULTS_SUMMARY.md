# GR4AD — Results & Methods Summary (paper draft)

> Consolidated from verified artifacts under `experiments/` (results.json / significance.json) and
> `figures/`. Every number traces to a persisted artifact. Last updated 2026-06-17.

## 1. Method (what we claim)

**Contribution 1 — confidence-cascade diagnosis.** A frozen generative retriever (autoregressive
decoder over hierarchical/semantic item codes, beam search K=50) is *over-confident*: greedy decode
ECE explodes along the code chain (per-position ECE pos0→3 under greedy = `[0.007, 0.360, 0.901, 0.890]`
vs oracle `[0.007, 0.068, 0.035, 0.011]`), while chain recall collapses (`t0` 0.084 → `all-4` 0.029).
Most of the catalog is *recoverable* in the beam (Oracle R@50 ≫ vanilla R@1) but the generator's own
scores rank it poorly — the **Scoring-Efficiency (SE) gap** (R@1/R@50 ≈ 9–19%).

**Contribution 2 — post-hoc reranking scorer (the contribution we test).** A small scorer reranks the
frozen generator's beam-50 candidates: `final = avg_logprob + λ · scorer_term`. Two variants share one
locked configuration: **Listwise** (self-attention, d=128, 4 heads, 2 layers, FFN 256, dropout 0.1,
user-CLS token, +beam_score) trained with binary BCE; **Pointwise Focal** (γ=2.0, bottleneck 64).

**Claim = the scorer METHOD is MODEL-AGNOSTIC.** The same method + hyperparameters + **λ-selection
PROCEDURE** improve any base generator across diverse (architecture × domain) settings. λ is the only
per-setting free value, chosen by an *identical* validation procedure.

### λ-selection (the control variable — leakage-free)
λ is selected on a **held-out** 30% user split that the scorer NEVER trained on (pool = 70% trains the
scorer; holdout = 30% selects λ by argmax R@1 over the locked grid
`{0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0}`). Selecting λ on training-included data over-trusts an
overfit scorer — catastrophic on high-SE settings (Toys flipped **−13% → +3.2%** once λ was chosen
honestly). 5-seed Listwise `[42,123,7,999,2024]` uses one shared λ (argmax of mean held-out R@1).

### Metric protocol (honest framing)
All rows report **Recall@K / MRR / nDCG@K of their own ranked list** via one shared `_rank_metrics`.
Generative rows (TIGER-equiv / Focal / Listwise / Oracle) are **beam-recall-bounded** (rank within the
K=50 beam, so R@50 = Oracle). Discriminative baselines (NRMS / SASRec) rank the **full catalog**.
Absolute R@K is therefore **not directly comparable** across the two — this is the field-standard
cross-method table (as in TIGER/GRAM) and is stated as a table footnote.

## 2. Main Table — News (MIND-small, user 70/15/15 seed42, full-catalog)

| Method | R@1 | R@10 | R@50 |
|---|---|---|---|
| NRMS (full-catalog) | 0.0015 | 0.0174 | 0.0610 |
| TIGER-equiv (our GenRec, vanilla) | 0.0331 | 0.1692 | 0.3671 |
| + Pointwise Focal (λ=0.1) | 0.0345 (+4.1%) | 0.1710 | — |
| **+ Listwise (5-seed, λ=0.1)** | **0.0352 (+6.3%)** \*\* | 0.1723 | — |
| Oracle | 0.3671 | 0.3671 | 0.3671 |

Listwise vs vanilla: McNemar @1 **p=3.5e-4**, @10 **p=1.3e-6**; per-seed **5/5 significant** at @1
(median p=4.5e-5) and @10 (median p=2.7e-6). NRMS (full-catalog discriminative) is far below the
beam-bounded generative rows (see §1 framing); McNemar vs vanilla p<1e-200.

## 3. Main Table — E-commerce (Beauty / Sports / Toys, GRAM T5 AR, leave-one-out, full-catalog)

R@10:

| Method | Beauty | Sports | Toys |
|---|---|---|---|
| SASRec (full-catalog) | 0.0133 | 0.0104 | 0.0112 |
| TIGER-equiv (GRAM vanilla) | 0.0881 | 0.0561 | 0.0927 |
| + Pointwise Focal | 0.0923 | 0.0601 | 0.0957 |
| + Listwise (5-seed) | 0.0944 | 0.0606 | 0.0979 |
| Oracle | 0.1731 | 0.1207 | 0.1662 |

R@1 lift of Listwise vs vanilla (held-out λ=0.02 all three): Beauty **+5.0%**, Sports **+6.5%**,
Toys **+3.2%**. Significance (McNemar): @10 **highly significant in all three** with **5/5 seeds**
(Beauty p=2.1e-8, Sports p=2.2e-8, Toys p=2.7e-11); @1 not significant under LOO single-target
sparsity (1–2/5 seeds) — by design we treat **@10/McNemar as primary for LOO**. SASRec base gate:
vanilla R@10 0.056 (Sports) ≈ GRAM-published 0.0554, Toys 0.093 ≈ 0.096.

## 4. Headline: model-agnostic, leakage-free

| Setting | base arch | domain | code-len | Listwise R@1 lift | primary significance |
|---|---|---|---|---|---|
| MIND | our GenRec | news | 4 | **+6.3%** | @1 & @10, 5/5 seeds |
| Beauty | GRAM T5 AR | e-com | 7 | **+5.0%** | @10, 5/5 seeds |
| Sports | GRAM T5 AR | e-com | 7 | **+6.5%** | @10, 5/5 seeds |
| Toys | GRAM T5 AR | e-com | 5 | **+3.2%** | @10, 5/5 seeds |

All four POSITIVE and @10-significant under ONE identical leakage-free held-out λ procedure, across
news + 3 e-com domains × 2 base architectures × 3 code-lengths (4/7/5).

## 5. Figures / diagnostics

- **Confidence cascade** (`figures/confidence_cascade.*`): chain-recall collapse + greedy-ECE explosion
  (pos3 greedy 0.890 vs oracle 0.011) — motivates reranking.
- **Per-candidate ECE reliability** (`figures/ece_reliability.*`): the generator's absolute confidence
  `exp(avg_logprob)` is badly mis-calibrated (**ECE 0.283**); the scorer's BCE-calibrated P is nearly
  perfect (**ECE 0.0009**) — the scorer recalibrates the over-confident generator. (Fair baseline: NOT
  a softmax-over-beam normalization, which would be a 1/K artifact.)
- **loss × label diagnostic** (`experiments/main_tables/loss_label_table.md`, MIND baseline_retrain):
  BCE-binary **0.0381 = grid max** over {bce,listmle,approx_ndcg} × {binary,soft} — justifies the
  locked BCE-binary choice. (Diagnostic uses the script's own beam + 7-pt grid + val-selection;
  relative comparison only.)
- **λ-sensitivity** (`figures/lambda_sensitivity.*`): held-out R@1 vs λ for all 4 settings; λ\*↔SE —
  MIND (λ\*=0.1, SE 9.0→9.6%), Beauty (0.02, 15.2→15.9%), Toys (0.02, 18.8→19.4%), Sports
  (0.02, 11.5→12.3%). Higher-SE settings prefer smaller λ\* (qualitative trend; absolute λ not
  cross-setting comparable — disclosed).

## 6. Rigor / reproducibility notes

- **Significance:** per-seed McNemar (report k/5) + per-user Wilcoxon; McNemar primary for Amazon LOO.
- **Beam:** per-sample (not per-user — the per-user bug gave 2.31% vs correct 3.31% on MIND), K=50,
  cache fingerprinted by (dataset, checkpoint, code_length, K, n).
- **Baselines:** lean PyTorch NRMS + SASRec (all-position), full-catalog eval, paired to the generative
  vanilla by real user-id (Amazon LOO) / (user_id, occurrence) key (MIND per-sample, robust to
  cross-process string-hash set-iteration order).
- **Review:** 4 adversarial-review rounds (3 on baselines, 1 on Phase E figures) — caught & fixed the
  λ-leakage, the NRMS per-sample mis-pairing, and the softmax-ECE artifact, among others.

## 7. Out of scope (deferred)
Beyond-accuracy (coverage/ILD/novelty), case study, MIND-large / Adressa / Yelp.
Related: [[project-heldout-lambda-selection]], [[project-baselines-nrms-sasrec]].
