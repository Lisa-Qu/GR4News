# GR4AD Phase E — Figures + Final Tables

**Date:** 2026-06-16
**Status:** Approved (design)

## Intent

Produce the paper's contribution-1 figures + ablation table + the two assembled main tables, all on
`baseline_retrain` + per-sample beam with held-out λ-selection numbers (matching the headline). Five
numbered deliverables; each code step goes through brainstorming→plan→subagent→adversarial review
(Claude reviewers, codex out of quota).

## Deliverables (1–5)

### 1. Confidence-cascade figure
- **Data:** `experiments/confidence_cascade_diagnosis/cascade_data.json` EXISTS, generated on
  `baseline_retrain` (verified: its CKPT_PATH + docstring). Keys: `chain`, `delta_conf_oracle`,
  `delta_conf_greedy`, `ece_oracle`, `ece_greedy`, `oracle_recall`, `token_accuracy`.
- **Code:** plot-only script `figures/plot_confidence_cascade.py` → position-0→3 confidence cascade
  (Oracle vs greedy chains). No GPU, no re-run.
- **Output:** `figures/confidence_cascade.{png,pdf}`.

### 2. ECE reliability diagram
- **Change:** `genrec_v2/ece_analysis.py` currently loads `genrec_v2_ablation_v2/B_nohot/best_model.pt`
  (`:35`) — REPOINT to `experiments/genrec_v2_exposure_bias/baseline_retrain/best_model.pt`. Add the
  scorer's calibrated P so the figure shows TWO reliability curves (vanilla beam-prob vs scorer-calibrated)
  + their ECE numbers.
- **Code + run:** re-run on per-sample beam (cached/lightweight, GPU). Plot script.
- **Output:** `figures/ece_reliability.{png,pdf}` + the ECE values into the run's results.

### 3. loss×label diagnostic table
- **Data:** NOT present for the scorer (the existing `genrec_v2_ranking_loss/` is a GENERATOR CE/DPO
  ablation, not the scorer grid).
- **Code + run:** run `genrec_v2/run_listwise_scorer.py` over the grid {bce, listmle, approx_ndcg} ×
  {binary, soft} on MIND val cached beam (held-out λ-selection, same control vars) → R@1 (+R@10) table.
  Justifies the locked BCE-binary choice (expect BCE-binary competitive/best).
- **Output:** `experiments/main_tables/loss_label_table.{md,tex}`.

### 4. λ-sensitivity plot
- **Data:** the four held-out runs printed per-λ curves to logs (`/tmp/{mind_heldout_lambda,
  beauty_heldout_lambda,toys_full_heldout,sports_full_heldout}.txt`) as
  `lambda=X held-out (mean) R@1=Y` lines. Parse those (decision: parse logs, no re-run).
- **Code:** parse the 4 logs → plot held-out R@1 vs λ for all four settings on one axis + annotate each
  setting's λ* and its Scoring Efficiency (λ*↔SE trend). No GPU.
- **Robustness note:** the parser is tied to the log line format `lambda=<float> ... R@1=<float>`; if a
  log is missing/format-changed, fail loudly (assert ≥1 curve parsed per setting).
- **Output:** `figures/lambda_sensitivity.{png,pdf}`.

### 5. Two final main tables
- **Data:** all `experiments/*/results.json` + `*/significance.json` exist (4 scorer settings + NRMS +
  3 SASRec). Pure assembly.
- **Code:** read jsons → format News table (NRMS / TIGER-equiv / +Focal / +Listwise(5-seed) / Oracle)
  and E-commerce table (SASRec / TIGER-equiv / +Focal / +Listwise / Oracle), with significance stars
  and the **beam-bound caveat footnote** (generative rows beam-recall-bounded; baselines full-catalog;
  absolute R@K not directly comparable). markdown + LaTeX.
- **Output:** `experiments/main_tables/{news_table,ecom_table}.{md,tex}`.

## Non-Goals (YAGNI)
- No new experiments beyond #2/#3 runs; no re-running scorers/baselines (numbers final).
- No interactive dashboards; static PNG+PDF only.
- λ-sensitivity does NOT re-run sweeps (parse existing logs).

## Conventions / Control
- All on `baseline_retrain` + per-sample beam (MIND) / GRAM AR epoch_30 (Amazon); held-out λ numbers.
- Figures: matplotlib, PNG (preview) + PDF (paper vector) to `figures/`.
- Tables: markdown (preview) + LaTeX (paper) to `experiments/main_tables/`.
- MLflow auto-log the #2/#3 runs.

## Process
Each code deliverable: brainstorming (this doc) → writing-plans (one Phase-E plan, 5 tasks) → subagent
implements → Claude adversarial review → fix → run. Deploy via scp; py_compile on server py3.10.

## Verification
- #1 cascade monotone pos0→3; #2 ECE(scorer) < ECE(vanilla); #3 BCE-binary competitive in the grid;
  #4 each setting's parsed λ* matches its results.json chosen λ; #5 every table cell traces to a
  results.json value, Oracle == R@50, significance stars match significance.json.

Related: [[project-heldout-lambda-selection]], [[project-baselines-nrms-sasrec]].
