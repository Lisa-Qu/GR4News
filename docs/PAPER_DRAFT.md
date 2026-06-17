# [Title TBD] Recalibrating Overconfident Generative Retrievers: A Model-Agnostic Post-hoc Reranking Scorer

> **First draft (2026-06-17).** Venue-agnostic prose; adapt to the target template (project targets
> CCF-B — confirm exact venue for page limit / required sections). **All `\cite{...}` keys are
> PLACEHOLDERS — none are verified; do NOT compile a .bib from these. See §Citations-to-verify.**
> Every experimental number traces to `experiments/*/results.json|significance.json` (see
> `docs/RESULTS_SUMMARY.md`).

## One-sentence contribution
A frozen generative retriever's beam log-probabilities are systematically over-confident and rank
in-beam candidates poorly; a small **post-hoc reranking scorer**, applied with one identical
method (architecture + loss + a leakage-free held-out λ-selection procedure), recalibrates those
scores and improves Recall across diverse base generators — it is **model-agnostic**.

---

## Abstract (5-sentence formula)
Generative retrieval models decode item identifiers token-by-token and rank candidates by sequence
log-probability, but we show these scores are badly mis-calibrated: a frozen generator places most
of the catalog's relevant items inside its beam yet ranks them poorly, a *confidence cascade* in
which per-position calibration error explodes along the decoded code while chain recall collapses.
Closing this gap is hard because the generator is expensive to retrain and its overconfidence is
intrinsic to autoregressive decoding. We introduce a lightweight **post-hoc reranking scorer** that
leaves the generator frozen, rescoring its beam-50 candidates by fusing the beam log-probability with
a learned relevance score, `final = avg_logprob + λ·scorer_term`, where λ is chosen by an identical
**held-out** validation procedure that we show is essential to avoid over-trusting an overfit scorer.
Across four settings spanning two base architectures (our k-means GenRec and GRAM's T5 autoregressive
decoder), two domains (news, e-commerce), and three code-lengths (4/7/5), the *same* method improves
Recall@1 by 3.2–6.3% with the listwise variant, significant at Recall@10 in all four settings
(5/5 seeds, McNemar p<10⁻⁶), and reduces per-candidate calibration error from 0.283 to 0.0009.

---

## 1. Introduction

Generative retrieval has emerged as a compact alternative to dual-encoder retrieval: an
encoder–decoder model autoregressively generates a target item's discrete identifier (a sequence of
semantic or hierarchical codes), and beam search produces a ranked candidate list `\cite{tiger,
gram, idgenrec}`. The model's own sequence log-probability serves as the ranking score. This is
elegant — one model both generates and ranks — but it couples *recall* (is the target in the beam?)
to *ranking* (is it near the top?) through a single signal that the decoder was never trained to
calibrate.

**We first diagnose a confidence cascade.** On a frozen generator we measure, per decoded position,
the expected calibration error (ECE) of the decoder's token confidence. Under greedy decoding ECE
rises from 0.007 at the first position to 0.890 by the last, while the fraction of beams whose entire
code chain is consistent with the target collapses from 0.084 to 0.029. Crucially, the target is
*recoverable*: Oracle Recall (target anywhere in the beam) far exceeds vanilla Recall@1 — the
Scoring Efficiency R@1/R@50 is only 9–19% across settings. The generator finds the right items but
ranks them with an overconfident, mis-calibrated score.

**We then close the gap with a model-agnostic post-hoc scorer.** Rather than retrain the (expensive,
frozen) generator, we rerank its beam-50 candidates with a small scorer trained only on validation
beams. The reranked score fuses the generator's evidence with the learned relevance,
`final = avg_logprob + λ·scorer_term`. The fusion weight λ is the single per-setting free parameter;
we select it by an *identical* procedure on a held-out split of validation users that the scorer
never trained on. We find this held-out selection is not a detail but a requirement: selecting λ on
the scorer's own training users over-trusts an overfit scorer and *reverses* the gain on high-Scoring-
Efficiency settings (Toys: −13% → +3.2% once λ is chosen honestly).

**Contributions.**
- We characterize a **confidence cascade** in generative retrievers: token-level overconfidence
  compounds along the decoded code, so beam log-probability is a weak ranking signal despite high
  beam recall (§3, Fig. 1).
- We propose a **model-agnostic post-hoc reranking scorer** (listwise and pointwise variants) with a
  **leakage-free held-out λ-selection** procedure, and show the procedure — not just the architecture
  — is the control variable that makes the method transfer (§4).
- We validate model-agnosticism across **four settings** (2 base architectures × 2 domains × 3
  code-lengths): listwise Recall@1 +3.2–6.3%, Recall@10 significant in all four (5/5 seeds,
  McNemar p<10⁻⁶), and per-candidate ECE 0.283→0.0009 (§5).

## 2. Related Work
*(Organize by methodology; cite generously. All keys below are PLACEHOLDERS to verify.)*

**Generative retrieval / recommendation.** Generating item identifiers and ranking by sequence
likelihood `\cite{tiger, gram, idgenrec, lcrec, letter}`. These works focus on *building* the
generator (better identifiers, training objectives); we instead treat any such generator as frozen
and address the *ranking* of its beam, which is complementary.

**Reranking and two-stage retrieval.** Classical retrieve-then-rerank `\cite{rankerretriever}`; our
scorer is a post-hoc reranker over a *generative* first stage and fuses (not replaces) the generator's
score.

**Calibration of neural rankers / sequence models.** ECE and reliability `\cite{guo_calibration}`;
we measure per-candidate calibration of a generative retriever and show a learned scorer recalibrates
it.

**Loss functions for ranking.** Pointwise focal `\cite{focal_loss}`, listwise BCE, ListMLE
`\cite{listmle}`, ApproxNDCG `\cite{approxndcg}`; our diagnostic compares these and finds binary BCE
best for this rerank setting.

**Discriminative baselines.** SASRec `\cite{sasrec}` (sequential), NRMS `\cite{nrms}` (news); we
compare under a matched full-catalog protocol.

## 3. The Confidence Cascade (Diagnosis)
*(Method + Fig. 1: `figures/confidence_cascade.*`.)* On the frozen MIND generator we decode the 4-token
code and, at each position, bin the decoder's max-softmax confidence against token correctness to get
per-position ECE, under both greedy and oracle (teacher-forced) decoding. Greedy ECE = [0.007, 0.360,
0.901, 0.890] vs oracle [0.007, 0.068, 0.035, 0.011]: errors compound only when the model conditions on
its own (overconfident) predictions. Chain recall drops t0→all-4 from 0.084 to 0.029, yet Oracle R@50
(0.367) ≫ R@1 (0.033). Takeaway: the ranking signal, not recall, is the bottleneck — motivating a
post-hoc rescorer.

## 4. Method

**Setup.** A frozen generator produces, per query, a beam of K=50 candidate codes with sequence
log-probabilities. Let `s` be the candidate's average token log-probability (`avg_logprob`). A scorer
`f` maps the candidate's decoder hidden states (and the user representation) to a relevance term; we
rerank by
`final = s + λ · f(·)`.
λ=0 recovers the vanilla generator ranking; λ→∞ is scorer-only.

**Scorer variants (held constant across all settings).**
- *Listwise* — self-attention over the beam (d=128, 4 heads, 2 layers, FFN 256, dropout 0.1, a
  user-CLS token, and the beam score as a feature), trained with binary BCE.
- *Pointwise Focal* — per-candidate MLP with a 64-d bottleneck, focal BCE (γ=2.0).

**Held-out λ-selection (the control variable).** We split validation *users* into a pool (70%, trains
the scorer) and a disjoint holdout (30%, selects λ). λ is chosen by argmax Recall@1 on the holdout over
a fixed grid {0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0}; the 5-seed listwise uses one shared λ
(argmax of mean holdout Recall@1). Selecting λ on data the scorer trained on over-trusts an overfit
scorer and inflates λ; on high-Scoring-Efficiency Toys this *reversed* the test gain from +3.2% to
−13%. **The method's transfer claim rests on holding this procedure — not a fixed λ value — constant.**

**What "model-agnostic" means here.** The CONTROL is the scorer method + hyperparameters + λ-selection
procedure (byte-identical everywhere); the base generator is the TREATMENT (free to differ in
architecture, tokenizer, code-length, domain, eval split). The chosen λ *value* is an outcome of the
identical procedure and may differ per setting (like a per-dataset learning rate); only the procedure
must match, with no test leakage.

## 5. Experiments

**Settings (treatment axis).** (1) MIND-small news, our k-means GenRec, code-len 4, user 70/15/15
split; (2–4) Amazon Beauty/Sports/Toys, GRAM T5 autoregressive decoder, code-len 7/7/5, leave-one-out.
Beam K=50 (per-sample), seeds {42,123,7,999,2024}.

**Protocol + honest framing.** All rows report Recall@K / MRR / nDCG@K of their own ranked list via
one shared metric implementation. Generative rows (vanilla / Focal / Listwise / Oracle) are
**beam-recall-bounded** (rank within the K=50 beam, so R@50 = Oracle); the discriminative baselines
NRMS/SASRec rank the **full catalog**. Absolute R@K is therefore not directly comparable across the two
families — this is the standard cross-method table and is footnoted as such; the within-family
generative comparison (vanilla → scorer) is the controlled test of our contribution.

### 5.1 Main results

**News (MIND-small, user-split, full-catalog).**

| Method | R@1 | R@10 | R@50 |
|---|---|---|---|
| NRMS (full-catalog) | 0.0015 | 0.0174 | 0.0610 |
| Vanilla generator (TIGER-equiv) | 0.0331 | 0.1692 | 0.3671 |
| + Pointwise Focal (λ=0.1) | 0.0345 (+4.1%) | 0.1710 | — |
| **+ Listwise (5-seed, λ=0.1)** | **0.0352 (+6.3%)** | 0.1723 | — |
| Oracle | 0.3671 | 0.3671 | 0.3671 |

Listwise vs vanilla: McNemar Recall@1 p=3.5×10⁻⁴, Recall@10 p=1.3×10⁻⁶; **5/5 seeds** significant at
both @1 and @10.

**E-commerce (Beauty/Sports/Toys, leave-one-out, full-catalog), Recall@10.**

| Method | Beauty | Sports | Toys |
|---|---|---|---|
| SASRec (full-catalog) | 0.0133 | 0.0104 | 0.0112 |
| Vanilla generator | 0.0881 | 0.0561 | 0.0927 |
| + Pointwise Focal | 0.0923 | 0.0601 | 0.0957 |
| + Listwise (5-seed) | 0.0944 | 0.0606 | 0.0979 |
| Oracle | 0.1731 | 0.1207 | 0.1662 |

Listwise Recall@1 lift vs vanilla (held-out λ=0.02 all three): Beauty +5.0%, Sports +6.5%, Toys +3.2%.
Recall@10 McNemar: Beauty p=2.1×10⁻⁸, Sports p=2.2×10⁻⁸, Toys p=2.7×10⁻¹¹, **5/5 seeds each**.
Recall@1 is not significant under leave-one-out single-target sparsity (we report Recall@10 / McNemar as
primary for LOO).

**Model-agnostic summary.** All four settings (2 architectures × 2 domains × code-lengths 4/7/5) yield a
positive, Recall@10-significant listwise lift under one identical leakage-free procedure.

### 5.2 Calibration (Fig. 2: `figures/ece_reliability.*`)
Per-candidate reliability on MIND: the generator's absolute confidence `exp(avg_logprob)` is badly
overconfident (ECE 0.283), while the listwise scorer's calibrated probability achieves ECE 0.0009 — the
scorer recalibrates the over-confident generator (consistent with §3). *(Vanilla P is the generator's
own per-candidate confidence, not a softmax-over-beam, which would be a 1/K normalization artifact.)*

### 5.3 Ablations
- **Loss × label** (Table, MIND): over {bce, listmle, approx_ndcg} × {binary, soft}, binary BCE is the
  grid maximum (0.0381 R@1), justifying the locked choice.
- **λ-sensitivity** (Fig. 3: `figures/lambda_sensitivity.*`): held-out R@1 vs λ for all four settings;
  the chosen λ* shrinks as Scoring Efficiency rises (MIND λ*=0.1 at SE 9%, Amazon λ*=0.02 at SE 12–19%) —
  higher-SE generators need a lighter touch. (Absolute λ is not cross-setting comparable; we report this
  as a qualitative trend.)
- **Held-out vs leaky λ-selection**: selecting λ on training-included data reverses Toys from +3.2% to
  −13%, demonstrating the procedure's necessity.

## 6. Limitations
- The generative rows are beam-recall-bounded; we do not claim absolute-metric superiority over
  full-catalog discriminative models, only a controlled within-generator improvement (and report both
  honestly).
- Recall@1 significance is weak under leave-one-out (single relevant target); we rely on Recall@10.
- λ is not dimensionless across base models, so its absolute value is not cross-setting comparable; we
  report λ*↔SE only as a trend.
- Four settings, two base architectures; broader generality (MIND-large scale, cross-lingual Adressa,
  Yelp) is future work.

## 7. Conclusion
A frozen generative retriever's overconfident beam scores leave a large, recoverable ranking gap. A
small post-hoc scorer with a leakage-free held-out λ-selection closes it across diverse base
generators, domains, and code-lengths — a model-agnostic, retrain-free improvement.

---

## Citations to verify (NONE are confirmed — fetch BibTeX programmatically before compiling)
| key | what we believe it is | must verify |
|---|---|---|
| tiger | Rajput et al., "Recommender Systems with Generative Retrieval" (TIGER), NeurIPS 2023 | title/authors/year/venue |
| gram | GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion, ACL 2025 (arXiv 2506.01673) | verified to exist via web (2026-06); confirm BibTeX |
| idgenrec | IDGenRec (LLM generative rec with textual IDs) | exists? authors/year |
| lcrec, letter | LC-Rec / LETTER (learned item identifiers) | exists? |
| sasrec | Kang & McAuley, "Self-Attentive Sequential Recommendation" (SASRec), ICDM 2018 | confirm |
| nrms | Wu et al., "Neural News Recommendation with Multi-Head Self-Attention" (NRMS), EMNLP 2019 | confirm |
| focal_loss | Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017 | confirm |
| listmle | Xia et al., "Listwise Approach to Learning to Rank" (ListMLE), ICML 2008 | confirm |
| approxndcg | Qin et al., ApproxNDCG | confirm exact ref |
| guo_calibration | Guo et al., "On Calibration of Modern Neural Networks", ICML 2017 | confirm |

## Open framing questions (flagged, not blocking)
1. **Venue** (affects template/length/required sections): project says CCF-B — which exactly (RecSys /
   CIKM / ECIR / WWW-short)? Determines page limit + whether a Reproducibility/Ethics section is required.
2. **Title** — placeholder; finalize once framing confirmed.
3. **Figure 1** — currently the cascade two-panel; consider a method/teaser schematic as Fig. 1 instead
   (reviewers read Fig. 1 first).
