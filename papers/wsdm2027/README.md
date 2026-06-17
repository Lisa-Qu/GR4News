# GR4AD — WSDM 2027 submission source

ACM `acmart` (sigconf) double-blind review draft.

## Files
- `main.tex` — full paper (acmart sigconf, `anonymous,review`). Title is a placeholder.
- `references.bib` — 11 citations, each web-verified (existence + canonical fields, 2026-06-17).
- `figures/{confidence_cascade,ece_reliability,lambda_sensitivity}.pdf` — vector figures (from `../../figures/`).

## Status (honest)
- **NOT yet compiled.** Local TeX Live is `2025basic` and lacks `acmart.cls`; installing it needs
  admin (`tlmgr` cannot write the system tree) — not done.
- **Verified without compiling:** all 11 `\cite{}` keys exactly match the keys in `references.bib`
  (no undefined citations); all three `\includegraphics` figures are present in `figures/`.
- Every experimental number traces to `../../experiments/*/results.json|significance.json`
  (see `../../docs/RESULTS_SUMMARY.md`).

## Compile
**Overleaf (recommended, acmart pre-installed):** upload this directory, set the main document to
`main.tex`, compile (pdfLaTeX). Overleaf bundles `acmart.cls` + `ACM-Reference-Format.bst`.

**Local (needs admin once):**
```bash
sudo tlmgr install acmart   # or use full TeX Live
latexmk -pdf main.tex
```

## Before submission (TODO)
- Confirm WSDM 2027 deadline on the official site (`wsdm-conference.org/2027`); trackers list
  full-paper **~Aug 24, 2026** (abstract ~Aug 17) — verify.
- Re-fetch each `references.bib` entry's BibTeX programmatically (CrossRef DOI / arXiv / DBLP) to lock
  author spelling, page numbers, DOIs (current fields are web-verified but hand-assembled).
- Finalize title; consider a method/teaser schematic as Fig. 1 (reviewers read Fig. 1 first) — the
  cascade figure could move to §3.
- Add the GenAI-usage / reproducibility statements if WSDM 2027 requires them.
- WSDM page limit: confirm (recent WSDM long papers ~9–10 pp ACM 2-col incl. references) and trim/expand.
