### News (MIND-small, user-split, full-catalog)

| Method | R@1 | R@10 | R@50 |
|---|---|---|---|
| NRMS (full-catalog) | 0.0015 | 0.0174 | 0.0610 |
| TIGER-equiv (vanilla) | 0.0331 | 0.1692 | 0.3671 |
| +Pointwise Focal | 0.0345 (+4.1%) | 0.1710 | — |
| +Listwise (5-seed)** | 0.0352 (+6.3%) | 0.1723 | — |
| Oracle | 0.3671 | 0.3671 | 0.3671 |

Footnote: generative rows (TIGER-equiv/Focal/Listwise/Oracle) are beam-recall-bounded (rank within the K=50 beam); NRMS/SASRec rank the full catalog. Absolute R@K is not directly comparable across the two; ** p<0.01, * p<0.05 (McNemar vs vanilla). (5/5 seeds p<0.05 per-seed McNemar)
