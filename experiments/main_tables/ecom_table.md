### E-commerce (Beauty/Sports/Toys, LOO, full-catalog)

| Method | Beauty R@10 | Sports R@10 | Toys R@10 |
|---|---|---|---|
| SASRec (full-catalog) | 0.0133 | 0.0104 | 0.0112 |
| TIGER-equiv (vanilla) | 0.0881 | 0.0561 | 0.0927 |
| +Listwise (5-seed) | 0.0944 | 0.0606 | 0.0979 |
| Oracle | 0.1731 | 0.1207 | 0.1662 |

Footnote: generative rows (TIGER-equiv/Focal/Listwise/Oracle) are beam-recall-bounded (rank within the K=50 beam); NRMS/SASRec rank the full catalog. Absolute R@K is not directly comparable across the two; ** p<0.01, * p<0.05 (McNemar vs vanilla).
