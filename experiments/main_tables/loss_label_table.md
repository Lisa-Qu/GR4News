### Loss × Label diagnostic (MIND, baseline_retrain, best val-selected R@1)

| loss \ label | binary | soft |
|---|---|---|
| bce | **0.0381** | 0.0351 |
| listmle | 0.0337 | 0.0337 |
| approx_ndcg | 0.0252 | 0.0155 |

Caption: loss×label diagnostic on MIND, baseline_retrain checkpoint (run_listwise_scorer's own beam collection + 7-pt λ grid + val-selection — NOT the headline held-out procedure/dense grid; relative loss×label comparison only). Each cell = best R@1 across the λ sweep (including the scorer-only 'pure' config); **bold** = grid max. BCE-binary is the locked headline choice.
