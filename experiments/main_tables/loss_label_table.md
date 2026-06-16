### Loss × Label diagnostic (MIND, best held-out R@1)

| loss \ label | binary | soft |
|---|---|---|
| bce | **0.0344** | 0.0332 |
| listmle | 0.0323 | 0.0323 |
| approx_ndcg | 0.0185 | 0.0249 |

Caption: loss×label diagnostic on MIND (this script's own beam collection + 7-pt λ grid; relative loss×label comparison only — the headline uses held-out λ + a dense grid). Each cell = best held-out R@1 across the λ sweep; **bold** = grid max. BCE-binary is the locked headline choice.
