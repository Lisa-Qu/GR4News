# GR4AD-Lite: Generative Ad Recommendation Service

## Dataset

**Criteo Attribution Modeling for Bidding Dataset**
- 30 days Criteo production traffic, 16.5M impressions
- Fields: click (binary), conversion (binary), conversion_value, cost, timestamp, 9 categorical features
- eCPM derivation: eCPM = pCTR x pCVR x conversion_value x 1000
- Source: https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/

Backup: CriteoPrivateAd (2025, HuggingFace) — larger, with bid features

---

## Phase 1: Semantic ID (RQ-VAE)

| Item | Detail |
|------|--------|
| Goal | Encode each ad into 4-digit semantic token sequence |
| Method | Residual Quantization VAE, codebook size 256 per level, 4 levels |
| Input | Ad categorical features + text embeddings (Sentence-BERT) |
| Output | ad_id -> [c1, c2, c3, c4], vocabulary size 256x4=1024 |
| Reference code | [snap-research/GRID](https://github.com/snap-research/GRID) RQ-VAE module, [EdoardoBotta/RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) |
| Framework | PyTorch, trained on single GPU |

---

## Phase 2: Encoder-Decoder + Dual Objective

| Item | Detail |
|------|--------|
| Encoder | 6-layer Transformer, input = user click history (sequence of semantic IDs) |
| Decoder | **LazyAR**: 9 layers total, first 6 layers shared across beams (non-AR), last 3 layers per-beam autoregressive |
| Head 1 | Next-token prediction: CrossEntropy over 256-dim codebook at each of 4 positions |
| Head 2 | eCPM regression: linear head on decoder hidden state -> scalar eCPM prediction |
| Training | **VSL (Value-aware Supervised Learning)**: joint loss = CE_relevance + lambda * MSE_ecpm, lambda=0.1 |
| Inference | Beam search (beam=32), score = log P(token sequence) + alpha * predicted_eCPM, alpha tuned on validation |
| Reference | GR4AD paper Section 3-4, [XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER) for encoder-decoder base |

---

## Phase 3: Serving Backend

```
FastAPI
├── POST /recommend          # input: user_id -> output: top-K ad semantic IDs + eCPM scores
├── POST /recommend/batch    # batch inference
├── GET  /health
├── GET  /metrics            # Prometheus format
└── POST /ab-test/assign     # A/B group assignment (hash-based)

Infrastructure:
├── BentoML                  # model packaging + adaptive batching
├── Redis                    # user embedding cache, hot ad cache, TTL=10min
├── MLflow                   # experiment tracking (VQ loss, CE loss, eCPM MSE, recall@K)
├── Prometheus + Grafana     # latency p50/p99, QPS, cache hit rate, eCPM distribution
└── Docker Compose           # one-command startup: api + redis + prometheus + grafana
```

Latency target: <100ms p99 (single request), 500+ QPS per GPU

---

## Phase 4: A/B Test Simulation

- Baseline: two-tower retrieval + FAISS ANN (standard approach)
- Treatment: GR4AD beam search generation
- Metrics: Recall@50, NDCG@10, eCPM lift, latency comparison
- Output: comparison table + matplotlib charts in README

---

## Phase 5: Polish

- Architecture diagram (draw.io)
- Benchmark results table in README
- Streamlit demo: input user_id -> show recommended ads with eCPM scores, beam search visualization
- GitHub Actions CI: lint + unit tests + model smoke test

---

## Tech Stack

| Layer | Tool |
|-------|------|
| ML | PyTorch, Sentence-BERT |
| Semantic ID | RQ-VAE (custom, ref GRID) |
| Model | Transformer Encoder + LazyAR Decoder |
| Serving | FastAPI + BentoML |
| Cache | Redis |
| Tracking | MLflow |
| Monitoring | Prometheus + Grafana |
| Container | Docker Compose |
| CI | GitHub Actions |
| Demo | Streamlit |
| Baseline | FAISS (two-tower comparison) |

---

## File Structure

```
GR4AD/
├── data/                    # data download & preprocessing scripts
├── model/
│   ├── rq_vae.py           # RQ-VAE for semantic ID generation
│   ├── encoder.py          # Transformer encoder (user history)
│   ├── lazy_ar_decoder.py  # LazyAR decoder with shared layers
│   ├── vsl_loss.py         # Value-aware dual loss function
│   └── beam_search.py      # Value-aware beam search
├── serving/
│   ├── app.py              # FastAPI endpoints
│   ├── bentofile.yaml      # BentoML service config
│   └── cache.py            # Redis caching layer
├── baseline/
│   └── two_tower.py        # Two-tower + FAISS baseline
├── experiments/
│   └── ab_simulation.py    # A/B test simulation
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## References

- GR4AD paper: https://arxiv.org/abs/2602.22732
- TIGER: https://arxiv.org/abs/2305.05065
- GRID toolkit: https://github.com/snap-research/GRID
- RQ-VAE-Recommender: https://github.com/EdoardoBotta/RQ-VAE-Recommender
- Criteo Attribution Dataset: https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/
- OneRec (Kuaishou): https://github.com/Kuaishou-OneRec/OpenOneRec


---
---

# Resume Bullets (3 points)

> **Generative Ad Recommendation System (GR4AD-Lite)** | PyTorch, FastAPI, Redis, Docker, BentoML, Prometheus
>
> - Replicated GR4AD (KDD 2025) generative recommendation pipeline: implemented **RQ-VAE** for 4-level semantic ad tokenization and **LazyAR decoder** (shared-layer autoregressive Transformer) with **value-aware beam search** jointly optimizing relevance and eCPM on Criteo ad dataset
> - Built production-grade serving backend with **FastAPI + BentoML** (adaptive batching, <100ms p99 latency), **Redis** caching layer for user/ad embeddings, and simulated **A/B testing framework** comparing generative retrieval vs. two-tower + FAISS baseline
> - Deployed full **MLOps stack** with Docker Compose (one-command startup), **MLflow** experiment tracking, **Prometheus + Grafana** real-time monitoring (QPS, latency percentiles, cache hit rate), and GitHub Actions CI/CD pipeline

Keywords hit: generative recommendation, semantic ID, autoregressive decoding, beam search, eCPM, multi-objective optimization, real-time serving, feature caching, A/B testing, MLOps, model serving, latency optimization, Docker, monitoring
