# mind_genrec

An isolated project path for `MIND-small` first and `MIND-large` later.

## Current Status

- this package is separate from the legacy GRAM code path
- the project now has working `semantic ID`, `AR generator`, `beam search`, `two-tower baseline`, and offline evaluation paths
- the generator path now supports both `AR` and `LazyAR` decoder choices
- serving can already load runtime artifacts, a trained generator checkpoint, and a trained baseline checkpoint
- no existing project source files were modified

## What Is Already Implemented

- `serving/`
  - FastAPI routes
  - request and response schemas
  - request orchestration
  - registry and cache plumbing
  - optional Redis-backed cache
- `data/`
  - MIND record dataclasses
  - raw `news.tsv` parsing helpers
  - raw `behaviors.tsv` parsing helpers
  - normalized training-sample builders
- `model/`
  - generator interface
  - hashing item encoder
  - residual quantizer
  - semantic ID mapper
  - user encoder
  - AR decoder
  - LazyAR decoder
  - beam search
- `training/`
  - semantic ID training and export entry point
  - generator training entry point
  - two-tower baseline training entry point
- `evaluation/`
  - offline generator evaluation
  - offline baseline evaluation
  - generator-vs-baseline comparison report
  - ranking metrics
  - explicit `full_corpus_retrieval` evaluation mode
- `baseline/`
  - exact-search vector index
  - two-tower baseline model
  - checkpoint-backed baseline retriever

## What Is Not Implemented Yet

- FAISS or another production ANN backend
- final experiment runner over real MIND-small / MIND-large
- candidate-set ranking evaluation for standard MIND benchmark reporting

## Testing

- unit-style smoke tests live under `mind_genrec/tests/`
- the current minimal suite covers:
  - evaluation-summary comparison
  - toy end-to-end pipeline
  - baseline-only serving fallback
  - LazyAR training + checkpoint loading smoke path

## Placeholder Policy

The current placeholder components are intentionally conservative:

- `/health` reports `degraded` until real runtime artifacts are attached
- `/health` now separates `service_ready`, `generator_ready`, and `baseline_ready`
- `/health` surfaces whether `semantic ID artifacts` are actually loaded
- `/recommend` may return empty results with warnings instead of plausible fake recommendations
- cache keys now include runtime model identity, so switching checkpoints does not reuse stale results

## Redis Cache

The serving path now supports two cache modes:

- default: in-memory TTL cache
- optional: Redis cache when `MIND_GENREC_REDIS_URL` is set

Relevant environment variables:

- `MIND_GENREC_REDIS_URL`
- `MIND_GENREC_CACHE_TTL_SECONDS`
- `MIND_GENREC_NEWS_JSONL`
- `MIND_GENREC_SEMANTIC_DIR`
- `MIND_GENREC_GENERATOR_CKPT`
- `MIND_GENREC_BASELINE_CKPT`
- `MIND_GENREC_DEVICE`

## Backend Quickstart

The serving stack now has an explicit runtime entrypoint:

```bash
python -m mind_genrec.serving.run_server \
  --news-jsonl /abs/path/to/news.jsonl \
  --semantic-dir /abs/path/to/semantic_ids \
  --generator-ckpt /abs/path/to/best_generator.pt \
  --baseline-ckpt /abs/path/to/best_baseline.pt \
  --device cpu
```

The same runtime can also be configured fully by environment variables.
See:

- `mind_genrec/.env.example`
- `mind_genrec/serving/settings.py`

Core serving variables:

- `MIND_GENREC_HOST`
- `MIND_GENREC_PORT`
- `MIND_GENREC_RELOAD`
- `MIND_GENREC_LOG_LEVEL`
- `MIND_GENREC_NEWS_JSONL`
- `MIND_GENREC_SEMANTIC_DIR`
- `MIND_GENREC_GENERATOR_CKPT`
- `MIND_GENREC_BASELINE_CKPT`
- `MIND_GENREC_REDIS_URL`
- `MIND_GENREC_CACHE_TTL_SECONDS`
- `MIND_GENREC_DEVICE`

## Docker Skeleton

The repo now includes:

- `mind_genrec/docker/Dockerfile`
- `mind_genrec/docker/docker-compose.yml`

This is a first deployment skeleton for:

- FastAPI app container
- Redis container

Before using `docker compose`, update:

- artifact mount paths
- env file values

so they point to the run directory you actually want to serve.

## Next Planned Work

1. run the full experiment on real `MIND-small`
2. compare `generator` and `two-tower baseline` on held-out `test.jsonl`
3. decide whether to replace the current exact baseline index with `FAISS`
4. decide whether to add candidate-set ranking evaluation alongside open retrieval
