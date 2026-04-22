"""CLI entry point for running the MIND FastAPI service."""

from __future__ import annotations

import argparse

import uvicorn

from mind_genrec.serving.app import create_app
from mind_genrec.serving.settings import ServingSettings


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for serving startup."""

    defaults = ServingSettings.from_env()
    parser = argparse.ArgumentParser(description="Run the mind_genrec FastAPI service.")
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--reload", action="store_true", default=defaults.reload)
    parser.add_argument("--log-level", default=defaults.log_level)
    parser.add_argument("--cache-ttl-seconds", type=int, default=defaults.cache_ttl_seconds)
    parser.add_argument("--redis-url", default=defaults.redis_url)
    parser.add_argument("--news-jsonl", default=defaults.news_jsonl_path)
    parser.add_argument("--semantic-dir", default=defaults.semantic_artifact_dir)
    parser.add_argument("--generator-ckpt", default=defaults.generator_checkpoint_path)
    parser.add_argument("--baseline-ckpt", default=defaults.baseline_checkpoint_path)
    parser.add_argument("--device", default=defaults.device)
    return parser


def main() -> None:
    """Launch the Uvicorn server with explicit runtime settings."""

    args = build_parser().parse_args()
    settings = ServingSettings(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        cache_ttl_seconds=args.cache_ttl_seconds,
        redis_url=args.redis_url,
        news_jsonl_path=args.news_jsonl,
        semantic_artifact_dir=args.semantic_dir,
        generator_checkpoint_path=args.generator_ckpt,
        baseline_checkpoint_path=args.baseline_ckpt,
        device=args.device,
    )
    app = create_app(settings=settings)
    uvicorn.run(app, **settings.uvicorn_kwargs())


if __name__ == "__main__":
    main()
