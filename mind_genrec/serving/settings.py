"""Runtime settings for the MIND serving stack."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServingSettings:
    """Environment-backed runtime settings for FastAPI serving."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    cache_ttl_seconds: int = 600
    redis_url: str | None = None
    news_jsonl_path: str | None = None
    semantic_artifact_dir: str | None = None
    generator_checkpoint_path: str | None = None
    baseline_checkpoint_path: str | None = None
    device: str | None = None

    @classmethod
    def from_env(cls) -> "ServingSettings":
        """Build settings from environment variables."""

        return cls(
            host=os.getenv("MIND_GENREC_HOST", "0.0.0.0"),
            port=int(os.getenv("MIND_GENREC_PORT", "8000")),
            reload=_read_bool("MIND_GENREC_RELOAD", False),
            log_level=os.getenv("MIND_GENREC_LOG_LEVEL", "info").lower(),
            cache_ttl_seconds=int(os.getenv("MIND_GENREC_CACHE_TTL_SECONDS", "600")),
            redis_url=os.getenv("MIND_GENREC_REDIS_URL"),
            news_jsonl_path=os.getenv("MIND_GENREC_NEWS_JSONL"),
            semantic_artifact_dir=os.getenv("MIND_GENREC_SEMANTIC_DIR"),
            generator_checkpoint_path=os.getenv("MIND_GENREC_GENERATOR_CKPT"),
            baseline_checkpoint_path=os.getenv("MIND_GENREC_BASELINE_CKPT"),
            device=os.getenv("MIND_GENREC_DEVICE"),
        )

    def model_registry_kwargs(self) -> dict[str, str | None]:
        """Return ModelRegistry kwargs derived from settings."""

        return {
            "news_jsonl_path": self.news_jsonl_path,
            "semantic_artifact_dir": self.semantic_artifact_dir,
            "generator_checkpoint_path": self.generator_checkpoint_path,
            "baseline_checkpoint_path": self.baseline_checkpoint_path,
            "device": self.device,
        }

    def cache_kwargs(self) -> dict[str, int | str | None]:
        """Return cache construction kwargs derived from settings."""

        return {
            "ttl_seconds": self.cache_ttl_seconds,
            "redis_url": self.redis_url,
        }

    def uvicorn_kwargs(self) -> dict[str, object]:
        """Return Uvicorn runtime kwargs."""

        return {
            "host": self.host,
            "port": self.port,
            "reload": self.reload,
            "log_level": self.log_level,
        }
