"""Optional MLflow logging for pipeline runs."""

from __future__ import annotations

from contextlib import AbstractContextManager
import json
from pathlib import Path
from typing import Any


def _flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten one nested dict into scalar-like MLflow param keys."""

    flat: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=full_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[full_key] = value
        else:
            flat[full_key] = json.dumps(value, ensure_ascii=False)
    return flat


def _sanitize_metric_key(key: str) -> str:
    """Replace characters MLflow rejects in metric names (@) with safe alternatives."""
    return key.replace("@", "_at_")


def _flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested metrics and keep numeric values only."""

    flat: dict[str, float] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, prefix=full_key))
        elif isinstance(value, bool):
            flat[full_key] = float(value)
        elif isinstance(value, (int, float)):
            flat[full_key] = float(value)
    return flat


class MlflowRunLogger(AbstractContextManager["MlflowRunLogger"]):
    """Small optional wrapper around MLflow.

    When disabled, every method becomes a no-op.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        tracking_uri: str | None = None,
        experiment_name: str = "mind_genrec",
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self._enabled = enabled
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tags = tags or {}
        self._mlflow = None
        self._active_run = None

        if self._enabled:
            try:
                import mlflow
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "MLflow logging was enabled, but the 'mlflow' package is not installed."
                ) from exc
            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)
            self._mlflow = mlflow

    @classmethod
    def from_config(
        cls,
        *,
        tracking_config: dict[str, Any] | None,
        default_experiment_name: str,
        default_run_name: str | None,
        tags: dict[str, str] | None = None,
    ) -> "MlflowRunLogger":
        """Build one logger from pipeline config."""

        tracking = tracking_config or {}
        return cls(
            enabled=bool(tracking.get("use_mlflow", False)),
            tracking_uri=tracking.get("tracking_uri"),
            experiment_name=str(tracking.get("experiment_name", default_experiment_name)),
            run_name=tracking.get("run_name") or default_run_name,
            tags=tags,
        )

    def __enter__(self) -> "MlflowRunLogger":
        if self._enabled and self._mlflow is not None:
            self._active_run = self._mlflow.start_run(run_name=self._run_name)
            if self._tags:
                self._mlflow.set_tags(self._tags)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._enabled and self._mlflow is not None and self._active_run is not None:
            self._mlflow.end_run(status="FAILED" if exc is not None else "FINISHED")
            self._active_run = None
        return None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log_params(self, payload: dict[str, Any], prefix: str = "") -> None:
        if not self._enabled or self._mlflow is None:
            return
        flat = _flatten_dict(payload, prefix=prefix)
        for key, value in flat.items():
            self._mlflow.log_param(key, value)

    def log_metrics(self, payload: dict[str, Any], prefix: str = "", step: int | None = None) -> None:
        if not self._enabled or self._mlflow is None:
            return
        flat = _flatten_metrics(payload, prefix=prefix)
        sanitized = {_sanitize_metric_key(k): v for k, v in flat.items()}
        if sanitized:
            self._mlflow.log_metrics(sanitized, step=step)

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        if not self._enabled or self._mlflow is None:
            return
        self._mlflow.log_dict(payload, artifact_file)

    def log_text(self, text: str, artifact_file: str) -> None:
        if not self._enabled or self._mlflow is None:
            return
        self._mlflow.log_text(text, artifact_file)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if not self._enabled or self._mlflow is None:
            return
        self._mlflow.log_artifact(str(Path(path)), artifact_path=artifact_path)
