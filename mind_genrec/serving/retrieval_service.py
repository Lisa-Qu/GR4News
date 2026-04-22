"""Backend service orchestration for generative recommendation requests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from mind_genrec.serving.cache import CacheBackend, RedisCache, ResilientCache, TTLCache
from mind_genrec.serving.model_registry import ModelRegistry
from mind_genrec.serving.schemas import (
    BatchRecommendationRequest,
    BatchRecommendationResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)


_LATENCY_BUCKETS_MS = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0)


class _LatencyHistogram:
    """Minimal Prometheus-compatible latency histogram (no external dependency)."""

    def __init__(self, buckets: tuple[float, ...] = _LATENCY_BUCKETS_MS) -> None:
        self._buckets = buckets
        self._counts = [0] * len(buckets)
        self._sum = 0.0
        self._total = 0

    def observe(self, value_ms: float) -> None:
        self._sum += value_ms
        self._total += 1
        for i, upper in enumerate(self._buckets):
            if value_ms <= upper:
                self._counts[i] += 1
                return  # only the first matching bucket

    def prometheus_lines(self, name: str) -> list[str]:
        lines = [
            f"# HELP {name} Request latency in milliseconds.",
            f"# TYPE {name} histogram",
        ]
        cumulative = 0
        for i, upper in enumerate(self._buckets):
            cumulative += self._counts[i]
            lines.append(f'{name}_bucket{{le="{upper:.0f}"}} {cumulative}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {self._total}')
        lines.append(f"{name}_sum {self._sum:.3f}")
        lines.append(f"{name}_count {self._total}")
        return lines


@dataclass
class _ServiceMetrics:
    requests_total: int = 0
    batch_requests_total: int = 0
    cache_hits_total: int = 0
    latency_histogram: _LatencyHistogram = field(default_factory=_LatencyHistogram)


@dataclass(frozen=True)
class _CachedRecommendationPayload:
    """Cache payload without request-scoped metadata."""

    model_name: str
    served_by_placeholder: bool
    warnings: tuple[str, ...]
    items: tuple[RecommendationItem, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "model_name": self.model_name,
                "served_by_placeholder": self.served_by_placeholder,
                "warnings": list(self.warnings),
                "items": [item.model_dump() for item in self.items],
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> "_CachedRecommendationPayload":
        raw = json.loads(payload)
        return cls(
            model_name=raw["model_name"],
            served_by_placeholder=bool(raw["served_by_placeholder"]),
            warnings=tuple(raw.get("warnings", [])),
            items=tuple(RecommendationItem(**item) for item in raw.get("items", [])),
        )


@dataclass(frozen=True)
class _SelectedRetriever:
    """One runtime retrieval path chosen for the request."""

    model_name: str
    served_by_placeholder: bool
    mode: str
    retriever: object


class RetrievalService:
    """High-level request handler for the FastAPI layer."""

    def __init__(
        self,
        registry: ModelRegistry,
        cache: CacheBackend[_CachedRecommendationPayload] | None = None,
    ) -> None:
        self._registry = registry
        self._cache = cache
        self._metrics = _ServiceMetrics()

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        started_at = perf_counter()
        bundle = self._registry.get_active_bundle()
        selected = self._select_retriever(bundle)
        cache_key = self._build_cache_key(
            request,
            model_name=selected.model_name,
            runtime_signature=bundle.runtime_signature,
        )
        cache_hit = False
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._metrics.requests_total += 1
                self._metrics.cache_hits_total += 1
                cache_hit = True
                return RecommendationResponse(
                    request_id=str(uuid4()),
                    model_name=cached.model_name,
                    latency_ms=(perf_counter() - started_at) * 1000.0,
                    cache_hit=cache_hit,
                    served_by_placeholder=cached.served_by_placeholder,
                    warnings=list(cached.warnings),
                    items=list(cached.items),
                )

        payload = self._build_payload(request, bundle=bundle, selected=selected)
        response = RecommendationResponse(
            request_id=str(uuid4()),
            model_name=payload.model_name,
            latency_ms=(perf_counter() - started_at) * 1000.0,
            cache_hit=cache_hit,
            served_by_placeholder=payload.served_by_placeholder,
            warnings=list(payload.warnings),
            items=list(payload.items),
        )

        self._metrics.requests_total += 1
        self._metrics.latency_histogram.observe(response.latency_ms)
        if self._cache is not None:
            self._cache.set(cache_key, payload)
        return response

    def recommend_batch(
        self,
        request: BatchRecommendationRequest,
    ) -> BatchRecommendationResponse:
        self._metrics.batch_requests_total += 1
        responses = [self.recommend(single_request) for single_request in request.requests]
        return BatchRecommendationResponse(
            request_count=len(request.requests),
            responses=responses,
        )

    def health_snapshot(
        self,
    ) -> tuple[str, bool, bool, bool, bool, str, int, bool, int, int, list[str]]:
        bundle = self._registry.get_active_bundle()
        selected = self._select_retriever(bundle)
        if bundle.generator_ready:
            status = "ok"
        elif bundle.service_ready:
            status = "degraded"
        else:
            status = "unavailable"
        return (
            status,
            bundle.service_ready,
            bundle.generator_ready,
            bundle.baseline_ready,
            bundle.uses_placeholder_components,
            selected.model_name,
            bundle.catalog_size,
            bundle.semantic_mapping_loaded,
            bundle.semantic_unique_code_count,
            bundle.semantic_collided_code_count,
            bundle.missing_components(),
        )

    def metrics_text(self) -> str:
        lines = [
            "# HELP mind_genrec_requests_total Total recommendation requests.",
            "# TYPE mind_genrec_requests_total counter",
            f"mind_genrec_requests_total {self._metrics.requests_total}",
            "# HELP mind_genrec_batch_requests_total Total batch requests.",
            "# TYPE mind_genrec_batch_requests_total counter",
            f"mind_genrec_batch_requests_total {self._metrics.batch_requests_total}",
            "# HELP mind_genrec_cache_hits_total Total cache hits.",
            "# TYPE mind_genrec_cache_hits_total counter",
            f"mind_genrec_cache_hits_total {self._metrics.cache_hits_total}",
        ]
        lines += self._metrics.latency_histogram.prometheus_lines(
            "mind_genrec_request_latency_ms"
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _build_cache_key(
        request: RecommendationRequest,
        *,
        model_name: str,
        runtime_signature: str,
    ) -> str:
        payload = f"{runtime_signature}:{model_name}:" + "|".join(request.history) + f":{request.top_k}"
        if request.user_id:
            payload = f"{request.user_id}:{payload}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_payload(
        self,
        request: RecommendationRequest,
        *,
        bundle,
        selected: _SelectedRetriever,
    ) -> _CachedRecommendationPayload:
        """Execute the underlying recommendation path without request metadata."""
        warnings = self._build_runtime_warnings(bundle)

        items: list[RecommendationItem] = []
        if selected.mode == "generator":
            candidates = selected.retriever.recommend(request.history, request.top_k)
            for candidate in candidates:
                record = bundle.catalog.get_item(candidate.news_id)
                items.append(
                    RecommendationItem(
                        news_id=candidate.news_id,
                        score=candidate.score,
                        semantic_id=candidate.semantic_id,
                        category=record.category if record else None,
                        subcategory=record.subcategory if record else None,
                        title=record.title if record else None,
                    )
                )
        else:
            candidates = selected.retriever.retrieve(request.history, request.top_k)
            for candidate in candidates:
                record = bundle.catalog.get_item(candidate.news_id)
                items.append(
                    RecommendationItem(
                        news_id=candidate.news_id,
                        score=candidate.score,
                        semantic_id=None,
                        category=record.category if record else None,
                        subcategory=record.subcategory if record else None,
                        title=record.title if record else None,
                    )
                )

        return _CachedRecommendationPayload(
            model_name=selected.model_name,
            served_by_placeholder=selected.served_by_placeholder,
            warnings=tuple(warnings),
            items=tuple(items),
        )

    @staticmethod
    def _select_retriever(bundle) -> _SelectedRetriever:
        """Choose the retriever used for `/recommend`."""

        if not bool(getattr(bundle.generator, "is_placeholder", False)):
            return _SelectedRetriever(
                model_name=bundle.generator.model_name,
                served_by_placeholder=False,
                mode="generator",
                retriever=bundle.generator,
            )
        if bundle.baseline is not None and not bool(getattr(bundle.baseline, "is_placeholder", False)):
            return _SelectedRetriever(
                model_name=bundle.baseline.model_name,
                served_by_placeholder=False,
                mode="baseline",
                retriever=bundle.baseline,
            )
        return _SelectedRetriever(
            model_name=bundle.generator.model_name,
            served_by_placeholder=True,
            mode="generator",
            retriever=bundle.generator,
        )

    @staticmethod
    def _build_runtime_warnings(bundle) -> list[str]:
        """Return user-visible warnings when placeholder components are active."""

        warnings: list[str] = []
        if getattr(bundle.generator, "is_placeholder", False):
            warnings.append(
                "The active generator is still a placeholder. Recommendations are not model-backed yet."
            )
        if bundle.catalog_size == 0:
            warnings.append(
                "No MIND item catalog is loaded yet. Item metadata and real candidate resolution are unavailable."
            )
        if not bundle.semantic_mapping_loaded:
            warnings.append(
                "Semantic ID mappings are not loaded yet. The semantic retrieval path is not active."
            )
        if bundle.baseline is None or getattr(bundle.baseline, "is_placeholder", False):
            warnings.append(
                "The baseline retriever is still a placeholder and is not participating in recommendation serving."
            )
        return warnings


def build_recommendation_cache(
    *,
    ttl_seconds: int,
    redis_url: str | None = None,
) -> CacheBackend[_CachedRecommendationPayload]:
    """Create the serving cache backend.

    Redis is used only when a URL is provided; otherwise the in-memory cache is
    kept as the default development fallback.
    """

    fallback = TTLCache[_CachedRecommendationPayload](ttl_seconds=ttl_seconds)
    if redis_url:
        return ResilientCache(
            RedisCache(
                redis_url=redis_url,
                ttl_seconds=ttl_seconds,
                serializer=lambda payload: payload.to_json(),
                deserializer=_CachedRecommendationPayload.from_json,
                namespace="mind_genrec:recommendation",
            ),
            fallback=fallback,
        )
    return fallback
