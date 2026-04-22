"""FastAPI app for the isolated MIND backend-first serving stack."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from mind_genrec.serving.model_registry import ModelRegistry
from mind_genrec.serving.retrieval_service import RetrievalService, build_recommendation_cache
from mind_genrec.serving.schemas import (
    BatchRecommendationRequest,
    BatchRecommendationResponse,
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from mind_genrec.serving.settings import ServingSettings


def create_app(
    *,
    settings: ServingSettings | None = None,
    registry: ModelRegistry | None = None,
    service: RetrievalService | None = None,
) -> FastAPI:
    """Create the FastAPI application with explicit runtime settings."""

    resolved_settings = settings or ServingSettings.from_env()
    resolved_registry = registry or ModelRegistry(**resolved_settings.model_registry_kwargs())
    resolved_service = service or RetrievalService(
        registry=resolved_registry,
        cache=build_recommendation_cache(**resolved_settings.cache_kwargs()),
    )

    app = FastAPI(title="mind_genrec", version="0.1.0")
    app.state.settings = resolved_settings
    app.state.registry = resolved_registry
    app.state.retrieval_service = resolved_service

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        (
            status,
            ready,
            generator_ready,
            baseline_ready,
            uses_placeholder_components,
            model_name,
            catalog_size,
            semantic_mapping_loaded,
            semantic_unique_code_count,
            semantic_collided_code_count,
            missing_components,
        ) = (
            resolved_service.health_snapshot()
        )
        return HealthResponse(
            status=status,
            ready=ready,
            service_ready=ready,
            generator_ready=generator_ready,
            baseline_ready=baseline_ready,
            uses_placeholder_components=uses_placeholder_components,
            model_name=model_name,
            catalog_size=catalog_size,
            semantic_mapping_loaded=semantic_mapping_loaded,
            semantic_unique_code_count=semantic_unique_code_count,
            semantic_collided_code_count=semantic_collided_code_count,
            missing_components=missing_components,
        )

    @app.post("/recommend", response_model=RecommendationResponse)
    def recommend(request: RecommendationRequest) -> RecommendationResponse:
        return resolved_service.recommend(request)

    @app.post("/recommend/batch", response_model=BatchRecommendationResponse)
    def recommend_batch(
        request: BatchRecommendationRequest,
    ) -> BatchRecommendationResponse:
        return resolved_service.recommend_batch(request)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(resolved_service.metrics_text())

    return app


app = create_app()
