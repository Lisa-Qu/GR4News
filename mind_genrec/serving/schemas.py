"""Pydantic schemas for the MIND generative recommendation API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Single recommendation query."""

    user_id: str | None = None
    history: list[str] = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class BatchRecommendationRequest(BaseModel):
    """Batch recommendation query."""

    requests: list[RecommendationRequest] = Field(..., min_length=1)


class RecommendationItem(BaseModel):
    """One recommended item returned by the API."""

    news_id: str
    score: float
    semantic_id: list[int] | None = None
    category: str | None = None
    subcategory: str | None = None
    title: str | None = None


class RecommendationResponse(BaseModel):
    """Single recommendation response."""

    request_id: str
    model_name: str
    latency_ms: float
    cache_hit: bool = False
    served_by_placeholder: bool = False
    warnings: list[str] = Field(default_factory=list)
    items: list[RecommendationItem]


class BatchRecommendationResponse(BaseModel):
    """Batch recommendation response."""

    request_count: int
    responses: list[RecommendationResponse]


class HealthResponse(BaseModel):
    """Health payload."""

    status: str
    ready: bool
    service_ready: bool = False
    generator_ready: bool = False
    baseline_ready: bool = False
    uses_placeholder_components: bool
    model_name: str
    catalog_size: int
    semantic_mapping_loaded: bool = False
    semantic_unique_code_count: int = 0
    semantic_collided_code_count: int = 0
    missing_components: list[str] = Field(default_factory=list)
