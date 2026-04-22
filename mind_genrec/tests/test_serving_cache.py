"""Smoke tests for serving cache backends."""

from __future__ import annotations

import unittest

from mind_genrec.serving.cache import TTLCache
from mind_genrec.serving.retrieval_service import _CachedRecommendationPayload, build_recommendation_cache
from mind_genrec.serving.schemas import RecommendationItem


class ServingCacheTest(unittest.TestCase):
    def test_cached_payload_json_roundtrip(self) -> None:
        payload = _CachedRecommendationPayload(
            model_name="lazy-ar-semantic-generator-beam",
            served_by_placeholder=False,
            warnings=("semantic mapping loaded",),
            items=(
                RecommendationItem(
                    news_id="N1",
                    score=0.9,
                    semantic_id=[1, 2, 3, 4],
                    category="news",
                    subcategory="local",
                    title="hello",
                ),
            ),
        )
        restored = _CachedRecommendationPayload.from_json(payload.to_json())
        self.assertEqual(restored.model_name, payload.model_name)
        self.assertEqual(restored.served_by_placeholder, payload.served_by_placeholder)
        self.assertEqual(restored.warnings, payload.warnings)
        self.assertEqual(restored.items[0].news_id, "N1")
        self.assertEqual(restored.items[0].semantic_id, [1, 2, 3, 4])

    def test_build_recommendation_cache_falls_back_to_ttl(self) -> None:
        cache = build_recommendation_cache(ttl_seconds=30, redis_url=None)
        self.assertIsInstance(cache, TTLCache)


if __name__ == "__main__":
    unittest.main()
