"""Unit tests for FastAPI startup and endpoint contracts."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from mind_genrec.serving.app import create_app
from mind_genrec.serving.settings import ServingSettings


def _make_client() -> TestClient:
    """Create a TestClient with no real artifacts (all placeholder)."""
    settings = ServingSettings()  # all paths None → placeholder bundle
    app = create_app(settings=settings)
    return TestClient(app, raise_server_exceptions=True)


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _make_client()

    def test_health_returns_200(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_response_has_required_fields(self) -> None:
        data = self.client.get("/health").json()
        for field in ("status", "ready", "service_ready", "generator_ready",
                      "baseline_ready", "model_name", "catalog_size"):
            self.assertIn(field, data, f"missing field: {field}")

    def test_health_placeholder_state(self) -> None:
        data = self.client.get("/health").json()
        # Without any checkpoint paths, service is not ready
        self.assertFalse(data["service_ready"])
        self.assertFalse(data["generator_ready"])
        self.assertTrue(data["uses_placeholder_components"])

    def test_health_status_field_is_string(self) -> None:
        data = self.client.get("/health").json()
        self.assertIsInstance(data["status"], str)
        self.assertIn(data["status"], {"ok", "degraded", "unavailable"})


class TestRecommendEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _make_client()

    def test_recommend_returns_200(self) -> None:
        resp = self.client.post("/recommend", json={"history": ["N1", "N2"], "top_k": 5})
        self.assertEqual(resp.status_code, 200)

    def test_recommend_response_has_required_fields(self) -> None:
        data = self.client.post("/recommend", json={"history": ["N1"], "top_k": 3}).json()
        for field in ("request_id", "model_name", "latency_ms", "cache_hit",
                      "served_by_placeholder", "items"):
            self.assertIn(field, data, f"missing field: {field}")

    def test_recommend_placeholder_returns_empty_items(self) -> None:
        data = self.client.post("/recommend", json={"history": ["N1"], "top_k": 5}).json()
        # StubGenerativeRetriever always returns []
        self.assertEqual(data["items"], [])
        self.assertTrue(data["served_by_placeholder"])

    def test_recommend_request_id_is_string(self) -> None:
        data = self.client.post("/recommend", json={"history": ["N1"], "top_k": 5}).json()
        self.assertIsInstance(data["request_id"], str)
        self.assertTrue(len(data["request_id"]) > 0)

    def test_recommend_latency_is_nonnegative(self) -> None:
        data = self.client.post("/recommend", json={"history": ["N1"], "top_k": 5}).json()
        self.assertGreaterEqual(data["latency_ms"], 0.0)

    def test_recommend_top_k_respected(self) -> None:
        # Placeholder returns 0 items regardless, but top_k field should be accepted
        resp = self.client.post("/recommend", json={"history": ["N1", "N2"], "top_k": 10})
        self.assertEqual(resp.status_code, 200)

    def test_recommend_empty_history_rejected(self) -> None:
        resp = self.client.post("/recommend", json={"history": [], "top_k": 5})
        self.assertEqual(resp.status_code, 422)

    def test_recommend_top_k_zero_rejected(self) -> None:
        resp = self.client.post("/recommend", json={"history": ["N1"], "top_k": 0})
        self.assertEqual(resp.status_code, 422)

    def test_recommend_cache_hit_on_second_request(self) -> None:
        payload = {"history": ["N1", "N2"], "top_k": 5}
        self.client.post("/recommend", json=payload)
        data = self.client.post("/recommend", json=payload).json()
        self.assertTrue(data["cache_hit"])


class TestBatchRecommendEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _make_client()

    def test_batch_recommend_returns_200(self) -> None:
        payload = {"requests": [
            {"history": ["N1"], "top_k": 3},
            {"history": ["N2", "N3"], "top_k": 5},
        ]}
        resp = self.client.post("/recommend/batch", json=payload)
        self.assertEqual(resp.status_code, 200)

    def test_batch_response_count_matches_input(self) -> None:
        payload = {"requests": [
            {"history": ["N1"], "top_k": 3},
            {"history": ["N2"], "top_k": 3},
            {"history": ["N3"], "top_k": 3},
        ]}
        data = self.client.post("/recommend/batch", json=payload).json()
        self.assertEqual(data["request_count"], 3)
        self.assertEqual(len(data["responses"]), 3)


class TestMetricsEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _make_client()

    def test_metrics_returns_200(self) -> None:
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)

    def test_metrics_contains_prometheus_labels(self) -> None:
        text = self.client.get("/metrics").text
        self.assertIn("mind_genrec_requests_total", text)
        self.assertIn("mind_genrec_cache_hits_total", text)


if __name__ == "__main__":
    unittest.main()
