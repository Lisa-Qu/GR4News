"""Tests for serving settings and app factory wiring."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from mind_genrec.serving.app import create_app
from mind_genrec.serving.settings import ServingSettings


class ServingSettingsTest(unittest.TestCase):
    def test_from_env_reads_core_fields(self) -> None:
        env = {
            "MIND_GENREC_HOST": "127.0.0.1",
            "MIND_GENREC_PORT": "9000",
            "MIND_GENREC_RELOAD": "true",
            "MIND_GENREC_LOG_LEVEL": "debug",
            "MIND_GENREC_CACHE_TTL_SECONDS": "42",
            "MIND_GENREC_REDIS_URL": "redis://127.0.0.1:6379/0",
            "MIND_GENREC_NEWS_JSONL": "/tmp/news.jsonl",
            "MIND_GENREC_SEMANTIC_DIR": "/tmp/semantic",
            "MIND_GENREC_GENERATOR_CKPT": "/tmp/generator.pt",
            "MIND_GENREC_BASELINE_CKPT": "/tmp/baseline.pt",
            "MIND_GENREC_DEVICE": "cpu",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = ServingSettings.from_env()

        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 9000)
        self.assertTrue(settings.reload)
        self.assertEqual(settings.log_level, "debug")
        self.assertEqual(settings.cache_ttl_seconds, 42)
        self.assertEqual(settings.redis_url, "redis://127.0.0.1:6379/0")
        self.assertEqual(settings.news_jsonl_path, "/tmp/news.jsonl")
        self.assertEqual(settings.semantic_artifact_dir, "/tmp/semantic")
        self.assertEqual(settings.generator_checkpoint_path, "/tmp/generator.pt")
        self.assertEqual(settings.baseline_checkpoint_path, "/tmp/baseline.pt")
        self.assertEqual(settings.device, "cpu")

    def test_create_app_exposes_settings_on_state(self) -> None:
        settings = ServingSettings(host="127.0.0.1", port=8123, cache_ttl_seconds=30)
        app = create_app(settings=settings)

        self.assertIs(app.state.settings, settings)
        self.assertEqual(app.state.settings.port, 8123)
        self.assertEqual(app.state.settings.cache_ttl_seconds, 30)


if __name__ == "__main__":
    unittest.main()
