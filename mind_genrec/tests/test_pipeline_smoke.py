"""Minimal end-to-end smoke test for the toy MIND pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from mind_genrec.pipeline.run_mind_pipeline import run_pipeline
from mind_genrec.serving.model_registry import ModelRegistry
from mind_genrec.serving.retrieval_service import RetrievalService
from mind_genrec.serving.schemas import RecommendationRequest


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_and_baseline_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_root = root / "raw"
            train_dir = raw_root / "train"
            valid_dir = raw_root / "valid"
            train_dir.mkdir(parents=True, exist_ok=True)
            valid_dir.mkdir(parents=True, exist_ok=True)

            news_lines = [
                "N1\tnews\tlocal\tCity council approves park\tA new park was approved downtown.\thttps://a\t[]\t[]",
                "N2\tnews\tworld\tElection debate draws crowd\tCandidates debated economic policy.\thttps://b\t[]\t[]",
                "N3\tsports\tfootball\tQuarterback leads comeback\tThe team rallied in the fourth quarter.\thttps://c\t[]\t[]",
                "N4\tfinance\tmarkets\tStocks rise after report\tMarkets climbed on jobs data.\thttps://d\t[]\t[]",
                "N5\tnews\tlocal\tMayor opens community center\tResidents attended the opening event.\thttps://e\t[]\t[]",
            ]
            behaviors_train = [
                "1\tU1\t11/11/2019 9:00:00 AM\tN1 N5\tN2-0 N3-1 N4-0",
                "2\tU2\t11/11/2019 9:05:00 AM\tN2 N4\tN1-0 N4-1 N5-0",
                "3\tU3\t11/11/2019 9:10:00 AM\tN3 N1\tN3-1 N2-0 N5-0",
                "4\tU4\t11/11/2019 9:12:00 AM\tN4 N2\tN4-1 N1-0 N3-0",
            ]
            behaviors_valid = [
                "5\tU5\t11/12/2019 9:00:00 AM\tN1 N3\tN3-1 N4-0 N2-0",
                "6\tU6\t11/12/2019 9:03:00 AM\tN2 N4\tN4-1 N1-0 N5-0",
            ]

            for split_dir, behavior_lines in ((train_dir, behaviors_train), (valid_dir, behaviors_valid)):
                (split_dir / "news.tsv").write_text("\n".join(news_lines) + "\n", encoding="utf-8")
                (split_dir / "behaviors.tsv").write_text("\n".join(behavior_lines) + "\n", encoding="utf-8")

            config = {
                "project_name": "mind_genrec",
                "dataset_name": "toy",
                "data": {"max_history_length": 20},
                "semantic_id": {
                    "encoder_type": "hashing",
                    "code_length": 4,
                    "codebook_size": 16,
                    "embedding_dim": 32,
                    "max_iterations": 5,
                    "sample_size": 64,
                    "batch_size": 32,
                    "seed": 7,
                },
                "model": {
                    "decoder_type": "ar",
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "dropout": 0.1,
                },
                "training": {
                    "batch_size": 4,
                    "learning_rate": 0.001,
                    "epochs": 1,
                },
                "baseline": {
                    "hidden_dim": 32,
                    "output_dim": 16,
                    "num_heads": 4,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "temperature": 0.07,
                    "batch_size": 4,
                    "learning_rate": 0.001,
                    "epochs": 1,
                },
                "evaluation": {
                    "top_ks": [5],
                    "batch_size": 4,
                },
            }
            config_path = root / "toy_config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            work_dir = root / "work"

            summary = run_pipeline(
                config_path=config_path,
                train_dir=train_dir,
                valid_dir=valid_dir,
                work_dir=work_dir,
                max_train_samples=4,
                max_valid_samples=2,
                device="cpu",
            )

            self.assertEqual(summary["preprocess_summary"]["news_count"], 5)
            self.assertGreater(summary["preprocess_summary"]["validation_sample_count"], 0)
            self.assertGreater(summary["preprocess_summary"]["test_sample_count"], 0)
            self.assertTrue((work_dir / "comparison_summary.json").exists())
            self.assertTrue((work_dir / "comparison_report.md").exists())

            registry = ModelRegistry(
                news_jsonl_path=str(work_dir / "normalized" / "news.jsonl"),
                semantic_artifact_dir=str(work_dir / "semantic_ids"),
                baseline_checkpoint_path=str(work_dir / "baseline" / "best_baseline.pt"),
                device="cpu",
            )
            service = RetrievalService(registry=registry)
            health = service.health_snapshot()
            self.assertEqual(health[0], "degraded")
            self.assertTrue(health[1])
            self.assertFalse(health[2])
            self.assertTrue(health[3])
            response = service.recommend(RecommendationRequest(history=["N1", "N3"], top_k=3))
            self.assertEqual(response.model_name, "two-tower-baseline")
            self.assertFalse(response.served_by_placeholder)
            self.assertGreaterEqual(len(response.items), 1)

    def test_pipeline_lazy_ar_generator_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_root = root / "raw"
            train_dir = raw_root / "train"
            valid_dir = raw_root / "valid"
            train_dir.mkdir(parents=True, exist_ok=True)
            valid_dir.mkdir(parents=True, exist_ok=True)

            news_lines = [
                "N1\tnews\tlocal\tCity council approves park\tA new park was approved downtown.\thttps://a\t[]\t[]",
                "N2\tnews\tworld\tElection debate draws crowd\tCandidates debated economic policy.\thttps://b\t[]\t[]",
                "N3\tsports\tfootball\tQuarterback leads comeback\tThe team rallied in the fourth quarter.\thttps://c\t[]\t[]",
                "N4\tfinance\tmarkets\tStocks rise after report\tMarkets climbed on jobs data.\thttps://d\t[]\t[]",
                "N5\tnews\tlocal\tMayor opens community center\tResidents attended the opening event.\thttps://e\t[]\t[]",
            ]
            behaviors_train = [
                "1\tU1\t11/11/2019 9:00:00 AM\tN1 N5\tN2-0 N3-1 N4-0",
                "2\tU2\t11/11/2019 9:05:00 AM\tN2 N4\tN1-0 N4-1 N5-0",
                "3\tU3\t11/11/2019 9:10:00 AM\tN3 N1\tN3-1 N2-0 N5-0",
                "4\tU4\t11/11/2019 9:12:00 AM\tN4 N2\tN4-1 N1-0 N3-0",
            ]
            behaviors_valid = [
                "5\tU5\t11/12/2019 9:00:00 AM\tN1 N3\tN3-1 N4-0 N2-0",
                "6\tU6\t11/12/2019 9:03:00 AM\tN2 N4\tN4-1 N1-0 N5-0",
            ]

            for split_dir, behavior_lines in ((train_dir, behaviors_train), (valid_dir, behaviors_valid)):
                (split_dir / "news.tsv").write_text("\n".join(news_lines) + "\n", encoding="utf-8")
                (split_dir / "behaviors.tsv").write_text("\n".join(behavior_lines) + "\n", encoding="utf-8")

            config = {
                "project_name": "mind_genrec",
                "dataset_name": "toy",
                "data": {"max_history_length": 20},
                "semantic_id": {
                    "encoder_type": "hashing",
                    "code_length": 4,
                    "codebook_size": 16,
                    "embedding_dim": 32,
                    "max_iterations": 5,
                    "sample_size": 64,
                    "batch_size": 32,
                    "seed": 7,
                },
                "model": {
                    "decoder_type": "lazy_ar",
                    "lazy_parallel_layers": 1,
                    "hidden_dim": 32,
                    "num_layers": 3,
                    "num_heads": 4,
                    "dropout": 0.1,
                },
                "training": {
                    "batch_size": 4,
                    "learning_rate": 0.001,
                    "epochs": 1,
                },
                "baseline": {
                    "hidden_dim": 32,
                    "output_dim": 16,
                    "num_heads": 4,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "temperature": 0.07,
                    "batch_size": 4,
                    "learning_rate": 0.001,
                    "epochs": 1,
                },
                "evaluation": {
                    "mode": "full_corpus_retrieval",
                    "validation_ratio": 0.5,
                    "top_ks": [5],
                    "batch_size": 4,
                },
            }
            config_path = root / "toy_lazy_config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            work_dir = root / "work"

            summary = run_pipeline(
                config_path=config_path,
                train_dir=train_dir,
                valid_dir=valid_dir,
                work_dir=work_dir,
                max_train_samples=4,
                max_valid_samples=1,
                max_test_samples=1,
                device="cpu",
            )

            self.assertEqual(
                summary["generator_metadata"]["model_config"]["decoder_type"],
                "lazy_ar",
            )

            registry = ModelRegistry(
                news_jsonl_path=str(work_dir / "normalized" / "news.jsonl"),
                semantic_artifact_dir=str(work_dir / "semantic_ids"),
                generator_checkpoint_path=str(work_dir / "generator" / "best_generator.pt"),
                device="cpu",
            )
            service = RetrievalService(registry=registry)
            response = service.recommend(RecommendationRequest(history=["N1", "N3"], top_k=3))
            self.assertEqual(response.model_name, "lazy-ar-semantic-generator-beam")
            self.assertFalse(response.served_by_placeholder)
            self.assertGreaterEqual(len(response.items), 1)


if __name__ == "__main__":
    unittest.main()
