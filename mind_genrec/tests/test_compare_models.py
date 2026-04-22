"""Unit tests for evaluation comparison helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mind_genrec.evaluation.compare_models import compare_evaluation_summaries


class CompareModelsTest(unittest.TestCase):
    def test_compare_evaluation_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            generator_summary = {
                "code_metrics": {
                    "loss": 1.0,
                    "token_accuracy": 0.8,
                },
                "ranking_metrics": {
                    "hit_rate@5": 0.6,
                    "mrr@5": 0.4,
                },
            }
            baseline_summary = {
                "baseline_metrics": {
                    "loss": 0.9,
                    "in_batch_accuracy": 0.5,
                },
                "ranking_metrics": {
                    "hit_rate@5": 0.5,
                    "mrr@5": 0.3,
                },
            }
            generator_path = tmp_path / "generator.json"
            baseline_path = tmp_path / "baseline.json"
            output_path = tmp_path / "comparison.json"
            markdown_path = tmp_path / "comparison.md"
            generator_path.write_text(json.dumps(generator_summary), encoding="utf-8")
            baseline_path.write_text(json.dumps(baseline_summary), encoding="utf-8")

            summary = compare_evaluation_summaries(
                generator_summary_path=generator_path,
                baseline_summary_path=baseline_path,
                output_path=output_path,
                markdown_path=markdown_path,
            )

            self.assertEqual(summary["top_ks"], [5])
            self.assertAlmostEqual(
                summary["ranking_comparison"]["hit_rate@5"]["delta_generator_minus_baseline"],
                0.1,
            )
            self.assertTrue(output_path.exists())
            self.assertTrue(markdown_path.exists())
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Ranking Metrics", markdown)


if __name__ == "__main__":
    unittest.main()
