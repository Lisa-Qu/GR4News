"""Compare generator and baseline evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_top_ks(generator_summary: dict[str, object], baseline_summary: dict[str, object]) -> list[int]:
    keys = set()
    for metrics_key in ("ranking_metrics",):
        for summary in (generator_summary, baseline_summary):
            metrics = summary.get(metrics_key, {})
            if isinstance(metrics, dict):
                for key in metrics:
                    if "@" not in key:
                        continue
                    _, raw_k = key.split("@", 1)
                    if raw_k.isdigit():
                        keys.add(int(raw_k))
    return sorted(keys)


def _build_metric_delta(
    generator_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for key in sorted(set(generator_metrics) | set(baseline_metrics)):
        generator_value = float(generator_metrics.get(key, 0.0))
        baseline_value = float(baseline_metrics.get(key, 0.0))
        comparison[key] = {
            "generator": generator_value,
            "baseline": baseline_value,
            "delta_generator_minus_baseline": generator_value - baseline_value,
        }
    return comparison


def _build_markdown_table(
    *,
    ranking_comparison: dict[str, dict[str, float]],
    generator_code_metrics: dict[str, float],
    baseline_model_metrics: dict[str, float],
) -> str:
    lines = [
        "# Evaluation Comparison",
        "",
        "## Ranking Metrics",
        "",
        "| Metric | Generator | Baseline | Delta (Generator - Baseline) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric_name, payload in ranking_comparison.items():
        lines.append(
            f"| {metric_name} | {payload['generator']:.6f} | {payload['baseline']:.6f} | "
            f"{payload['delta_generator_minus_baseline']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Model-Specific Metrics",
            "",
            "### Generator",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for metric_name, value in sorted(generator_code_metrics.items()):
        lines.append(f"| {metric_name} | {float(value):.6f} |")

    lines.extend(
        [
            "",
            "### Baseline",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for metric_name, value in sorted(baseline_model_metrics.items()):
        lines.append(f"| {metric_name} | {float(value):.6f} |")

    lines.append("")
    return "\n".join(lines)


def compare_evaluation_summaries(
    *,
    generator_summary_path: str | Path,
    baseline_summary_path: str | Path,
    output_path: str | Path,
    markdown_path: str | Path | None = None,
) -> dict[str, object]:
    """Compare generator and baseline evaluation summaries."""

    generator_summary = json.loads(Path(generator_summary_path).read_text(encoding="utf-8"))
    baseline_summary = json.loads(Path(baseline_summary_path).read_text(encoding="utf-8"))

    generator_ranking = dict(generator_summary.get("ranking_metrics", {}))
    baseline_ranking = dict(baseline_summary.get("ranking_metrics", {}))
    generator_code_metrics = dict(generator_summary.get("code_metrics", {}))
    baseline_model_metrics = dict(baseline_summary.get("baseline_metrics", {}))

    ranking_comparison = _build_metric_delta(generator_ranking, baseline_ranking)
    summary = {
        "generator_summary_path": str(Path(generator_summary_path).resolve()),
        "baseline_summary_path": str(Path(baseline_summary_path).resolve()),
        "top_ks": _parse_top_ks(generator_summary, baseline_summary),
        "ranking_comparison": ranking_comparison,
        "generator_code_metrics": generator_code_metrics,
        "baseline_model_metrics": baseline_model_metrics,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if markdown_path is not None:
        markdown = _build_markdown_table(
            ranking_comparison=ranking_comparison,
            generator_code_metrics=generator_code_metrics,
            baseline_model_metrics=baseline_model_metrics,
        )
        markdown_target = Path(markdown_path)
        markdown_target.parent.mkdir(parents=True, exist_ok=True)
        markdown_target.write_text(markdown, encoding="utf-8")

    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Compare generator and baseline evaluation summaries.")
    parser.add_argument("--generator-summary-path", required=True)
    parser.add_argument("--baseline-summary-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--markdown-path")
    return parser


def main() -> None:
    """Entry point."""

    args = build_parser().parse_args()
    summary = compare_evaluation_summaries(
        generator_summary_path=args.generator_summary_path,
        baseline_summary_path=args.baseline_summary_path,
        output_path=args.output_path,
        markdown_path=args.markdown_path,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
