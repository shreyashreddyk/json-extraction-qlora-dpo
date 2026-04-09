"""Scaffold CLI for baseline, SFT, and DPO evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.artifacts import mirror_small_artifact
from json_ft.runtime import format_runtime_summary, resolve_repo_artifact_targets, resolve_runtime_context
from json_ft.utils import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--run-name", default="baseline-eval")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--model-name-or-path", default="placeholder-model")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/eval/heldout_eval.jsonl"))
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--mirror-metrics-to-repo", action="store_true")
    parser.add_argument("--mirror-report-to-repo", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    context = resolve_runtime_context(
        repo_root=repo_root,
        stage="eval",
        run_name=args.run_name,
        runtime_root=args.runtime_root,
    )
    metrics_output = args.metrics_output or (context.metrics_dir / f"{args.run_name}_metrics.json")
    report_output = args.report_output or (context.reports_dir / f"{args.run_name}_report.json")
    repo_targets = resolve_repo_artifact_targets(repo_root)

    print("Model evaluation scaffold")
    print(f"Config: {args.config}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Metrics output: {metrics_output}")
    print(format_runtime_summary(context))

    metrics_path = write_json(
        metrics_output,
        {
            "stage": "eval",
            "run_name": args.run_name,
            "model_name_or_path": args.model_name_or_path,
            "dataset_path": str(args.dataset_path),
            "config": str(args.config),
            "status": "scaffold_ready",
            "json_validity_rate": None,
            "schema_pass_rate": None,
            "note": "Evaluation is not implemented yet. This file reserves the final metrics shape.",
        },
    )
    report_path = write_json(
        report_output,
        {
            "stage": "eval",
            "run_name": args.run_name,
            "metrics_path": str(metrics_path),
            "status": "scaffold_ready",
        },
    )

    if args.mirror_metrics_to_repo:
        mirrored = mirror_small_artifact(metrics_path, repo_targets["metrics"] / metrics_path.name)
        print(f"Mirrored metrics artifact: {mirrored}")
    if args.mirror_report_to_repo:
        mirrored = mirror_small_artifact(report_path, repo_targets["reports"] / report_path.name)
        print(f"Mirrored report artifact: {mirrored}")

    print(f"Report output: {report_path}")
    print("TODO: implement inference execution, schema validation, and aggregate metric reporting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

