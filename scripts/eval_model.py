"""Evaluate a baseline, SFT, or DPO model on the held-out JSON extraction manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.artifacts import mirror_small_artifact
from json_ft.evaluation import (
    resolve_eval_output_paths,
    resolve_eval_settings,
    run_model_evaluation,
)
from json_ft.runtime import (
    format_runtime_backend_summary,
    format_runtime_summary,
    resolve_repo_artifact_targets,
    resolve_runtime_context,
)
from json_ft.schemas import build_support_ticket_schema
from json_ft.utils import write_json, write_jsonl, write_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--run-name", default="baseline-qwen2.5-1.5b")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--stage-label", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--merged-model-path", default=None)
    parser.add_argument("--model-manifest", type=Path, default=None)
    parser.add_argument("--prior-stage-predictions", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--prompt-source", choices=("messages", "prompt"), default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--mirror-metrics-to-repo", action="store_true")
    parser.add_argument("--mirror-report-to-repo", action="store_true")
    parser.add_argument("--mirror-predictions-to-repo", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    settings = resolve_eval_settings(
        config_path=args.config,
        repo_root=repo_root,
        stage_label=args.stage_label,
        backend=args.backend,
        model_name_or_path=args.model_name_or_path,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        merged_model_path=args.merged_model_path,
        model_manifest_path=args.model_manifest,
        prior_stage_predictions_path=args.prior_stage_predictions,
        dataset_path=args.dataset_path,
        prompt_source=args.prompt_source,
        sample_limit=args.sample_limit,
        eval_batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        revision=args.revision,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    context = resolve_runtime_context(
        repo_root=repo_root,
        stage="eval",
        run_name=args.run_name,
        runtime_root=args.runtime_root,
    )
    output_paths = resolve_eval_output_paths(
        context=context,
        settings=settings,
        run_name=args.run_name,
        metrics_output=args.metrics_output,
        report_output=args.report_output,
        predictions_output=args.predictions_output,
    )
    repo_targets = resolve_repo_artifact_targets(repo_root)
    schema = build_support_ticket_schema()

    print("Running model evaluation")
    print(f"Config: {args.config}")
    print(f"Stage label: {settings.stage_label}")
    print(f"Model: {settings.model_name_or_path}")
    print(f"Base model: {settings.base_model or '<none>'}")
    print(f"Adapter path: {settings.adapter_path or '<none>'}")
    print(f"Model manifest: {settings.model_manifest_path or '<none>'}")
    print(f"Prior-stage predictions: {settings.prior_stage_predictions_path or '<none>'}")
    print(f"Backend: {settings.backend}")
    print(f"Prompt source: {settings.prompt_source}")
    print(f"Dataset path: {settings.dataset_path}")
    print(f"Eval batch size: {settings.eval_batch_size}")
    print(format_runtime_summary(context))
    print(
        format_runtime_backend_summary(
            explicit_device_map=settings.model["device_map"],
            cuda_default="cuda",
        )
    )

    metrics_payload, diagnostics_payload, report_text, prediction_rows, bucket_rows = run_model_evaluation(
        run_name=args.run_name,
        settings=settings,
        schema=schema,
    )

    metrics_path = write_json(output_paths["metrics"], metrics_payload)
    diagnostics_path = write_json(output_paths["diagnostics"], diagnostics_payload)
    report_path = write_text(output_paths["report"], report_text)
    predictions_path = write_jsonl(output_paths["predictions"], prediction_rows)
    buckets_path = write_jsonl(output_paths["buckets"], bucket_rows)

    print(f"Metrics output: {metrics_path}")
    print(f"Diagnostics output: {diagnostics_path}")
    print(f"Report output: {report_path}")
    print(f"Predictions output: {predictions_path}")
    print(f"Bucket output: {buckets_path}")

    if args.mirror_metrics_to_repo:
        mirrored = mirror_small_artifact(metrics_path, repo_targets["metrics"] / metrics_path.name)
        print(f"Mirrored metrics artifact: {mirrored}")
        mirrored = mirror_small_artifact(diagnostics_path, repo_targets["metrics"] / diagnostics_path.name)
        print(f"Mirrored diagnostics artifact: {mirrored}")
    if args.mirror_report_to_repo:
        mirrored = mirror_small_artifact(report_path, repo_targets["reports"] / report_path.name)
        print(f"Mirrored report artifact: {mirrored}")
    if args.mirror_predictions_to_repo:
        mirrored = mirror_small_artifact(predictions_path, repo_targets["reports"] / predictions_path.name)
        print(f"Mirrored predictions artifact: {mirrored}")
        mirrored = mirror_small_artifact(buckets_path, repo_targets["reports"] / buckets_path.name)
        print(f"Mirrored bucket artifact: {mirrored}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
