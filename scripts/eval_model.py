"""Evaluate a baseline, SFT, or DPO model on the held-out JSON extraction manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.artifacts import mirror_small_artifact
from json_ft.inference import InferenceRequest, build_inference_backend
from json_ft.metrics import CATEGORICAL_EXACT_MATCH_FIELDS, EvaluationRecord, evaluate_records
from json_ft.runtime import format_runtime_summary, resolve_repo_artifact_targets, resolve_runtime_context
from json_ft.schemas import build_support_ticket_schema
from json_ft.utils import load_yaml, read_jsonl, write_json, write_jsonl, write_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--run-name", default="baseline-qwen2.5-1.5b")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--prompt-source", choices=("messages", "prompt"), default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
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


def _config_value(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _load_eval_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Eval config does not exist: {path.resolve()}")
    return load_yaml(path)


def _resolve_run_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    model_config = config.get("model", {})
    generation_config = config.get("generation", {})
    artifact_config = config.get("artifacts", {})

    return {
        "backend": _config_value(args.backend, config.get("backend"), default="local-transformers"),
        "model_name_or_path": _config_value(
            args.model_name_or_path,
            config.get("model_name_or_path"),
            default="Qwen/Qwen2.5-1.5B-Instruct",
        ),
        "dataset_path": Path(
            _config_value(
                args.dataset_path,
                config.get("dataset_path"),
                default="data/manifests/support_tickets_eval_manifest.jsonl",
            )
        ),
        "prompt_source": _config_value(args.prompt_source, config.get("prompt_source"), default="messages"),
        "sample_limit": _config_value(args.sample_limit, config.get("sample_limit")),
        "generation": {
            "max_new_tokens": _config_value(
                args.max_new_tokens,
                generation_config.get("max_new_tokens"),
                default=256,
            ),
            "temperature": _config_value(
                args.temperature,
                generation_config.get("temperature"),
                default=0.0,
            ),
            "top_p": _config_value(args.top_p, generation_config.get("top_p"), default=1.0),
            "do_sample": bool(args.do_sample or generation_config.get("do_sample", False)),
        },
        "model": {
            "revision": _config_value(args.revision, model_config.get("revision")),
            "trust_remote_code": bool(
                args.trust_remote_code or model_config.get("trust_remote_code", False)
            ),
            "torch_dtype": _config_value(args.torch_dtype, model_config.get("torch_dtype"), default="auto"),
            "device_map": _config_value(args.device_map, model_config.get("device_map")),
        },
        "artifacts": {
            "metrics_filename": artifact_config.get("metrics_filename", "{run_name}_metrics.json"),
            "report_filename": artifact_config.get("report_filename", "{run_name}_report.md"),
            "predictions_filename": artifact_config.get(
                "predictions_filename",
                "{run_name}_predictions.jsonl",
            ),
        },
    }


def _resolved_output_paths(
    context: Any,
    settings: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Path]:
    artifact_config = settings["artifacts"]
    return {
        "metrics": args.metrics_output
        or (context.metrics_dir / artifact_config["metrics_filename"].format(run_name=args.run_name)),
        "report": args.report_output
        or (context.reports_dir / artifact_config["report_filename"].format(run_name=args.run_name)),
        "predictions": args.predictions_output
        or (context.reports_dir / artifact_config["predictions_filename"].format(run_name=args.run_name)),
    }


def _load_eval_rows(dataset_path: Path, sample_limit: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    if sample_limit is not None:
        return rows[:sample_limit]
    return rows


def _build_request(
    row: dict[str, Any],
    prompt_source: str,
    generation: dict[str, Any],
) -> InferenceRequest:
    if prompt_source == "messages":
        return InferenceRequest(
            messages=row.get("messages"),
            prompt=row.get("prompt"),
            record_id=row.get("record_id"),
            max_new_tokens=generation["max_new_tokens"],
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            do_sample=generation["do_sample"],
            prompt_source=prompt_source,
        )
    return InferenceRequest(
        prompt=row.get("prompt"),
        record_id=row.get("record_id"),
        max_new_tokens=generation["max_new_tokens"],
        temperature=generation["temperature"],
        top_p=generation["top_p"],
        do_sample=generation["do_sample"],
        prompt_source=prompt_source,
    )


def _prediction_artifact_row(
    row: dict[str, Any],
    response: Any,
) -> dict[str, Any]:
    validation = response.validation
    return {
        "record_id": row.get("record_id"),
        "source_dataset": row.get("source_dataset"),
        "input_text": row.get("input_text"),
        "metadata": row.get("metadata", {}),
        "reference_payload": row.get("reference_payload"),
        "raw_output": response.text,
        "parsed_payload": response.parsed_payload,
        "parse_error": response.parse_error,
        "json_recovery_used": response.json_recovery_used,
        "schema_is_valid": validation.is_valid if validation is not None else False,
        "missing_fields": list(validation.missing_fields) if validation is not None else [],
        "unexpected_fields": list(validation.unexpected_fields) if validation is not None else [],
        "validation_issues": [
            {
                "path": list(issue.path),
                "issue_type": issue.issue_type,
                "message": issue.message,
            }
            for issue in (validation.issues if validation is not None else ())
        ],
        "latency_ms": response.latency_ms,
        "backend": response.backend,
        "model_name_or_path": response.model_name_or_path,
        "prompt_source": response.prompt_source,
        "generation": response.generation_kwargs,
    }


def _failure_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in prediction_rows:
        if row["parse_error"]:
            failures.append(row)
            continue
        if not row["schema_is_valid"]:
            failures.append(row)
            continue
        if row["parsed_payload"] != row["reference_payload"]:
            failures.append(row)
    return failures


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _render_report(
    *,
    run_name: str,
    config_path: Path,
    dataset_path: Path,
    settings: dict[str, Any],
    metrics_payload: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
) -> str:
    failures = _failure_rows(prediction_rows)
    lines = [
        f"# Baseline Evaluation Report: {run_name}",
        "",
        "## Run Summary",
        "",
        f"- Model: `{settings['model_name_or_path']}`",
        f"- Backend: `{settings['backend']}`",
        f"- Prompt source: `{settings['prompt_source']}`",
        f"- Dataset path: `{dataset_path}`",
        f"- Config path: `{config_path}`",
        f"- Evaluated records: `{metrics_payload['record_count']}`",
        "",
        "## Headline Metrics",
        "",
        f"- JSON validity rate: `{_format_float(metrics_payload['json_validity_rate'])}`",
        f"- Schema validation pass rate: `{_format_float(metrics_payload['schema_validation_pass_rate'])}`",
        f"- Hallucinated field rate: `{_format_float(metrics_payload['hallucinated_field_rate'])}`",
        f"- JSON recovery rate: `{_format_float(metrics_payload['json_recovery_rate'])}`",
        f"- Field-level micro F1: `{_format_float(metrics_payload['field_level']['micro']['f1'])}`",
        f"- Field-level macro F1: `{_format_float(metrics_payload['field_level']['macro']['f1'])}`",
        f"- Mean latency (ms): `{_format_float(metrics_payload['latency_ms']['mean'])}`",
        "",
        "## Exact Match by Field",
        "",
    ]
    for field_name in CATEGORICAL_EXACT_MATCH_FIELDS:
        lines.append(
            f"- `{field_name}`: `{_format_float(metrics_payload['categorical_exact_match'][field_name])}`"
        )

    lines.extend(
        [
            "",
            "## Field-Level Precision / Recall / F1",
            "",
        ]
    )
    for field_name, field_metrics in metrics_payload["field_level"]["per_field"].items():
        lines.append(
            "- "
            f"`{field_name}`: "
            f"P=`{_format_float(field_metrics['precision'])}`, "
            f"R=`{_format_float(field_metrics['recall'])}`, "
            f"F1=`{_format_float(field_metrics['f1'])}`, "
            f"support=`{field_metrics['support']}`"
        )

    lines.extend(
        [
            "",
            "## Failure Summary",
            "",
            f"- Parse failures: `{metrics_payload['counts']['parse_failure_count']}`",
            f"- Schema failures: `{metrics_payload['counts']['schema_failure_count']}`",
            f"- Hallucinated predictions: `{metrics_payload['counts']['hallucinated_prediction_count']}`",
            f"- Rows with semantic mismatch after parsing: `{len(failures)}`",
            "",
            "## Example Failures",
            "",
        ]
    )

    if not failures:
        lines.append("No failures were recorded for this run.")
        return "\n".join(lines) + "\n"

    for row in failures[:5]:
        lines.extend(
            [
                f"### `{row['record_id']}`",
                "",
                f"- Parse error: `{row['parse_error'] or 'none'}`",
                f"- Schema valid: `{row['schema_is_valid']}`",
                f"- Unexpected fields: `{', '.join(row['unexpected_fields']) or 'none'}`",
                "",
                "Input:",
                "",
                "```text",
                row["input_text"],
                "```",
                "",
                "Model output:",
                "",
                "```text",
                row["raw_output"],
                "```",
                "",
                "Reference payload:",
                "",
                "```json",
                json.dumps(row["reference_payload"], indent=2, sort_keys=True, ensure_ascii=True),
                "```",
                "",
                "Parsed payload:",
                "",
                "```json",
                json.dumps(row["parsed_payload"], indent=2, sort_keys=True, ensure_ascii=True)
                if row["parsed_payload"] is not None
                else "null",
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_eval_config(args.config)
    settings = _resolve_run_settings(args, config)

    repo_root = Path(__file__).resolve().parents[1]
    context = resolve_runtime_context(
        repo_root=repo_root,
        stage="eval",
        run_name=args.run_name,
        runtime_root=args.runtime_root,
    )
    output_paths = _resolved_output_paths(context, settings, args)
    repo_targets = resolve_repo_artifact_targets(repo_root)
    schema = build_support_ticket_schema()

    rows = _load_eval_rows(settings["dataset_path"], settings["sample_limit"])
    backend = build_inference_backend(
        settings["backend"],
        settings["model_name_or_path"],
        revision=settings["model"]["revision"],
        trust_remote_code=settings["model"]["trust_remote_code"],
        torch_dtype=settings["model"]["torch_dtype"],
        device_map=settings["model"]["device_map"],
        schema=schema,
    )

    print("Running model evaluation")
    print(f"Config: {args.config}")
    print(f"Model: {settings['model_name_or_path']}")
    print(f"Backend: {settings['backend']}")
    print(f"Prompt source: {settings['prompt_source']}")
    print(f"Dataset path: {settings['dataset_path']}")
    print(f"Eval rows: {len(rows)}")
    print(format_runtime_summary(context))

    evaluation_records: list[EvaluationRecord] = []
    prediction_rows: list[dict[str, Any]] = []
    for row in rows:
        response = backend.generate(_build_request(row, settings["prompt_source"], settings["generation"]))
        evaluation_records.append(
            EvaluationRecord(
                record_id=str(row.get("record_id")),
                reference_payload=row["reference_payload"],
                raw_output=response.text,
                parsed_payload=response.parsed_payload,
                validation=response.validation,
                latency_ms=response.latency_ms,
                json_recovery_used=response.json_recovery_used,
            )
        )
        prediction_rows.append(_prediction_artifact_row(row, response))

    aggregate_metrics = evaluate_records(evaluation_records, schema)
    metrics_payload = {
        "stage": "baseline_eval",
        "run_name": args.run_name,
        "config_path": str(args.config),
        "dataset_path": str(settings["dataset_path"]),
        "model_name_or_path": settings["model_name_or_path"],
        "backend": settings["backend"],
        "prompt_source": settings["prompt_source"],
        "sample_limit": settings["sample_limit"],
        "generation": settings["generation"],
        **aggregate_metrics,
    }
    report_text = _render_report(
        run_name=args.run_name,
        config_path=args.config,
        dataset_path=settings["dataset_path"],
        settings=settings,
        metrics_payload=metrics_payload,
        prediction_rows=prediction_rows,
    )

    metrics_path = write_json(output_paths["metrics"], metrics_payload)
    report_path = write_text(output_paths["report"], report_text)
    predictions_path = write_jsonl(output_paths["predictions"], prediction_rows)

    print(f"Metrics output: {metrics_path}")
    print(f"Report output: {report_path}")
    print(f"Predictions output: {predictions_path}")

    if args.mirror_metrics_to_repo:
        mirrored = mirror_small_artifact(metrics_path, repo_targets["metrics"] / metrics_path.name)
        print(f"Mirrored metrics artifact: {mirrored}")
    if args.mirror_report_to_repo:
        mirrored = mirror_small_artifact(report_path, repo_targets["reports"] / report_path.name)
        print(f"Mirrored report artifact: {mirrored}")
    if args.mirror_predictions_to_repo:
        mirrored = mirror_small_artifact(predictions_path, repo_targets["reports"] / predictions_path.name)
        print(f"Mirrored predictions artifact: {mirrored}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
