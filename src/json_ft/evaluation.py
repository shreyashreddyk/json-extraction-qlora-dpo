"""Shared single-run evaluation helpers for baseline, SFT, and DPO comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import json

from .inference import InferenceRequest, build_inference_backend
from .manifests import LatestModelManifest
from .metrics import CATEGORICAL_EXACT_MATCH_FIELDS, EvaluationRecord, evaluate_records
from .schemas import SchemaConstraint
from .utils import load_yaml, read_jsonl


def _config_value(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _resolve_repo_path(repo_root: Path, value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _load_optional_manifest(path: Path | None) -> dict[str, Any] | LatestModelManifest | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if {"stage", "status", "base_model", "adapter_path"} <= payload.keys():
        try:
            return LatestModelManifest(**payload)
        except TypeError:
            return payload
    return payload


@dataclass(frozen=True)
class EvaluationSettings:
    """Resolved settings for one model-eval run."""

    config_path: Path
    stage_label: str
    backend: str
    model_name_or_path: str
    base_model: str | None
    adapter_path: str | None
    merged_model_path: str | None
    model_manifest_path: Path | None
    dataset_path: Path
    prompt_source: str
    sample_limit: int | None
    generation: dict[str, Any]
    model: dict[str, Any]
    artifacts: dict[str, Any]


def load_eval_config(path: Path) -> dict[str, Any]:
    """Load the YAML evaluation config."""

    if not path.exists():
        raise FileNotFoundError(f"Eval config does not exist: {path.resolve()}")
    return load_yaml(path)


def resolve_eval_settings(
    *,
    config_path: Path,
    repo_root: Path,
    stage_label: str | None = None,
    backend: str | None = None,
    model_name_or_path: str | None = None,
    base_model: str | None = None,
    adapter_path: str | None = None,
    merged_model_path: str | None = None,
    model_manifest_path: Path | None = None,
    dataset_path: Path | None = None,
    prompt_source: str | None = None,
    sample_limit: int | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    do_sample: bool = False,
    revision: str | None = None,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    trust_remote_code: bool = False,
) -> EvaluationSettings:
    """Resolve CLI overrides plus config defaults into one eval run config."""

    config = load_eval_config(config_path)
    model_config = config.get("model", {})
    generation_config = config.get("generation", {})
    artifact_config = config.get("artifacts", {})

    resolved_manifest_path = _resolve_repo_path(repo_root, model_manifest_path)
    manifest_payload = _load_optional_manifest(resolved_manifest_path)
    manifest_stage = None
    manifest_base_model = None
    manifest_adapter_path = None
    manifest_merged_path = None
    if isinstance(manifest_payload, LatestModelManifest):
        manifest_stage = manifest_payload.stage
        manifest_base_model = manifest_payload.base_model
        manifest_adapter_path = manifest_payload.adapter_path
        manifest_merged_path = manifest_payload.merged_export_path
    elif isinstance(manifest_payload, dict):
        manifest_stage = manifest_payload.get("stage")
        manifest_base_model = manifest_payload.get("base_model")
        manifest_adapter_path = manifest_payload.get("adapter_path")
        manifest_merged_path = manifest_payload.get("merged_export_path")

    resolved_base_model = _config_value(base_model, manifest_base_model)
    resolved_adapter_path = _config_value(adapter_path, model_config.get("adapter_path"), manifest_adapter_path)
    resolved_merged_model_path = _config_value(
        merged_model_path,
        model_config.get("merged_model_path"),
        manifest_merged_path,
    )
    resolved_model_name = _config_value(
        model_name_or_path,
        config.get("model_name_or_path"),
        resolved_merged_model_path,
        resolved_base_model,
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )

    return EvaluationSettings(
        config_path=config_path,
        stage_label=_config_value(stage_label, config.get("stage_label"), manifest_stage, default="baseline"),
        backend=_config_value(backend, config.get("backend"), default="local-transformers"),
        model_name_or_path=str(resolved_model_name),
        base_model=resolved_base_model,
        adapter_path=str(_resolve_repo_path(repo_root, resolved_adapter_path)) if resolved_adapter_path else None,
        merged_model_path=str(_resolve_repo_path(repo_root, resolved_merged_model_path))
        if resolved_merged_model_path
        else None,
        model_manifest_path=resolved_manifest_path,
        dataset_path=_resolve_repo_path(
            repo_root,
            _config_value(dataset_path, config.get("dataset_path"), default="data/manifests/support_tickets_eval_manifest.jsonl"),
        )
        or (repo_root / "data" / "manifests" / "support_tickets_eval_manifest.jsonl"),
        prompt_source=_config_value(prompt_source, config.get("prompt_source"), default="messages"),
        sample_limit=_config_value(sample_limit, config.get("sample_limit")),
        generation={
            "max_new_tokens": _config_value(max_new_tokens, generation_config.get("max_new_tokens"), default=256),
            "temperature": _config_value(temperature, generation_config.get("temperature"), default=0.0),
            "top_p": _config_value(top_p, generation_config.get("top_p"), default=1.0),
            "do_sample": bool(do_sample or generation_config.get("do_sample", False)),
        },
        model={
            "revision": _config_value(revision, model_config.get("revision")),
            "trust_remote_code": bool(trust_remote_code or model_config.get("trust_remote_code", False)),
            "torch_dtype": _config_value(torch_dtype, model_config.get("torch_dtype"), default="auto"),
            "device_map": _config_value(device_map, model_config.get("device_map")),
        },
        artifacts={
            "metrics_filename": artifact_config.get("metrics_filename", "{run_name}_metrics.json"),
            "report_filename": artifact_config.get("report_filename", "{run_name}_report.md"),
            "predictions_filename": artifact_config.get(
                "predictions_filename",
                "{run_name}_predictions.jsonl",
            ),
        },
    )


def resolve_eval_output_paths(
    *,
    context: Any,
    settings: EvaluationSettings,
    run_name: str,
    metrics_output: Path | None = None,
    report_output: Path | None = None,
    predictions_output: Path | None = None,
) -> dict[str, Path]:
    """Resolve output artifact paths for one eval run."""

    artifact_config = settings.artifacts
    return {
        "metrics": metrics_output
        or (context.metrics_dir / artifact_config["metrics_filename"].format(run_name=run_name)),
        "report": report_output
        or (context.reports_dir / artifact_config["report_filename"].format(run_name=run_name)),
        "predictions": predictions_output
        or (context.reports_dir / artifact_config["predictions_filename"].format(run_name=run_name)),
    }


def load_eval_rows(dataset_path: Path, sample_limit: int | None) -> list[dict[str, Any]]:
    """Load held-out eval rows from the shared manifest contract."""

    rows = read_jsonl(dataset_path)
    if sample_limit is not None:
        return rows[:sample_limit]
    return rows


def build_inference_request(
    row: dict[str, Any],
    prompt_source: str,
    generation: dict[str, Any],
) -> InferenceRequest:
    """Build one inference request from an eval row."""

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


def prediction_artifact_row(
    row: dict[str, Any],
    response: Any,
    *,
    stage_label: str,
    base_model: str | None,
    adapter_path: str | None,
    merged_model_path: str | None,
) -> dict[str, Any]:
    """Render one prediction row with stage provenance."""

    validation = response.validation
    return {
        "record_id": row.get("record_id"),
        "stage_label": stage_label,
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
        "base_model": base_model,
        "adapter_path": adapter_path,
        "merged_model_path": merged_model_path,
        "prompt_source": response.prompt_source,
        "generation": response.generation_kwargs,
    }


def failure_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that failed parse, schema, or semantic exactness."""

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


def render_single_run_report(
    *,
    run_name: str,
    settings: EvaluationSettings,
    metrics_payload: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
) -> str:
    """Render a markdown report for one evaluation run."""

    failures = failure_rows(prediction_rows)
    stage_heading = settings.stage_label.replace("_", " ").title()
    lines = [
        f"# {stage_heading} Evaluation Report: {run_name}",
        "",
        "## Run Summary",
        "",
        f"- Stage label: `{settings.stage_label}`",
        f"- Model: `{settings.model_name_or_path}`",
        f"- Base model: `{settings.base_model or 'n/a'}`",
        f"- Adapter path: `{settings.adapter_path or 'n/a'}`",
        f"- Merged model path: `{settings.merged_model_path or 'n/a'}`",
        f"- Backend: `{settings.backend}`",
        f"- Prompt source: `{settings.prompt_source}`",
        f"- Dataset path: `{settings.dataset_path}`",
        f"- Config path: `{settings.config_path}`",
        f"- Model manifest path: `{settings.model_manifest_path or 'n/a'}`",
        f"- Evaluated records: `{metrics_payload['record_count']}`",
        "",
        "## Syntax Metrics",
        "",
        f"- JSON validity rate: `{_format_float(metrics_payload['json_validity_rate'])}`",
        f"- Schema validation pass rate: `{_format_float(metrics_payload['schema_validation_pass_rate'])}`",
        f"- Hallucinated field rate: `{_format_float(metrics_payload['hallucinated_field_rate'])}`",
        f"- JSON recovery rate: `{_format_float(metrics_payload['json_recovery_rate'])}`",
        "",
        "## Semantic Metrics",
        "",
        f"- Field-level micro F1: `{_format_float(metrics_payload['field_level']['micro']['f1'])}`",
        f"- Field-level macro F1: `{_format_float(metrics_payload['field_level']['macro']['f1'])}`",
        "",
        "## Latency",
        "",
        f"- Mean latency (ms): `{_format_float(metrics_payload['latency_ms']['mean'])}`",
        f"- P95 latency (ms): `{_format_float(metrics_payload['latency_ms']['p95'])}`",
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


def run_model_evaluation(
    *,
    run_name: str,
    settings: EvaluationSettings,
    schema: SchemaConstraint,
    backend: Any | None = None,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Execute one evaluation run and return metrics, report text, and predictions."""

    rows = load_eval_rows(settings.dataset_path, settings.sample_limit)
    active_backend = backend or build_inference_backend(
        settings.backend,
        settings.model_name_or_path,
        adapter_path=settings.adapter_path,
        revision=settings.model["revision"],
        trust_remote_code=settings.model["trust_remote_code"],
        torch_dtype=settings.model["torch_dtype"],
        device_map=settings.model["device_map"],
        schema=schema,
    )

    evaluation_records: list[EvaluationRecord] = []
    prediction_rows: list[dict[str, Any]] = []
    for row in rows:
        response = active_backend.generate(
            build_inference_request(row, settings.prompt_source, settings.generation)
        )
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
        prediction_rows.append(
            prediction_artifact_row(
                row,
                response,
                stage_label=settings.stage_label,
                base_model=settings.base_model,
                adapter_path=settings.adapter_path,
                merged_model_path=settings.merged_model_path,
            )
        )

    aggregate_metrics = evaluate_records(evaluation_records, schema)
    metrics_payload = {
        "stage": settings.stage_label,
        "run_name": run_name,
        "config_path": str(settings.config_path),
        "dataset_path": str(settings.dataset_path),
        "model_name_or_path": settings.model_name_or_path,
        "base_model": settings.base_model,
        "adapter_path": settings.adapter_path,
        "merged_model_path": settings.merged_model_path,
        "model_manifest_path": str(settings.model_manifest_path) if settings.model_manifest_path else None,
        "backend": settings.backend,
        "prompt_source": settings.prompt_source,
        "sample_limit": settings.sample_limit,
        "generation": settings.generation,
        **aggregate_metrics,
    }
    report_text = render_single_run_report(
        run_name=run_name,
        settings=settings,
        metrics_payload=metrics_payload,
        prediction_rows=prediction_rows,
    )
    return metrics_payload, report_text, prediction_rows
