"""Shared single-run evaluation helpers for baseline, SFT, and DPO comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import json

from .inference import InferenceRequest, build_inference_backend
from .manifests import LatestModelManifest
from .metrics import (
    CATEGORICAL_EXACT_MATCH_FIELDS,
    EvaluationRecord,
    LIST_PRF_FIELDS,
    STRUCTURED_PRF_FIELDS,
    evaluate_records,
)
from .schemas import SchemaConstraint
from .stage_metadata import build_data_pipeline_metadata
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
    prior_stage_predictions_path: Path | None
    dataset_path: Path
    build_summary_path: Path | None
    composition_summary_path: Path | None
    prompt_source: str
    sample_limit: int | None
    eval_batch_size: int
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
    prior_stage_predictions_path: Path | None = None,
    dataset_path: Path | None = None,
    prompt_source: str | None = None,
    sample_limit: int | None = None,
    eval_batch_size: int | None = None,
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
    inference_config = config.get("inference", {})
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
        prior_stage_predictions_path=_resolve_repo_path(repo_root, prior_stage_predictions_path),
        dataset_path=_resolve_repo_path(
            repo_root,
            _config_value(dataset_path, config.get("dataset_path"), default="data/manifests/support_tickets_eval_manifest.jsonl"),
        )
        or (repo_root / "data" / "manifests" / "support_tickets_eval_manifest.jsonl"),
        build_summary_path=_resolve_repo_path(
            repo_root,
            config.get("build_summary_path", "data/manifests/support_tickets_dataset_build_summary.json"),
        ),
        composition_summary_path=_resolve_repo_path(
            repo_root,
            config.get("composition_summary_path", "artifacts/metrics/support_tickets_dataset_composition.json"),
        ),
        prompt_source=_config_value(prompt_source, config.get("prompt_source"), default="messages"),
        sample_limit=_config_value(sample_limit, config.get("sample_limit")),
        eval_batch_size=int(_config_value(eval_batch_size, inference_config.get("eval_batch_size"), default=4)),
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
            "diagnostics_filename": artifact_config.get("diagnostics_filename", "{run_name}_diagnostics.json"),
            "report_filename": artifact_config.get("report_filename", "{run_name}_report.md"),
            "predictions_filename": artifact_config.get(
                "predictions_filename",
                "{run_name}_predictions.jsonl",
            ),
            "buckets_filename": artifact_config.get("buckets_filename", "{run_name}_example_buckets.jsonl"),
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
        "diagnostics": context.metrics_dir / artifact_config["diagnostics_filename"].format(run_name=run_name),
        "report": report_output
        or (context.reports_dir / artifact_config["report_filename"].format(run_name=run_name)),
        "predictions": predictions_output
        or (context.reports_dir / artifact_config["predictions_filename"].format(run_name=run_name)),
        "buckets": context.reports_dir / artifact_config["buckets_filename"].format(run_name=run_name),
    }


def load_eval_rows(dataset_path: Path, sample_limit: int | None) -> list[dict[str, Any]]:
    """Load held-out eval rows from the shared manifest contract."""

    rows = read_jsonl(dataset_path)
    if sample_limit is not None:
        return rows[:sample_limit]
    return rows


def _chunk_rows(rows: Sequence[dict[str, Any]], batch_size: int) -> list[Sequence[dict[str, Any]]]:
    size = max(1, int(batch_size))
    return [rows[index : index + size] for index in range(0, len(rows), size)]


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


NULLABLE_CUSTOMER_FIELDS = (
    "customer.name",
    "customer.account_id",
    "customer.plan_tier",
)


def _nested_value(payload: dict[str, Any] | None, field_path: str) -> Any:
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _list_values(payload: dict[str, Any] | None, field_path: str) -> set[str]:
    value = _nested_value(payload, field_path)
    if not isinstance(value, list):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _semantic_score(row: dict[str, Any]) -> float:
    predicted = row.get("parsed_payload")
    reference = row.get("reference_payload") or {}
    structured_match_count = sum(
        1
        for field_name in STRUCTURED_PRF_FIELDS
        if _nested_value(predicted, field_name) == _nested_value(reference, field_name)
    )
    predicted_actions = _list_values(predicted, LIST_PRF_FIELDS[0])
    reference_actions = _list_values(reference, LIST_PRF_FIELDS[0])
    overlap = predicted_actions & reference_actions
    tp = len(overlap)
    fp = len(predicted_actions - overlap)
    fn = len(reference_actions - overlap)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    actions_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return (structured_match_count + actions_f1) / (len(STRUCTURED_PRF_FIELDS) + 1)


def _syntax_tuple(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("parsed_payload") is not None),
        int(bool(row.get("schema_is_valid", False))),
        int(not row.get("unexpected_fields")),
    )


def _categorical_confusion_like_summary(prediction_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    summary: dict[str, list[dict[str, Any]]] = {}
    for field_name in CATEGORICAL_EXACT_MATCH_FIELDS:
        counts: dict[tuple[str, str], int] = {}
        for row in prediction_rows:
            reference_value = str(_nested_value(row.get("reference_payload"), field_name))
            predicted_value = str(_nested_value(row.get("parsed_payload"), field_name))
            key = (reference_value, predicted_value)
            counts[key] = counts.get(key, 0) + 1
        summary[field_name] = [
            {"reference": ref, "predicted": pred, "count": count}
            for (ref, pred), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        ]
    return summary


def _per_field_error_counts(prediction_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    field_names = [*STRUCTURED_PRF_FIELDS, *LIST_PRF_FIELDS]
    counts: dict[str, dict[str, int]] = {
        field_name: {"missing": 0, "mismatch": 0, "null_handling_mistake": 0}
        for field_name in field_names
    }
    for row in prediction_rows:
        predicted = row.get("parsed_payload")
        reference = row.get("reference_payload") or {}
        for field_name in field_names:
            predicted_value = _nested_value(predicted, field_name)
            reference_value = _nested_value(reference, field_name)
            if reference_value is not None and predicted_value is None:
                counts[field_name]["missing"] += 1
            if predicted_value != reference_value:
                counts[field_name]["mismatch"] += 1
            if field_name in NULLABLE_CUSTOMER_FIELDS and reference_value is None and predicted_value not in (None, "", []):
                counts[field_name]["null_handling_mistake"] += 1
    return counts


def _bucket_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if row.get("parse_error"):
        labels.append("syntax_failures")
    elif not row.get("schema_is_valid", False):
        labels.append("syntax_failures")

    if row.get("unexpected_fields"):
        labels.append("hallucinated_keys")

    predicted = row.get("parsed_payload")
    reference = row.get("reference_payload") or {}
    if predicted is not None and predicted != reference:
        labels.append("semantic_failures")

    for field_name in NULLABLE_CUSTOMER_FIELDS:
        reference_value = _nested_value(reference, field_name)
        predicted_value = _nested_value(predicted, field_name)
        if reference_value is None and predicted_value not in (None, "", []):
            labels.append("null_handling_mistakes")
            break

    predicted_actions = _list_values(predicted, LIST_PRF_FIELDS[0])
    reference_actions = _list_values(reference, LIST_PRF_FIELDS[0])
    if reference_actions and predicted_actions and predicted_actions != reference_actions:
        overlap = predicted_actions & reference_actions
        if overlap and overlap != reference_actions:
            labels.append("partial_action_extraction")
    return list(dict.fromkeys(labels))


def build_example_buckets(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach bucket labels used for qualitative stage review."""

    bucket_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        labels = _bucket_labels(row)
        bucket_rows.append(
            {
                "record_id": row.get("record_id"),
                "stage_label": row.get("stage_label"),
                "bucket_labels": labels,
                "primary_bucket": labels[0] if labels else "clean",
                "source_dataset": row.get("source_dataset"),
                "raw_output": row.get("raw_output"),
                "parsed_payload": row.get("parsed_payload"),
                "reference_payload": row.get("reference_payload"),
                "unexpected_fields": row.get("unexpected_fields"),
                "parse_error": row.get("parse_error"),
            }
        )
    return bucket_rows


def _bucket_counts(bucket_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in bucket_rows:
        for label in row.get("bucket_labels", []):
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _load_prior_stage_predictions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows = read_jsonl(path)
    return {str(row.get("record_id")): row for row in rows}


def _top_regressions_vs_prior_stage(
    prediction_rows: list[dict[str, Any]],
    prior_stage_predictions_path: Path | None,
) -> list[dict[str, Any]]:
    prior_rows = _load_prior_stage_predictions(prior_stage_predictions_path)
    regressions: list[dict[str, Any]] = []
    for row in prediction_rows:
        record_id = str(row.get("record_id"))
        prior_row = prior_rows.get(record_id)
        if prior_row is None:
            continue
        prior_syntax = _syntax_tuple(prior_row)
        current_syntax = _syntax_tuple(row)
        prior_score = _semantic_score(prior_row)
        current_score = _semantic_score(row)
        if current_syntax < prior_syntax or current_score + 1e-9 < prior_score:
            regressions.append(
                {
                    "record_id": record_id,
                    "prior_stage_label": prior_row.get("stage_label"),
                    "current_stage_label": row.get("stage_label"),
                    "prior_syntax": prior_syntax,
                    "current_syntax": current_syntax,
                    "prior_semantic_score": prior_score,
                    "current_semantic_score": current_score,
                    "reason": (
                        "Current stage regressed on syntax."
                        if current_syntax < prior_syntax
                        else "Current stage regressed on semantic score."
                    ),
                    "prior_output": prior_row.get("raw_output"),
                    "current_output": row.get("raw_output"),
                }
            )
    return sorted(
        regressions,
        key=lambda item: (
            item["current_semantic_score"] - item["prior_semantic_score"],
            item["current_syntax"],
        ),
    )[:5]


def build_eval_diagnostics(
    *,
    prediction_rows: list[dict[str, Any]],
    settings: EvaluationSettings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build richer per-run diagnostics alongside aggregate metrics."""

    bucket_rows = build_example_buckets(prediction_rows)
    diagnostics = {
        "qualitative_summary": (
            "Use the example buckets and per-field diagnostics for qualitative review. "
            "The free-text summary field remains qualitative and is not scored as an aggregate metric."
        ),
        "categorical_confusion_like": _categorical_confusion_like_summary(prediction_rows),
        "per_field_error_counts": _per_field_error_counts(prediction_rows),
        "bucket_counts": _bucket_counts(bucket_rows),
        "top_regressions_vs_prior_stage": _top_regressions_vs_prior_stage(
            prediction_rows,
            settings.prior_stage_predictions_path,
        ),
    }
    return diagnostics, bucket_rows


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def render_single_run_report(
    *,
    run_name: str,
    settings: EvaluationSettings,
    metrics_payload: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    diagnostics_payload: dict[str, Any],
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
            f"- Null-handling mistakes: `{diagnostics_payload['bucket_counts'].get('null_handling_mistakes', 0)}`",
            f"- Partial action extraction rows: `{diagnostics_payload['bucket_counts'].get('partial_action_extraction', 0)}`",
            "",
            "## Diagnostics",
            "",
            f"- Qualitative summary note: {diagnostics_payload['qualitative_summary']}",
            f"- Prior-stage regressions tracked: `{len(diagnostics_payload['top_regressions_vs_prior_stage'])}`",
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
) -> tuple[dict[str, Any], dict[str, Any], str, list[dict[str, Any]], list[dict[str, Any]]]:
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
    total_rows = len(rows)
    print(f"[eval] Loaded {total_rows} evaluation rows from {settings.dataset_path}.")
    row_batches = _chunk_rows(rows, settings.eval_batch_size)
    total_batches = len(row_batches)
    supports_batching = hasattr(active_backend, "generate_batch")
    for batch_index, row_batch in enumerate(row_batches, start=1):
        batch_start = ((batch_index - 1) * settings.eval_batch_size) + 1
        batch_end = batch_start + len(row_batch) - 1
        print(f"[eval] Evaluating batch {batch_index}/{total_batches}: rows {batch_start}-{batch_end}/{total_rows}")
        requests = [
            build_inference_request(row, settings.prompt_source, settings.generation)
            for row in row_batch
        ]
        if supports_batching and len(requests) > 1:
            responses = active_backend.generate_batch(requests)
        else:
            responses = [active_backend.generate(request) for request in requests]

        for row, response in zip(row_batch, responses, strict=True):
            record_id = str(row.get("record_id"))
            evaluation_records.append(
                EvaluationRecord(
                    record_id=record_id,
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
    diagnostics_payload, bucket_rows = build_eval_diagnostics(
        prediction_rows=prediction_rows,
        settings=settings,
    )
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
        "prior_stage_predictions_path": (
            str(settings.prior_stage_predictions_path) if settings.prior_stage_predictions_path else None
        ),
        "backend": settings.backend,
        "prompt_source": settings.prompt_source,
        "sample_limit": settings.sample_limit,
        "generation": settings.generation,
        "qualitative_summary": diagnostics_payload["qualitative_summary"],
        "data_pipeline": build_data_pipeline_metadata(
            repo_root=settings.config_path.parents[1],
            build_summary_path=settings.build_summary_path,
            composition_summary_path=settings.composition_summary_path,
        ),
        **aggregate_metrics,
    }
    report_text = render_single_run_report(
        run_name=run_name,
        settings=settings,
        metrics_payload=metrics_payload,
        prediction_rows=prediction_rows,
        diagnostics_payload=diagnostics_payload,
    )
    return metrics_payload, diagnostics_payload, report_text, prediction_rows, bucket_rows
