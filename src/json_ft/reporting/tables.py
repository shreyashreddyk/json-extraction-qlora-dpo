"""Table builders for final analysis and reporting."""

from __future__ import annotations

from typing import Any

from .loaders import ReportingBundle


def _safe_round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def _stage_order(bundle: ReportingBundle) -> list[tuple[str, Any]]:
    return [
        ("baseline", bundle.baseline),
        ("sft", bundle.sft),
        ("dpo", bundle.dpo),
    ]


def build_dataset_composition_table(bundle: ReportingBundle) -> list[dict[str, object]]:
    """Return per-source dataset composition rows."""

    composition = bundle.composition_summary or {}
    rows = composition.get("rows", [])
    table_rows: list[dict[str, object]] = []
    for row in rows:
        table_rows.append(
            {
                "source_dataset": row.get("source_dataset"),
                "split": row.get("split"),
                "row_count": row.get("row_count", 0),
                "synthetic_row_count": row.get("synthetic_row_count", 0),
                "synthetic_row_rate": _safe_round(row.get("synthetic_row_rate", 0.0), 4),
            }
        )
    return table_rows


def build_stage_metrics_table(bundle: ReportingBundle) -> list[dict[str, object]]:
    """Return one stage comparison table with syntax and semantic metrics."""

    rows: list[dict[str, object]] = []
    for stage_name, stage in _stage_order(bundle):
        metrics = stage.metrics
        if not metrics:
            continue
        rows.append(
            {
                "stage": stage_name,
                "json_validity_rate": _safe_round(metrics.get("json_validity_rate")),
                "schema_validation_pass_rate": _safe_round(metrics.get("schema_validation_pass_rate")),
                "hallucinated_field_rate": _safe_round(metrics.get("hallucinated_field_rate")),
                "json_recovery_rate": _safe_round(metrics.get("json_recovery_rate")),
                "field_level_micro_f1": _safe_round(metrics.get("field_level", {}).get("micro", {}).get("f1")),
                "field_level_macro_f1": _safe_round(metrics.get("field_level", {}).get("macro", {}).get("f1")),
                "latency_mean_ms": _safe_round(metrics.get("latency_ms", {}).get("mean"), 2),
                "latency_p95_ms": _safe_round(metrics.get("latency_ms", {}).get("p95"), 2),
            }
        )
    return rows


def build_stage_delta_table(bundle: ReportingBundle) -> list[dict[str, object]]:
    """Return saved stage deltas for syntax and semantics."""

    comparison = bundle.comparison_summary or {}
    deltas = comparison.get("deltas", {})
    table_rows: list[dict[str, object]] = []
    for delta_name in ("sft_vs_baseline", "dpo_vs_sft", "dpo_vs_baseline"):
        payload = deltas.get(delta_name, {})
        row = {"comparison": delta_name}
        for metric_name, value in (payload.get("syntax", {}) or {}).items():
            row[metric_name] = _safe_round(value)
        for metric_name, value in (payload.get("semantic", {}) or {}).items():
            row[metric_name] = _safe_round(value)
        table_rows.append(row)
    return table_rows


def build_field_level_table(bundle: ReportingBundle, metric: str = "f1") -> list[dict[str, object]]:
    """Return per-field metrics across baseline, SFT, and DPO."""

    supported_metrics = {"precision", "recall", "f1", "exact_match"}
    if metric not in supported_metrics:
        raise ValueError(f"Unsupported metric: {metric}")

    field_names: list[str] = []
    if metric == "exact_match":
        field_names = list((bundle.baseline.metrics or {}).get("categorical_exact_match", {}).keys())
    else:
        field_names = list(
            (bundle.baseline.metrics or {}).get("field_level", {}).get("per_field", {}).keys()
        )

    rows: list[dict[str, object]] = []
    for field_name in field_names:
        baseline_value = _metric_value(bundle.baseline.metrics, field_name, metric)
        sft_value = _metric_value(bundle.sft.metrics, field_name, metric)
        dpo_value = _metric_value(bundle.dpo.metrics, field_name, metric)
        rows.append(
            {
                "field": field_name,
                "baseline": _safe_round(baseline_value),
                "sft": _safe_round(sft_value),
                "dpo": _safe_round(dpo_value),
                "sft_delta_vs_baseline": _safe_round(_delta(baseline_value, sft_value)),
                "dpo_delta_vs_sft": _safe_round(_delta(sft_value, dpo_value)),
                "dpo_delta_vs_baseline": _safe_round(_delta(baseline_value, dpo_value)),
            }
        )
    return rows


def _metric_value(metrics: dict[str, Any] | None, field_name: str, metric: str) -> float | None:
    if not metrics:
        return None
    if metric == "exact_match":
        value = (metrics.get("categorical_exact_match") or {}).get(field_name)
        return float(value) if isinstance(value, (int, float)) else None
    value = (metrics.get("field_level", {}).get("per_field", {}).get(field_name, {}) or {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def build_pair_quality_table(bundle: ReportingBundle) -> list[dict[str, object]]:
    """Return preference pair-quality summary rows."""

    summary = bundle.preference_summary or {}
    diagnostics = bundle.preference_diagnostics or {}
    if not summary and not diagnostics:
        return []

    pair_quality_by_source = diagnostics.get("pair_quality_by_source_dataset") or summary.get(
        "pair_quality_by_source_dataset",
        {},
    )
    rows: list[dict[str, object]] = [
        {
            "scope": "overall",
            "source_row_count": summary.get("source_row_count"),
            "pair_count": summary.get("pair_count"),
            "pair_emission_rate": _safe_round(summary.get("pair_emission_rate")),
            "candidate_json_valid_rate": _safe_round(summary.get("candidate_json_valid_rate")),
            "candidate_schema_pass_rate": _safe_round(summary.get("candidate_schema_pass_rate")),
            "chosen_schema_valid_rate": _safe_round(summary.get("chosen_schema_valid_rate")),
            "rejected_schema_valid_rate": _safe_round(summary.get("rejected_schema_valid_rate")),
            "average_numeric_score_gap": _safe_round(
                (summary.get("average_chosen_vs_rejected_score_gap") or {}).get("numeric_score_gap")
            ),
        }
    ]
    for source_dataset, source_metrics in sorted(pair_quality_by_source.items()):
        rows.append(
            {
                "scope": source_dataset,
                "source_row_count": source_metrics.get("source_row_count"),
                "pair_count": source_metrics.get("pair_count"),
                "pair_emission_rate": _safe_round(source_metrics.get("pair_emission_rate")),
                "skipped_counts": source_metrics.get("skipped_counts", {}),
            }
        )
    return rows


def build_failure_bucket_table(bundle: ReportingBundle) -> list[dict[str, object]]:
    """Return saved bucket counts by stage."""

    labels = sorted(
        {
            *(((bundle.baseline.diagnostics or {}).get("bucket_counts") or {}).keys()),
            *(((bundle.sft.diagnostics or {}).get("bucket_counts") or {}).keys()),
            *(((bundle.dpo.diagnostics or {}).get("bucket_counts") or {}).keys()),
        }
    )
    rows: list[dict[str, object]] = []
    for label in labels:
        rows.append(
            {
                "bucket": label,
                "baseline": (bundle.baseline.diagnostics or {}).get("bucket_counts", {}).get(label, 0),
                "sft": (bundle.sft.diagnostics or {}).get("bucket_counts", {}).get(label, 0),
                "dpo": (bundle.dpo.diagnostics or {}).get("bucket_counts", {}).get(label, 0),
            }
        )
    return rows
