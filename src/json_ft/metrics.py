"""Deterministic metric helpers for schema-constrained extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import json

from .formatting import strip_code_fences
from .schemas import SchemaConstraint, ValidationResult, validate_extraction_payload

CATEGORICAL_EXACT_MATCH_FIELDS = (
    "issue_category",
    "priority",
    "product_area",
    "sentiment",
    "requires_human_followup",
    "customer.plan_tier",
)

STRUCTURED_PRF_FIELDS = (
    "issue_category",
    "priority",
    "product_area",
    "customer.name",
    "customer.account_id",
    "customer.plan_tier",
    "sentiment",
    "requires_human_followup",
)

LIST_PRF_FIELDS = ("actions_requested",)

_MISSING = object()


@dataclass(frozen=True)
class EvaluationRecord:
    """Single prediction/reference pair used for aggregate scoring."""

    record_id: str
    reference_payload: dict[str, Any]
    raw_output: str
    parsed_payload: dict[str, Any] | None
    validation: ValidationResult | None
    latency_ms: float
    json_recovery_used: bool = False


@dataclass
class MetricCounts:
    """Simple precision/recall/F1 accumulator."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        if denominator == 0:
            return 0.0
        return self.tp / denominator

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        if denominator == 0:
            return 0.0
        return self.tp / denominator

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metric summary."""

        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.tp + self.fn,
        }


def _safe_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _nested_value(payload: dict[str, Any] | None, field_path: str) -> Any:
    if payload is None:
        return _MISSING
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _value_token(field_path: str, value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return f"{field_path}={encoded}"


def _list_tokens(field_path: str, value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    cleaned_values = {str(item).strip() for item in value if str(item).strip()}
    return {f"{field_path}[]={json.dumps(item, ensure_ascii=True)}" for item in cleaned_values}


def _field_counts_for_record(
    predicted_payload: dict[str, Any] | None,
    reference_payload: dict[str, Any],
    field_path: str,
) -> MetricCounts:
    predicted_value = _nested_value(predicted_payload, field_path)
    reference_value = _nested_value(reference_payload, field_path)
    predicted_token = None if predicted_value is _MISSING else _value_token(field_path, predicted_value)
    reference_token = None if reference_value is _MISSING else _value_token(field_path, reference_value)

    counts = MetricCounts()
    if predicted_token is not None and predicted_token == reference_token:
        counts.tp += 1
        return counts
    if predicted_token is not None:
        counts.fp += 1
    if reference_token is not None:
        counts.fn += 1
    return counts


def _list_counts_for_record(
    predicted_payload: dict[str, Any] | None,
    reference_payload: dict[str, Any],
    field_path: str,
) -> MetricCounts:
    predicted_tokens = _list_tokens(field_path, _nested_value(predicted_payload, field_path))
    reference_tokens = _list_tokens(field_path, _nested_value(reference_payload, field_path))
    overlap = predicted_tokens & reference_tokens

    return MetricCounts(
        tp=len(overlap),
        fp=len(predicted_tokens - overlap),
        fn=len(reference_tokens - overlap),
    )


def json_validity_rate(candidates: Iterable[str]) -> float:
    """Measure the share of model outputs that parse as JSON objects."""

    items = list(candidates)
    if not items:
        return 0.0
    valid_count = 0
    for candidate in items:
        try:
            parsed = json.loads(strip_code_fences(candidate))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            valid_count += 1
    return valid_count / len(items)


def schema_pass_rate(
    payloads: Iterable[dict],
    schema: SchemaConstraint,
) -> float:
    """Measure the share of payloads that pass full schema validation."""

    items = list(payloads)
    if not items:
        return 0.0
    passed = sum(1 for payload in items if validate_extraction_payload(payload, schema).is_valid)
    return passed / len(items)


def categorical_exact_match(
    predictions: Iterable[dict],
    references: Iterable[dict],
    field_name: str,
) -> float:
    """Measure exact match rate for a single categorical field."""

    prediction_items = list(predictions)
    reference_items = list(references)
    if not prediction_items or len(prediction_items) != len(reference_items):
        return 0.0
    matched = 0
    for prediction, reference in zip(prediction_items, reference_items, strict=True):
        if _nested_value(prediction, field_name) == _nested_value(reference, field_name):
            matched += 1
    return matched / len(reference_items)


def evaluate_records(
    records: Iterable[EvaluationRecord],
    schema: SchemaConstraint,
) -> dict[str, Any]:
    """Compute aggregate baseline metrics across scored evaluation rows."""

    items = list(records)
    total_count = len(items)
    latencies_ms = [record.latency_ms for record in items]
    parsed_predictions = [
        record.parsed_payload for record in items if record.parsed_payload is not None
    ]

    exact_match_metrics = {
        field_name: categorical_exact_match(
            [record.parsed_payload or {} for record in items],
            [record.reference_payload for record in items],
            field_name,
        )
        for field_name in CATEGORICAL_EXACT_MATCH_FIELDS
    }

    per_field_counts: dict[str, MetricCounts] = {
        field_name: MetricCounts() for field_name in (*STRUCTURED_PRF_FIELDS, *LIST_PRF_FIELDS)
    }
    for record in items:
        for field_name in STRUCTURED_PRF_FIELDS:
            counts = _field_counts_for_record(
                record.parsed_payload,
                record.reference_payload,
                field_name,
            )
            per_field_counts[field_name].tp += counts.tp
            per_field_counts[field_name].fp += counts.fp
            per_field_counts[field_name].fn += counts.fn
        for field_name in LIST_PRF_FIELDS:
            counts = _list_counts_for_record(
                record.parsed_payload,
                record.reference_payload,
                field_name,
            )
            per_field_counts[field_name].tp += counts.tp
            per_field_counts[field_name].fp += counts.fp
            per_field_counts[field_name].fn += counts.fn

    micro_counts = MetricCounts()
    for counts in per_field_counts.values():
        micro_counts.tp += counts.tp
        micro_counts.fp += counts.fp
        micro_counts.fn += counts.fn

    macro_precisions = [counts.precision for counts in per_field_counts.values()]
    macro_recalls = [counts.recall for counts in per_field_counts.values()]
    macro_f1s = [counts.f1 for counts in per_field_counts.values()]

    json_valid_count = sum(1 for record in items if record.parsed_payload is not None)
    schema_valid_count = sum(
        1 for record in items if record.validation is not None and record.validation.is_valid
    )
    json_recovery_count = sum(1 for record in items if record.json_recovery_used)
    hallucinated_prediction_count = sum(
        1
        for record in items
        if record.validation is not None and bool(record.validation.unexpected_fields)
    )
    parse_failure_count = sum(1 for record in items if record.parsed_payload is None)

    return {
        "record_count": total_count,
        "json_validity_rate": json_valid_count / total_count if total_count else 0.0,
        "schema_validation_pass_rate": schema_valid_count / total_count if total_count else 0.0,
        "hallucinated_field_rate": (
            hallucinated_prediction_count / total_count if total_count else 0.0
        ),
        "json_recovery_rate": json_recovery_count / total_count if total_count else 0.0,
        "counts": {
            "json_valid_count": json_valid_count,
            "schema_validation_pass_count": schema_valid_count,
            "parse_failure_count": parse_failure_count,
            "schema_failure_count": total_count - schema_valid_count,
            "hallucinated_prediction_count": hallucinated_prediction_count,
            "json_recovery_count": json_recovery_count,
        },
        "categorical_exact_match": exact_match_metrics,
        "field_level": {
            "micro": micro_counts.to_dict(),
            "macro": {
                "precision": _safe_average(macro_precisions),
                "recall": _safe_average(macro_recalls),
                "f1": _safe_average(macro_f1s),
            },
            "per_field": {
                field_name: counts.to_dict() for field_name, counts in per_field_counts.items()
            },
        },
        "latency_ms": {
            "mean": _safe_average(latencies_ms),
            "min": min(latencies_ms) if latencies_ms else 0.0,
            "max": max(latencies_ms) if latencies_ms else 0.0,
            "p50": _percentile(latencies_ms, 0.50),
            "p95": _percentile(latencies_ms, 0.95),
        },
        "schema": {
            "name": schema.name,
            "version": schema.version,
        },
        "parsed_prediction_count": len(parsed_predictions),
    }
