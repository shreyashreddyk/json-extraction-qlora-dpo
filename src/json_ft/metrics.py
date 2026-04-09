"""Deterministic metric helpers for schema-constrained extraction."""

from __future__ import annotations

import json
from typing import Iterable

from .schemas import SchemaConstraint, validate_extraction_payload


def json_validity_rate(candidates: Iterable[str]) -> float:
    """Measure the share of model outputs that parse as JSON."""

    items = list(candidates)
    if not items:
        return 0.0
    valid_count = 0
    for candidate in items:
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
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
        if prediction.get(field_name) == reference.get(field_name):
            matched += 1
    return matched / len(reference_items)
