"""Scoring helpers used to reason about preference pairs and failure analysis."""

from __future__ import annotations

from .schemas import SchemaConstraint, validate_extraction_payload


def score_payload_against_schema(payload: dict, schema: SchemaConstraint) -> int:
    """Assign a simple score based on overall schema compliance."""

    validation = validate_extraction_payload(payload, schema)
    score = 100
    score -= len(validation.missing_fields) * 25
    score -= len(validation.unexpected_fields) * 10
    other_issues = len(validation.issues) - len(validation.missing_fields) - len(
        validation.unexpected_fields
    )
    score -= max(other_issues, 0) * 15
    return max(score, 0)


def choose_better_payload(chosen: dict, rejected: dict, schema: SchemaConstraint) -> bool:
    """Return True when the chosen payload outranks the rejected payload."""

    return score_payload_against_schema(chosen, schema) >= score_payload_against_schema(
        rejected,
        schema,
    )
