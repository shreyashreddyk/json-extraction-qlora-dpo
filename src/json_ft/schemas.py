"""Schema contracts and validation helpers for structured extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SchemaConstraint:
    """Small placeholder schema contract used during the scaffold phase."""

    name: str
    version: str
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()
    field_descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def allowed_fields(self) -> set[str]:
        """Return the fields accepted by the current schema."""

        return set(self.required_fields) | set(self.optional_fields)


@dataclass(frozen=True)
class ValidationResult:
    """Summary of schema-level validation checks."""

    is_valid: bool
    missing_fields: tuple[str, ...]
    unexpected_fields: tuple[str, ...]


def build_placeholder_schema() -> SchemaConstraint:
    """Return a minimal example schema for early tests and documentation."""

    return SchemaConstraint(
        name="support_ticket_extraction",
        version="0.1.0",
        required_fields=("customer_name", "issue_type", "priority"),
        optional_fields=("summary", "actions_requested"),
        field_descriptions={
            "customer_name": "Name of the customer or account holder.",
            "issue_type": "Primary category for the issue described in text.",
            "priority": "Urgency bucket chosen from the task taxonomy.",
            "summary": "Concise natural-language summary of the issue.",
            "actions_requested": "List of requested follow-up actions.",
        },
    )


def validate_extraction_payload(
    payload: dict[str, Any],
    schema: SchemaConstraint | None = None,
) -> ValidationResult:
    """Validate required and unexpected keys for a candidate JSON payload."""

    active_schema = schema or build_placeholder_schema()
    missing_fields = tuple(
        field_name for field_name in active_schema.required_fields if field_name not in payload
    )
    unexpected_fields = tuple(
        sorted(field_name for field_name in payload if field_name not in active_schema.allowed_fields)
    )
    return ValidationResult(
        is_valid=not missing_fields and not unexpected_fields,
        missing_fields=missing_fields,
        unexpected_fields=unexpected_fields,
    )

