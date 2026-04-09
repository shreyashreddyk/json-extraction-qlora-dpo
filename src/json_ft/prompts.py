"""Prompt template helpers for schema-constrained extraction."""

from __future__ import annotations

from .schemas import SchemaConstraint, build_placeholder_schema


def render_schema_overview(schema: SchemaConstraint | None = None) -> str:
    """Render a readable schema overview for prompt templates."""

    active_schema = schema or build_placeholder_schema()
    lines = [
        f"Schema name: {active_schema.name}",
        f"Schema version: {active_schema.version}",
        "Required fields:",
    ]
    lines.extend(f"- {field_name}" for field_name in active_schema.required_fields)
    if active_schema.optional_fields:
        lines.append("Optional fields:")
        lines.extend(f"- {field_name}" for field_name in active_schema.optional_fields)
    return "\n".join(lines)


def render_extraction_prompt(input_text: str, schema: SchemaConstraint | None = None) -> str:
    """Build a baseline extraction prompt aligned to the active schema."""

    schema_block = render_schema_overview(schema)
    return (
        "You are a structured extraction assistant.\n"
        "Return only valid JSON that satisfies the schema below.\n\n"
        f"{schema_block}\n\n"
        "Input text:\n"
        f"{input_text.strip()}\n"
    )

