"""Prompt template helpers for schema-constrained support-ticket extraction."""

from __future__ import annotations

from enum import Enum

from .schemas import (
    IssueCategory,
    PlanTier,
    PriorityLevel,
    ProductArea,
    SchemaConstraint,
    SentimentLabel,
    build_support_ticket_schema,
)


def _enum_values(enum_type: type[Enum]) -> str:
    """Render enum choices as a compact prompt-friendly string."""

    return " | ".join(item.value for item in enum_type)


def render_schema_overview(schema: SchemaConstraint | None = None) -> str:
    """Render a concise but explicit schema overview for prompts and docs."""

    active_schema = schema or build_support_ticket_schema()
    return "\n".join(
        [
            f"Schema name: {active_schema.name}",
            f"Schema version: {active_schema.version}",
            "Top-level JSON object fields:",
            "- summary: string",
            f"- issue_category: {_enum_values(IssueCategory)}",
            f"- priority: {_enum_values(PriorityLevel)}",
            f"- product_area: {_enum_values(ProductArea)}",
            "- customer: object with fields name, account_id, plan_tier",
            f"- customer.plan_tier: {_enum_values(PlanTier)} | null",
            f"- sentiment: {_enum_values(SentimentLabel)}",
            "- requires_human_followup: boolean",
            "- actions_requested: list[string]",
        ]
    )


def render_system_instruction(schema: SchemaConstraint | None = None) -> str:
    """Build the system instruction shared across prompt-completion and chat data."""

    schema_block = render_schema_overview(schema)
    return (
        "You are a structured support-ticket extraction assistant.\n"
        "Return only valid JSON that follows the schema exactly.\n"
        "Use null for unknown customer fields.\n"
        "Use [] when the customer did not request any explicit action.\n"
        "Do not add keys that are not defined by the schema.\n\n"
        f"{schema_block}"
    )


def render_user_prompt(input_text: str) -> str:
    """Render the user-facing extraction request around the raw ticket text."""

    return (
        "Extract the support ticket into the schema described above.\n\n"
        "Ticket text:\n"
        f"{input_text.strip()}"
    )


def render_extraction_prompt(input_text: str, schema: SchemaConstraint | None = None) -> str:
    """Build a prompt-completion friendly extraction prompt."""

    return f"{render_system_instruction(schema)}\n\n{render_user_prompt(input_text)}\n"
