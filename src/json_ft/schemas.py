"""Schema contracts and validation helpers for support-ticket extraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .formatting import strip_code_fences

SCHEMA_NAME = "support_ticket_extraction"
SCHEMA_VERSION = "1.0.0"


class IssueCategory(str, Enum):
    """Primary issue taxonomy for routing support tickets."""

    BILLING = "billing"
    ACCOUNT_ACCESS = "account_access"
    TECHNICAL_BUG = "technical_bug"
    FEATURE_REQUEST = "feature_request"
    INTEGRATION = "integration"
    GENERAL_QUESTION = "general_question"
    OTHER = "other"


class PriorityLevel(str, Enum):
    """Urgency bucket inferred from the ticket text."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ProductArea(str, Enum):
    """Product surface most relevant to the reported issue."""

    WEB_APP = "web_app"
    MOBILE_APP = "mobile_app"
    API = "api"
    BILLING_PORTAL = "billing_portal"
    ACCOUNT_PORTAL = "account_portal"
    INTEGRATIONS = "integrations"
    OTHER = "other"
    UNKNOWN = "unknown"


class SentimentLabel(str, Enum):
    """Coarse customer sentiment for failure analysis and prioritization."""

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    MIXED = "mixed"


class PlanTier(str, Enum):
    """Customer plan tier when it is known from the text."""

    FREE = "free"
    PRO = "pro"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class CustomerContext(BaseModel):
    """Customer metadata nested inside the extracted support ticket payload."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str | None = Field(
        default=None,
        description="Customer or account-contact name when it is explicitly mentioned.",
    )
    account_id: str | None = Field(
        default=None,
        description="Customer account identifier when it is explicitly mentioned.",
    )
    plan_tier: PlanTier | None = Field(
        default=None,
        description="Subscription plan when it is explicitly mentioned.",
    )

    @field_validator("name", "account_id")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """Collapse blank strings to null after whitespace stripping."""

        if value == "":
            return None
        return value


class SupportTicketExtraction(BaseModel):
    """Strict support-ticket JSON extraction schema for v1 dataset preparation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(
        ...,
        description="Short summary of the issue and requested outcome.",
        min_length=1,
    )
    issue_category: IssueCategory = Field(
        ...,
        description="Primary support taxonomy label for the ticket.",
    )
    priority: PriorityLevel = Field(
        ...,
        description="Urgency bucket derived from the described impact.",
    )
    product_area: ProductArea = Field(
        ...,
        description="Product surface most relevant to the issue.",
    )
    customer: CustomerContext = Field(
        ...,
        description="Customer metadata captured from the ticket.",
    )
    sentiment: SentimentLabel = Field(
        ...,
        description="Overall customer sentiment expressed in the ticket.",
    )
    requires_human_followup: bool = Field(
        ...,
        description="Whether the request likely needs a human support action.",
    )
    actions_requested: list[str] = Field(
        default_factory=list,
        description="Explicit actions the customer wants support to take.",
    )

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        """Keep summary non-empty after normalization."""

        if not value:
            raise ValueError("summary must not be empty")
        return value

    @field_validator("actions_requested", mode="before")
    @classmethod
    def default_actions_requested(cls, value: Any) -> Any:
        """Normalize missing or null requested actions to an empty list."""

        if value is None:
            return []
        return value

    @field_validator("actions_requested")
    @classmethod
    def validate_actions_requested(cls, values: list[str]) -> list[str]:
        """Strip actions and reject empty placeholders."""

        cleaned = [value.strip() for value in values if value.strip()]
        if len(cleaned) != len(values):
            raise ValueError("actions_requested entries must be non-empty strings")
        return cleaned


@dataclass(frozen=True)
class SchemaConstraint:
    """Lightweight schema metadata used across prompts, metrics, and docs."""

    name: str
    version: str
    model: type[SupportTicketExtraction]

    @property
    def required_fields(self) -> tuple[str, ...]:
        """Return required top-level fields for the schema."""

        return tuple(
            field_name
            for field_name, field_info in self.model.model_fields.items()
            if field_info.is_required()
        )

    @property
    def optional_fields(self) -> tuple[str, ...]:
        """Return optional top-level fields for the schema."""

        return tuple(
            field_name
            for field_name, field_info in self.model.model_fields.items()
            if not field_info.is_required()
        )

    @property
    def field_descriptions(self) -> dict[str, str]:
        """Expose human-readable descriptions for top-level fields."""

        return {
            field_name: field_info.description or ""
            for field_name, field_info in self.model.model_fields.items()
        }


@dataclass(frozen=True)
class ValidationIssue:
    """Single schema validation issue extracted from a pydantic error."""

    path: tuple[str, ...]
    issue_type: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    """Structured summary of schema validation checks."""

    is_valid: bool
    normalized_payload: dict[str, Any] | None
    issues: tuple[ValidationIssue, ...]
    missing_fields: tuple[str, ...]
    unexpected_fields: tuple[str, ...]


def build_support_ticket_schema() -> SchemaConstraint:
    """Return the strict support-ticket extraction schema metadata."""

    return SchemaConstraint(
        name=SCHEMA_NAME,
        version=SCHEMA_VERSION,
        model=SupportTicketExtraction,
    )


def schema_metadata(schema: SchemaConstraint | None = None) -> dict[str, Any]:
    """Return compact metadata for the active schema."""

    active_schema = schema or build_support_ticket_schema()
    return {
        "name": active_schema.name,
        "version": active_schema.version,
        "required_fields": list(active_schema.required_fields),
        "optional_fields": list(active_schema.optional_fields),
        "field_descriptions": active_schema.field_descriptions,
    }


def export_support_ticket_json_schema() -> dict[str, Any]:
    """Return the JSON Schema representation for the extraction payload."""

    return SupportTicketExtraction.model_json_schema()


def parse_candidate_json(candidate: str) -> dict[str, Any]:
    """Parse a JSON object from plain text or fenced Markdown code blocks."""

    parsed = json.loads(strip_code_fences(candidate))
    if not isinstance(parsed, dict):
        raise ValueError("candidate JSON must be an object")
    return parsed


def load_support_ticket_model(
    payload: SupportTicketExtraction | dict[str, Any] | str,
    schema: SchemaConstraint | None = None,
) -> SupportTicketExtraction:
    """Validate a payload and return the normalized pydantic model."""

    active_schema = schema or build_support_ticket_schema()
    if isinstance(payload, active_schema.model):
        return payload
    candidate_payload = parse_candidate_json(payload) if isinstance(payload, str) else payload
    return active_schema.model.model_validate(candidate_payload)


def dump_support_ticket_payload(
    payload: SupportTicketExtraction | dict[str, Any] | str,
    schema: SchemaConstraint | None = None,
) -> dict[str, Any]:
    """Validate a payload and return a normalized dictionary representation."""

    return load_support_ticket_model(payload, schema).model_dump(mode="json")


def format_support_ticket_json(
    payload: SupportTicketExtraction | dict[str, Any] | str,
    schema: SchemaConstraint | None = None,
) -> str:
    """Serialize a normalized payload to stable pretty-printed JSON."""

    normalized_payload = dump_support_ticket_payload(payload, schema)
    return json.dumps(normalized_payload, indent=2, sort_keys=True, ensure_ascii=True)


def validate_extraction_payload(
    payload: dict[str, Any] | str,
    schema: SchemaConstraint | None = None,
) -> ValidationResult:
    """Validate a candidate payload against the support-ticket schema."""

    active_schema = schema or build_support_ticket_schema()
    try:
        normalized_payload = dump_support_ticket_payload(payload, active_schema)
    except (ValidationError, ValueError, TypeError) as exc:
        if isinstance(exc, ValidationError):
            issues = tuple(
                ValidationIssue(
                    path=tuple(str(part) for part in error["loc"]),
                    issue_type=str(error["type"]),
                    message=str(error["msg"]),
                )
                for error in exc.errors()
            )
        else:
            issues = (
                ValidationIssue(
                    path=(),
                    issue_type=type(exc).__name__,
                    message=str(exc),
                ),
            )
        missing_fields = tuple(
            ".".join(issue.path) for issue in issues if issue.issue_type == "missing"
        )
        unexpected_fields = tuple(
            ".".join(issue.path) for issue in issues if issue.issue_type == "extra_forbidden"
        )
        return ValidationResult(
            is_valid=False,
            normalized_payload=None,
            issues=issues,
            missing_fields=missing_fields,
            unexpected_fields=unexpected_fields,
        )
    return ValidationResult(
        is_valid=True,
        normalized_payload=normalized_payload,
        issues=(),
        missing_fields=(),
        unexpected_fields=(),
    )
