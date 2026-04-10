"""Reusable package for schema-constrained JSON extraction experiments."""

from .dataset_adapters import (
    JsonExtractionSample,
    MessagesSFTExample,
    NemotronSFTExample,
    PreferenceExample,
    SFTExample,
)
from .manifests import LatestModelManifest, load_latest_model_manifest, save_latest_model_manifest
from .metrics import (
    EvaluationRecord,
    categorical_exact_match,
    evaluate_records,
    json_validity_rate,
    schema_pass_rate,
)
from .runtime import RuntimeContext, detect_colab, resolve_runtime_context
from .schemas import (
    CustomerContext,
    IssueCategory,
    PlanTier,
    PriorityLevel,
    ProductArea,
    SchemaConstraint,
    SentimentLabel,
    SupportTicketExtraction,
    ValidationIssue,
    ValidationResult,
    build_support_ticket_schema,
)

__all__ = [
    "CustomerContext",
    "EvaluationRecord",
    "IssueCategory",
    "JsonExtractionSample",
    "LatestModelManifest",
    "MessagesSFTExample",
    "NemotronSFTExample",
    "PlanTier",
    "PreferenceExample",
    "PriorityLevel",
    "ProductArea",
    "RuntimeContext",
    "SchemaConstraint",
    "SentimentLabel",
    "SFTExample",
    "SupportTicketExtraction",
    "ValidationIssue",
    "ValidationResult",
    "build_support_ticket_schema",
    "categorical_exact_match",
    "detect_colab",
    "evaluate_records",
    "json_validity_rate",
    "load_latest_model_manifest",
    "resolve_runtime_context",
    "save_latest_model_manifest",
    "schema_pass_rate",
]
