"""Reusable package for schema-constrained JSON extraction experiments.

The modules in this package are intentionally lightweight during the scaffold
phase. They define boundaries, contracts, and a few deterministic helpers so
the repository is ready for dataset preparation and later model work.
"""

from .dataset_adapters import PreferenceExample, RawRecord, SFTExample
from .manifests import LatestModelManifest, load_latest_model_manifest, save_latest_model_manifest
from .metrics import categorical_exact_match, json_validity_rate, schema_pass_rate
from .runtime import RuntimeContext, detect_colab, resolve_runtime_context
from .schemas import SchemaConstraint, ValidationResult, build_placeholder_schema

__all__ = [
    "PreferenceExample",
    "RawRecord",
    "SFTExample",
    "LatestModelManifest",
    "load_latest_model_manifest",
    "save_latest_model_manifest",
    "RuntimeContext",
    "detect_colab",
    "resolve_runtime_context",
    "SchemaConstraint",
    "ValidationResult",
    "build_placeholder_schema",
    "categorical_exact_match",
    "json_validity_rate",
    "schema_pass_rate",
]
