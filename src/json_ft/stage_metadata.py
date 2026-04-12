"""Shared helpers for attaching data-pipeline provenance to stage artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json


def _resolve_repo_path(repo_root: str | Path, path_value: str | Path | None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (Path(repo_root).resolve() / path).resolve()


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return read_json(path)


def build_data_pipeline_metadata(
    *,
    repo_root: str | Path,
    build_summary_path: str | Path | None,
    composition_summary_path: str | Path | None,
) -> dict[str, Any]:
    """Load the latest dataset-build metadata referenced by a stage."""

    resolved_build_summary_path = _resolve_repo_path(repo_root, build_summary_path)
    resolved_composition_summary_path = _resolve_repo_path(repo_root, composition_summary_path)
    build_summary = _load_optional_json(resolved_build_summary_path)
    composition_summary = _load_optional_json(resolved_composition_summary_path)

    composition_payload = composition_summary or {}
    if "summary" in composition_payload and isinstance(composition_payload["summary"], dict):
        composition_excerpt = composition_payload["summary"]
    else:
        composition_excerpt = composition_payload

    summary_excerpt: dict[str, Any] = {}
    if build_summary:
        summary_excerpt = {
            "profile": build_summary.get("profile"),
            "total_rows": build_summary.get("total_rows"),
            "split_counts": build_summary.get("split_counts"),
            "source_counts": build_summary.get("source_counts"),
            "source_group_counts": build_summary.get("source_group_counts"),
            "synthetic_row_rate": build_summary.get("synthetic_row_rate"),
            "nullable_field_null_rates": build_summary.get("nullable_field_null_rates"),
            "schema": build_summary.get("schema"),
            "leakage_checks": build_summary.get("leakage_checks"),
        }

    return {
        "build_summary_path": str(resolved_build_summary_path) if resolved_build_summary_path else None,
        "composition_summary_path": (
            str(resolved_composition_summary_path) if resolved_composition_summary_path else None
        ),
        "build_summary": summary_excerpt,
        "source_composition_snapshot": {
            "source_counts": composition_excerpt.get("source_counts"),
            "source_group_counts": composition_excerpt.get("source_group_counts"),
            "synthetic_row_rate": composition_excerpt.get("synthetic_row_rate"),
            "issue_category_counts": composition_excerpt.get("issue_category_counts"),
            "priority_counts": composition_excerpt.get("priority_counts"),
        },
    }
