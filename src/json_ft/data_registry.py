"""Config-driven dataset registry for multi-source data builds."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .utils import load_yaml


class SourceType(str, Enum):
    """Supported source loading backends."""

    LOCAL_JSONL = "local_jsonl"
    LOCAL_CSV = "local_csv"
    HUGGINGFACE = "huggingface"
    GENERATED = "generated"


class SourceGroup(str, Enum):
    """Logical source group used for sampling and reporting."""

    DOMAIN_TASK_DATA = "domain_task_data"
    SCHEMA_DISCIPLINE_DATA = "schema_discipline_data"
    SYNTHETIC_AUGMENTATION_DATA = "synthetic_augmentation_data"


@dataclass(frozen=True)
class DatasetSource:
    """Single registry entry for one dataset source."""

    dataset_name: str
    source_type: SourceType
    license_note: str
    source_uri_or_path: str
    adapter_name: str
    source_group: SourceGroup
    default_inclusion_weight: float
    enabled_by_default: bool
    source_notes: str
    hf_split: str | None = None
    local_fixture_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_local_fixture_path(self, repo_root: Path) -> Path | None:
        """Resolve an optional repo-local fixture fallback path."""

        if not self.local_fixture_path:
            return None
        return (repo_root / self.local_fixture_path).resolve()

    def resolve_source_path(self, repo_root: Path, raw_root: Path | None = None) -> Path | None:
        """Resolve a local path when the source URI represents a file."""

        if self.source_type == SourceType.GENERATED:
            return None

        candidate = self.source_uri_or_path
        replacements = {
            "repo_root": str(repo_root.resolve()),
            "raw_root": str((raw_root or repo_root).resolve()),
        }
        for key, value in replacements.items():
            candidate = candidate.replace(f"{{{key}}}", value)

        path = Path(candidate)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return path


def _coerce_source_type(value: str) -> SourceType:
    try:
        return SourceType(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported source_type: {value}") from exc


def _coerce_source_group(value: str) -> SourceGroup:
    try:
        return SourceGroup(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported source_group: {value}") from exc


def load_dataset_registry(path: str | Path) -> list[DatasetSource]:
    """Load and validate the dataset registry YAML."""

    payload = load_yaml(path)
    rows = payload.get("sources", [])
    if not isinstance(rows, list):
        raise ValueError("configs/data_sources.yaml must define a top-level 'sources' list")

    sources: list[DatasetSource] = []
    seen_names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each dataset source entry must be a mapping")
        dataset_name = row["dataset_name"]
        if dataset_name in seen_names:
            raise ValueError(f"Duplicate dataset_name in registry: {dataset_name}")
        seen_names.add(dataset_name)
        known_keys = {
            "dataset_name",
            "source_type",
            "license_note",
            "source_uri_or_path",
            "adapter_name",
            "source_group",
            "default_inclusion_weight",
            "enabled_by_default",
            "source_notes",
            "hf_split",
            "local_fixture_path",
        }
        sources.append(
            DatasetSource(
                dataset_name=dataset_name,
                source_type=_coerce_source_type(str(row["source_type"])),
                license_note=str(row["license_note"]),
                source_uri_or_path=str(row["source_uri_or_path"]),
                adapter_name=str(row["adapter_name"]),
                source_group=_coerce_source_group(str(row["source_group"])),
                default_inclusion_weight=float(row["default_inclusion_weight"]),
                enabled_by_default=bool(row["enabled_by_default"]),
                source_notes=str(row["source_notes"]),
                hf_split=str(row["hf_split"]) if row.get("hf_split") is not None else None,
                local_fixture_path=(
                    str(row["local_fixture_path"]) if row.get("local_fixture_path") is not None else None
                ),
                extra={key: value for key, value in row.items() if key not in known_keys},
            )
        )
    return sources


def registry_by_name(sources: list[DatasetSource]) -> dict[str, DatasetSource]:
    """Return the registry keyed by dataset name."""

    return {source.dataset_name: source for source in sources}
