"""Repo-side manifests for promoted model pointers and run metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class LatestModelManifest:
    """Metadata pointer for the latest successful promoted model stage."""

    stage: str
    status: str
    base_model: str
    adapter_path: str
    merged_export_path: str | None = None
    schema_version: str | None = None
    config_paths: list[str] = field(default_factory=list)
    metrics_paths: list[str] = field(default_factory=list)
    report_paths: list[str] = field(default_factory=list)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def manifest_path(repo_root: str | Path) -> Path:
    """Return the repo-side latest-model manifest location."""

    return Path(repo_root).resolve() / "artifacts" / "checkpoints" / "latest_model.json"


def save_latest_model_manifest(repo_root: str | Path, manifest: LatestModelManifest) -> Path:
    """Write the latest-model manifest to the repo artifact directory."""

    path = manifest_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_latest_model_manifest(repo_root: str | Path) -> LatestModelManifest | None:
    """Load the latest-model manifest when it exists."""

    path = manifest_path(repo_root)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LatestModelManifest(**payload)


def manifest_to_dict(manifest: LatestModelManifest) -> dict[str, Any]:
    """Expose a manifest as a JSON-serializable dictionary."""

    return asdict(manifest)
