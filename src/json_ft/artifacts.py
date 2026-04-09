"""Helpers for mirroring selected small runtime outputs back into the repo."""

from __future__ import annotations

from pathlib import Path
import shutil


def mirror_small_artifact(source: str | Path, destination: str | Path, max_size_bytes: int = 1_000_000) -> Path:
    """Copy a small artifact from runtime storage into the repo.

    The helper intentionally rejects large files so checkpoints and dumps do not
    accidentally land in Git-tracked locations.
    """

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.stat().st_size > max_size_bytes:
        raise ValueError(f"Refusing to mirror large artifact: {source_path}")
    shutil.copy2(source_path, destination_path)
    return destination_path
