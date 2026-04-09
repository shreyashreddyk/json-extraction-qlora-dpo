"""Shared utility helpers for paths, file IO, and reproducible scaffolding."""

from __future__ import annotations

from pathlib import Path
import json


def repo_root(start: str | Path | None = None) -> Path:
    """Return the repository root based on the location of this module."""

    if start is not None:
        return Path(start).resolve()
    return Path(__file__).resolve().parents[2]


def ensure_directory(path: str | Path) -> Path:
    """Create a directory when needed and return its resolved path."""

    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_text(path: str | Path) -> str:
    """Read a UTF-8 text file."""

    return Path(path).read_text(encoding="utf-8")


def write_json(path: str | Path, payload: dict) -> Path:
    """Write a JSON file with stable formatting."""

    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return resolved
