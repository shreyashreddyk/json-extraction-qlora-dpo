"""Shared utility helpers for paths, file IO, and reproducible scaffolding."""

from __future__ import annotations

from pathlib import Path
from typing import Any
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


def write_text(path: str | Path, contents: str) -> Path:
    """Write a UTF-8 text file."""

    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(contents, encoding="utf-8")
    return resolved


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {Path(path).resolve()}")
    return payload


def write_json(path: str | Path, payload: dict) -> Path:
    """Write a JSON file with stable formatting."""

    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return resolved


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to load repository config files. "
            "Install it in your environment before running config-driven scripts."
        ) from exc

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {Path(path).resolve()}")
    return payload


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dictionaries."""

    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"JSONL path does not exist: {resolved}")
    rows: list[dict] = []
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object on line {line_number} in {resolved}")
        rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> Path:
    """Write dictionaries to JSONL with stable key ordering."""

    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    contents = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rows)
    resolved.write_text(f"{contents}\n" if contents else "", encoding="utf-8")
    return resolved
