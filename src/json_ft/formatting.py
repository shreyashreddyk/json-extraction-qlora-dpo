"""Formatting utilities shared across prompts, datasets, and evaluation."""

from __future__ import annotations

import json


def format_json_payload(payload: dict) -> str:
    """Return a stable pretty-printed JSON string."""

    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences often produced by baseline models."""

    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")]
    return cleaned.strip()

