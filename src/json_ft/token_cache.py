"""Optional rendered-token cache helpers for repeated SFT runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json

from .utils import read_json, write_json


@dataclass(frozen=True)
class TokenCacheStats:
    """Compact token statistics for one manifest view."""

    record_count: int
    avg_token_count: float
    max_token_count: int
    min_token_count: int
    total_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "avg_token_count": self.avg_token_count,
            "max_token_count": self.max_token_count,
            "min_token_count": self.min_token_count,
            "total_token_count": self.total_token_count,
        }


def _sha1_json(payload: dict[str, Any]) -> str:
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def build_token_cache_key(
    *,
    manifest_path: str | Path,
    rows: list[dict[str, Any]],
    model_name_or_path: str,
    max_seq_length: int,
    packing: bool,
    completion_only_loss: bool,
    mode: str,
    sample_percent: float | None = None,
    sample_seed: int | None = None,
) -> str:
    """Build a stable cache key for one rendered manifest view."""

    manifest_fingerprint = {
        "manifest_path": str(Path(manifest_path).resolve()),
        "record_count": len(rows),
        "record_ids": [str(row.get("record_id", "")) for row in rows],
    }
    payload = {
        "manifest": manifest_fingerprint,
        "model_name_or_path": model_name_or_path,
        "max_seq_length": max_seq_length,
        "packing": packing,
        "completion_only_loss": completion_only_loss,
        "mode": mode,
        "sample_percent": sample_percent,
        "sample_seed": sample_seed,
    }
    return _sha1_json(payload)[:16]


def summarize_token_counts(token_counts: list[int]) -> TokenCacheStats:
    """Summarize token lengths for one rendered view."""

    if not token_counts:
        return TokenCacheStats(
            record_count=0,
            avg_token_count=0.0,
            max_token_count=0,
            min_token_count=0,
            total_token_count=0,
        )
    return TokenCacheStats(
        record_count=len(token_counts),
        avg_token_count=sum(token_counts) / len(token_counts),
        max_token_count=max(token_counts),
        min_token_count=min(token_counts),
        total_token_count=sum(token_counts),
    )


def load_cached_token_payload(cache_dir: str | Path) -> dict[str, Any] | None:
    """Load a cache payload when it already exists."""

    metadata_path = Path(cache_dir).resolve() / "metadata.json"
    if not metadata_path.exists():
        return None
    return read_json(metadata_path)


def write_cached_token_payload(cache_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Persist rendered token metadata for reuse across runs."""

    resolved_cache_dir = Path(cache_dir).resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    return write_json(resolved_cache_dir / "metadata.json", payload)
