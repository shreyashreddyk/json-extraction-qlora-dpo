"""Deterministic row-subsetting helpers shared across training stages."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from random import Random
from typing import Any


@dataclass(frozen=True)
class SampleSelectionMetadata:
    """Stable metadata describing one row-subsetting decision."""

    original_row_count: int
    selected_row_count: int
    selected_fraction: float
    sample_mode: str
    sample_seed: int | None
    sample_percent: float | None
    sample_limit: int | None
    percent_row_count: int | None
    absolute_limit_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_row_count": self.original_row_count,
            "selected_row_count": self.selected_row_count,
            "selected_fraction": self.selected_fraction,
            "sample_mode": self.sample_mode,
            "sample_seed": self.sample_seed,
            "sample_percent": self.sample_percent,
            "sample_limit": self.sample_limit,
            "percent_row_count": self.percent_row_count,
            "absolute_limit_applied": self.absolute_limit_applied,
        }


@dataclass(frozen=True)
class SampleSelection:
    """Selected rows plus stable metadata for summaries and manifests."""

    rows: list[Any]
    metadata: SampleSelectionMetadata


def _validate_sample_percent(sample_percent: float | None) -> float | None:
    if sample_percent in (None, 1.0):
        return sample_percent
    if sample_percent <= 0 or sample_percent > 1.0:
        raise ValueError("sample_percent must be within (0, 1.0].")
    return float(sample_percent)


def select_rows(
    rows: list[Any],
    *,
    sample_limit: int | None = None,
    sample_percent: float | None = None,
    sample_seed: int = 17,
) -> SampleSelection:
    """Select a deterministic subset of rows using percent and/or absolute limit."""

    validated_percent = _validate_sample_percent(sample_percent)
    original_row_count = len(rows)
    percent_row_count: int | None = None
    selected_rows = list(rows)

    if validated_percent not in (None, 1.0):
        percent_row_count = max(1, floor(original_row_count * validated_percent)) if original_row_count else 0
        indexed_rows = list(enumerate(rows))
        Random(sample_seed).shuffle(indexed_rows)
        indexed_rows = indexed_rows[:percent_row_count]
        indexed_rows.sort(key=lambda item: item[0])
        selected_rows = [row for _, row in indexed_rows]

    absolute_limit_applied = sample_limit is not None
    if sample_limit is not None:
        selected_rows = selected_rows[: int(sample_limit)]

    selected_row_count = len(selected_rows)
    selected_fraction = (
        selected_row_count / original_row_count if original_row_count else 0.0
    )
    sample_mode = "deterministic_percent" if validated_percent not in (None, 1.0) else "full"

    return SampleSelection(
        rows=selected_rows,
        metadata=SampleSelectionMetadata(
            original_row_count=original_row_count,
            selected_row_count=selected_row_count,
            selected_fraction=selected_fraction,
            sample_mode=sample_mode,
            sample_seed=sample_seed if validated_percent not in (None, 1.0) else None,
            sample_percent=validated_percent,
            sample_limit=sample_limit,
            percent_row_count=percent_row_count,
            absolute_limit_applied=absolute_limit_applied,
        ),
    )
