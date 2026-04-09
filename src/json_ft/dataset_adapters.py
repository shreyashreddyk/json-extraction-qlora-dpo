"""Dataset record shapes and conversion helpers for future training data."""

from __future__ import annotations

from dataclasses import dataclass

from .formatting import format_json_payload
from .prompts import render_extraction_prompt


@dataclass(frozen=True)
class RawRecord:
    """Raw natural-language example paired with a gold JSON object."""

    record_id: str
    input_text: str
    target_json: dict


@dataclass(frozen=True)
class SFTExample:
    """Instruction tuning example produced from a raw record."""

    record_id: str
    prompt: str
    completion: str


@dataclass(frozen=True)
class PreferenceExample:
    """Chosen vs rejected pair aligned to a single prompt."""

    record_id: str
    prompt: str
    chosen: str
    rejected: str


def build_sft_example(record: RawRecord) -> SFTExample:
    """Convert a raw record into a prompt-completion training example."""

    return SFTExample(
        record_id=record.record_id,
        prompt=render_extraction_prompt(record.input_text),
        completion=format_json_payload(record.target_json),
    )


def build_preference_example(
    record_id: str,
    prompt: str,
    chosen_payload: dict,
    rejected_payload: dict,
) -> PreferenceExample:
    """Create a preference pair from two structured candidate payloads."""

    return PreferenceExample(
        record_id=record_id,
        prompt=prompt,
        chosen=format_json_payload(chosen_payload),
        rejected=format_json_payload(rejected_payload),
    )

