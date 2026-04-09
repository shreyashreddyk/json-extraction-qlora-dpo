"""Dataset adapters and export helpers for support-ticket extraction data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .prompts import render_extraction_prompt, render_system_instruction, render_user_prompt
from .schemas import (
    SchemaConstraint,
    SupportTicketExtraction,
    build_support_ticket_schema,
    dump_support_ticket_payload,
    format_support_ticket_json,
    load_support_ticket_model,
)


class DatasetSplit(str, Enum):
    """Explicit splits used for deterministic manifest generation."""

    TRAIN = "train"
    EVAL = "eval"


class JsonExtractionSample(BaseModel):
    """Canonical task-specific record used across SFT and evaluation prep."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    record_id: str
    split: DatasetSplit
    source_dataset: str
    input_text: str = Field(..., min_length=1)
    target: SupportTicketExtraction
    metadata: dict[str, Any] = Field(default_factory=dict)


class NemotronSftSourceRecord(BaseModel):
    """Source record shaped like NVIDIA Nemotron JSONL SFT data."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    record_id: str
    split: DatasetSplit
    source_dataset: str
    input: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    system: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class SFTExample:
    """Prompt-completion style example compatible with TRL SFTTrainer."""

    record_id: str
    prompt: str
    completion: str


@dataclass(frozen=True)
class MessagesSFTExample:
    """Conversational example for TRL chat-template aware SFT training."""

    record_id: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class NemotronSFTExample:
    """Nemotron JSONL-style SFT record with input, output, and system fields."""

    record_id: str
    input: str
    output: str
    system: str | None


@dataclass(frozen=True)
class PreferenceExample:
    """Future DPO-ready record with placeholders for labeling decisions."""

    record_id: str
    prompt: str
    reference_completion: str
    chosen: str | None
    rejected: str | None
    labeling_status: str


def adapt_json_extraction_record(record: dict[str, Any]) -> JsonExtractionSample:
    """Validate a canonical task-specific support-ticket sample."""

    return JsonExtractionSample.model_validate(record)


def adapt_nemotron_sft_record(record: dict[str, Any]) -> JsonExtractionSample:
    """Convert a Nemotron-style SFT record into the canonical sample shape."""

    source_record = NemotronSftSourceRecord.model_validate(record)
    metadata = dict(source_record.metadata)
    if source_record.system:
        metadata["source_system"] = source_record.system
    return JsonExtractionSample(
        record_id=source_record.record_id,
        split=source_record.split,
        source_dataset=source_record.source_dataset,
        input_text=source_record.input,
        target=load_support_ticket_model(source_record.output),
        metadata=metadata,
    )


def adapt_source_record(record: dict[str, Any], source_format: str) -> JsonExtractionSample:
    """Dispatch source records into the shared canonical sample shape."""

    if source_format == "json_extraction":
        return adapt_json_extraction_record(record)
    if source_format == "nemotron_sft":
        return adapt_nemotron_sft_record(record)
    raise ValueError(f"Unsupported source format: {source_format}")


def build_sft_example(
    sample: JsonExtractionSample,
    schema: SchemaConstraint | None = None,
) -> SFTExample:
    """Convert a canonical sample into prompt-completion training format."""

    active_schema = schema or build_support_ticket_schema()
    return SFTExample(
        record_id=sample.record_id,
        prompt=render_extraction_prompt(sample.input_text, active_schema),
        completion=format_support_ticket_json(sample.target, active_schema),
    )


def build_messages_sft_example(
    sample: JsonExtractionSample,
    schema: SchemaConstraint | None = None,
) -> MessagesSFTExample:
    """Convert a canonical sample into conversational training format."""

    active_schema = schema or build_support_ticket_schema()
    return MessagesSFTExample(
        record_id=sample.record_id,
        messages=[
            {"role": "system", "content": render_system_instruction(active_schema)},
            {"role": "user", "content": render_user_prompt(sample.input_text)},
            {
                "role": "assistant",
                "content": format_support_ticket_json(sample.target, active_schema),
            },
        ],
    )


def build_nemotron_sft_example(
    sample: JsonExtractionSample,
    schema: SchemaConstraint | None = None,
) -> NemotronSFTExample:
    """Convert a canonical sample into Nemotron JSONL SFT format."""

    active_schema = schema or build_support_ticket_schema()
    return NemotronSFTExample(
        record_id=sample.record_id,
        input=render_user_prompt(sample.input_text),
        output=format_support_ticket_json(sample.target, active_schema),
        system=render_system_instruction(active_schema),
    )


def build_preference_placeholder(
    sample: JsonExtractionSample,
    schema: SchemaConstraint | None = None,
) -> PreferenceExample:
    """Create an unlabeled preference example for future DPO annotation."""

    sft_example = build_sft_example(sample, schema)
    return PreferenceExample(
        record_id=sample.record_id,
        prompt=sft_example.prompt,
        reference_completion=sft_example.completion,
        chosen=None,
        rejected=None,
        labeling_status="todo",
    )


def build_preference_example(
    record_id: str,
    prompt: str,
    chosen_payload: dict[str, Any],
    rejected_payload: dict[str, Any],
    schema: SchemaConstraint | None = None,
) -> PreferenceExample:
    """Create a fully labeled preference pair from two candidate payloads."""

    active_schema = schema or build_support_ticket_schema()
    return PreferenceExample(
        record_id=record_id,
        prompt=prompt,
        reference_completion=format_support_ticket_json(chosen_payload, active_schema),
        chosen=format_support_ticket_json(chosen_payload, active_schema),
        rejected=format_support_ticket_json(rejected_payload, active_schema),
        labeling_status="labeled",
    )


def prompt_completion_record(sample: JsonExtractionSample) -> dict[str, Any]:
    """Return a JSON-serializable prompt-completion training record."""

    example = build_sft_example(sample)
    return {
        "record_id": example.record_id,
        "prompt": example.prompt,
        "completion": example.completion,
    }


def messages_record(sample: JsonExtractionSample) -> dict[str, Any]:
    """Return a JSON-serializable conversational training record."""

    example = build_messages_sft_example(sample)
    return {
        "record_id": example.record_id,
        "messages": example.messages,
    }


def nemotron_sft_record(sample: JsonExtractionSample) -> dict[str, Any]:
    """Return a JSON-serializable Nemotron SFT record."""

    example = build_nemotron_sft_example(sample)
    return asdict(example)


def preference_placeholder_record(sample: JsonExtractionSample) -> dict[str, Any]:
    """Return a JSON-serializable placeholder preference record."""

    return asdict(build_preference_placeholder(sample))


def eval_manifest_record(sample: JsonExtractionSample) -> dict[str, Any]:
    """Return the canonical evaluation manifest row for held-out scoring."""

    prompt_completion = build_sft_example(sample)
    messages = build_messages_sft_example(sample)
    return {
        "record_id": sample.record_id,
        "split": sample.split.value,
        "source_dataset": sample.source_dataset,
        "input_text": sample.input_text,
        "reference_payload": dump_support_ticket_payload(sample.target),
        "reference_json": prompt_completion.completion,
        "prompt": prompt_completion.prompt,
        "messages": messages.messages,
        "metadata": sample.metadata,
    }
