"""Deterministic train-only text hardening for support-ticket extraction."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
import hashlib
import re

from .dataset_adapters import JsonExtractionSample
from .schemas import CustomerContext, dump_support_ticket_payload, load_support_ticket_model


AUGMENTATION_ORDER = (
    "paraphrase_ticket_text",
    "longer_noisier_ticket",
    "missing_field_ticket",
    "category_confusable_ticket",
    "irrelevant_prose_ticket",
    "ambiguous_customer_info_ticket",
    "correct_null_ticket",
)


def _replace_customer_context(text: str, sample: JsonExtractionSample) -> tuple[str, dict[str, Any]] | None:
    customer = sample.target.customer
    updated_text = text
    removed_fields: dict[str, Any] = {}

    if customer.name:
        pattern = re.escape(customer.name)
        if re.search(pattern, updated_text):
            updated_text = re.sub(pattern, "our team", updated_text, count=1)
            removed_fields["name"] = None
    if customer.account_id:
        pattern = re.escape(customer.account_id)
        if re.search(pattern, updated_text):
            updated_text = re.sub(pattern, "the affected account", updated_text, count=1)
            removed_fields["account_id"] = None
    if customer.plan_tier:
        plan_pattern = re.escape(customer.plan_tier.value)
        updated_text = re.sub(plan_pattern, "current plan", updated_text, flags=re.IGNORECASE, count=1)
        if updated_text != text:
            removed_fields["plan_tier"] = None

    if updated_text == text:
        return None
    return updated_text, removed_fields


def _updated_target(sample: JsonExtractionSample, **customer_updates: Any):
    payload = dump_support_ticket_payload(sample.target)
    customer_payload = dict(payload["customer"])
    customer_payload.update(customer_updates)
    payload["customer"] = customer_payload
    return load_support_ticket_model(payload)


def build_augmented_sample(
    sample: JsonExtractionSample,
    augmentation_kind: str,
    index: int,
) -> JsonExtractionSample | None:
    """Return one deterministic augmentation or None when it does not apply."""

    base_text = sample.input_text
    target = sample.target

    if augmentation_kind == "paraphrase_ticket_text":
        text = (
            "Support request rewrite:\n"
            f"{base_text}\n\n"
            "Restated briefly: the customer wants the issue handled without changing the expected JSON facts."
        )
        new_target = target
    elif augmentation_kind == "longer_noisier_ticket":
        text = (
            f"{base_text}\n\n"
            "Additional context: this came up during an internal handoff, and the customer included several"
            " environment details that do not change the core request. Please focus on the actual support"
            " issue, not the scheduling noise."
        )
        new_target = target
    elif augmentation_kind == "missing_field_ticket":
        replacement = _replace_customer_context(base_text, sample)
        if replacement is None:
            return None
        text, customer_updates = replacement
        text = (
            f"{text}\n\n"
            "Some identifying details are intentionally omitted in this version, so null customer fields can be correct."
        )
        new_target = _updated_target(sample, **customer_updates)
    elif augmentation_kind == "category_confusable_ticket":
        text = (
            f"{base_text}\n\n"
            "Clarification: there may be other minor annoyances around the account, but the primary request stays the same"
            " and should not be reclassified because of incidental mentions."
        )
        new_target = target
    elif augmentation_kind == "irrelevant_prose_ticket":
        text = (
            "Hi team, thanks in advance for taking a look.\n\n"
            f"{base_text}\n\n"
            "Unrelated note: we are also updating some internal docs next week, which is not part of this ticket."
        )
        new_target = target
    elif augmentation_kind == "ambiguous_customer_info_ticket":
        replacement = _replace_customer_context(base_text, sample)
        if replacement is None:
            return None
        text, customer_updates = replacement
        text = (
            f"{text}\n\n"
            "The customer references their workspace indirectly here, so the extractor should prefer nulls over guesses."
        )
        new_target = _updated_target(sample, **customer_updates)
    elif augmentation_kind == "correct_null_ticket":
        if any(getattr(sample.target.customer, field) is not None for field in ("name", "account_id", "plan_tier")):
            return None
        text = (
            f"{base_text}\n\n"
            "No extra customer identifiers are available beyond what is written here."
        )
        new_target = target
    else:
        raise ValueError(f"Unsupported augmentation_kind: {augmentation_kind}")

    metadata = dict(sample.metadata)
    metadata.update(
        {
            "adapter_name": "synthetic_hardening_v1",
            "source_group": "synthetic_augmentation_data",
            "source_type": "generated",
            "source_uri_or_path": "generated://synthetic_hardening_v1",
            "source_record_id": f"{sample.record_id}__aug_{index:03d}",
            "license_note": "Generated train-only augmentation derived from accepted canonical task rows.",
            "synthetic": True,
            "augmentation_kind": augmentation_kind,
            "parent_record_id": sample.record_id,
            "lineage_root_id": sample.metadata.get("lineage_root_id", sample.record_id),
            "augmentation_index": index,
            "mapping_version": "v1",
            "ingested_at_utc": datetime.now(UTC).isoformat(),
            "raw_hash": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        }
    )
    return JsonExtractionSample(
        record_id=f"{sample.record_id}__aug_{index:03d}",
        split=sample.split,
        source_dataset="synthetic_hardening_v1",
        input_text=text,
        target=new_target,
        metadata=metadata,
    )


def generate_augmentations(
    samples: list[JsonExtractionSample],
    *,
    max_generated_rows: int,
) -> list[JsonExtractionSample]:
    """Generate deterministic train-only augmentations with a global cap."""

    generated: list[JsonExtractionSample] = []
    index = 0
    for sample in samples:
        for augmentation_kind in AUGMENTATION_ORDER:
            if len(generated) >= max_generated_rows:
                return generated
            augmented = build_augmented_sample(sample, augmentation_kind, index)
            index += 1
            if augmented is not None:
                generated.append(augmented)
    return generated
