import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.dataset_adapters import (
    adapt_json_extraction_record,
    build_preference_example,
    build_preference_placeholder,
)
from json_ft.prompts import render_extraction_prompt
from json_ft.scoring import choose_better_payload
from json_ft.schemas import build_support_ticket_schema


def sample_record() -> dict:
    return {
        "record_id": "support-train-001",
        "split": "train",
        "source_dataset": "synthetic_support_tickets_v1",
        "input_text": "Ava Cole says the billing portal charged the April invoice twice and wants a refund.",
        "target": {
            "summary": "Customer was charged twice for the April invoice and requests a refund.",
            "issue_category": "billing",
            "priority": "high",
            "product_area": "billing_portal",
            "customer": {
                "name": "Ava Cole",
                "account_id": "AC-100",
                "plan_tier": "pro",
            },
            "sentiment": "negative",
            "requires_human_followup": True,
            "actions_requested": ["Refund the duplicate invoice charge"],
        },
    }


class PreferencePairBuilderTest(unittest.TestCase):
    def test_preference_example_serializes_chosen_and_rejected_payloads(self) -> None:
        example = build_preference_example(
            record_id="record-1",
            prompt="Prompt",
            chosen_payload=sample_record()["target"],
            rejected_payload={
                "summary": "Customer was charged twice.",
                "issue_category": "billing",
                "priority": "high",
                "product_area": "billing_portal",
                "customer": {"name": "Ava Cole", "account_id": "AC-100", "plan_tier": "pro"},
                "sentiment": "negative",
                "requires_human_followup": True,
                "actions_requested": [],
            },
        )

        self.assertEqual(example.labeling_status, "labeled")
        assert example.chosen is not None
        assert example.rejected is not None
        self.assertIn('"priority": "high"', example.chosen)
        self.assertIn('"actions_requested": []', example.rejected)

    def test_preference_placeholder_keeps_reference_completion_and_empty_labels(self) -> None:
        example = build_preference_placeholder(adapt_json_extraction_record(sample_record()))

        self.assertEqual(example.labeling_status, "todo")
        self.assertIsNone(example.chosen)
        self.assertIsNone(example.rejected)
        self.assertIn('"issue_category": "billing"', example.reference_completion)

    def test_choose_better_payload_prefers_schema_complete_output(self) -> None:
        schema = build_support_ticket_schema()
        chosen = sample_record()["target"]
        rejected = {
            "summary": "Customer was charged twice for the April invoice and requests a refund.",
            "issue_category": "billing",
            "priority": "high",
            "product_area": "billing_portal",
            "customer": {"name": "Ava Cole", "account_id": "AC-100", "plan_tier": "pro"},
            "sentiment": "negative",
            "requires_human_followup": True,
        }

        self.assertTrue(choose_better_payload(chosen, rejected, schema))

    def test_preference_prompt_can_reuse_training_prompt(self) -> None:
        prompt = render_extraction_prompt(sample_record()["input_text"])
        self.assertIn("Return only valid JSON", prompt)
