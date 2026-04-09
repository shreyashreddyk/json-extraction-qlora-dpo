import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.dataset_adapters import (
    adapt_json_extraction_record,
    adapt_nemotron_sft_record,
    messages_record,
    nemotron_sft_record,
    prompt_completion_record,
)
from json_ft.formatting import strip_code_fences
from json_ft.prompts import render_extraction_prompt


def fixture_record() -> dict:
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


class PromptFormattingTest(unittest.TestCase):
    def test_strip_code_fences_handles_json_blocks(self) -> None:
        fenced = "```json\n{\n  \"ok\": true\n}\n```"
        self.assertEqual(strip_code_fences(fenced), '{\n  "ok": true\n}')

    def test_render_extraction_prompt_mentions_schema_rules(self) -> None:
        prompt = render_extraction_prompt("Customer reports a duplicate charge in billing.")

        self.assertIn("summary", prompt)
        self.assertIn("issue_category", prompt)
        self.assertIn("requires_human_followup", prompt)
        self.assertIn("Use null for unknown customer fields", prompt)
        self.assertIn("Use [] when the customer did not request any explicit action", prompt)

    def test_prompt_completion_record_keeps_gold_json_only_in_completion(self) -> None:
        sample = adapt_json_extraction_record(fixture_record())

        record = prompt_completion_record(sample)

        self.assertIn('"issue_category": "billing"', record["completion"])
        self.assertNotIn('"issue_category": "billing"', record["prompt"])
        self.assertIn("Ticket text:", record["prompt"])

    def test_messages_record_places_gold_json_in_assistant_turn_only(self) -> None:
        sample = adapt_json_extraction_record(fixture_record())

        record = messages_record(sample)

        self.assertEqual([message["role"] for message in record["messages"]], ["system", "user", "assistant"])
        self.assertIn('"priority": "high"', record["messages"][2]["content"])
        self.assertNotIn('"priority": "high"', record["messages"][0]["content"])
        self.assertNotIn('"priority": "high"', record["messages"][1]["content"])

    def test_nemotron_export_keeps_json_in_output_field(self) -> None:
        sample = adapt_json_extraction_record(fixture_record())

        record = nemotron_sft_record(sample)

        self.assertIn('"customer"', record["output"])
        self.assertNotIn('"customer"', record["input"])
        self.assertIn("Return only valid JSON", record["system"])

    def test_nemotron_source_record_is_normalized_into_canonical_sample(self) -> None:
        record = adapt_nemotron_sft_record(
            {
                "record_id": "nemotron-001",
                "split": "eval",
                "source_dataset": "nemotron_fixture",
                "input": "Customer cannot access the account portal after MFA reset.",
                "output": """```json
                {
                  "summary": "Customer cannot access the account portal after an MFA reset.",
                  "issue_category": "account_access",
                  "priority": "urgent",
                  "product_area": "account_portal",
                  "customer": {
                    "name": "Ava Cole",
                    "account_id": "AC-100",
                    "plan_tier": "pro"
                  },
                  "sentiment": "negative",
                  "requires_human_followup": true,
                  "actions_requested": ["Restore admin access"]
                }
                ```""",
                "system": "Return only JSON.",
            }
        )

        self.assertEqual(record.record_id, "nemotron-001")
        self.assertEqual(record.target.priority.value, "urgent")
        self.assertEqual(record.metadata["source_system"], "Return only JSON.")
