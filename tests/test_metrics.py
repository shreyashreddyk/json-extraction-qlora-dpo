import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.metrics import categorical_exact_match, json_validity_rate, schema_pass_rate
from json_ft.schemas import build_support_ticket_schema


class MetricsTest(unittest.TestCase):
    def test_json_validity_rate_counts_parseable_payloads(self) -> None:
        value = json_validity_rate(['{"ok": true}', "not-json"])
        self.assertEqual(value, 0.5)

    def test_schema_pass_rate_uses_support_ticket_schema(self) -> None:
        schema = build_support_ticket_schema()
        payloads = [
            {
                "summary": "Customer reports a duplicate charge and wants a refund.",
                "issue_category": "billing",
                "priority": "high",
                "product_area": "billing_portal",
                "customer": {"name": "Ava", "account_id": "AC-100", "plan_tier": "pro"},
                "sentiment": "negative",
                "requires_human_followup": True,
                "actions_requested": ["Refund the duplicate invoice charge"],
            },
            {
                "summary": "Customer cannot access the account after an MFA reset.",
                "issue_category": "account_access",
                "priority": "urgent",
                "customer": {"name": "Ben", "account_id": "BIZ-204", "plan_tier": "business"},
                "sentiment": "negative",
                "requires_human_followup": True,
            },
        ]

        self.assertEqual(schema_pass_rate(payloads, schema), 0.5)

    def test_categorical_exact_match_compares_single_field(self) -> None:
        predictions = [{"priority": "high"}, {"priority": "low"}]
        references = [{"priority": "high"}, {"priority": "medium"}]

        self.assertEqual(categorical_exact_match(predictions, references, "priority"), 0.5)
