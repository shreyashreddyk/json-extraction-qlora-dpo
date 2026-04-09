import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.metrics import categorical_exact_match, json_validity_rate, schema_pass_rate
from json_ft.schemas import build_placeholder_schema


class MetricsTest(unittest.TestCase):
    def test_json_validity_rate_counts_parseable_payloads(self) -> None:
        value = json_validity_rate(['{"ok": true}', "not-json"])
        self.assertEqual(value, 0.5)

    def test_schema_pass_rate_uses_placeholder_schema(self) -> None:
        schema = build_placeholder_schema()
        payloads = [
            {"customer_name": "Ava", "issue_type": "billing", "priority": "high"},
            {"customer_name": "Ben", "issue_type": "access"},
        ]

        self.assertEqual(schema_pass_rate(payloads, schema), 0.5)

    def test_categorical_exact_match_compares_single_field(self) -> None:
        predictions = [{"priority": "high"}, {"priority": "low"}]
        references = [{"priority": "high"}, {"priority": "medium"}]

        self.assertEqual(categorical_exact_match(predictions, references, "priority"), 0.5)
