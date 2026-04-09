import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.dataset_adapters import build_preference_example
from json_ft.scoring import choose_better_payload
from json_ft.schemas import build_placeholder_schema


class PreferencePairBuilderTest(unittest.TestCase):
    def test_preference_example_serializes_chosen_and_rejected_payloads(self) -> None:
        example = build_preference_example(
            record_id="record-1",
            prompt="Prompt",
            chosen_payload={"customer_name": "Ava", "issue_type": "billing", "priority": "high"},
            rejected_payload={"customer_name": "Ava", "issue_type": "billing"},
        )

        self.assertIn('"priority": "high"', example.chosen)
        self.assertIn('"issue_type": "billing"', example.rejected)

    def test_choose_better_payload_prefers_schema_complete_output(self) -> None:
        schema = build_placeholder_schema()
        chosen = {"customer_name": "Ava", "issue_type": "billing", "priority": "high"}
        rejected = {"customer_name": "Ava", "issue_type": "billing"}

        self.assertTrue(choose_better_payload(chosen, rejected, schema))
