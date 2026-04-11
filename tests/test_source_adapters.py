import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.schemas import IssueCategory, PriorityLevel
from json_ft.source_adapters import (
    adapt_cfpb_complaint_csv_v1,
    adapt_hf_customer_support_ticket_v1,
    adapt_hf_it_helpdesk_ticket_v1,
    adapt_hf_schema_discipline_json_v1,
    adapt_json_extraction_source_row,
)
from json_ft.utils import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


class SourceAdaptersTest(unittest.TestCase):
    def test_json_extraction_adapter_preserves_existing_target(self) -> None:
        row = read_jsonl(REPO_ROOT / "data" / "fixtures" / "support_tickets.jsonl")[0]
        draft = adapt_json_extraction_source_row(row)

        self.assertEqual(draft.record_id, "support-train-001")
        self.assertEqual(draft.target.issue_category, IssueCategory.BILLING)
        self.assertEqual(draft.target.priority, PriorityLevel.HIGH)

    def test_it_helpdesk_adapter_maps_access_issue(self) -> None:
        row = read_jsonl(REPO_ROOT / "data" / "fixtures" / "source_adapter_samples" / "console_ai_it_helpdesk_synthetic_tickets.jsonl")[0]
        draft = adapt_hf_it_helpdesk_ticket_v1(row)

        self.assertEqual(draft.target.issue_category, IssueCategory.ACCOUNT_ACCESS)
        self.assertEqual(draft.target.priority, PriorityLevel.HIGH)
        self.assertIn("VPN access", draft.target.summary)

    def test_customer_support_adapter_maps_general_question(self) -> None:
        row = read_jsonl(REPO_ROOT / "data" / "fixtures" / "source_adapter_samples" / "prady06_customer_support_tickets.jsonl")[1]
        draft = adapt_hf_customer_support_ticket_v1(row)

        self.assertEqual(draft.target.issue_category, IssueCategory.GENERAL_QUESTION)
        self.assertEqual(draft.target.priority, PriorityLevel.LOW)
        self.assertFalse(draft.target.requires_human_followup)

    def test_cfpb_adapter_maps_financial_complaint(self) -> None:
        import csv

        with (REPO_ROOT / "data" / "fixtures" / "source_adapter_samples" / "cfpb_consumer_complaints.csv").open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        draft = adapt_cfpb_complaint_csv_v1(row)

        self.assertEqual(draft.target.issue_category, IssueCategory.BILLING)
        self.assertEqual(draft.target.customer.account_id, None)
        self.assertTrue(draft.target.requires_human_followup)

    def test_schema_discipline_adapter_rejects_incompatible_output(self) -> None:
        rows = read_jsonl(REPO_ROOT / "data" / "fixtures" / "source_adapter_samples" / "suneeldk_text_json.jsonl")
        compatible = adapt_hf_schema_discipline_json_v1(rows[0])
        self.assertEqual(compatible.target.issue_category, IssueCategory.BILLING)
        with self.assertRaises(Exception):
            adapt_hf_schema_discipline_json_v1(rows[1])


if __name__ == "__main__":
    unittest.main()
