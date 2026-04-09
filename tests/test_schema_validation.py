import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.schemas import (
    build_support_ticket_schema,
    dump_support_ticket_payload,
    export_support_ticket_json_schema,
    parse_candidate_json,
    validate_extraction_payload,
)


def valid_payload() -> dict:
    return {
        "summary": "Customer cannot sign in after an MFA reset and needs access restored.",
        "issue_category": "account_access",
        "priority": "urgent",
        "product_area": "account_portal",
        "customer": {
            "name": "Ava Cole",
            "account_id": "AC-100",
            "plan_tier": "pro",
        },
        "sentiment": "negative",
        "requires_human_followup": True,
        "actions_requested": ["Restore admin access", "Reset MFA enrollment"],
    }


class SchemaValidationTest(unittest.TestCase):
    def test_valid_payload_normalizes_successfully(self) -> None:
        result = validate_extraction_payload(valid_payload(), build_support_ticket_schema())

        self.assertTrue(result.is_valid)
        assert result.normalized_payload is not None
        self.assertEqual(result.normalized_payload["priority"], "urgent")
        self.assertEqual(result.normalized_payload["customer"]["plan_tier"], "pro")

    def test_schema_validation_rejects_missing_required_field(self) -> None:
        payload = valid_payload()
        payload.pop("summary")

        result = validate_extraction_payload(payload, build_support_ticket_schema())

        self.assertFalse(result.is_valid)
        self.assertEqual(result.missing_fields, ("summary",))

    def test_schema_validation_rejects_invalid_enum_values(self) -> None:
        payload = valid_payload()
        payload["priority"] = "p1-now"

        result = validate_extraction_payload(payload, build_support_ticket_schema())

        self.assertFalse(result.is_valid)
        self.assertTrue(any(issue.path == ("priority",) for issue in result.issues))

    def test_schema_validation_rejects_unexpected_nested_field(self) -> None:
        payload = valid_payload()
        payload["customer"]["email"] = "unexpected@example.com"

        result = validate_extraction_payload(payload, build_support_ticket_schema())

        self.assertFalse(result.is_valid)
        self.assertEqual(result.unexpected_fields, ("customer.email",))

    def test_parse_candidate_json_handles_fenced_json(self) -> None:
        candidate = """```json
        {
          "summary": "Customer cannot sign in after an MFA reset and needs access restored.",
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
          "actions_requested": ["Restore admin access", "Reset MFA enrollment"]
        }
        ```"""

        parsed = parse_candidate_json(candidate)

        self.assertEqual(parsed["customer"]["account_id"], "AC-100")

    def test_dump_payload_converts_blank_customer_fields_to_null(self) -> None:
        payload = valid_payload()
        payload["customer"]["name"] = "  "
        payload["customer"]["account_id"] = ""
        payload["customer"]["plan_tier"] = None

        normalized = dump_support_ticket_payload(payload)

        self.assertIsNone(normalized["customer"]["name"])
        self.assertIsNone(normalized["customer"]["account_id"])
        self.assertIsNone(normalized["customer"]["plan_tier"])

    def test_export_support_ticket_json_schema_lists_required_fields(self) -> None:
        schema = export_support_ticket_json_schema()

        self.assertEqual(schema["title"], "SupportTicketExtraction")
        self.assertIn("summary", schema["required"])
        self.assertIn("customer", schema["required"])
