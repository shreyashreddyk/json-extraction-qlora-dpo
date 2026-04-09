import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.schemas import build_placeholder_schema, validate_extraction_payload


class SchemaValidationTest(unittest.TestCase):
    def test_schema_validation_rejects_missing_required_field(self) -> None:
        schema = build_placeholder_schema()
        payload = {
            "customer_name": "Ava",
            "issue_type": "billing",
        }

        result = validate_extraction_payload(payload, schema)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.missing_fields, ("priority",))

    def test_schema_validation_rejects_unexpected_field(self) -> None:
        schema = build_placeholder_schema()
        payload = {
            "customer_name": "Ava",
            "issue_type": "billing",
            "priority": "high",
            "mystery_field": "unexpected",
        }

        result = validate_extraction_payload(payload, schema)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.unexpected_fields, ("mystery_field",))
