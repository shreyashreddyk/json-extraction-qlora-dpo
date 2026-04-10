import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.metrics import (
    EvaluationRecord,
    categorical_exact_match,
    evaluate_records,
    json_validity_rate,
    schema_pass_rate,
)
from json_ft.schemas import build_support_ticket_schema, validate_extraction_payload


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


class MetricsTest(unittest.TestCase):
    def test_json_validity_rate_counts_parseable_payloads(self) -> None:
        value = json_validity_rate(['{"ok": true}', "not-json", "```json\n{}\n```"])
        self.assertEqual(value, 2 / 3)

    def test_schema_pass_rate_uses_support_ticket_schema(self) -> None:
        schema = build_support_ticket_schema()
        payloads = [
            valid_payload(),
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

    def test_categorical_exact_match_supports_nested_fields(self) -> None:
        predictions = [
            {"customer": {"plan_tier": "pro"}},
            {"customer": {"plan_tier": None}},
        ]
        references = [
            {"customer": {"plan_tier": "pro"}},
            {"customer": {"plan_tier": "business"}},
        ]

        self.assertEqual(categorical_exact_match(predictions, references, "customer.plan_tier"), 0.5)

    def test_evaluate_records_reports_hallucinations_and_exact_match(self) -> None:
        schema = build_support_ticket_schema()
        perfect = valid_payload()

        hallucinated = valid_payload()
        hallucinated["priority"] = "high"
        hallucinated["extra_field"] = "unexpected"

        records = [
            EvaluationRecord(
                record_id="record-1",
                reference_payload=valid_payload(),
                raw_output="{}",
                parsed_payload=perfect,
                validation=validate_extraction_payload(perfect, schema),
                latency_ms=12.0,
            ),
            EvaluationRecord(
                record_id="record-2",
                reference_payload=valid_payload(),
                raw_output="{}",
                parsed_payload=hallucinated,
                validation=validate_extraction_payload(hallucinated, schema),
                latency_ms=18.0,
                json_recovery_used=True,
            ),
        ]

        result = evaluate_records(records, schema)

        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["json_validity_rate"], 1.0)
        self.assertEqual(result["schema_validation_pass_rate"], 0.5)
        self.assertEqual(result["hallucinated_field_rate"], 0.5)
        self.assertEqual(result["json_recovery_rate"], 0.5)
        self.assertEqual(result["categorical_exact_match"]["priority"], 0.5)
        self.assertEqual(result["counts"]["hallucinated_prediction_count"], 1)

    def test_evaluate_records_computes_field_level_prf_and_actions_overlap(self) -> None:
        schema = build_support_ticket_schema()
        reference = valid_payload()
        predicted = valid_payload()
        predicted["customer"] = {"account_id": "AC-100", "plan_tier": "pro"}
        predicted["actions_requested"] = ["Restore admin access"]

        result = evaluate_records(
            [
                EvaluationRecord(
                    record_id="record-1",
                    reference_payload=reference,
                    raw_output="{}",
                    parsed_payload=predicted,
                    validation=validate_extraction_payload(predicted, schema),
                    latency_ms=9.0,
                )
            ],
            schema,
        )

        customer_name_metrics = result["field_level"]["per_field"]["customer.name"]
        actions_metrics = result["field_level"]["per_field"]["actions_requested"]

        self.assertEqual(customer_name_metrics["tp"], 0)
        self.assertEqual(customer_name_metrics["fn"], 1)
        self.assertEqual(customer_name_metrics["precision"], 0.0)
        self.assertEqual(customer_name_metrics["recall"], 0.0)
        self.assertAlmostEqual(actions_metrics["precision"], 1.0)
        self.assertAlmostEqual(actions_metrics["recall"], 0.5)
        self.assertAlmostEqual(actions_metrics["f1"], 2 / 3)
