import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.utils import read_json, read_text, write_json, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT_PATH = REPO_ROOT / "scripts" / "compare_stages.py"


def load_compare_script_module():
    spec = importlib.util.spec_from_file_location("compare_stages_script", COMPARE_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE_ONE = {
    "issue_category": "billing",
    "priority": "urgent",
    "product_area": "billing_portal",
    "customer": {"name": "Ava Cole", "account_id": "AC-100", "plan_tier": "pro"},
    "sentiment": "negative",
    "requires_human_followup": True,
    "actions_requested": ["Refund the duplicate charge", "Confirm auto-pay"],
}

REFERENCE_TWO = {
    "issue_category": "general_question",
    "priority": "low",
    "product_area": "unknown",
    "customer": {"name": None, "account_id": None, "plan_tier": None},
    "sentiment": "neutral",
    "requires_human_followup": False,
    "actions_requested": ["Provide documentation about regional data residency availability"],
}


def prediction_row(record_id: str, reference_payload: dict, parsed_payload: dict | None, *, raw_output: str, schema_is_valid: bool = True, unexpected_fields: list[str] | None = None) -> dict:
    return {
        "record_id": record_id,
        "input_text": f"input {record_id}",
        "reference_payload": reference_payload,
        "parsed_payload": parsed_payload,
        "raw_output": raw_output,
        "parse_error": None if parsed_payload is not None else "parse error",
        "schema_is_valid": schema_is_valid,
        "unexpected_fields": unexpected_fields or [],
    }


def metrics_payload(stage: str, micro_f1: float, macro_f1: float, schema_pass: float) -> dict:
    return {
        "stage": stage,
        "run_name": f"{stage}-run",
        "model_name_or_path": f"{stage}-model",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_path": None if stage == "baseline" else f"/tmp/{stage}",
        "merged_model_path": None,
        "model_manifest_path": None,
        "json_validity_rate": 1.0,
        "schema_validation_pass_rate": schema_pass,
        "hallucinated_field_rate": 0.0,
        "json_recovery_rate": 0.0,
        "field_level": {
            "micro": {"f1": micro_f1},
            "macro": {"f1": macro_f1},
        },
        "categorical_exact_match": {
            "issue_category": micro_f1,
            "priority": micro_f1,
            "product_area": micro_f1,
            "sentiment": micro_f1,
            "requires_human_followup": micro_f1,
            "customer.plan_tier": micro_f1,
        },
        "latency_ms": {"mean": 100.0, "p95": 150.0},
    }


class ComparisonReportTest(unittest.TestCase):
    def test_compare_stages_writes_consolidated_summary_and_report(self) -> None:
        module = load_compare_script_module()

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            runtime_root = tmp_path / "runtime"

            baseline_metrics_path = tmp_path / "baseline_metrics.json"
            sft_metrics_path = tmp_path / "sft_metrics.json"
            dpo_metrics_path = tmp_path / "dpo_metrics.json"
            baseline_predictions_path = tmp_path / "baseline_predictions.jsonl"
            sft_predictions_path = tmp_path / "sft_predictions.jsonl"
            dpo_predictions_path = tmp_path / "dpo_predictions.jsonl"

            write_json(baseline_metrics_path, metrics_payload("baseline", 0.40, 0.42, 1.0))
            write_json(sft_metrics_path, metrics_payload("sft", 0.70, 0.72, 1.0))
            write_json(dpo_metrics_path, metrics_payload("dpo", 0.68, 0.70, 0.95))

            baseline_predictions = [
                prediction_row(
                    "support-eval-001",
                    REFERENCE_ONE,
                    {
                        **REFERENCE_ONE,
                        "priority": "high",
                    },
                    raw_output='{"priority": "high"}',
                ),
                prediction_row(
                    "support-eval-002",
                    REFERENCE_TWO,
                    {
                        **REFERENCE_TWO,
                        "requires_human_followup": True,
                    },
                    raw_output='{"requires_human_followup": true}',
                ),
            ]
            sft_predictions = [
                prediction_row(
                    "support-eval-001",
                    REFERENCE_ONE,
                    {
                        **REFERENCE_ONE,
                        "priority": "high",
                    },
                    raw_output='{"priority": "high"}',
                ),
                prediction_row(
                    "support-eval-002",
                    REFERENCE_TWO,
                    REFERENCE_TWO,
                    raw_output='{"priority": "low"}',
                ),
            ]
            dpo_predictions = [
                prediction_row(
                    "support-eval-001",
                    REFERENCE_ONE,
                    REFERENCE_ONE,
                    raw_output='{"priority": "urgent"}',
                ),
                prediction_row(
                    "support-eval-002",
                    REFERENCE_TWO,
                    {
                        **REFERENCE_TWO,
                        "requires_human_followup": True,
                    },
                    raw_output='{"requires_human_followup": true}',
                ),
            ]

            write_jsonl(baseline_predictions_path, baseline_predictions)
            write_jsonl(sft_predictions_path, sft_predictions)
            write_jsonl(dpo_predictions_path, dpo_predictions)

            exit_code = module.main(
                [
                    "--run-name",
                    "comparison-smoke",
                    "--runtime-root",
                    str(runtime_root),
                    "--baseline-metrics",
                    str(baseline_metrics_path),
                    "--baseline-predictions",
                    str(baseline_predictions_path),
                    "--sft-metrics",
                    str(sft_metrics_path),
                    "--sft-predictions",
                    str(sft_predictions_path),
                    "--dpo-metrics",
                    str(dpo_metrics_path),
                    "--dpo-predictions",
                    str(dpo_predictions_path),
                    "--mirror-summary-to-repo",
                    "--mirror-report-to-repo",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = read_json(runtime_root / "persistent" / "metrics" / "comparison-smoke_comparison_summary.json")
            report = read_text(runtime_root / "persistent" / "reports" / "comparison-smoke_comparison_report.md")

            self.assertIn("dpo_vs_sft", summary["deltas"])
            self.assertEqual(summary["classification_basis"], "DPO relative to SFT")
            self.assertEqual(len(summary["row_evidence"]["semantic_gain"]), 1)
            self.assertEqual(len(summary["row_evidence"]["semantic_regression"]), 1)
            self.assertIn("# Consolidated Comparison Report: comparison-smoke", report)
            self.assertIn("Semantic Gain", report)
            self.assertIn("Semantic Regression", report)
            mirrored_summary = REPO_ROOT / "artifacts" / "metrics" / "comparison-smoke_comparison_summary.json"
            mirrored_report = REPO_ROOT / "artifacts" / "reports" / "comparison-smoke_comparison_report.md"
            self.assertTrue(mirrored_summary.exists())
            self.assertTrue(mirrored_report.exists())
            mirrored_summary.unlink()
            mirrored_report.unlink()


if __name__ == "__main__":
    unittest.main()
