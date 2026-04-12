import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys_path = str(Path(__file__).resolve().parents[1] / "src")
import sys

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from json_ft.reporting import (
    CaseStudy,
    build_dataset_composition_table,
    build_failure_bucket_table,
    build_pair_quality_table,
    build_stage_delta_table,
    build_stage_metrics_table,
    extract_case_studies,
    generate_report_plots,
    load_reporting_bundle,
    render_final_markdown_report,
)
from json_ft.reporting.loaders import ReportingBundle, StageArtifacts
from json_ft.utils import read_json, read_text, write_json, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT_PATH = REPO_ROOT / "scripts" / "export_final_report.py"


def _load_export_script_module():
    spec = importlib.util.spec_from_file_location("export_final_report_script", EXPORT_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_artifacts(stage_name: str, rows: list[dict] | None = None) -> StageArtifacts:
    metrics = {
        "stage": stage_name,
        "json_validity_rate": 1.0,
        "schema_validation_pass_rate": 1.0,
        "hallucinated_field_rate": 0.0,
        "json_recovery_rate": 0.0,
        "field_level": {
            "micro": {"f1": 1.0 if stage_name != "baseline" else 0.4},
            "macro": {"f1": 1.0 if stage_name != "baseline" else 0.4},
            "per_field": {
                "issue_category": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                "priority": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            },
        },
        "categorical_exact_match": {
            "issue_category": 1.0,
            "priority": 1.0,
            "product_area": 1.0,
            "sentiment": 1.0,
            "requires_human_followup": 1.0,
            "customer.plan_tier": 1.0,
        },
        "latency_ms": {"mean": 10.0, "p95": 15.0},
    }
    return StageArtifacts(
        stage_name=stage_name,
        metrics_path=None,
        metrics=metrics,
        diagnostics_path=None,
        diagnostics={"bucket_counts": {"semantic_failures": 1, "syntax_failures": 0}},
        report_path=None,
        predictions_path=None,
        predictions=rows,
    )


def _build_bundle_for_cases() -> ReportingBundle:
    reference = {
        "issue_category": "billing",
        "priority": "high",
        "product_area": "billing_portal",
        "customer": {"name": None, "account_id": None, "plan_tier": None},
        "sentiment": "negative",
        "requires_human_followup": True,
        "actions_requested": [],
        "summary": "Billing issue",
    }
    action_reference = {
        **reference,
        "actions_requested": ["Send refund confirmation"],
    }
    baseline_rows = [
        {
            "record_id": "r1",
            "input_text": "billing text",
            "reference_payload": reference,
            "parsed_payload": {**reference, "priority": "low"},
            "raw_output": '{"priority": "low"}',
            "schema_is_valid": True,
            "unexpected_fields": [],
            "parse_error": None,
        },
        {
            "record_id": "r2",
            "input_text": "routing text",
            "reference_payload": action_reference,
            "parsed_payload": {**reference, "product_area": "web_app"},
            "raw_output": '{"product_area": "web_app"}',
            "schema_is_valid": True,
            "unexpected_fields": [],
            "parse_error": None,
        },
        {
            "record_id": "r3",
            "input_text": "regression text",
            "reference_payload": reference,
            "parsed_payload": {**reference, "issue_category": "other"},
            "raw_output": '{"issue_category": "other"}',
            "schema_is_valid": True,
            "unexpected_fields": [],
            "parse_error": None,
        },
        {
            "record_id": "r4",
            "input_text": "syntax text",
            "reference_payload": reference,
            "parsed_payload": {**reference, "priority": "low"},
            "raw_output": '{"priority": "low"}',
            "schema_is_valid": True,
            "unexpected_fields": [],
            "parse_error": None,
        },
        {
            "record_id": "r5",
            "input_text": "hard failure text",
            "reference_payload": reference,
            "parsed_payload": {
                **reference,
                "priority": "low",
                "issue_category": "other",
                "product_area": "api",
                "sentiment": "neutral",
                "requires_human_followup": False,
            },
            "raw_output": '{"bad": "baseline"}',
            "schema_is_valid": True,
            "unexpected_fields": [],
            "parse_error": None,
        },
    ]
    sft_rows = [
        {**baseline_rows[0], "record_id": "r1", "parsed_payload": reference, "raw_output": '{"priority": "high"}'},
        {
            **baseline_rows[1],
            "record_id": "r2",
            "parsed_payload": {**action_reference, "actions_requested": []},
            "raw_output": '{"actions_requested": []}',
        },
        {**baseline_rows[2], "record_id": "r3", "parsed_payload": reference, "raw_output": '{"priority": "high"}'},
        {**baseline_rows[3], "record_id": "r4", "schema_is_valid": False, "raw_output": '{"priority": "low"}'},
        {**baseline_rows[4], "record_id": "r5"},
    ]
    dpo_rows = [
        {**baseline_rows[0], "record_id": "r1", "parsed_payload": reference, "raw_output": '{"priority": "high"}'},
        {
            **baseline_rows[1],
            "record_id": "r2",
            "parsed_payload": action_reference,
            "raw_output": '{"priority": "high", "product_area": "billing_portal", "actions_requested": ["Send refund confirmation"]}',
        },
        {
            **baseline_rows[2],
            "record_id": "r3",
            "parsed_payload": {**reference, "issue_category": "other"},
            "raw_output": '{"issue_category": "other"}',
        },
        {
            **baseline_rows[3],
            "record_id": "r4",
            "parsed_payload": {**reference, "priority": "low"},
            "schema_is_valid": True,
            "raw_output": '{"priority": "low"}',
        },
        {**baseline_rows[4], "record_id": "r5"},
    ]
    return ReportingBundle(
        repo_root=REPO_ROOT,
        source_root=REPO_ROOT,
        runtime_root=REPO_ROOT / "runtime",
        build_summary_path=None,
        build_summary=None,
        composition_summary_path=None,
        composition_summary={"rows": [], "summary": {}},
        comparison_summary_path=None,
        comparison_summary={"deltas": {}},
        canonical_manifest_path=None,
        canonical_manifest_rows=None,
        eval_manifest_path=None,
        eval_manifest_rows=None,
        sft_manifest_path=None,
        sft_manifest_rows=None,
        baseline=_stage_artifacts("baseline", baseline_rows),
        sft=_stage_artifacts("sft", sft_rows),
        dpo=_stage_artifacts("dpo", dpo_rows),
        availability={},
    )


class ReportingBundleTest(unittest.TestCase):
    def test_load_reporting_bundle_prefers_repo_artifacts_and_runtime_fallbacks(self) -> None:
        bundle = load_reporting_bundle(REPO_ROOT)
        self.assertIsNotNone(bundle.composition_summary)
        self.assertIsNotNone(bundle.build_summary)
        self.assertIsNotNone(bundle.baseline.metrics)
        self.assertTrue(bundle.availability["baseline_predictions"])
        self.assertFalse(bundle.availability["sft_predictions"])

    def test_stage_and_dataset_tables_use_saved_repo_metrics(self) -> None:
        bundle = load_reporting_bundle(REPO_ROOT)
        stage_rows = build_stage_metrics_table(bundle)
        delta_rows = build_stage_delta_table(bundle)
        dataset_rows = build_dataset_composition_table(bundle)
        failure_rows = build_failure_bucket_table(bundle)

        self.assertEqual(stage_rows[0]["stage"], "baseline")
        self.assertEqual(stage_rows[0]["field_level_micro_f1"], 0.303)
        self.assertEqual(delta_rows[1]["comparison"], "dpo_vs_sft")
        self.assertTrue(any(row["source_dataset"] == "prady06_customer_support_tickets" for row in dataset_rows))
        self.assertTrue(any(row["bucket"] == "null_handling_mistakes" for row in failure_rows))

    def test_pair_quality_table_is_empty_when_preference_artifacts_are_missing(self) -> None:
        bundle = load_reporting_bundle(REPO_ROOT)
        self.assertEqual(build_pair_quality_table(bundle), [])


class CaseStudyExtractionTest(unittest.TestCase):
    def test_extract_case_studies_classifies_required_categories(self) -> None:
        bundle = _build_bundle_for_cases()
        case_studies = extract_case_studies(bundle, max_per_category=2)

        self.assertEqual(case_studies["baseline_bad_to_sft_good"][0].record_id, "r1")
        self.assertEqual(case_studies["sft_good_to_dpo_better"][0].record_id, "r2")
        self.assertEqual(case_studies["sft_good_to_dpo_worse"][0].record_id, "r3")
        self.assertEqual(case_studies["syntax_cleaned_up_semantics_unchanged"][0].record_id, "r4")
        self.assertEqual(case_studies["unchanged_hard_failures"][0].record_id, "r5")


class MarkdownExportTest(unittest.TestCase):
    def test_render_final_markdown_report_writes_required_sections(self) -> None:
        bundle = _build_bundle_for_cases()
        case_studies = extract_case_studies(bundle, max_per_category=1)
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "final_project_report.md"
            rendered = render_final_markdown_report(bundle, case_studies, output_path)
            text = read_text(rendered)

        self.assertIn("# Final Project Report", text)
        self.assertIn("## Project Summary", text)
        self.assertIn("## Key Metrics", text)
        self.assertIn("## Honest Conclusion", text)
        self.assertIn("Baseline Bad -> SFT Good", text)


class ExportCliSmokeTest(unittest.TestCase):
    def test_export_final_report_cli_writes_markdown_without_plot_dependencies(self) -> None:
        module = _load_export_script_module()
        with TemporaryDirectory() as tmp_dir:
            output_markdown = Path(tmp_dir) / "final.md"
            exit_code = module.main(["--output-markdown", str(output_markdown)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_markdown.exists())
            self.assertIn("# Final Project Report", read_text(output_markdown))


@unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "matplotlib not installed")
class PlotRenderingTest(unittest.TestCase):
    def test_generate_report_plots_writes_pngs(self) -> None:
        bundle = load_reporting_bundle(REPO_ROOT)
        with TemporaryDirectory() as tmp_dir:
            plot_paths = generate_report_plots(bundle, Path(tmp_dir))

        self.assertTrue(plot_paths)
        self.assertTrue(all(Path(path).suffix == ".png" for path in plot_paths.values()))


if __name__ == "__main__":
    unittest.main()
