import unittest
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.data_build import _sample_rows, assign_split, build_dataset_manifests
from json_ft.dataset_adapters import DatasetSplit, adapt_source_record
from json_ft.utils import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


class DataBuildTest(unittest.TestCase):
    def _write_temp_build_config(self, directory: Path) -> Path:
        build_config = directory / "data_build.yaml"
        build_config.write_text(
            "\n".join(
                [
                    "seed: 17",
                    f"raw_root: {REPO_ROOT / 'data' / 'fixtures' / 'source_adapter_samples'}",
                    "runtime_root: null",
                    "eval_ratio: 0.2",
                    "train_target_count: null",
                    "eval_target_count: null",
                    "max_source_share: 0.45",
                    "max_synthetic_share: 0.30",
                    "eval_allow_synthetic: false",
                    "schema_discipline_enabled: false",
                    "source_group_weights:",
                    "  domain_task_data: 1.0",
                    "  schema_discipline_data: 0.2",
                    "  synthetic_augmentation_data: 0.0",
                    "source_weight_overrides:",
                    "  synthetic_support_tickets_v1: 0.4",
                    "  console_ai_it_helpdesk_synthetic_tickets: 1.0",
                    "  prady06_customer_support_tickets: 0.9",
                    "  cfpb_consumer_complaints: 0.8",
                    "augmentation:",
                    "  enabled: true",
                    "  synthetic_source_name: synthetic_hardening_v1",
                    "outputs:",
                    f"  canonical_output: {directory / 'support_tickets_canonical.jsonl'}",
                    f"  prompt_completion_output: {directory / 'support_tickets_sft_prompt_completion.jsonl'}",
                    f"  messages_output: {directory / 'support_tickets_sft_messages.jsonl'}",
                    f"  eval_output: {directory / 'support_tickets_eval_manifest.jsonl'}",
                    f"  summary_output: {directory / 'support_tickets_dataset_build_summary.json'}",
                    f"  composition_json_output: {directory / 'support_tickets_dataset_composition.json'}",
                    f"  composition_csv_output: {directory / 'support_tickets_dataset_composition.csv'}",
                    f"  composition_markdown_output: {directory / 'support_tickets_dataset_composition.md'}",
                    "profiles:",
                    "  dev:",
                    "    train_target_count: 12",
                    "    eval_target_count: 4",
                    "    max_source_share: 0.45",
                    "    max_synthetic_share: 0.25",
                    "  full:",
                    "    train_target_count: null",
                    "    eval_target_count: null",
                    "    max_source_share: 0.45",
                    "    max_synthetic_share: 0.30",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return build_config

    def test_assign_split_is_deterministic_and_honors_explicit_split(self) -> None:
        first = assign_split(
            source_dataset="example",
            source_record_id="row-1",
            split_hint=None,
            eval_ratio=0.2,
        )
        second = assign_split(
            source_dataset="example",
            source_record_id="row-1",
            split_hint=None,
            eval_ratio=0.2,
        )
        explicit = assign_split(
            source_dataset="example",
            source_record_id="row-2",
            split_hint="eval",
            eval_ratio=0.2,
        )

        self.assertEqual(first, second)
        self.assertEqual(explicit, DatasetSplit.EVAL)

    def test_weighted_sampling_is_deterministic_and_caps_source_share(self) -> None:
        grouped = defaultdict(list)
        for index in range(8):
            grouped["source_a"].append(
                adapt_source_record(
                    {
                        "record_id": f"a-{index}",
                        "split": "train",
                        "source_dataset": "source_a",
                        "input_text": f"row {index}",
                        "target": {
                            "summary": "Sample row",
                            "issue_category": "billing",
                            "priority": "medium",
                            "product_area": "billing_portal",
                            "customer": {"name": None, "account_id": None, "plan_tier": None},
                            "sentiment": "neutral",
                            "requires_human_followup": True,
                            "actions_requested": [],
                        },
                        "metadata": {},
                    },
                    "json_extraction",
                )
            )
        for index in range(4):
            grouped["source_b"].append(
                adapt_source_record(
                    {
                        "record_id": f"b-{index}",
                        "split": "train",
                        "source_dataset": "source_b",
                        "input_text": f"row {index}",
                        "target": {
                            "summary": "Sample row",
                            "issue_category": "general_question",
                            "priority": "low",
                            "product_area": "web_app",
                            "customer": {"name": None, "account_id": None, "plan_tier": None},
                            "sentiment": "neutral",
                            "requires_human_followup": False,
                            "actions_requested": [],
                        },
                        "metadata": {},
                    },
                    "json_extraction",
                )
            )
        selected_first = _sample_rows(
            grouped,
            target_count=6,
            weights={"source_a": 1.0, "source_b": 1.0},
            max_source_share=0.5,
            seed=17,
        )
        selected_second = _sample_rows(
            grouped,
            target_count=6,
            weights={"source_a": 1.0, "source_b": 1.0},
            max_source_share=0.5,
            seed=17,
        )

        self.assertEqual([sample.record_id for sample in selected_first], [sample.record_id for sample in selected_second])
        self.assertLessEqual(sum(sample.source_dataset == "source_a" for sample in selected_first), 3)

    def test_build_dataset_manifests_writes_multi_source_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            build_config_path = self._write_temp_build_config(tmp_path)
            result = build_dataset_manifests(
                repo_root=REPO_ROOT,
                registry_config_path=REPO_ROOT / "configs" / "data_sources.yaml",
                build_config_path=build_config_path,
                profile_name="dev",
                runtime_root=tmp_path / "runtime",
            )

            summary = result["summary"]
            canonical_rows = read_jsonl(result["export_paths"]["canonical_output"])
            eval_rows = read_jsonl(result["export_paths"]["eval_output"])

            self.assertGreater(summary["total_rows"], 10)
            self.assertIn("cfpb_consumer_complaints", summary["source_counts"])
            self.assertTrue(summary["leakage_checks"]["is_lineage_clean"])
            self.assertTrue(all(not row["metadata"].get("synthetic", False) for row in eval_rows))
            self.assertIn("source_group", canonical_rows[0]["metadata"])
            self.assertIn("source_uri_or_path", canonical_rows[0]["metadata"])
            self.assertIn("mapping_version", canonical_rows[0]["metadata"])

    def test_schema_discipline_source_can_be_enabled_explicitly(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            build_config_path = self._write_temp_build_config(tmp_path)
            result = build_dataset_manifests(
                repo_root=REPO_ROOT,
                registry_config_path=REPO_ROOT / "configs" / "data_sources.yaml",
                build_config_path=build_config_path,
                profile_name="dev",
                include_sources=["suneeldk_text_json"],
                runtime_root=tmp_path / "runtime",
            )
            summary = result["summary"]
            self.assertGreater(summary["adapter_reject_count"], 0)
            self.assertIn("hf_schema_discipline_json_v1", summary["adapter_reject_counts_by_source"])


if __name__ == "__main__":
    unittest.main()
