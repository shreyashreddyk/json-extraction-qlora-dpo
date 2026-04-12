import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.inference import InferenceResponse, analyze_inference_text
from json_ft.preference import (
    build_preference_run,
    load_preference_samples,
    resolve_preference_config,
    resolve_preference_output_paths,
    write_preference_artifacts,
)
from json_ft.utils import read_json, read_jsonl, write_jsonl, write_text


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "data" / "fixtures" / "support_tickets.jsonl"
PREFERENCE_SCRIPT_PATH = REPO_ROOT / "scripts" / "build_preference_pairs.py"


def load_preference_script_module():
    spec = importlib.util.spec_from_file_location("build_preference_pairs_script", PREFERENCE_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_config_text(input_path: Path) -> str:
    return "\n".join(
        [
            "model:",
            "  base_model: fake-model",
            "  adapter_path: /tmp/fake-adapter",
            "  revision: null",
            "  trust_remote_code: false",
            "  torch_dtype: auto",
            "  device_map: null",
            "pair_generation:",
            f"  input_path: {input_path}",
            "  source_format: json_extraction",
            "  source_split: train",
            "  prompt_source: messages",
            "  candidate_count: 3",
            "  inference_batch_size: 1",
            "  sample_limit: null",
            "  sample_percent: null",
            "  sample_seed: 17",
            "  quality_gates:",
            "    minimum_score_gap: 0.2",
            "    max_similarity_ratio: 0.96",
            "    reject_same_failure_mode: true",
            "    require_chosen_schema_valid: true",
            "  generation:",
            "    max_new_tokens: 128",
            "    temperature: 0.8",
            "    top_p: 0.95",
            "    do_sample: true",
            "    base_seed: 11",
            "profiles:",
            "  dev:",
            "    pair_generation:",
            "      candidate_count: 3",
            "      sample_limit: null",
            "  full:",
            "    pair_generation:",
            "      candidate_count: 3",
            "      sample_limit: null",
            "      quality_gates:",
            "        minimum_score_gap: 0.2",
            "  colab_full:",
            "    pair_generation:",
            "      candidate_count: 3",
            "artifacts:",
            '  pairs_filename: "{run_name}_dpo_pairs.jsonl"',
            '  audit_filename: "{run_name}_preference_audit.jsonl"',
            '  summary_filename: "{run_name}_preference_summary.json"',
            '  diagnostics_filename: "{run_name}_preference_diagnostics.json"',
            '  pair_emission_curve_filename: "{run_name}_preference_pair_emission.png"',
            '  skipped_reasons_curve_filename: "{run_name}_preference_skipped_reasons.png"',
            '  score_gap_curve_filename: "{run_name}_preference_score_gap.png"',
            '  source_quality_curve_filename: "{run_name}_preference_source_quality.png"',
            "training:",
            "  beta: 0.1",
            "",
        ]
    )


class FakeBackend:
    def __init__(self, responses: dict[tuple[str, int | None], str]) -> None:
        self.responses = responses
        self.generate_call_count = 0
        self.generate_batch_call_count = 0

    def generate(self, request):  # pragma: no cover - exercised through builder tests
        self.generate_call_count += 1
        text = self.responses[(request.record_id, request.seed)]
        parsed_payload, parse_error, validation, recovery_used = analyze_inference_text(text)
        return InferenceResponse(
            text=text,
            backend="fake-backend",
            latency_ms=5.0,
            prompt_source=request.prompt_source,
            model_name_or_path="fake-model",
            parsed_payload=parsed_payload,
            parse_error=parse_error,
            validation=validation,
            generation_kwargs={"seed": request.seed},
            json_recovery_used=recovery_used,
        )

    def generate_batch(self, requests):  # pragma: no cover - exercised through builder tests
        self.generate_batch_call_count += 1
        return [self.generate(request) for request in requests]


class PreferencePairBuilderTest(unittest.TestCase):
    def test_resolve_preference_config_recovers_sft_adapter_from_dpo_latest_manifest(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fixture_rows = read_jsonl(FIXTURE_PATH)[:1]
            input_path = tmp_path / "source.jsonl"
            config_path = tmp_path / "dpo.yaml"
            latest_manifest_path = tmp_path / "latest_model.json"
            dpo_manifest_path = tmp_path / "dpo_manifest.json"
            source_manifest_path = tmp_path / "sft_manifest.json"
            sft_adapter_path = tmp_path / "runtime" / "checkpoints" / "sft" / "adapter"
            sft_adapter_path.mkdir(parents=True)

            write_jsonl(input_path, fixture_rows)
            write_text(
                config_path,
                "\n".join(
                    [
                        "model:",
                        f"  latest_model_manifest: {latest_manifest_path}",
                        "  base_model: null",
                        "  adapter_path: null",
                        "pair_generation:",
                        f"  input_path: {input_path}",
                        "  source_format: json_extraction",
                        "  source_split: train",
                        "  prompt_source: messages",
                        "  candidate_count: 3",
                        "  sample_limit: 1",
                        "  quality_gates:",
                        "    minimum_score_gap: 0.2",
                        "  generation:",
                        "    max_new_tokens: 64",
                        "    temperature: 0.8",
                        "    top_p: 0.95",
                        "    do_sample: true",
                        "    base_seed: 11",
                        "profiles:",
                        "  dev: {}",
                        "artifacts:",
                        '  pairs_filename: "{run_name}_dpo_pairs.jsonl"',
                        '  audit_filename: "{run_name}_preference_audit.jsonl"',
                        '  summary_filename: "{run_name}_preference_summary.json"',
                        '  diagnostics_filename: "{run_name}_preference_diagnostics.json"',
                        '  pair_emission_curve_filename: "{run_name}_preference_pair_emission.png"',
                        '  skipped_reasons_curve_filename: "{run_name}_preference_skipped_reasons.png"',
                        '  score_gap_curve_filename: "{run_name}_preference_score_gap.png"',
                        '  source_quality_curve_filename: "{run_name}_preference_source_quality.png"',
                        "",
                    ]
                ),
            )

            write_text(
                source_manifest_path,
                json.dumps(
                    {
                        "stage": "sft",
                        "base_model": "fake-sft-base",
                        "adapter_path": str(sft_adapter_path),
                    }
                ),
            )
            write_text(
                dpo_manifest_path,
                json.dumps(
                    {
                        "stage": "dpo",
                        "base_model": "fake-sft-base",
                        "source_sft_manifest_path": str(source_manifest_path),
                        "source_adapter_path": str(sft_adapter_path),
                    }
                ),
            )
            write_text(
                latest_manifest_path,
                json.dumps(
                    {
                        "stage": "dpo",
                        "status": "ready",
                        "base_model": "fake-sft-base",
                        "adapter_path": str(tmp_path / "runtime" / "checkpoints" / "dpo" / "adapter"),
                        "schema_version": "1.0.0",
                        "timestamp_utc": "2026-04-10T00:00:00+00:00",
                        "report_paths": [str(dpo_manifest_path)],
                    }
                ),
            )

            config = resolve_preference_config(
                config_path=config_path,
                repo_root=REPO_ROOT,
                profile_name="dev",
            )

            self.assertEqual(config.model_name_or_path, "fake-sft-base")
            self.assertEqual(Path(config.adapter_path).resolve(), sft_adapter_path.resolve())

    def test_split_filtering_keeps_train_rows_by_default(self) -> None:
        samples, metadata = load_preference_samples(
            input_path=FIXTURE_PATH,
            source_format="json_extraction",
            source_split="train",
        )

        self.assertEqual(len(samples), 7)
        self.assertTrue(all(sample.split.value == "train" for sample in samples))
        self.assertEqual(metadata.selected_row_count, 7)

    def test_resolve_preference_config_accepts_inference_batch_size_override(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fixture_rows = read_jsonl(FIXTURE_PATH)[:1]
            input_path = tmp_path / "source.jsonl"
            config_path = tmp_path / "dpo.yaml"
            write_jsonl(input_path, fixture_rows)
            write_text(config_path, build_config_text(input_path))

            config = resolve_preference_config(
                config_path=config_path,
                repo_root=REPO_ROOT,
                profile_name="dev",
                inference_batch_size=8,
            )

            self.assertEqual(config.inference_batch_size, 8)

    def test_builder_deduplicates_candidates_and_selects_pair(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fixture_rows = read_jsonl(FIXTURE_PATH)[:1]
            input_path = tmp_path / "source.jsonl"
            config_path = tmp_path / "dpo.yaml"
            write_jsonl(input_path, fixture_rows)
            write_text(config_path, build_config_text(input_path))
            config = resolve_preference_config(
                config_path=config_path,
                repo_root=REPO_ROOT,
                profile_name="dev",
            )
            samples, source_subset_metadata = load_preference_samples(
                input_path=config.input_path,
                source_format=config.source_format,
                source_split=config.source_split,
                sample_limit=config.sample_limit,
                sample_percent=config.sample_percent,
                sample_seed=config.sample_seed,
            )
            gold_json = json.dumps(fixture_rows[0]["target"], indent=2, sort_keys=True)
            invalid_json = "not-json"
            backend = FakeBackend(
                {
                    ("support-train-001", 11): gold_json,
                    ("support-train-001", 12): gold_json,
                    ("support-train-001", 13): invalid_json,
                }
            )

            pair_rows, audit_rows, summary, diagnostics = build_preference_run(
                samples=samples,
                backend=backend,
                config=config,
                source_subset_metadata=source_subset_metadata,
            )

            self.assertEqual(len(pair_rows), 1)
            self.assertEqual(audit_rows[0]["candidate_count_after_dedup"], 2)
            self.assertEqual(audit_rows[0]["chosen_index"], 0)
            self.assertEqual(audit_rows[0]["rejected_index"], 2)
            self.assertEqual(summary["pair_count"], 1)
            self.assertGreater(summary["candidate_json_valid_rate"], 0.0)
            self.assertIn("score_gap_distribution", diagnostics)
            self.assertIn("prompt", pair_rows[0])
            self.assertIn("chosen", pair_rows[0])
            self.assertIn("rejected", pair_rows[0])

    def test_builder_uses_generate_batch_when_enabled(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fixture_rows = read_jsonl(FIXTURE_PATH)[:2]
            input_path = tmp_path / "source.jsonl"
            config_path = tmp_path / "dpo.yaml"
            write_jsonl(input_path, fixture_rows)
            write_text(config_path, build_config_text(input_path))
            config = resolve_preference_config(
                config_path=config_path,
                repo_root=REPO_ROOT,
                profile_name="dev",
                inference_batch_size=2,
            )
            samples, source_subset_metadata = load_preference_samples(
                input_path=config.input_path,
                source_format=config.source_format,
                source_split=config.source_split,
                sample_limit=config.sample_limit,
                sample_percent=config.sample_percent,
                sample_seed=config.sample_seed,
            )
            responses = {}
            for sample in samples:
                gold_json = json.dumps(
                    next(row for row in fixture_rows if row["record_id"] == sample.record_id)["target"],
                    indent=2,
                    sort_keys=True,
                )
                responses[(sample.record_id, 11)] = gold_json
                responses[(sample.record_id, 12)] = gold_json
                responses[(sample.record_id, 13)] = "not-json"
            backend = FakeBackend(responses)

            build_preference_run(
                samples=samples,
                backend=backend,
                config=config,
                source_subset_metadata=source_subset_metadata,
            )

            self.assertEqual(backend.generate_batch_call_count, 3)
            self.assertEqual(backend.generate_call_count, 6)

    def test_builder_skips_rows_without_distinct_pair(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fixture_rows = read_jsonl(FIXTURE_PATH)[:1]
            input_path = tmp_path / "source.jsonl"
            config_path = tmp_path / "dpo.yaml"
            write_jsonl(input_path, fixture_rows)
            write_text(config_path, build_config_text(input_path))
            config = resolve_preference_config(
                config_path=config_path,
                repo_root=REPO_ROOT,
                profile_name="dev",
            )
            samples, source_subset_metadata = load_preference_samples(
                input_path=config.input_path,
                source_format=config.source_format,
                source_split=config.source_split,
                sample_limit=config.sample_limit,
                sample_percent=config.sample_percent,
                sample_seed=config.sample_seed,
            )
            gold_json = json.dumps(fixture_rows[0]["target"], indent=2, sort_keys=True)
            backend = FakeBackend(
                {
                    ("support-train-001", 11): gold_json,
                    ("support-train-001", 12): gold_json,
                    ("support-train-001", 13): gold_json,
                }
            )

            pair_rows, audit_rows, summary, diagnostics = build_preference_run(
                samples=samples,
                backend=backend,
                config=config,
                source_subset_metadata=source_subset_metadata,
            )

            self.assertEqual(pair_rows, [])
            self.assertEqual(audit_rows[0]["skip_reason"], "insufficient_distinct_candidates")
            self.assertEqual(summary["skipped_counts"]["insufficient_distinct_candidates"], 1)
            self.assertEqual(diagnostics["skipped_counts"]["insufficient_distinct_candidates"], 1)

    def test_preference_artifacts_write_expected_shapes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_paths = resolve_preference_output_paths(
                output_dir=tmp_path,
                run_name="pref-smoke",
                artifact_names={
                    "pairs_filename": "{run_name}_dpo_pairs.jsonl",
                    "audit_filename": "{run_name}_preference_audit.jsonl",
                    "summary_filename": "{run_name}_preference_summary.json",
                    "diagnostics_filename": "{run_name}_preference_diagnostics.json",
                    "pair_emission_curve_filename": "{run_name}_preference_pair_emission.png",
                    "skipped_reasons_curve_filename": "{run_name}_preference_skipped_reasons.png",
                    "score_gap_curve_filename": "{run_name}_preference_score_gap.png",
                    "source_quality_curve_filename": "{run_name}_preference_source_quality.png",
                },
            )

            pairs_path, audit_path, summary_path, diagnostics_path, _plot_paths = write_preference_artifacts(
                paths=output_paths,
                pair_rows=[{"prompt": "Prompt", "chosen": "{}", "rejected": "not-json"}],
                audit_rows=[{"record_id": "r1", "skip_reason": None}],
                summary={"pair_count": 1, "source_row_count": 1},
                diagnostics={"score_gap_distribution": [], "pair_quality_by_source_dataset": {}},
            )

            persisted_pair = read_jsonl(pairs_path)[0]
            self.assertEqual(persisted_pair["prompt"], "Prompt")
            self.assertEqual(persisted_pair["chosen"], "{}")
            self.assertEqual(persisted_pair["rejected"], "not-json")
            self.assertEqual(read_jsonl(audit_path)[0]["record_id"], "r1")
            self.assertEqual(read_json(summary_path)["pair_count"], 1)
            self.assertEqual(read_json(diagnostics_path)["pair_quality_by_source_dataset"], {})

    def test_preference_sampling_percent_is_deterministic(self) -> None:
        first, first_metadata = load_preference_samples(
            input_path=FIXTURE_PATH,
            source_format="json_extraction",
            source_split="train",
            sample_percent=0.5,
            sample_seed=17,
        )
        second, second_metadata = load_preference_samples(
            input_path=FIXTURE_PATH,
            source_format="json_extraction",
            source_split="train",
            sample_percent=0.5,
            sample_seed=17,
        )

        self.assertEqual([sample.record_id for sample in first], [sample.record_id for sample in second])
        self.assertEqual(first_metadata.selected_row_count, second_metadata.selected_row_count)

    def test_preference_cli_smoke_writes_runtime_artifacts(self) -> None:
        module = load_preference_script_module()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fixture_rows = read_jsonl(FIXTURE_PATH)[:1]
            input_path = tmp_path / "source.jsonl"
            config_path = tmp_path / "dpo.yaml"
            runtime_root = tmp_path / "runtime"
            write_jsonl(input_path, fixture_rows)
            write_text(config_path, build_config_text(input_path))

            gold_json = json.dumps(fixture_rows[0]["target"], indent=2, sort_keys=True)
            backend = FakeBackend(
                {
                    ("support-train-001", 11): gold_json,
                    ("support-train-001", 12): gold_json,
                    ("support-train-001", 13): "not-json",
                }
            )

            with patch.object(module, "build_inference_backend", return_value=backend):
                exit_code = module.main(
                    [
                        "--config",
                        str(config_path),
                        "--profile",
                        "dev",
                        "--run-name",
                        "pref-cli-smoke",
                        "--runtime-root",
                        str(runtime_root),
                    ]
                )

            self.assertEqual(exit_code, 0)
            output_dir = runtime_root / "persistent" / "preferences" / "pref-cli-smoke"
            pairs_path = output_dir / "pref-cli-smoke_dpo_pairs.jsonl"
            audit_path = output_dir / "pref-cli-smoke_preference_audit.jsonl"
            summary_path = output_dir / "pref-cli-smoke_preference_summary.json"
            diagnostics_path = output_dir / "pref-cli-smoke_preference_diagnostics.json"

            self.assertTrue(pairs_path.exists())
            self.assertTrue(audit_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(diagnostics_path.exists())
            self.assertEqual(read_json(summary_path)["pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
