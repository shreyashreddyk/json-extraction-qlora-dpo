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
            "  sample_limit: null",
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
            "artifacts:",
            '  pairs_filename: "{run_name}_dpo_pairs.jsonl"',
            '  audit_filename: "{run_name}_preference_audit.jsonl"',
            '  summary_filename: "{run_name}_preference_summary.json"',
            "training:",
            "  beta: 0.1",
            "",
        ]
    )


class FakeBackend:
    def __init__(self, responses: dict[tuple[str, int | None], str]) -> None:
        self.responses = responses

    def generate(self, request):  # pragma: no cover - exercised through builder tests
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


class PreferencePairBuilderTest(unittest.TestCase):
    def test_split_filtering_keeps_train_rows_by_default(self) -> None:
        samples = load_preference_samples(
            input_path=FIXTURE_PATH,
            source_format="json_extraction",
            source_split="train",
        )

        self.assertEqual(len(samples), 7)
        self.assertTrue(all(sample.split.value == "train" for sample in samples))

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
            samples = load_preference_samples(
                input_path=config.input_path,
                source_format=config.source_format,
                source_split=config.source_split,
                sample_limit=config.sample_limit,
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

            pair_rows, audit_rows, summary = build_preference_run(
                samples=samples,
                backend=backend,
                config=config,
            )

            self.assertEqual(len(pair_rows), 1)
            self.assertEqual(audit_rows[0]["candidate_count_after_dedup"], 2)
            self.assertEqual(audit_rows[0]["chosen_index"], 0)
            self.assertEqual(audit_rows[0]["rejected_index"], 2)
            self.assertEqual(summary["pair_count"], 1)
            self.assertGreater(summary["candidate_json_valid_rate"], 0.0)
            self.assertIn("prompt", pair_rows[0])
            self.assertIn("chosen", pair_rows[0])
            self.assertIn("rejected", pair_rows[0])

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
            samples = load_preference_samples(
                input_path=config.input_path,
                source_format=config.source_format,
                source_split=config.source_split,
                sample_limit=config.sample_limit,
            )
            gold_json = json.dumps(fixture_rows[0]["target"], indent=2, sort_keys=True)
            backend = FakeBackend(
                {
                    ("support-train-001", 11): gold_json,
                    ("support-train-001", 12): gold_json,
                    ("support-train-001", 13): gold_json,
                }
            )

            pair_rows, audit_rows, summary = build_preference_run(
                samples=samples,
                backend=backend,
                config=config,
            )

            self.assertEqual(pair_rows, [])
            self.assertEqual(audit_rows[0]["skip_reason"], "insufficient_distinct_candidates")
            self.assertEqual(summary["skipped_counts"]["insufficient_distinct_candidates"], 1)

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
                },
            )

            pairs_path, audit_path, summary_path = write_preference_artifacts(
                paths=output_paths,
                pair_rows=[{"prompt": "Prompt", "chosen": "{}", "rejected": "not-json"}],
                audit_rows=[{"record_id": "r1", "skip_reason": None}],
                summary={"pair_count": 1, "source_row_count": 1},
            )

            self.assertEqual(read_jsonl(pairs_path)[0], {"prompt": "Prompt", "chosen": "{}", "rejected": "not-json"})
            self.assertEqual(read_jsonl(audit_path)[0]["record_id"], "r1")
            self.assertEqual(read_json(summary_path)["pair_count"], 1)

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

            self.assertTrue(pairs_path.exists())
            self.assertTrue(audit_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(read_json(summary_path)["pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
