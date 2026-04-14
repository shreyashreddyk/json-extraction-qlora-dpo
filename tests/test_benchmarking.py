import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from json_ft.benchmark_reporting import render_benchmark_report
from json_ft.benchmarking import (
    benchmark_checkpoint_paths,
    build_benchmark_promptsets,
    build_workload_mix_rows,
    check_vllm_health,
    compute_prompt_budget,
    load_benchmark_checkpoint_state,
    load_benchmark_step_checkpoints,
    load_checkpointed_benchmark_bundle,
    resolve_serving_target,
    save_benchmark_checkpoint_state,
    save_benchmark_step_checkpoint,
    validate_benchmark_checkpoint_resume,
)
from json_ft.utils import read_json, read_jsonl, write_json, write_jsonl, write_text


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_HEALTH_SCRIPT = REPO_ROOT / "scripts" / "check_vllm_health.py"
BUILD_PROMPTSETS_SCRIPT = REPO_ROOT / "scripts" / "build_benchmark_promptsets.py"
RENDER_REPORT_SCRIPT = REPO_ROOT / "scripts" / "render_benchmark_report.py"


def load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return text

    def _encode(self, text: str) -> list[int]:
        encoded: list[int] = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            encoded.append(self._token_to_id[token])
        return encoded

    def __call__(self, text: str, add_special_tokens: bool = True):
        return {"input_ids": self._encode(text)}

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(self._id_to_token[token] for token in tokens)


class BenchmarkingCoreTest(unittest.TestCase):
    def test_resolve_serving_target_prefers_latest_model_manifest_for_lora(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = tmp_path / "repo"
            manifest_dir = repo_root / "artifacts" / "checkpoints"
            manifest_dir.mkdir(parents=True)
            config_path = repo_root / "configs" / "inference.yaml"
            config_path.parent.mkdir(parents=True)

            write_json(
                manifest_dir / "latest_model.json",
                {
                    "stage": "dpo",
                    "status": "ready",
                    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "adapter_path": "/content/drive/MyDrive/json-ft-runs/persistent/checkpoints/dpo/adapter",
                    "merged_export_path": None,
                    "schema_version": "1.0.0",
                },
            )
            write_text(
                config_path,
                "\n".join(
                    [
                        "model_resolution:",
                        "  latest_model_manifest: artifacts/checkpoints/latest_model.json",
                        "  preferred_target: base_plus_lora",
                        "  lora_alias: support-ticket-ft",
                        "",
                    ]
                ),
            )

            target, _ = resolve_serving_target(
                config_path=config_path,
                repo_root=repo_root,
            )

        self.assertEqual(target.target_kind, "base_plus_lora")
        self.assertEqual(target.request_model_name, "support-ticket-ft")
        self.assertEqual(target.served_model_name_or_path, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(
            target.adapter_path,
            "/Users/shreyashreddy/Library/CloudStorage/GoogleDrive-kshreyashreddy@gmail.com/My Drive/json-ft-runs/persistent/checkpoints/dpo/adapter",
        )

    def test_compute_prompt_budget_trims_only_user_message(self) -> None:
        tokenizer = FakeTokenizer()
        messages = [
            {"role": "system", "content": "system instructions stay intact"},
            {
                "role": "user",
                "content": "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12",
            },
        ]

        trimmed_messages, budget = compute_prompt_budget(
            tokenizer=tokenizer,
            messages=messages,
            budgeting_config={
                "max_model_len": 12,
                "desired_max_tokens": 6,
                "minimum_output_tokens": 4,
                "safety_margin_tokens": 1,
                "trim_head_fraction": 0.5,
            },
        )

        self.assertTrue(budget.trim_applied)
        self.assertEqual(trimmed_messages[0]["content"], "system instructions stay intact")
        self.assertNotEqual(trimmed_messages[1]["content"], messages[1]["content"])
        self.assertGreaterEqual(budget.final_output_tokens, 1)

    def test_build_benchmark_promptsets_writes_natural_and_stress_rows(self) -> None:
        rows = [
            {
                "record_id": "a",
                "split": "eval",
                "source_dataset": "synthetic",
                "input_text": "Short issue.",
                "reference_payload": {"priority": "low"},
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Short issue."},
                    {"role": "assistant", "content": "{}"},
                ],
                "metadata": {},
            },
            {
                "record_id": "b",
                "split": "eval",
                "source_dataset": "synthetic",
                "input_text": "Longer issue body with more words for the long bucket classification path.",
                "reference_payload": {"priority": "high"},
                "messages": [
                    {"role": "system", "content": "sys"},
                    {
                        "role": "user",
                        "content": "Longer issue body with more words for the long bucket classification path.",
                    },
                    {"role": "assistant", "content": "{}"},
                ],
                "metadata": {},
            },
            {
                "record_id": "c",
                "split": "eval",
                "source_dataset": "synthetic",
                "input_text": "Medium issue body for testing.",
                "reference_payload": {"priority": "medium"},
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Medium issue body for testing."},
                    {"role": "assistant", "content": "{}"},
                ],
                "metadata": {},
            },
        ]
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "eval.jsonl"
            write_jsonl(dataset_path, rows)

            with patch("json_ft.benchmarking._load_tokenizer", return_value=FakeTokenizer()):
                result = build_benchmark_promptsets(
                    dataset_path=dataset_path,
                    target=type("Target", (), {"base_model": "fake-model", "served_model_name_or_path": "fake-model"})(),
                    promptset_config={"seed": 17, "stress_sample_limit": 2},
                    output_dir=tmp_path / "promptsets",
                )

            manifest = read_json(result["manifest_path"])
            natural_rows = read_jsonl(Path(manifest["natural_prompt_rows_path"]))
            stress_rows = read_jsonl(Path(manifest["stress_prompt_rows_path"]))

        self.assertEqual(len(natural_rows), 3)
        self.assertGreaterEqual(len(stress_rows), 1)
        self.assertTrue(all(row["length_family"] == "natural" for row in natural_rows))
        self.assertTrue(all(row["benchmark_only"] for row in stress_rows))

    def test_build_workload_mix_rows_uses_natural_short_and_stress_long(self) -> None:
        natural_rows = [
            {"record_id": "n1", "bucket_label": "short", "length_family": "natural"},
            {"record_id": "n2", "bucket_label": "short", "length_family": "natural"},
            {"record_id": "n3", "bucket_label": "long", "length_family": "natural"},
        ]
        stress_rows = [
            {"record_id": "s1", "bucket_label": "long", "length_family": "stress"},
            {"record_id": "s2", "bucket_label": "long", "length_family": "stress"},
        ]

        rows = build_workload_mix_rows(
            natural_rows=natural_rows,
            stress_rows=stress_rows,
            mix_name="stress_mix_50_50_natural_short_stress_long",
            total_count=4,
            seed=17,
        )

        short_count = sum(1 for row in rows if row["length_family"] == "natural")
        stress_count = sum(1 for row in rows if row["length_family"] == "stress")
        self.assertEqual(short_count, 2)
        self.assertEqual(stress_count, 2)

    def test_benchmark_step_checkpoint_round_trip(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            checkpoint_paths = benchmark_checkpoint_paths(run_dir)
            payload = {
                "checkpoint_version": 1,
                "experiment_family": "config_search",
                "workload_name": "stress_mix_50_50_natural_short_stress_long",
                "server_config_id": "search-btok2048-seq32",
                "concurrency": 16,
                "step_id": "config_search-search-btok2048-seq32-stress_mix_50_50_natural_short_stress_long-c16",
                "summary_row": {"server_config_id": "search-btok2048-seq32", "throughput_rps": 1.0},
                "correctness_row": {"experiment_id": "config_search-1"},
                "raw_rows": [{"request_id": "r1"}],
            }

            checkpoint_path = save_benchmark_step_checkpoint(
                checkpoint_paths["steps_dir"],
                step_id=payload["step_id"],
                payload=payload,
            )
            loaded = load_benchmark_step_checkpoints(checkpoint_paths["steps_dir"])
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["step_id"], payload["step_id"])
            self.assertEqual(loaded[0]["summary_row"]["server_config_id"], "search-btok2048-seq32")

    def test_load_checkpointed_benchmark_bundle_reconstructs_partial_bundle(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            checkpoint_paths = benchmark_checkpoint_paths(run_dir)
            state_payload = {
                "checkpoint_version": 1,
                "run_name": "vllm-benchmark-lab",
                "fingerprint": "fingerprint-1",
                "config_path": "/tmp/config.yaml",
                "dataset_path": "/tmp/dataset.jsonl",
                "target": {"target_kind": "base_plus_lora"},
                "promptset_manifest": {"counts": {"natural": 1, "stress": 1}},
                "created_at_utc": "2026-04-13T00:00:00+00:00",
                "updated_at_utc": "2026-04-13T00:00:00+00:00",
                "completed_steps": [],
                "step_artifacts": {},
                "complete": False,
                "bundle_path": str(run_dir / "bundle.json"),
                "checkpoint_dir": str(checkpoint_paths["checkpoint_dir"]),
                "checkpoint_steps_dir": str(checkpoint_paths["steps_dir"]),
            }
            save_benchmark_checkpoint_state(checkpoint_paths["state_path"], state_payload)
            save_benchmark_step_checkpoint(
                checkpoint_paths["steps_dir"],
                step_id="config_search-search-btok2048-seq32-stress_mix_50_50_natural_short_stress_long-c16",
                payload={
                    "checkpoint_version": 1,
                    "experiment_family": "config_search",
                    "workload_name": "stress_mix_50_50_natural_short_stress_long",
                    "server_config_id": "search-btok2048-seq32",
                    "concurrency": 16,
                    "step_id": "config_search-search-btok2048-seq32-stress_mix_50_50_natural_short_stress_long-c16",
                    "summary_row": {"server_config_id": "search-btok2048-seq32", "throughput_rps": 1.0},
                    "correctness_row": {"experiment_id": "config_search-1"},
                    "raw_rows": [{"request_id": "r1"}],
                },
            )

            bundle = load_checkpointed_benchmark_bundle(run_dir)

        self.assertIsNotNone(bundle)
        self.assertTrue(bundle["partial"])
        self.assertEqual(len(bundle["summary_rows"]), 1)
        self.assertEqual(len(bundle["correctness_rows"]), 1)
        self.assertEqual(len(bundle["config_search_rows"]), 1)
        self.assertEqual(len(bundle["raw_request_rows"]), 1)

    def test_validate_benchmark_checkpoint_resume_rejects_fingerprint_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            validate_benchmark_checkpoint_resume({"fingerprint": "fingerprint-a"}, "fingerprint-b")

    def test_check_vllm_health_script_writes_json(self) -> None:
        module = load_script(CHECK_HEALTH_SCRIPT, "check_vllm_health_script")
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "health.json"
            with patch.object(
                module,
                "check_vllm_health",
                return_value={
                    "api_base": "http://127.0.0.1:8000",
                    "health_ok": True,
                    "models_ok": True,
                    "metrics_ok": True,
                    "served_models": ["support-ticket-ft"],
                    "ok": True,
                    "errors": [],
                },
            ):
                exit_code = module.main(["--output-path", str(output_path)])
                payload = read_json(output_path)

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])

    def test_build_promptsets_script_smoke(self) -> None:
        module = load_script(BUILD_PROMPTSETS_SCRIPT, "build_promptsets_script")
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root = REPO_ROOT
            runtime_root = tmp_path / "runtime"
            with patch.object(module, "resolve_serving_target") as patched_target, patch.object(
                module,
                "build_benchmark_promptsets",
            ) as patched_build:
                patched_target.return_value = (
                    type("Target", (), {"target_kind": "base_plus_lora"})(),
                    {"benchmark": {"promptsets": {}, "dataset_path": "data/manifests/support_tickets_eval_manifest.jsonl"}},
                )
                patched_build.return_value = {
                    "manifest_path": tmp_path / "promptset_manifest.json",
                    "natural_rows": [1, 2],
                    "stress_rows": [3],
                }
                exit_code = module.main(
                    [
                        "--config",
                        str(repo_root / "configs" / "inference.yaml"),
                        "--runtime-root",
                        str(runtime_root),
                    ]
                )

        self.assertEqual(exit_code, 0)

    def test_render_benchmark_report_script_smoke(self) -> None:
        module = load_script(RENDER_REPORT_SCRIPT, "render_benchmark_report_script")
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bundle_path = tmp_path / "bundle.json"
            write_json(
                bundle_path,
                {
                    "generated_at_utc": "2026-04-12T00:00:00+00:00",
                    "run_name": "bench",
                    "target": {"target_kind": "base_plus_lora", "request_model_name": "support-ticket-ft"},
                    "summary_rows": [],
                    "correctness_rows": [],
                    "config_search_rows": [],
                },
            )
            with patch.object(module, "render_benchmark_report") as patched_render:
                patched_render.return_value = {
                    "report_path": str(tmp_path / "reports" / "benchmark_report.md"),
                    "index_path": str(tmp_path / "reports" / "render_index.json"),
                }
                exit_code = module.main(["--bundle-path", str(bundle_path)])

        self.assertEqual(exit_code, 0)

    def test_render_benchmark_report_writes_markdown(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bundle = {
                "generated_at_utc": "2026-04-12T00:00:00+00:00",
                "run_name": "bench",
                "target": {"target_kind": "base_plus_lora", "request_model_name": "support-ticket-ft"},
                "summary_rows": [
                    {
                        "experiment_family": "mixed_workload_baseline_sweep",
                        "server_config_id": "default",
                        "concurrency": 8,
                        "throughput_rps": 3.2,
                        "latency_p50_ms": 1000.0,
                        "latency_p90_ms": 1500.0,
                        "latency_p99_ms": 2000.0,
                        "tail_inflation_p99_over_p50": 2.0,
                        "bucket_latency_ms": {"short": {"p99": 1800.0}, "medium": {"p99": 2100.0}, "long": {"p99": 2600.0}},
                    }
                ],
                "correctness_rows": [],
                "config_search_rows": [],
            }
            with patch("json_ft.benchmark_reporting._load_pyplot") as patched_plt:
                fake_plt = type(
                    "FakePlt",
                    (),
                    {
                        "subplots": lambda *args, **kwargs: (
                            type("Figure", (), {"tight_layout": lambda self: None, "savefig": lambda self, path, dpi=160: Path(path).write_bytes(b"fake")})(),
                            type(
                                "Axis",
                                (),
                                {
                                    "plot": lambda *args, **kwargs: None,
                                    "scatter": lambda *args, **kwargs: None,
                                    "imshow": lambda *args, **kwargs: None,
                                    "set_title": lambda *args, **kwargs: None,
                                    "set_xlabel": lambda *args, **kwargs: None,
                                    "set_ylabel": lambda *args, **kwargs: None,
                                    "grid": lambda *args, **kwargs: None,
                                    "legend": lambda *args, **kwargs: None,
                                    "set_xticks": lambda *args, **kwargs: None,
                                    "set_xticklabels": lambda *args, **kwargs: None,
                                    "set_yticks": lambda *args, **kwargs: None,
                                    "set_yticklabels": lambda *args, **kwargs: None,
                                },
                            )(),
                        ),
                        "colorbar": lambda *args, **kwargs: None,
                    },
                )()
                patched_plt.return_value = fake_plt
                rendered = render_benchmark_report(bundle, tmp_path / "reports")

            self.assertTrue(Path(rendered["report_path"]).exists())
            report_text = Path(rendered["report_path"]).read_text(encoding="utf-8")
            self.assertIn("# vLLM Benchmark Report", report_text)


if __name__ == "__main__":
    unittest.main()
