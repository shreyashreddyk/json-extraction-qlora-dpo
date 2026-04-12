import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.utils import read_json, write_jsonl, write_text
from json_ft.dpo import resolve_dpo_config


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DPO_SCRIPT_PATH = REPO_ROOT / "scripts" / "train_dpo.py"


def load_train_dpo_script_module():
    spec = importlib.util.spec_from_file_location("train_dpo_script", TRAIN_DPO_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_preference_row() -> dict:
    return {
        "prompt": "Extract the ticket.",
        "chosen": '{"priority": "high"}',
        "rejected": '{"priority": "low"}',
    }


def build_dpo_config(preference_manifest: Path) -> str:
    return "\n".join(
        [
            "model:",
            "  base_model: Qwen/Qwen2.5-1.5B-Instruct",
            "  adapter_path: /tmp/fake-sft-adapter",
            "  reference_strategy: adapter",
            "  revision: null",
            "  trust_remote_code: false",
            "  torch_dtype: auto",
            "  device_map: null",
            "quantization:",
            "  enabled: true",
            "  load_in_4bit: true",
            "  bnb_4bit_quant_type: nf4",
            "  bnb_4bit_use_double_quant: true",
            "  compute_dtype: auto",
            "pair_generation:",
            "  input_path: data/fixtures/support_tickets.jsonl",
            "  build_summary_path: data/manifests/support_tickets_dataset_build_summary.json",
            "  composition_summary_path: artifacts/metrics/support_tickets_dataset_composition.json",
            "  source_format: json_extraction",
            "  source_split: train",
            "  prompt_source: messages",
            "  candidate_count: 3",
            "  sample_limit: 3",
            "  quality_gates:",
            "    minimum_score_gap: 0.2",
            "  generation:",
            "    max_new_tokens: 128",
            "    temperature: 0.8",
            "    top_p: 0.95",
            "    do_sample: true",
            "    base_seed: 11",
            "training:",
            f"  preference_manifest: {preference_manifest}",
            "  eval_preference_manifest: null",
            "  train_sample_limit: null",
            "  eval_sample_limit: null",
            "  train_sample_percent: null",
            "  eval_sample_percent: null",
            "  sample_seed: 17",
            "  beta: 0.1",
            "  loss_type: sigmoid",
            "  learning_rate: 5.0e-6",
            "  num_train_epochs: 1",
            "  per_device_train_batch_size: 1",
            "  per_device_eval_batch_size: 1",
            "  gradient_accumulation_steps: 2",
            "  gradient_checkpointing: true",
            "  lr_scheduler_type: cosine",
            "  warmup_ratio: 0.05",
            "  weight_decay: 0.0",
            "  optim: paged_adamw_32bit",
            "  logging_steps: 1",
            "  save_total_limit: 2",
            "  report_to: []",
            '  eval_strategy: "no"',
            "  save_strategy: steps",
            "  save_steps: 1",
            "  max_prompt_length: 1024",
            "  max_completion_length: 256",
            "profiles:",
            "  dev:",
            "    training:",
            "      train_sample_limit: 1",
            "      max_steps: 3",
            "      gradient_accumulation_steps: 2",
            "      logging_steps: 1",
            "      save_strategy: steps",
            "      save_steps: 1",
            "  full:",
            "    training:",
            "      train_sample_limit: null",
            "  colab_full:",
            "    training:",
            "      train_sample_limit: null",
            "artifacts:",
            '  summary_filename: "{run_name}_dpo_summary.json"',
            '  history_filename: "{run_name}_dpo_history.json"',
            '  checkpoint_manifest_filename: "{run_name}_dpo_manifest.json"',
            '  loss_curve_filename: "{run_name}_dpo_loss_curve.png"',
            '  eval_loss_curve_filename: "{run_name}_dpo_eval_loss_curve.png"',
            '  rewards_chosen_curve_filename: "{run_name}_dpo_rewards_chosen_curve.png"',
            '  rewards_rejected_curve_filename: "{run_name}_dpo_rewards_rejected_curve.png"',
            '  rewards_accuracies_curve_filename: "{run_name}_dpo_rewards_accuracies_curve.png"',
            '  rewards_margins_curve_filename: "{run_name}_dpo_rewards_margins_curve.png"',
            "",
        ]
    )


class FakeFigure:
    def savefig(self, path, dpi=160):  # pragma: no cover - exercised via CLI
        Path(path).write_bytes(b"fake-image")


class FakePyplot:
    def figure(self, figsize=(8, 4.5)):  # pragma: no cover - exercised via CLI
        return FakeFigure()

    def plot(self, *args, **kwargs):  # pragma: no cover - exercised via CLI
        return None

    def title(self, *args, **kwargs):  # pragma: no cover - exercised via CLI
        return None

    def xlabel(self, *args, **kwargs):  # pragma: no cover - exercised via CLI
        return None

    def ylabel(self, *args, **kwargs):  # pragma: no cover - exercised via CLI
        return None

    def grid(self, *args, **kwargs):  # pragma: no cover - exercised via CLI
        return None

    def tight_layout(self):  # pragma: no cover - exercised via CLI
        return None

    def close(self, figure):  # pragma: no cover - exercised via CLI
        return None


class FakeTokenizer:
    def save_pretrained(self, path: str) -> None:  # pragma: no cover - exercised via CLI
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeTrainer:
    def __init__(self, checkpoint_root: Path) -> None:
        self.checkpoint_root = checkpoint_root
        self.state = SimpleNamespace(
            log_history=[
                {"train_loss": 0.92, "learning_rate": 5.0e-6, "step": 1, "epoch": 0.5},
                {"rewards/chosen": 0.23, "rewards/rejected": -0.12, "rewards/margins": 0.35, "rewards/accuracies": 1.0, "step": 1, "epoch": 0.5},
                {"train_loss": 0.61, "learning_rate": 4.5e-6, "step": 2, "epoch": 1.0},
            ],
            best_metric=0.35,
        )

    def train(self):  # pragma: no cover - exercised via CLI
        return SimpleNamespace(metrics={"train_runtime": 2.34, "train_loss": 0.61})

    def save_model(self, path: str) -> None:  # pragma: no cover - exercised via CLI
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "adapter_model.safetensors").write_text("weights", encoding="utf-8")

    def save_state(self) -> None:  # pragma: no cover - exercised via CLI
        payload = {"log_history": self.state.log_history, "best_metric": self.state.best_metric}
        (self.checkpoint_root / "trainer_state.json").write_text(json.dumps(payload), encoding="utf-8")


class TrainDpoCliTest(unittest.TestCase):
    def test_resolve_dpo_config_colab_full_alias_maps_to_full(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            preference_manifest = tmp_path / "prefs.jsonl"
            config_path = tmp_path / "dpo.yaml"

            write_jsonl(preference_manifest, [build_preference_row()])
            write_text(config_path, build_dpo_config(preference_manifest))

            config = resolve_dpo_config(
                config_path=config_path,
                repo_root=tmp_path,
                profile_name="colab_full",
                preference_manifest=preference_manifest,
            )

            self.assertEqual(config.profile_name, "full")
            self.assertIn("minimum_score_gap", config.quality_gates)

    def test_train_dpo_dry_run_writes_summary_and_manifest(self) -> None:
        module = load_train_dpo_script_module()

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            preference_manifest = tmp_path / "prefs.jsonl"
            config_path = tmp_path / "dpo.yaml"
            runtime_root = tmp_path / "runtime"

            write_jsonl(preference_manifest, [build_preference_row()])
            write_text(config_path, build_dpo_config(preference_manifest))

            exit_code = module.main(
                [
                    "--config",
                    str(config_path),
                    "--profile",
                    "dev",
                    "--run-name",
                    "dry-run-dpo",
                    "--runtime-root",
                    str(runtime_root),
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = read_json(runtime_root / "persistent" / "metrics" / "dry-run-dpo_dpo_summary.json")
            checkpoint_manifest = read_json(
                runtime_root / "persistent" / "checkpoints" / "dpo" / "dry-run-dpo" / "dry-run-dpo_dpo_manifest.json"
            )

            self.assertEqual(summary["status"], "dry_run_ready")
            self.assertEqual(summary["profile"], "dev")
            self.assertEqual(summary["train_record_count"], 1)
            self.assertEqual(summary["effective_batch_size"], 2)
            self.assertEqual(checkpoint_manifest["status"], "dry_run_ready")
            self.assertEqual(checkpoint_manifest["preference_manifest"], str(preference_manifest.resolve()))

    def test_train_dpo_dry_run_applies_sample_percent_cli_overrides(self) -> None:
        module = load_train_dpo_script_module()

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            preference_manifest = tmp_path / "prefs.jsonl"
            config_path = tmp_path / "dpo.yaml"
            runtime_root = tmp_path / "runtime"

            write_jsonl(preference_manifest, [build_preference_row() for _ in range(10)])
            write_text(config_path, build_dpo_config(preference_manifest))

            exit_code = module.main(
                [
                    "--config",
                    str(config_path),
                    "--profile",
                    "full",
                    "--run-name",
                    "override-dpo-sampling",
                    "--runtime-root",
                    str(runtime_root),
                    "--train-sample-percent",
                    "0.4",
                    "--sample-seed",
                    "29",
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = read_json(runtime_root / "persistent" / "metrics" / "override-dpo-sampling_dpo_summary.json")
            checkpoint_manifest = read_json(
                runtime_root
                / "persistent"
                / "checkpoints"
                / "dpo"
                / "override-dpo-sampling"
                / "override-dpo-sampling_dpo_manifest.json"
            )

            self.assertEqual(summary["train_record_count"], 4)
            self.assertEqual(summary["subset_selection"]["train"]["original_row_count"], 10)
            self.assertEqual(summary["subset_selection"]["train"]["selected_row_count"], 4)
            self.assertEqual(summary["subset_selection"]["train"]["sample_percent"], 0.4)
            self.assertEqual(checkpoint_manifest["subset_selection"]["train"]["sample_seed"], 29)

    def test_train_dpo_dry_run_applies_batch_cli_overrides(self) -> None:
        module = load_train_dpo_script_module()

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            preference_manifest = tmp_path / "prefs.jsonl"
            config_path = tmp_path / "dpo.yaml"
            runtime_root = tmp_path / "runtime"

            write_jsonl(preference_manifest, [build_preference_row() for _ in range(4)])
            write_text(config_path, build_dpo_config(preference_manifest))

            exit_code = module.main(
                [
                    "--config",
                    str(config_path),
                    "--profile",
                    "full",
                    "--run-name",
                    "override-dpo-batch",
                    "--runtime-root",
                    str(runtime_root),
                    "--per-device-train-batch-size",
                    "3",
                    "--per-device-eval-batch-size",
                    "2",
                    "--gradient-accumulation-steps",
                    "5",
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = read_json(runtime_root / "persistent" / "metrics" / "override-dpo-batch_dpo_summary.json")
            checkpoint_manifest = read_json(
                runtime_root
                / "persistent"
                / "checkpoints"
                / "dpo"
                / "override-dpo-batch"
                / "override-dpo-batch_dpo_manifest.json"
            )

            self.assertEqual(summary["training"]["per_device_train_batch_size"], 3)
            self.assertEqual(summary["training"]["per_device_eval_batch_size"], 2)
            self.assertEqual(summary["training"]["gradient_accumulation_steps"], 5)
            self.assertEqual(summary["effective_batch_size"], 15)
            self.assertEqual(checkpoint_manifest["training"]["per_device_train_batch_size"], 3)
            self.assertEqual(checkpoint_manifest["training"]["gradient_accumulation_steps"], 5)

    def test_train_dpo_fake_training_writes_history_reward_plots_and_manifest(self) -> None:
        module = load_train_dpo_script_module()

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            preference_manifest = tmp_path / "prefs.jsonl"
            config_path = tmp_path / "dpo.yaml"
            runtime_root = tmp_path / "runtime"

            write_jsonl(preference_manifest, [build_preference_row()])
            write_text(config_path, build_dpo_config(preference_manifest))

            def fake_build_trainer_bundle(*, artifacts, **kwargs):
                return SimpleNamespace(
                    trainer=FakeTrainer(artifacts.checkpoint_root),
                    model=SimpleNamespace(),
                    ref_model=SimpleNamespace(),
                    tokenizer=FakeTokenizer(),
                )

            with patch.object(module, "build_trainer_bundle", side_effect=fake_build_trainer_bundle):
                with patch("json_ft.training_plots._load_pyplot", return_value=FakePyplot()):
                    exit_code = module.main(
                        [
                            "--config",
                            str(config_path),
                            "--profile",
                            "dev",
                            "--run-name",
                            "fake-dpo",
                            "--runtime-root",
                            str(runtime_root),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            summary = read_json(runtime_root / "persistent" / "metrics" / "fake-dpo_dpo_summary.json")
            history = read_json(runtime_root / "persistent" / "metrics" / "fake-dpo_dpo_history.json")
            checkpoint_manifest = read_json(
                runtime_root / "persistent" / "checkpoints" / "dpo" / "fake-dpo" / "fake-dpo_dpo_manifest.json"
            )

            self.assertEqual(summary["status"], "completed")
            self.assertIn("train_runtime", summary["train_metrics"])
            self.assertEqual(summary["effective_batch_size"], 2)
            self.assertIsNotNone(summary["history_artifacts"]["loss_curve_path"])
            self.assertEqual(len(history["train_loss"]), 2)
            self.assertIn("rewards/chosen", history["scalar_series"])
            self.assertEqual(checkpoint_manifest["status"], "completed")
            self.assertTrue((runtime_root / "persistent" / "plots" / "fake-dpo_dpo_loss_curve.png").exists())
            self.assertTrue(
                (runtime_root / "persistent" / "plots" / "fake-dpo_dpo_rewards_chosen_curve.png").exists()
            )
            self.assertTrue(
                (runtime_root / "persistent" / "checkpoints" / "dpo" / "fake-dpo" / "adapter" / "adapter_model.safetensors").exists()
            )
            self.assertTrue(
                (runtime_root / "persistent" / "logs" / "dpo" / "fake-dpo" / "trainer_state.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
