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


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SFT_SCRIPT_PATH = REPO_ROOT / "scripts" / "train_sft.py"


def load_train_sft_script_module():
    spec = importlib.util.spec_from_file_location("train_sft_script", TRAIN_SFT_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_train_row() -> dict:
    return {
        "record_id": "train-001",
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": "Extract the ticket."},
            {"role": "assistant", "content": '{"priority": "high"}'},
        ],
    }


def build_eval_row() -> dict:
    return {
        "record_id": "eval-001",
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": "Extract the ticket."},
            {"role": "assistant", "content": '{"priority": "high"}'},
        ],
        "prompt": "Extract the ticket.",
        "reference_json": '{"priority": "high"}',
    }


def build_sft_config(train_manifest: Path, eval_manifest: Path) -> str:
    return "\n".join(
        [
            "model:",
            "  model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct",
            "  trust_remote_code: false",
            '  eos_token: "<|im_end|>"',
            "data:",
            f"  train_manifest: {train_manifest}",
            f"  eval_manifest: {eval_manifest}",
            "  dataset_format: messages",
            "  max_seq_length: 1024",
            "quantization:",
            "  enabled: true",
            "  load_in_4bit: true",
            "  bnb_4bit_quant_type: nf4",
            "  bnb_4bit_use_double_quant: true",
            "  compute_dtype: auto",
            "lora:",
            "  r: 16",
            "  alpha: 32",
            "  dropout: 0.05",
            "  bias: none",
            "  task_type: CAUSAL_LM",
            "  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]",
            "training:",
            "  learning_rate: 2.0e-4",
            "  lr_scheduler_type: cosine",
            "  warmup_ratio: 0.05",
            "  weight_decay: 0.0",
            "  optim: paged_adamw_32bit",
            "  gradient_checkpointing: true",
            "  packing: false",
            "  per_device_eval_batch_size: 1",
            "  save_total_limit: 2",
            "  report_to: []",
            "  completion_only_loss: true",
            "profiles:",
            "  dev:",
            "    data:",
            "      train_sample_limit: 1",
            "      eval_sample_limit: 1",
            "    training:",
            "      max_steps: 5",
            "      per_device_train_batch_size: 1",
            "      gradient_accumulation_steps: 2",
            "      logging_steps: 1",
            "      eval_strategy: steps",
            "      eval_steps: 1",
            "      save_strategy: steps",
            "      save_steps: 1",
            "  full:",
            "    data:",
            "      train_sample_limit: null",
            "      eval_sample_limit: null",
            "    training:",
            "      num_train_epochs: 3",
            "      per_device_train_batch_size: 1",
            "      gradient_accumulation_steps: 8",
            "      logging_steps: 5",
            "      eval_strategy: epoch",
            "      save_strategy: epoch",
            "artifacts:",
            '  summary_filename: "{run_name}_sft_summary.json"',
            '  history_filename: "{run_name}_sft_history.json"',
            '  loss_curve_filename: "{run_name}_sft_loss_curve.png"',
            '  eval_loss_curve_filename: "{run_name}_sft_eval_loss_curve.png"',
            '  checkpoint_manifest_filename: "{run_name}_adapter_manifest.json"',
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
                {"loss": 1.25, "learning_rate": 2.0e-4, "step": 1, "epoch": 0.5},
                {"loss": 0.92, "learning_rate": 1.5e-4, "step": 2, "epoch": 1.0},
                {"eval_loss": 0.81, "step": 2, "epoch": 1.0},
            ],
            best_metric=0.81,
        )

    def train(self):  # pragma: no cover - exercised via CLI
        return SimpleNamespace(metrics={"train_runtime": 1.23, "train_loss": 0.92})

    def save_model(self, path: str) -> None:  # pragma: no cover - exercised via CLI
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "adapter_model.safetensors").write_text("weights", encoding="utf-8")

    def save_state(self) -> None:  # pragma: no cover - exercised via CLI
        payload = {"log_history": self.state.log_history, "best_metric": self.state.best_metric}
        (self.checkpoint_root / "trainer_state.json").write_text(json.dumps(payload), encoding="utf-8")


class TrainSftCliTest(unittest.TestCase):
    def test_train_sft_dry_run_writes_summary_and_manifest(self) -> None:
        module = load_train_sft_script_module()

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_manifest = tmp_path / "train.jsonl"
            eval_manifest = tmp_path / "eval.jsonl"
            config_path = tmp_path / "sft.yaml"
            runtime_root = tmp_path / "runtime"

            write_jsonl(train_manifest, [build_train_row()])
            write_jsonl(eval_manifest, [build_eval_row()])
            write_text(config_path, build_sft_config(train_manifest, eval_manifest))

            exit_code = module.main(
                [
                    "--config",
                    str(config_path),
                    "--profile",
                    "dev",
                    "--run-name",
                    "dry-run-smoke",
                    "--runtime-root",
                    str(runtime_root),
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = read_json(runtime_root / "persistent" / "metrics" / "dry-run-smoke_sft_summary.json")
            checkpoint_manifest = read_json(
                runtime_root / "persistent" / "checkpoints" / "sft" / "dry-run-smoke" / "dry-run-smoke_adapter_manifest.json"
            )

            self.assertEqual(summary["status"], "dry_run_ready")
            self.assertEqual(summary["profile"], "dev")
            self.assertEqual(summary["train_record_count"], 1)
            self.assertEqual(summary["eval_record_count"], 1)
            self.assertEqual(checkpoint_manifest["status"], "dry_run_ready")
            self.assertEqual(checkpoint_manifest["adapter_path"], summary["adapter_path"])

    def test_train_sft_fake_training_writes_history_plots_and_adapter_manifest(self) -> None:
        module = load_train_sft_script_module()

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_manifest = tmp_path / "train.jsonl"
            eval_manifest = tmp_path / "eval.jsonl"
            config_path = tmp_path / "sft.yaml"
            runtime_root = tmp_path / "runtime"

            write_jsonl(train_manifest, [build_train_row()])
            write_jsonl(eval_manifest, [build_eval_row()])
            write_text(config_path, build_sft_config(train_manifest, eval_manifest))

            def fake_build_trainer_bundle(*, artifacts, **kwargs):
                return SimpleNamespace(
                    trainer=FakeTrainer(artifacts.checkpoint_root),
                    model=SimpleNamespace(),
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
                            "fake-train",
                            "--runtime-root",
                            str(runtime_root),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            summary = read_json(runtime_root / "persistent" / "metrics" / "fake-train_sft_summary.json")
            history = read_json(runtime_root / "persistent" / "metrics" / "fake-train_sft_history.json")
            checkpoint_manifest = read_json(
                runtime_root / "persistent" / "checkpoints" / "sft" / "fake-train" / "fake-train_adapter_manifest.json"
            )

            self.assertEqual(summary["status"], "completed")
            self.assertIn("train_runtime", summary["train_metrics"])
            self.assertEqual(len(history["train_loss"]), 2)
            self.assertEqual(len(history["eval_loss"]), 1)
            self.assertEqual(checkpoint_manifest["status"], "completed")
            self.assertTrue((runtime_root / "persistent" / "plots" / "fake-train_sft_loss_curve.png").exists())
            self.assertTrue((runtime_root / "persistent" / "plots" / "fake-train_sft_eval_loss_curve.png").exists())
            self.assertTrue(
                (runtime_root / "persistent" / "checkpoints" / "sft" / "fake-train" / "adapter" / "adapter_model.safetensors").exists()
            )
            self.assertTrue(
                (runtime_root / "persistent" / "logs" / "sft" / "fake-train" / "trainer_state.json").exists()
            )
