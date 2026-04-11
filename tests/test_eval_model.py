import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.inference import InferenceResponse, analyze_inference_text
from json_ft.utils import read_jsonl, read_text, write_jsonl, write_text


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_model.py"
EVAL_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "support_tickets_eval_manifest.jsonl"


def load_eval_script_module():
    spec = importlib.util.spec_from_file_location("eval_model_script", EVAL_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    def generate(self, request):  # pragma: no cover - exercised through CLI test
        text = self._responses[self._index]
        self._index += 1
        parsed_payload, parse_error, validation, recovery_used = analyze_inference_text(text)
        return InferenceResponse(
            text=text,
            backend="fake-backend",
            latency_ms=7.5,
            prompt_source=request.prompt_source,
            model_name_or_path="fake-model",
            parsed_payload=parsed_payload,
            parse_error=parse_error,
            validation=validation,
            generation_kwargs={
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "do_sample": request.do_sample,
            },
            json_recovery_used=recovery_used,
        )


class EvalModelCliTest(unittest.TestCase):
    def test_eval_cli_writes_metrics_report_and_predictions(self) -> None:
        rows = read_jsonl(EVAL_MANIFEST_PATH)[:2]
        responses = [row["reference_json"] for row in rows]
        module = load_eval_script_module()

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "eval_manifest.jsonl"
            config_path = tmp_path / "eval.yaml"
            metrics_path = tmp_path / "metrics.json"
            report_path = tmp_path / "report.md"
            predictions_path = tmp_path / "predictions.jsonl"
            runtime_root = tmp_path / "runtime"

            write_jsonl(dataset_path, rows)
            write_text(
                config_path,
                "\n".join(
                    [
                        "backend: local-transformers",
                        "model_name_or_path: fake-model",
                        f"dataset_path: {dataset_path}",
                        "prompt_source: messages",
                        "generation:",
                        "  max_new_tokens: 64",
                        "  temperature: 0.0",
                        "  top_p: 1.0",
                        "  do_sample: false",
                        "artifacts:",
                        '  metrics_filename: "{run_name}_metrics.json"',
                        '  report_filename: "{run_name}_report.md"',
                        '  predictions_filename: "{run_name}_predictions.jsonl"',
                        "",
                    ]
                ),
            )

            with patch("json_ft.evaluation.build_inference_backend", return_value=FakeBackend(responses)):
                exit_code = module.main(
                    [
                        "--config",
                        str(config_path),
                        "--run-name",
                        "smoke-eval",
                        "--stage-label",
                        "baseline",
                        "--runtime-root",
                        str(runtime_root),
                        "--metrics-output",
                        str(metrics_path),
                        "--report-output",
                        str(report_path),
                        "--predictions-output",
                        str(predictions_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            report = read_text(report_path)
            prediction_rows = read_jsonl(predictions_path)

            self.assertEqual(metrics["record_count"], 2)
            self.assertEqual(metrics["stage"], "baseline")
            self.assertEqual(metrics["json_validity_rate"], 1.0)
            self.assertEqual(metrics["schema_validation_pass_rate"], 1.0)
            self.assertIn("# Baseline Evaluation Report: smoke-eval", report)
            self.assertIn("JSON validity rate", report)
            self.assertEqual(len(prediction_rows), 2)
            self.assertEqual(prediction_rows[0]["backend"], "fake-backend")
            self.assertEqual(prediction_rows[0]["stage_label"], "baseline")
