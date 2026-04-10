import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.inference import (
    InferenceRequest,
    _build_generation_config,
    _build_generation_metadata,
    build_inference_backend,
    extract_first_json_object,
    parse_model_output_text,
)


class InferenceParsingTest(unittest.TestCase):
    def test_parse_model_output_accepts_plain_json(self) -> None:
        parsed, recovery_used = parse_model_output_text('{"priority": "high"}')

        self.assertEqual(parsed["priority"], "high")
        self.assertFalse(recovery_used)

    def test_parse_model_output_accepts_fenced_json(self) -> None:
        parsed, recovery_used = parse_model_output_text('```json\n{"priority": "high"}\n```')

        self.assertEqual(parsed["priority"], "high")
        self.assertFalse(recovery_used)

    def test_parse_model_output_recovers_wrapped_json(self) -> None:
        parsed, recovery_used = parse_model_output_text(
            'Here is the extracted payload:\n{"priority": "high", "customer": {"name": "Ava"}}\nThanks!'
        )

        self.assertEqual(parsed["customer"]["name"], "Ava")
        self.assertTrue(recovery_used)

    def test_extract_first_json_object_handles_nested_strings(self) -> None:
        extracted = extract_first_json_object(
            'Model output {"summary": "Contains a brace } inside text", "priority": "high"} trailing'
        )

        self.assertEqual(
            extracted,
            '{"summary": "Contains a brace } inside text", "priority": "high"}',
        )

    def test_parse_model_output_rejects_invalid_output(self) -> None:
        with self.assertRaises(ValueError):
            parse_model_output_text("No JSON present here.")

    def test_generation_metadata_omits_sampling_fields_when_disabled(self) -> None:
        request = InferenceRequest(max_new_tokens=64, temperature=0.0, top_p=1.0, do_sample=False)

        metadata = _build_generation_metadata(request, pad_token_id=7)

        self.assertEqual(
            metadata,
            {
                "max_new_tokens": 64,
                "do_sample": False,
                "pad_token_id": 7,
            },
        )

    def test_generation_metadata_includes_seed_when_present(self) -> None:
        request = InferenceRequest(
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            seed=23,
        )

        metadata = _build_generation_metadata(request, pad_token_id=5)

        self.assertEqual(metadata["seed"], 23)

    def test_generation_config_clears_sampling_only_fields_when_disabled(self) -> None:
        base_config = SimpleNamespace(
            max_new_tokens=32,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=20,
            pad_token_id=None,
        )
        request = InferenceRequest(max_new_tokens=64, temperature=0.0, top_p=1.0, do_sample=False)

        config = _build_generation_config(base_config, request, pad_token_id=11)

        self.assertEqual(config.max_new_tokens, 64)
        self.assertFalse(config.do_sample)
        self.assertEqual(config.pad_token_id, 11)
        self.assertIsNone(config.temperature)
        self.assertIsNone(config.top_p)
        self.assertIsNone(config.top_k)

    def test_generation_config_keeps_sampling_fields_when_enabled(self) -> None:
        base_config = SimpleNamespace(
            max_new_tokens=32,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=20,
            pad_token_id=None,
        )
        request = InferenceRequest(max_new_tokens=80, temperature=0.6, top_p=0.8, do_sample=True)

        config = _build_generation_config(base_config, request, pad_token_id=9)

        self.assertEqual(config.max_new_tokens, 80)
        self.assertTrue(config.do_sample)
        self.assertEqual(config.temperature, 0.6)
        self.assertEqual(config.top_p, 0.8)
        self.assertEqual(config.top_k, 20)

    def test_build_inference_backend_forwards_adapter_path(self) -> None:
        with patch("json_ft.inference.LocalTransformersInferenceBackend.from_model_name_or_path") as patched:
            patched.return_value = "fake-backend"

            backend = build_inference_backend(
                "local-transformers",
                "fake-model",
                adapter_path="/tmp/adapter",
            )

        self.assertEqual(backend, "fake-backend")
        _, kwargs = patched.call_args
        self.assertEqual(kwargs["adapter_path"], "/tmp/adapter")
