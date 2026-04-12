"""Inference helpers for local baseline evaluation and later serving work."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol
import json

from .formatting import strip_code_fences
from .schemas import (
    SchemaConstraint,
    ValidationResult,
    build_support_ticket_schema,
    parse_candidate_json,
    validate_extraction_payload,
)


@dataclass(frozen=True)
class InferenceRequest:
    """Single-model generation request for the extraction task."""

    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    record_id: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    prompt_source: str = "messages"
    seed: int | None = None


@dataclass(frozen=True)
class InferenceResponse:
    """Single-model generation response with evaluation-ready metadata."""

    text: str
    backend: str
    latency_ms: float
    prompt_source: str
    model_name_or_path: str
    parsed_payload: dict[str, Any] | None = None
    parse_error: str | None = None
    validation: ValidationResult | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    json_recovery_used: bool = False


class InferenceBackend(Protocol):
    """Small protocol used by the eval CLI and smoke tests."""

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate a single response for the extraction prompt."""

    def generate_batch(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        """Generate a batch of responses for the extraction prompt."""


def build_vllm_serve_command(
    model_name_or_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> list[str]:
    """Build the command that would be used to launch a vLLM OpenAI server."""

    return [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name_or_path,
        "--host",
        host,
        "--port",
        str(port),
    ]


def _assistant_free_messages(messages: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Drop assistant completions preserved in eval manifests before generation."""

    if not messages:
        return []
    trimmed_messages: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        if role == "assistant":
            break
        trimmed_messages.append(
            {
                "role": role,
                "content": str(message.get("content", "")),
            }
        )
    return trimmed_messages


def extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object embedded in free-form model text."""

    cleaned = strip_code_fences(text)
    start = cleaned.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        character = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
            continue
        if character == "{":
            depth += 1
            continue
        if character == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    return None


def parse_model_output_text(text: str) -> tuple[dict[str, Any], bool]:
    """Parse model text into a JSON object, using small recovery heuristics."""

    candidate = strip_code_fences(text)
    try:
        return parse_candidate_json(candidate), False
    except (TypeError, ValueError, json.JSONDecodeError):
        # The explicit recovery path below keeps the parse contract small and clear.
        pass

    extracted = extract_first_json_object(text)
    if not extracted:
        raise ValueError("Could not find a JSON object in the model output.")

    return parse_candidate_json(extracted), extracted != candidate


def analyze_inference_text(
    text: str,
    schema: SchemaConstraint | None = None,
) -> tuple[dict[str, Any] | None, str | None, ValidationResult | None, bool]:
    """Parse and validate raw model text for downstream evaluation."""

    active_schema = schema or build_support_ticket_schema()
    try:
        parsed_payload, recovery_used = parse_model_output_text(text)
    except ValueError as exc:
        return None, str(exc), None, False

    validation = validate_extraction_payload(parsed_payload, active_schema)
    return parsed_payload, None, validation, recovery_used


def _resolve_torch_dtype(torch_module: Any, value: str | None, use_cuda: bool) -> Any:
    """Resolve the config-facing dtype string into a runtime dtype value."""

    if value in (None, "", "auto"):
        if use_cuda:
            if hasattr(torch_module.cuda, "is_bf16_supported") and torch_module.cuda.is_bf16_supported():
                return torch_module.bfloat16
            return torch_module.float16
        return torch_module.float32
    if not hasattr(torch_module, value):
        raise ValueError(f"Unsupported torch dtype: {value}")
    return getattr(torch_module, value)


def _build_generation_metadata(
    request: InferenceRequest,
    pad_token_id: int | None,
) -> dict[str, Any]:
    """Record the generation settings that were intentionally applied."""

    metadata = {
        "max_new_tokens": request.max_new_tokens,
        "do_sample": request.do_sample,
        "pad_token_id": pad_token_id,
    }
    if request.do_sample:
        metadata["temperature"] = request.temperature
        metadata["top_p"] = request.top_p
    if request.seed is not None:
        metadata["seed"] = request.seed
    return metadata


def _build_generation_config(
    base_config: Any | None,
    request: InferenceRequest,
    pad_token_id: int | None,
) -> Any | None:
    """Prepare a per-request generation config and clear unused sampling knobs."""

    if base_config is None:
        return None

    config = deepcopy(base_config)
    config.max_new_tokens = request.max_new_tokens
    config.do_sample = request.do_sample
    config.pad_token_id = pad_token_id

    if request.do_sample:
        config.temperature = request.temperature
        config.top_p = request.top_p
        return config

    for attribute_name in ("temperature", "top_p", "top_k"):
        if hasattr(config, attribute_name):
            setattr(config, attribute_name, None)
    return config


class LocalTransformersInferenceBackend:
    """Small local `transformers` backend for untuned baseline evaluation."""

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Any,
        model: Any,
        schema: SchemaConstraint | None = None,
        backend_name: str = "local-transformers",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.model = model
        self.schema = schema or build_support_ticket_schema()
        self.backend_name = backend_name

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @classmethod
    def from_model_name_or_path(
        cls,
        model_name_or_path: str,
        *,
        adapter_path: str | None = None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        torch_dtype: str | None = "auto",
        device_map: str | None = None,
        schema: SchemaConstraint | None = None,
    ) -> LocalTransformersInferenceBackend:
        """Load a local causal LM and tokenizer lazily."""

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is required for local baseline inference. "
                "Install it in Colab or your local environment before running eval_model.py."
            ) from exc
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers is required for local baseline inference. "
                "Install it in Colab or your local environment before running eval_model.py."
            ) from exc
        if adapter_path:
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "peft is required to load adapter-backed inference. "
                    "Install it in Colab or your local environment before using adapter_path."
                ) from exc
        else:
            PeftModel = None

        use_cuda = bool(torch.cuda.is_available())
        resolved_dtype = _resolve_torch_dtype(torch, torch_dtype, use_cuda)
        print(f"[eval] Loading tokenizer from {model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        common_model_kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "device_map": device_map,
        }
        print(
            f"[eval] Loading model from {model_name_or_path} "
            f"(dtype={resolved_dtype}, device_map={device_map or '<none>'})..."
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=resolved_dtype,
                **common_model_kwargs,
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=resolved_dtype,
                **common_model_kwargs,
            )
        if adapter_path and PeftModel is not None:
            print(f"[eval] Loading adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)
        if device_map is None and use_cuda:
            print("[eval] Moving model to cuda...")
            model = model.to("cuda")
        model.eval()
        print("[eval] Model backend is ready.")
        return cls(
            model_name_or_path=adapter_path or model_name_or_path,
            tokenizer=tokenizer,
            model=model,
            schema=schema,
        )

    def render_prompt(self, request: InferenceRequest) -> str:
        """Render request data into the model's input prompt."""

        if request.messages:
            trimmed_messages = _assistant_free_messages(request.messages)
            if not trimmed_messages:
                raise ValueError("messages prompt source requires at least one non-assistant message")
            return self.tokenizer.apply_chat_template(
                trimmed_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if request.prompt:
            return request.prompt
        raise ValueError("Inference request must provide either messages or prompt text.")

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Run a single deterministic generation and return eval-ready metadata."""

        import torch

        if request.seed is not None:
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request.seed)

        prompt_text = self.render_prompt(request)
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            encoded = {key: value.to(model_device) for key, value in encoded.items()}

        generation_metadata = _build_generation_metadata(request, self.tokenizer.pad_token_id)
        generation_config = _build_generation_config(
            getattr(self.model, "generation_config", None),
            request,
            self.tokenizer.pad_token_id,
        )

        start = perf_counter()
        with torch.inference_mode():
            if generation_config is not None:
                generated = self.model.generate(**encoded, generation_config=generation_config)
            else:
                generated = self.model.generate(**encoded, **generation_metadata)
        latency_ms = (perf_counter() - start) * 1000.0

        prompt_token_count = encoded["input_ids"].shape[-1]
        generated_tokens = generated[0][prompt_token_count:]
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        parsed_payload, parse_error, validation, recovery_used = analyze_inference_text(
            output_text,
            self.schema,
        )
        return InferenceResponse(
            text=output_text,
            backend=self.backend_name,
            latency_ms=latency_ms,
            prompt_source=request.prompt_source,
            model_name_or_path=self.model_name_or_path,
            parsed_payload=parsed_payload,
            parse_error=parse_error,
            validation=validation,
            generation_kwargs=generation_metadata,
            json_recovery_used=recovery_used,
        )

    def generate_batch(self, requests: list[InferenceRequest]) -> list[InferenceResponse]:
        """Run one batched generation call and return one response per request."""

        import torch

        if not requests:
            return []

        first_request = requests[0]
        generation_key = (
            first_request.max_new_tokens,
            first_request.temperature,
            first_request.top_p,
            first_request.do_sample,
            first_request.prompt_source,
            first_request.seed,
        )
        for request in requests[1:]:
            request_key = (
                request.max_new_tokens,
                request.temperature,
                request.top_p,
                request.do_sample,
                request.prompt_source,
                request.seed,
            )
            if request_key != generation_key:
                raise ValueError("Batched evaluation requires requests with identical generation settings.")

        if first_request.seed is not None:
            torch.manual_seed(first_request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(first_request.seed)

        prompt_texts = [self.render_prompt(request) for request in requests]
        original_padding_side = getattr(self.tokenizer, "padding_side", None)
        if original_padding_side is not None:
            self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        finally:
            if original_padding_side is not None:
                self.tokenizer.padding_side = original_padding_side

        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            encoded = {key: value.to(model_device) for key, value in encoded.items()}

        generation_metadata = _build_generation_metadata(first_request, self.tokenizer.pad_token_id)
        generation_config = _build_generation_config(
            getattr(self.model, "generation_config", None),
            first_request,
            self.tokenizer.pad_token_id,
        )

        start = perf_counter()
        with torch.inference_mode():
            if generation_config is not None:
                generated = self.model.generate(**encoded, generation_config=generation_config)
            else:
                generated = self.model.generate(**encoded, **generation_metadata)
        batch_latency_ms = (perf_counter() - start) * 1000.0

        prompt_token_count = encoded["input_ids"].shape[-1]
        responses: list[InferenceResponse] = []
        for index, request in enumerate(requests):
            generated_tokens = generated[index][prompt_token_count:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            parsed_payload, parse_error, validation, recovery_used = analyze_inference_text(
                output_text,
                self.schema,
            )
            responses.append(
                InferenceResponse(
                    text=output_text,
                    backend=self.backend_name,
                    latency_ms=batch_latency_ms,
                    prompt_source=request.prompt_source,
                    model_name_or_path=self.model_name_or_path,
                    parsed_payload=parsed_payload,
                    parse_error=parse_error,
                    validation=validation,
                    generation_kwargs=generation_metadata,
                    json_recovery_used=recovery_used,
                )
            )
        return responses


class OfflinePlaceholderInferenceBackend:
    """Deterministic backend retained for smoke tests and scaffold validation."""

    def __init__(self, schema: SchemaConstraint | None = None) -> None:
        self.schema = schema or build_support_ticket_schema()

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        response = offline_placeholder_inference(request, self.schema)
        parsed_payload, parse_error, validation, recovery_used = analyze_inference_text(
            response.text,
            self.schema,
        )
        return InferenceResponse(
            text=response.text,
            backend=response.backend,
            latency_ms=0.0,
            prompt_source=request.prompt_source,
            model_name_or_path="offline-placeholder",
            parsed_payload=parsed_payload,
            parse_error=parse_error,
            validation=validation,
            generation_kwargs={},
            json_recovery_used=recovery_used,
        )


def build_inference_backend(
    backend: str,
    model_name_or_path: str,
    *,
    adapter_path: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    torch_dtype: str | None = "auto",
    device_map: str | None = None,
    schema: SchemaConstraint | None = None,
) -> InferenceBackend:
    """Resolve the configured inference backend."""

    if backend == "local-transformers":
        return LocalTransformersInferenceBackend.from_model_name_or_path(
            model_name_or_path,
            adapter_path=adapter_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            schema=schema,
        )
    if backend == "offline-placeholder":
        return OfflinePlaceholderInferenceBackend(schema=schema)
    raise ValueError(f"Unsupported inference backend: {backend}")


def offline_placeholder_inference(
    request: InferenceRequest,
    schema: SchemaConstraint | None = None,
) -> InferenceResponse:
    """Return a deterministic placeholder response for scaffold validation."""

    _ = schema or build_support_ticket_schema()
    return InferenceResponse(
        text="{\n  \"status\": \"placeholder\"\n}",
        backend="offline-placeholder",
        latency_ms=0.0,
        prompt_source=request.prompt_source,
        model_name_or_path="offline-placeholder",
    )
