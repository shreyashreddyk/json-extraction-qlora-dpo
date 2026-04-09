"""Inference helpers and serving placeholders for later vLLM integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceRequest:
    """Minimal offline inference request shape."""

    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.0


@dataclass(frozen=True)
class InferenceResponse:
    """Minimal offline inference response shape."""

    text: str
    backend: str


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


def offline_placeholder_inference(request: InferenceRequest) -> InferenceResponse:
    """Return a deterministic placeholder response for scaffold validation."""

    return InferenceResponse(
        text="{\n  \"status\": \"placeholder\"\n}",
        backend="offline-placeholder",
    )

