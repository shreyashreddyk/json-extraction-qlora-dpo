"""vLLM serving, promptset construction, and benchmark helpers."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable
from urllib import error, parse, request
import csv
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import time

from .inference import analyze_inference_text
from .manifests import LatestModelManifest
from .metrics import CATEGORICAL_EXACT_MATCH_FIELDS
from .prompts import render_system_instruction, render_user_prompt
from .runtime import detect_colab
from .schemas import build_support_ticket_schema
from .utils import load_yaml, read_json, read_jsonl, write_json, write_jsonl, write_text

DEFAULT_LOCAL_GOOGLE_DRIVE_ROOT = (
    "/Users/shreyashreddy/Library/CloudStorage/GoogleDrive-kshreyashreddy@gmail.com/My Drive"
)
DEFAULT_COLAB_DRIVE_ROOT = "/content/drive/MyDrive"
DEFAULT_PROMOTED_LORA_ALIAS = "support-ticket-ft"
DEFAULT_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
PROMPTSET_VERSION = "v1"
STRESS_WORKLOAD_VERSION = "v1"


def _load_tokenizer(model_name_or_path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency-backed
        raise RuntimeError(
            "transformers is required for promptset tokenization and benchmark request budgeting."
        ) from exc

    return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _stable_sort_key(row: dict[str, Any], seed: int) -> tuple[int, str]:
    record_id = str(row.get("record_id") or row.get("lineage_record_id") or "")
    return (_stable_seed(record_id, seed), record_id)


def _resolve_path(repo_root: Path, value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    raw = Path(str(value))
    if raw.is_absolute():
        return raw
    return (repo_root / raw).resolve()


def _local_google_drive_root() -> Path:
    configured = os.environ.get("LOCAL_GOOGLE_DRIVE_ROOT", DEFAULT_LOCAL_GOOGLE_DRIVE_ROOT)
    return Path(configured).expanduser().resolve()


def normalize_drive_path(path_value: str | Path | None) -> str | None:
    """Map Drive paths between local desktop mirrors and Colab when possible."""

    if path_value in (None, ""):
        return None
    raw = str(path_value)
    local_root = str(_local_google_drive_root())
    colab_root = DEFAULT_COLAB_DRIVE_ROOT
    if detect_colab() and raw.startswith(local_root):
        suffix = raw[len(local_root) :].lstrip("/")
        return f"{colab_root}/{suffix}"
    if not detect_colab() and raw.startswith(colab_root):
        suffix = raw[len(colab_root) :].lstrip("/")
        return str(_local_google_drive_root() / suffix)
    return raw


def _coerce_manifest_payload(payload: dict[str, Any]) -> LatestModelManifest | dict[str, Any]:
    if {"stage", "status", "base_model", "adapter_path"} <= payload.keys():
        try:
            return LatestModelManifest(**payload)
        except TypeError:
            return payload
    return payload


def _load_optional_manifest(path: Path | None) -> LatestModelManifest | dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return _coerce_manifest_payload(read_json(path))


def _assistant_free_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    trimmed: list[dict[str, str]] = []
    for message in messages or []:
        role = str(message.get("role", "")).strip()
        if role == "assistant":
            break
        trimmed.append(
            {
                "role": role,
                "content": str(message.get("content", "")),
            }
        )
    return trimmed


def _render_request_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = _assistant_free_messages(row.get("messages"))
    if messages:
        return messages
    return [
        {"role": "system", "content": render_system_instruction()},
        {"role": "user", "content": render_user_prompt(str(row.get("input_text", "")))},
    ]


def render_request_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def prompt_token_length(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    rendered = render_request_prompt(tokenizer, messages)
    return len(tokenizer(rendered, add_special_tokens=True)["input_ids"])


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _decode_tokens(tokenizer: Any, tokens: list[int]) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=True)


@dataclass(frozen=True)
class ResolvedServingTarget:
    """Explicit deployment target for the vLLM server."""

    target_kind: str
    base_model: str
    served_model_name_or_path: str
    request_model_name: str
    adapter_path: str | None
    merged_model_path: str | None
    lora_alias: str | None
    source_manifest_path: str | None
    promotion_manifest_path: str | None
    schema_version: str | None


@dataclass(frozen=True)
class PromptBudget:
    """Request-level context budgeting outcome."""

    original_prompt_tokens: int
    final_prompt_tokens: int
    desired_output_tokens: int
    final_output_tokens: int
    trim_applied: bool
    trimmed_input_tokens: int
    trim_reason: str | None
    trimmed_user_content: str


@dataclass(frozen=True)
class ServerLaunchResult:
    """Runtime metadata for one launched vLLM process."""

    config_id: str
    command: list[str]
    api_base: str
    host: str
    port: int
    pid: int
    stdout_log_path: str
    stderr_log_path: str
    pid_path: str
    metadata_path: str


@dataclass(frozen=True)
class BenchmarkPaths:
    """Runtime layout for benchmark artifacts."""

    run_dir: Path
    promptsets_dir: Path
    raw_dir: Path
    tables_dir: Path
    plots_dir: Path
    reports_dir: Path
    correctness_dir: Path
    server_dir: Path


def benchmark_checkpoint_paths(run_dir: Path) -> dict[str, Path]:
    """Return the checkpoint layout for a benchmark run."""

    checkpoint_dir = run_dir / "checkpoints"
    steps_dir = checkpoint_dir / "steps"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)
    return {
        "checkpoint_dir": checkpoint_dir,
        "state_path": checkpoint_dir / "checkpoint_state.json",
        "steps_dir": steps_dir,
        "bundle_path": run_dir / "bundle.json",
    }


def load_inference_config(path: Path) -> dict[str, Any]:
    """Load the benchmark/inference YAML config."""

    if not path.exists():
        raise FileNotFoundError(f"Inference config does not exist: {path.resolve()}")
    return load_yaml(path)


def benchmark_paths(run_dir: Path) -> BenchmarkPaths:
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = BenchmarkPaths(
        run_dir=run_dir,
        promptsets_dir=run_dir / "promptsets",
        raw_dir=run_dir / "raw",
        tables_dir=run_dir / "tables",
        plots_dir=run_dir / "plots",
        reports_dir=run_dir / "reports",
        correctness_dir=run_dir / "correctness",
        server_dir=run_dir / "server",
    )
    for path in asdict(paths).values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return paths


def _atomic_write_text(path: Path, contents: str) -> Path:
    """Write text atomically to avoid half-written checkpoint files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(contents, encoding="utf-8")
    tmp_path.replace(path)
    return path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    return _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")


def compute_benchmark_fingerprint(
    *,
    target: ResolvedServingTarget,
    config: dict[str, Any],
    dataset_path: Path,
) -> str:
    """Create a stable fingerprint for resume validation."""

    payload = {
        "target": asdict(target),
        "config": config,
        "dataset_path": str(dataset_path.resolve()),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def load_benchmark_checkpoint_state(state_path: Path) -> dict[str, Any] | None:
    """Load the persisted checkpoint state if it exists."""

    if not state_path.exists():
        return None
    return read_json(state_path)


def validate_benchmark_checkpoint_resume(
    state: dict[str, Any] | None,
    fingerprint: str,
) -> None:
    """Fail fast when a checkpointed run name is being reused for a different config."""

    if state is None:
        return
    existing_fingerprint = state.get("fingerprint")
    if existing_fingerprint not in (None, fingerprint):
        raise ValueError(
            "Benchmark checkpoint fingerprint mismatch for the same run name. "
            "Use a new --run-name if the target or config changed."
        )


def save_benchmark_checkpoint_state(state_path: Path, payload: dict[str, Any]) -> Path:
    """Persist checkpoint state atomically."""

    return _atomic_write_json(state_path, payload)


def _safe_step_filename(step_id: str) -> str:
    safe_step_id = step_id.replace("/", "_").replace(":", "_")
    return f"{safe_step_id}.json"


def save_benchmark_step_checkpoint(
    steps_dir: Path,
    *,
    step_id: str,
    payload: dict[str, Any],
) -> Path:
    """Persist one completed benchmark step atomically."""

    step_path = steps_dir / _safe_step_filename(step_id)
    return _atomic_write_json(step_path, payload)


def load_benchmark_step_checkpoints(steps_dir: Path) -> list[dict[str, Any]]:
    """Load all valid checkpointed benchmark steps in filename order."""

    if not steps_dir.exists():
        return []
    step_payloads: list[dict[str, Any]] = []
    for step_path in sorted(steps_dir.glob("*.json")):
        try:
            payload = read_json(step_path)
        except Exception:
            continue
        if isinstance(payload, dict):
            payload.setdefault("checkpoint_path", str(step_path))
            step_payloads.append(payload)
    return step_payloads


def load_checkpointed_benchmark_bundle(run_dir: Path) -> dict[str, Any] | None:
    """Load a completed bundle or reconstruct a partial bundle from checkpoints."""

    bundle_path = run_dir / "bundle.json"
    if bundle_path.exists():
        return read_json(bundle_path)

    checkpoint_paths = benchmark_checkpoint_paths(run_dir)
    state = load_benchmark_checkpoint_state(checkpoint_paths["state_path"])
    step_payloads = load_benchmark_step_checkpoints(checkpoint_paths["steps_dir"])
    if not state and not step_payloads:
        return None

    summary_rows: list[dict[str, Any]] = []
    correctness_rows: list[dict[str, Any]] = []
    config_search_rows: list[dict[str, Any]] = []
    raw_request_rows: list[dict[str, Any]] = []
    for payload in step_payloads:
        summary_row = payload.get("summary_row")
        if isinstance(summary_row, dict):
            summary_rows.append(summary_row)
            if payload.get("experiment_family") == "config_search":
                config_search_rows.append(summary_row)
        correctness_row = payload.get("correctness_row")
        if isinstance(correctness_row, dict):
            correctness_rows.append(correctness_row)
        raw_rows = payload.get("raw_rows")
        if isinstance(raw_rows, list):
            raw_request_rows.extend([row for row in raw_rows if isinstance(row, dict)])

    return {
        "generated_at_utc": state.get("updated_at_utc") if isinstance(state, dict) else _now_utc(),
        "run_name": state.get("run_name") if isinstance(state, dict) else run_dir.name,
        "config_path": state.get("config_path") if isinstance(state, dict) else None,
        "run_dir": str(run_dir.resolve()),
        "target": state.get("target") if isinstance(state, dict) else {},
        "promptset_manifest": state.get("promptset_manifest") if isinstance(state, dict) else {},
        "summary_rows": summary_rows,
        "correctness_rows": correctness_rows,
        "config_search_rows": config_search_rows,
        "raw_request_rows": raw_request_rows,
        "partial": True,
        "checkpoint_state": state,
        "checkpoint_steps": step_payloads,
    }


def resolve_serving_target(
    *,
    config_path: Path,
    repo_root: Path,
    target_kind: str | None = None,
    base_model: str | None = None,
    adapter_path: str | None = None,
    merged_model_path: str | None = None,
    latest_model_manifest_path: Path | None = None,
    lora_alias: str | None = None,
) -> tuple[ResolvedServingTarget, dict[str, Any]]:
    """Resolve the promoted serving target from overrides, latest-model, or config."""

    config = load_inference_config(config_path)
    resolution_config = dict(config.get("model_resolution", {}) or {})
    fallback = dict(resolution_config.get("fallback", {}) or {})

    latest_manifest_path = (
        _resolve_path(repo_root, latest_model_manifest_path)
        or _resolve_path(repo_root, resolution_config.get("latest_model_manifest"))
        or (repo_root / "artifacts" / "checkpoints" / "latest_model.json")
    )
    latest_manifest_payload = _load_optional_manifest(latest_manifest_path)

    manifest_stage = None
    manifest_base_model = None
    manifest_adapter_path = None
    manifest_merged_path = None
    manifest_schema_version = None
    if isinstance(latest_manifest_payload, LatestModelManifest):
        manifest_stage = latest_manifest_payload.stage
        manifest_base_model = latest_manifest_payload.base_model
        manifest_adapter_path = latest_manifest_payload.adapter_path
        manifest_merged_path = latest_manifest_payload.merged_export_path
        manifest_schema_version = latest_manifest_payload.schema_version
    elif isinstance(latest_manifest_payload, dict):
        manifest_stage = latest_manifest_payload.get("stage")
        manifest_base_model = latest_manifest_payload.get("base_model")
        manifest_adapter_path = latest_manifest_payload.get("adapter_path")
        manifest_merged_path = latest_manifest_payload.get("merged_export_path")
        manifest_schema_version = latest_manifest_payload.get("schema_version")

    resolved_base_model = (
        base_model
        or manifest_base_model
        or fallback.get("base_model")
        or "Qwen/Qwen2.5-1.5B-Instruct"
    )
    resolved_adapter_path = normalize_drive_path(
        adapter_path
        or manifest_adapter_path
        or fallback.get("adapter_path")
    )
    resolved_merged_model_path = normalize_drive_path(
        merged_model_path
        or manifest_merged_path
        or fallback.get("merged_model_path")
    )
    resolved_target_kind = (
        target_kind
        or resolution_config.get("preferred_target")
        or fallback.get("deployment_mode")
    )
    if not resolved_target_kind:
        if resolved_adapter_path:
            resolved_target_kind = "base_plus_lora"
        elif resolved_merged_model_path:
            resolved_target_kind = "merged_model"
        else:
            resolved_target_kind = "base_only"

    resolved_lora_alias = (
        lora_alias
        or resolution_config.get("lora_alias")
        or fallback.get("lora_alias")
        or DEFAULT_PROMOTED_LORA_ALIAS
    )

    if resolved_target_kind == "base_plus_lora":
        if not resolved_adapter_path:
            raise ValueError("base_plus_lora serving requires an adapter path.")
        return (
            ResolvedServingTarget(
                target_kind=resolved_target_kind,
                base_model=resolved_base_model,
                served_model_name_or_path=resolved_base_model,
                request_model_name=resolved_lora_alias,
                adapter_path=resolved_adapter_path,
                merged_model_path=None,
                lora_alias=resolved_lora_alias,
                source_manifest_path=normalize_drive_path(
                    manifest_stage and latest_manifest_path and str(latest_manifest_path)
                ),
                promotion_manifest_path=normalize_drive_path(
                    str(latest_manifest_path) if latest_manifest_path and latest_manifest_path.exists() else None
                ),
                schema_version=manifest_schema_version,
            ),
            config,
        )

    if resolved_target_kind == "merged_model":
        if not resolved_merged_model_path:
            raise ValueError("merged_model serving requires a merged model path.")
        served_model_name = resolved_merged_model_path
        return (
            ResolvedServingTarget(
                target_kind=resolved_target_kind,
                base_model=resolved_base_model,
                served_model_name_or_path=served_model_name,
                request_model_name=served_model_name,
                adapter_path=None,
                merged_model_path=resolved_merged_model_path,
                lora_alias=None,
                source_manifest_path=normalize_drive_path(
                    manifest_stage and latest_manifest_path and str(latest_manifest_path)
                ),
                promotion_manifest_path=normalize_drive_path(
                    str(latest_manifest_path) if latest_manifest_path and latest_manifest_path.exists() else None
                ),
                schema_version=manifest_schema_version,
            ),
            config,
        )

    return (
        ResolvedServingTarget(
            target_kind="base_only",
            base_model=resolved_base_model,
            served_model_name_or_path=resolved_base_model,
            request_model_name=resolved_base_model,
            adapter_path=None,
            merged_model_path=None,
            lora_alias=None,
            source_manifest_path=normalize_drive_path(
                manifest_stage and latest_manifest_path and str(latest_manifest_path)
            ),
            promotion_manifest_path=normalize_drive_path(
                str(latest_manifest_path) if latest_manifest_path and latest_manifest_path.exists() else None
            ),
            schema_version=manifest_schema_version,
        ),
        config,
    )


def build_vllm_serve_command(
    target: ResolvedServingTarget,
    serving_config: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Build the explicit vLLM serve command for one serving target."""

    host = str(serving_config.get("host", "127.0.0.1"))
    port = int(serving_config.get("port", 8000))
    api_key = serving_config.get("api_key")
    command = [
        "vllm",
        "serve",
        target.served_model_name_or_path,
        "--host",
        host,
        "--port",
        str(port),
        "--generation-config",
        str(serving_config.get("generation_config", "vllm")),
    ]

    optional_flags = {
        "dtype": serving_config.get("dtype"),
        "gpu-memory-utilization": serving_config.get("gpu_memory_utilization"),
        "max-model-len": serving_config.get("max_model_len"),
        "max-num-batched-tokens": serving_config.get("max_num_batched_tokens"),
        "max-num-seqs": serving_config.get("max_num_seqs"),
        "tensor-parallel-size": serving_config.get("tensor_parallel_size"),
        "seed": serving_config.get("seed"),
    }
    for flag, value in optional_flags.items():
        if value in (None, ""):
            continue
        command.extend([f"--{flag}", str(value)])

    if api_key:
        command.extend(["--api-key", str(api_key)])

    if target.target_kind == "base_plus_lora":
        command.append("--enable-lora")
        command.extend(["--lora-modules", f"{target.request_model_name}={target.adapter_path}"])
        if serving_config.get("max_loras") not in (None, ""):
            command.extend(["--max-loras", str(serving_config["max_loras"])])
        if serving_config.get("max_lora_rank") not in (None, ""):
            command.extend(["--max-lora-rank", str(serving_config["max_lora_rank"])])

    metadata = {
        "host": host,
        "port": port,
        "api_base": serving_config.get("api_base") or f"http://{host}:{port}",
        "api_key": api_key,
        "serving_config": dict(serving_config),
        "target": asdict(target),
        "command": command,
    }
    return command, metadata


def _gpu_metadata() -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        return {
            "torch_available": False,
            "cuda_available": False,
        }

    payload: dict[str, Any] = {
        "torch_available": True,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        try:
            payload["gpu_count"] = int(torch.cuda.device_count())
            payload["gpu_name"] = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            payload["gpu_mem_free_bytes"] = int(free_bytes)
            payload["gpu_mem_total_bytes"] = int(total_bytes)
        except Exception:  # pragma: no cover - hardware-backed
            payload["gpu_name"] = "<unavailable>"
    return payload


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, Any]]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method=method)
    with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - local server
        status = int(getattr(response, "status", 200))
        raw = response.read().decode("utf-8")
    return status, json.loads(raw) if raw else {}


def check_vllm_health(
    api_base: str,
    *,
    expected_model_name: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Check health, model inventory, and metrics availability."""

    base = api_base.rstrip("/")
    result: dict[str, Any] = {
        "api_base": base,
        "checked_at_utc": _now_utc(),
        "expected_model_name": expected_model_name,
        "health_ok": False,
        "models_ok": False,
        "metrics_ok": False,
        "served_models": [],
        "errors": [],
    }

    try:
        req = request.Request(f"{base}/health", method="GET")
        with request.urlopen(req, timeout=timeout_seconds) as response:  # noqa: S310 - local server
            result["health_status"] = int(getattr(response, "status", 200))
            result["health_ok"] = result["health_status"] == 200
    except Exception as exc:  # pragma: no cover - network-backed
        result["errors"].append(f"health: {exc}")

    try:
        _, models_payload = _request_json(f"{base}/v1/models", timeout=timeout_seconds)
        names: list[str] = []
        for item in models_payload.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                names.append(str(item["id"]))
        result["served_models"] = names
        if expected_model_name:
            result["models_ok"] = expected_model_name in names
        else:
            result["models_ok"] = bool(names)
    except Exception as exc:  # pragma: no cover - network-backed
        result["errors"].append(f"models: {exc}")

    try:
        req = request.Request(f"{base}/metrics", method="GET")
        with request.urlopen(req, timeout=timeout_seconds) as response:  # noqa: S310 - local server
            result["metrics_status"] = int(getattr(response, "status", 200))
            metrics_text = response.read().decode("utf-8")
            result["metrics_ok"] = result["metrics_status"] == 200 and (
                "vllm:" in metrics_text.lower() or "vllm_" in metrics_text.lower()
            )
    except Exception as exc:  # pragma: no cover - network-backed
        result["errors"].append(f"metrics: {exc}")

    result["ok"] = bool(result["health_ok"] and result["models_ok"])
    return result


def wait_for_vllm_ready(
    api_base: str,
    *,
    expected_model_name: str | None = None,
    timeout_seconds: float = 180.0,
    poll_interval_seconds: float = 2.0,
) -> dict[str, Any]:
    """Poll health endpoints until the server is ready."""

    deadline = time.time() + timeout_seconds
    last_result: dict[str, Any] | None = None
    while time.time() < deadline:
        last_result = check_vllm_health(
            api_base,
            expected_model_name=expected_model_name,
            timeout_seconds=max(5.0, poll_interval_seconds),
        )
        if last_result.get("ok"):
            return last_result
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"Timed out waiting for vLLM server readiness: {last_result}")


def _terminate_process(process: subprocess.Popen[Any], timeout_seconds: float = 20.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
        return
    except subprocess.TimeoutExpired:  # pragma: no cover - process-backed
        pass
    process.kill()
    process.wait(timeout=timeout_seconds)


def launch_vllm_server(
    *,
    target: ResolvedServingTarget,
    serving_config: dict[str, Any],
    server_dir: Path,
    config_id: str,
    runtime_root: Path,
    config_path: Path,
) -> tuple[subprocess.Popen[Any], ServerLaunchResult]:
    """Launch a vLLM process and persist startup metadata."""

    server_dir.mkdir(parents=True, exist_ok=True)
    config_server_dir = server_dir / config_id
    config_server_dir.mkdir(parents=True, exist_ok=True)
    stdout_log_path = config_server_dir / "stdout.log"
    stderr_log_path = config_server_dir / "stderr.log"
    pid_path = config_server_dir / "server.pid"
    metadata_path = config_server_dir / "startup_metadata.json"

    command, launch_metadata = build_vllm_serve_command(target, serving_config)
    stdout_handle = stdout_log_path.open("w", encoding="utf-8")
    stderr_handle = stderr_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=stdout_handle,
        stderr=stderr_handle,
        cwd=runtime_root,
        start_new_session=True,
        text=True,
    )
    pid_path.write_text(str(process.pid), encoding="utf-8")
    startup_metadata = {
        "config_id": config_id,
        "command": command,
        "target": asdict(target),
        "config_path": str(config_path.resolve()),
        "runtime_root": str(runtime_root.resolve()),
        "started_at_utc": _now_utc(),
        "stdout_log_path": str(stdout_log_path.resolve()),
        "stderr_log_path": str(stderr_log_path.resolve()),
        "pid": process.pid,
        "gpu": _gpu_metadata(),
        **launch_metadata,
    }
    write_json(metadata_path, startup_metadata)
    result = ServerLaunchResult(
        config_id=config_id,
        command=command,
        api_base=str(launch_metadata["api_base"]),
        host=str(launch_metadata["host"]),
        port=int(launch_metadata["port"]),
        pid=process.pid,
        stdout_log_path=str(stdout_log_path.resolve()),
        stderr_log_path=str(stderr_log_path.resolve()),
        pid_path=str(pid_path.resolve()),
        metadata_path=str(metadata_path.resolve()),
    )
    return process, result


def stop_vllm_server(process: subprocess.Popen[Any] | None, pid_path: Path | None = None) -> None:
    """Terminate a vLLM process from a live handle or pid file."""

    if process is not None:
        _terminate_process(process)
        return
    if pid_path is None or not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _maybe_sample(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return list(rows)
    ordered = sorted(rows, key=lambda row: _stable_sort_key(row, seed))
    return ordered[:limit]


def _sentence_fragments(text: str) -> list[str]:
    cleaned = " ".join(text.replace("\n", " ").split())
    fragments = [fragment.strip() for fragment in cleaned.split(".") if fragment.strip()]
    return fragments or [cleaned]


def _benchmark_support_notes(row: dict[str, Any]) -> list[str]:
    input_text = str(row.get("input_text", "")).lower()
    notes = [
        "The issue persisted across repeated attempts in the same support session.",
        "A reproducible failure pattern was reported after basic retry steps.",
    ]
    if "billing" in input_text or "invoice" in input_text or "payment" in input_text:
        notes.extend(
            [
                "The customer confirmed the request is blocking normal billing review.",
                "The support queue flagged the case as needing clear next-step guidance.",
            ]
        )
    elif "api" in input_text or "integration" in input_text or "connector" in input_text:
        notes.extend(
            [
                "The customer reported that the integration flow still failed after credential checks.",
                "A follow-up note mentioned that the same issue appears during connector maintenance.",
            ]
        )
    elif "login" in input_text or "sign" in input_text or "account" in input_text:
        notes.extend(
            [
                "The customer stated that the account issue still occurs after a fresh login attempt.",
                "A follow-up message requested confirmation on whether additional account review is needed.",
            ]
        )
    else:
        notes.extend(
            [
                "The customer asked for a clear support recommendation instead of a generic status update.",
                "The support conversation suggested that self-serve troubleshooting did not resolve the problem.",
            ]
        )
    return notes


def _build_stress_input_text(row: dict[str, Any], variant_name: str) -> str:
    base_text = str(row.get("input_text", "")).strip()
    fragments = _sentence_fragments(base_text)
    repeated_context = "\n".join(f"- {note}" for note in _benchmark_support_notes(row))
    quote_head = fragments[0]
    quote_tail = fragments[-1]

    if variant_name == "ticket_repeat_context":
        return (
            f"{base_text}\n\n"
            "Additional customer follow-up context:\n"
            f"- Original issue summary: {quote_head}.\n"
            "- The same support request was retried after standard troubleshooting.\n"
            f"- Latest update from the user: {quote_tail}.\n"
            f"{repeated_context}\n"
        )
    if variant_name == "multi_turn_ticket_digest":
        return (
            f"{base_text}\n\n"
            "Thread digest:\n"
            "Customer message 1: The original issue is still active and blocks expected work.\n"
            f"Customer message 2: {quote_head}.\n"
            "Support note: Initial self-serve remediation did not resolve the issue.\n"
            f"Customer message 3: {quote_tail}.\n"
            f"{repeated_context}\n"
        )
    return (
        f"{base_text}\n\n"
        "Sanitized incident notes:\n"
        "- Timestamped retries showed the same visible symptom.\n"
        "- The customer requested a precise next-step recommendation.\n"
        "- No durable workaround was confirmed during the same support exchange.\n"
        f"- Ticket excerpt A: {quote_head}.\n"
        f"- Ticket excerpt B: {quote_tail}.\n"
        f"{repeated_context}\n"
    )


def _variant_name_for_row(row: dict[str, Any], seed: int) -> str:
    variants = ("ticket_repeat_context", "multi_turn_ticket_digest", "log_attachment_stub")
    index = _stable_seed(str(row.get("record_id", "")), seed) % len(variants)
    return variants[index]


def _stress_variant_row(row: dict[str, Any], seed: int) -> dict[str, Any]:
    variant_name = _variant_name_for_row(row, seed)
    stress_input_text = _build_stress_input_text(row, variant_name)
    messages = [
        {"role": "system", "content": render_system_instruction()},
        {"role": "user", "content": render_user_prompt(stress_input_text)},
    ]
    return {
        **row,
        "lineage_record_id": row.get("record_id"),
        "record_id": f"{row.get('record_id')}-stress-{variant_name}",
        "input_text": stress_input_text,
        "messages": messages,
        "prompt": None,
        "benchmark_variant": variant_name,
        "length_family": "stress",
        "benchmark_only": True,
    }


def _quantile_cutoffs(values: list[int]) -> tuple[int, int]:
    ordered = sorted(values)
    if not ordered:
        return 0, 0
    low_index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.33))
    high_index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.66))
    return ordered[low_index], ordered[high_index]


def _bucket_label(length: int, low_cutoff: int, high_cutoff: int) -> str:
    if length <= low_cutoff:
        return "short"
    if length <= high_cutoff:
        return "medium"
    return "long"


def build_benchmark_promptsets(
    *,
    dataset_path: Path,
    target: ResolvedServingTarget,
    promptset_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Build deterministic natural and stress promptsets from held-out rows."""

    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(promptset_config.get("seed", 17))
    natural_sample_limit = promptset_config.get("natural_sample_limit")
    natural_rows = _maybe_sample(read_jsonl(dataset_path), natural_sample_limit, seed)
    tokenizer_name = str(
        promptset_config.get("tokenizer_name_or_path")
        or target.base_model
        or target.served_model_name_or_path
    )
    tokenizer = _load_tokenizer(tokenizer_name)

    base_rows: list[dict[str, Any]] = []
    natural_lengths: list[int] = []
    for row in natural_rows:
        messages = _render_request_messages(row)
        token_length = prompt_token_length(tokenizer, messages)
        natural_lengths.append(token_length)
        base_rows.append(
            {
                "record_id": row.get("record_id"),
                "lineage_record_id": row.get("record_id"),
                "source_dataset": row.get("source_dataset"),
                "source_split": row.get("split"),
                "messages": messages,
                "reference_payload": row.get("reference_payload"),
                "metadata": row.get("metadata", {}),
                "input_text": row.get("input_text"),
                "prompt_token_length": token_length,
                "benchmark_variant": "natural",
                "length_family": "natural",
                "benchmark_only": False,
            }
        )

    low_cutoff, high_cutoff = _quantile_cutoffs(natural_lengths)
    natural_prompt_rows: list[dict[str, Any]] = []
    for row in base_rows:
        natural_prompt_rows.append(
            {
                **row,
                "bucket_label": _bucket_label(
                    int(row["prompt_token_length"]),
                    low_cutoff,
                    high_cutoff,
                ),
            }
        )

    stress_sample_limit = int(promptset_config.get("stress_sample_limit", 0) or 0)
    stress_source_rows = [
        row
        for row in natural_prompt_rows
        if row["bucket_label"] in {"short", "medium"}
    ]
    stress_source_rows = _maybe_sample(stress_source_rows, stress_sample_limit or len(stress_source_rows), seed)
    stress_rows: list[dict[str, Any]] = []
    for row in stress_source_rows:
        variant_row = _stress_variant_row(row, seed)
        variant_row["messages"] = _render_request_messages(variant_row)
        variant_row["prompt_token_length"] = prompt_token_length(tokenizer, variant_row["messages"])
        variant_row["bucket_label"] = _bucket_label(
            int(variant_row["prompt_token_length"]),
            low_cutoff,
            high_cutoff,
        )
        stress_rows.append(variant_row)

    natural_path = write_jsonl(output_dir / "natural_prompt_rows.jsonl", natural_prompt_rows)
    stress_path = write_jsonl(output_dir / "stress_prompt_rows.jsonl", stress_rows)

    manifest = {
        "version": PROMPTSET_VERSION,
        "stress_workload_version": STRESS_WORKLOAD_VERSION,
        "created_at_utc": _now_utc(),
        "dataset_path": str(dataset_path.resolve()),
        "tokenizer_name_or_path": tokenizer_name,
        "seed": seed,
        "natural_sample_limit": natural_sample_limit,
        "stress_sample_limit": stress_sample_limit or len(stress_source_rows),
        "bucket_cutoffs": {
            "short_max_tokens": low_cutoff,
            "medium_max_tokens": high_cutoff,
        },
        "counts": {
            "natural": len(natural_prompt_rows),
            "stress": len(stress_rows),
        },
        "natural_prompt_rows_path": str(natural_path),
        "stress_prompt_rows_path": str(stress_path),
        "target": asdict(target) if hasattr(target, "__dataclass_fields__") else dict(target.__dict__),
    }
    manifest_path = write_json(output_dir / "promptset_manifest.json", manifest)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "natural_rows": natural_prompt_rows,
        "stress_rows": stress_rows,
    }


def _select_rows(rows: list[dict[str, Any]], *, seed: int, count: int) -> list[dict[str, Any]]:
    if count >= len(rows):
        return sorted(rows, key=lambda row: _stable_sort_key(row, seed))
    ordered = sorted(rows, key=lambda row: _stable_sort_key(row, seed))
    return ordered[:count]


def _mix_counts(total_count: int, short_ratio: float, long_ratio: float) -> tuple[int, int]:
    short_count = int(round(total_count * short_ratio))
    long_count = int(round(total_count * long_ratio))
    if short_count + long_count > total_count:
        long_count = max(0, total_count - short_count)
    return short_count, long_count


def build_workload_mix_rows(
    *,
    natural_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    mix_name: str,
    total_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build deterministic workload mixes from natural and stress promptsets."""

    natural_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in natural_rows:
        natural_by_bucket[str(row.get("bucket_label", ""))].append(row)
    stress_long_rows = [row for row in stress_rows if row.get("bucket_label") == "long"]

    if mix_name == "smoke_stratified":
        selected: list[dict[str, Any]] = []
        per_bucket = max(1, total_count // 3)
        for bucket in ("short", "medium", "long"):
            selected.extend(_select_rows(natural_by_bucket[bucket], seed=seed + len(selected), count=per_bucket))
        return selected[:total_count]

    if mix_name in {"natural_short_only", "natural_medium_only", "natural_long_only"}:
        bucket = mix_name.split("_")[1]
        return _select_rows(natural_by_bucket[bucket], seed=seed, count=total_count)

    if mix_name.startswith("natural_mix_"):
        parts = mix_name.replace("natural_mix_", "").replace("_short_long", "")
        short_ratio, long_ratio = parts.split("_")
        short_count, long_count = _mix_counts(total_count, int(short_ratio) / 100.0, int(long_ratio) / 100.0)
        selected = []
        selected.extend(_select_rows(natural_by_bucket["short"], seed=seed, count=short_count))
        selected.extend(_select_rows(natural_by_bucket["long"], seed=seed + 1, count=long_count))
        if len(selected) < total_count:
            selected.extend(
                _select_rows(natural_by_bucket["medium"], seed=seed + 2, count=total_count - len(selected))
            )
        return selected[:total_count]

    if mix_name == "stress_long_only":
        return _select_rows(stress_long_rows, seed=seed, count=total_count)

    if mix_name.startswith("stress_mix_"):
        parts = mix_name.replace("stress_mix_", "").replace("_natural_short_stress_long", "")
        short_ratio, long_ratio = parts.split("_")
        short_count, long_count = _mix_counts(total_count, int(short_ratio) / 100.0, int(long_ratio) / 100.0)
        selected = []
        selected.extend(_select_rows(natural_by_bucket["short"], seed=seed, count=short_count))
        selected.extend(_select_rows(stress_long_rows, seed=seed + 1, count=long_count))
        return selected[:total_count]

    raise ValueError(f"Unsupported workload mix: {mix_name}")


def write_workload_mix_artifacts(
    *,
    mix_names: Iterable[str],
    total_count: int,
    seed: int,
    natural_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for mix_name in mix_names:
        rows = build_workload_mix_rows(
            natural_rows=natural_rows,
            stress_rows=stress_rows,
            mix_name=mix_name,
            total_count=total_count,
            seed=seed,
        )
        path = write_jsonl(output_dir / f"{mix_name}.jsonl", rows)
        paths[mix_name] = str(path)
    return paths


def compute_prompt_budget(
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
    budgeting_config: dict[str, Any],
) -> tuple[list[dict[str, str]], PromptBudget]:
    """Apply extraction-aware context budgeting to a request."""

    desired_max_tokens = int(budgeting_config.get("desired_max_tokens", 256))
    minimum_output_tokens = int(budgeting_config.get("minimum_output_tokens", 96))
    safety_margin_tokens = int(budgeting_config.get("safety_margin_tokens", 32))
    max_model_len = int(budgeting_config.get("max_model_len", 2048))
    trim_head_fraction = float(budgeting_config.get("trim_head_fraction", 0.55))

    original_prompt_tokens = prompt_token_length(tokenizer, messages)
    available_tokens = max_model_len - safety_margin_tokens - original_prompt_tokens
    if available_tokens >= minimum_output_tokens:
        final_output_tokens = max(1, min(desired_max_tokens, available_tokens))
        return (
            messages,
            PromptBudget(
                original_prompt_tokens=original_prompt_tokens,
                final_prompt_tokens=original_prompt_tokens,
                desired_output_tokens=desired_max_tokens,
                final_output_tokens=final_output_tokens,
                trim_applied=False,
                trimmed_input_tokens=0,
                trim_reason=None,
                trimmed_user_content=str(messages[-1]["content"]),
            ),
        )

    trimmed_messages = [dict(message) for message in messages]
    if not trimmed_messages or trimmed_messages[-1]["role"] != "user":
        raise ValueError("Context budgeting requires the final request message to be the user ticket.")

    original_user_content = str(trimmed_messages[-1]["content"])
    user_tokens = _tokenize_text(tokenizer, original_user_content)
    fixed_prefix_tokens = prompt_token_length(
        tokenizer,
        [*trimmed_messages[:-1], {"role": "user", "content": ""}],
    )
    max_user_tokens = max(0, max_model_len - safety_margin_tokens - fixed_prefix_tokens - minimum_output_tokens)
    if len(user_tokens) <= max_user_tokens:
        final_output_tokens = max(1, min(desired_max_tokens, minimum_output_tokens))
        return (
            trimmed_messages,
            PromptBudget(
                original_prompt_tokens=original_prompt_tokens,
                final_prompt_tokens=original_prompt_tokens,
                desired_output_tokens=desired_max_tokens,
                final_output_tokens=final_output_tokens,
                trim_applied=False,
                trimmed_input_tokens=0,
                trim_reason="output_shrunk_to_minimum",
                trimmed_user_content=original_user_content,
            ),
        )

    head_count = int(max_user_tokens * trim_head_fraction)
    tail_count = max(0, max_user_tokens - head_count)
    trimmed_tokens = user_tokens[:head_count]
    if tail_count > 0:
        trimmed_tokens.extend(user_tokens[-tail_count:])
    trimmed_user_content = _decode_tokens(tokenizer, trimmed_tokens)
    trimmed_messages[-1]["content"] = trimmed_user_content
    final_prompt_tokens = prompt_token_length(tokenizer, trimmed_messages)
    final_available_tokens = max(1, max_model_len - safety_margin_tokens - final_prompt_tokens)
    final_output_tokens = max(1, min(desired_max_tokens, final_available_tokens))
    return (
        trimmed_messages,
        PromptBudget(
            original_prompt_tokens=original_prompt_tokens,
            final_prompt_tokens=final_prompt_tokens,
            desired_output_tokens=desired_max_tokens,
            final_output_tokens=final_output_tokens,
            trim_applied=True,
            trimmed_input_tokens=max(0, len(user_tokens) - len(trimmed_tokens)),
            trim_reason="trimmed_user_ticket_text",
            trimmed_user_content=trimmed_user_content,
        ),
    )


def _estimate_text_token_count(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:  # pragma: no cover - tokenizer-backed
        return max(1, len(text.split()))


def execute_chat_completion_request(
    *,
    api_base: str,
    model_name: str,
    messages: list[dict[str, str]],
    output_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    request_seed: int | None,
    tokenizer: Any,
    measure_ttft: bool = True,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Send one chat-completions request to the local vLLM server."""

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": output_tokens,
        "temperature": temperature if do_sample else 0.0,
        "top_p": top_p if do_sample else 1.0,
        "stream": bool(measure_ttft),
    }
    if request_seed is not None:
        payload["seed"] = int(request_seed)

    url = f"{api_base.rstrip('/')}{DEFAULT_CHAT_COMPLETIONS_PATH}"
    encoded = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:  # noqa: S310 - local server
            status = int(getattr(response, "status", 200))
            if not measure_ttft:
                raw = response.read().decode("utf-8")
                latency_ms = (time.perf_counter() - started) * 1000.0
                payload = json.loads(raw)
                text = str(payload["choices"][0]["message"]["content"])
                usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
                completion_tokens = usage.get("completion_tokens")
                if completion_tokens is None:
                    completion_tokens = _estimate_text_token_count(tokenizer, text)
                return {
                    "success": True,
                    "http_status": status,
                    "latency_ms": latency_ms,
                    "ttft_ms": None,
                    "text": text,
                    "completion_tokens": int(completion_tokens),
                    "response_payload": payload,
                    "error_type": None,
                    "error_message": None,
                }

            text_parts: list[str] = []
            first_token_time: float | None = None
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta_text = (
                    chunk.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if delta_text and first_token_time is None:
                    first_token_time = time.perf_counter()
                if delta_text:
                    text_parts.append(str(delta_text))
            finished = time.perf_counter()
        text = "".join(text_parts)
        completion_tokens = _estimate_text_token_count(tokenizer, text)
        return {
            "success": True,
            "http_status": status,
            "latency_ms": (finished - started) * 1000.0,
            "ttft_ms": ((first_token_time or finished) - started) * 1000.0 if first_token_time else None,
            "text": text,
            "completion_tokens": int(completion_tokens),
            "response_payload": None,
            "error_type": None,
            "error_message": None,
        }
    except error.HTTPError as exc:  # pragma: no cover - network-backed
        error_body = exc.read().decode("utf-8", errors="replace")
        return {
            "success": False,
            "http_status": int(exc.code),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "ttft_ms": None,
            "text": "",
            "completion_tokens": 0,
            "response_payload": None,
            "error_type": "http_error",
            "error_message": error_body,
        }
    except Exception as exc:  # pragma: no cover - network-backed
        return {
            "success": False,
            "http_status": None,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "ttft_ms": None,
            "text": "",
            "completion_tokens": 0,
            "response_payload": None,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }


def _evaluate_categorical_matches(
    parsed_payload: dict[str, Any] | None,
    reference_payload: dict[str, Any] | None,
) -> dict[str, bool | None]:
    def nested_value(payload: dict[str, Any] | None, field_name: str) -> Any:
        current = payload
        for part in field_name.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    return {
        field_name: (
            None
            if parsed_payload is None or reference_payload is None
            else nested_value(parsed_payload, field_name) == nested_value(reference_payload, field_name)
        )
        for field_name in CATEGORICAL_EXACT_MATCH_FIELDS
    }


def run_benchmark_workload(
    *,
    workload_rows: list[dict[str, Any]],
    target: ResolvedServingTarget,
    api_base: str,
    tokenizer: Any,
    generation_config: dict[str, Any],
    budgeting_config: dict[str, Any],
    concurrency: int,
    experiment_id: str,
    server_config_id: str,
    workload_name: str,
    request_seed: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute one concurrent workload against the local vLLM server."""

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    schema = build_support_ticket_schema()
    total_requests = len(workload_rows)
    progress_step = max(1, total_requests // 10)
    completed = 0

    def execute_one(index_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, row = index_row
        original_messages = [dict(message) for message in row["messages"]]
        final_messages, budget = compute_prompt_budget(
            tokenizer=tokenizer,
            messages=original_messages,
            budgeting_config=budgeting_config,
        )
        request_result = execute_chat_completion_request(
            api_base=api_base,
            model_name=target.request_model_name,
            messages=final_messages,
            output_tokens=budget.final_output_tokens,
            temperature=float(generation_config.get("temperature", 0.0)),
            top_p=float(generation_config.get("top_p", 1.0)),
            do_sample=bool(generation_config.get("do_sample", False)),
            request_seed=request_seed + index if request_seed is not None else None,
            tokenizer=tokenizer,
            measure_ttft=bool(generation_config.get("measure_ttft", True)),
            timeout_seconds=float(generation_config.get("request_timeout_seconds", 180.0)),
        )
        parsed_payload, parse_error, validation, recovery_used = analyze_inference_text(
            str(request_result.get("text", "")),
            schema=schema,
        )
        categorical_matches = _evaluate_categorical_matches(parsed_payload, row.get("reference_payload"))
        latency_ms = float(request_result["latency_ms"])
        completion_tokens = int(request_result.get("completion_tokens") or 0)
        return {
            "request_id": f"{experiment_id}-{index:05d}",
            "experiment_id": experiment_id,
            "server_config_id": server_config_id,
            "workload_name": workload_name,
            "concurrency": concurrency,
            "record_id": row.get("record_id"),
            "lineage_record_id": row.get("lineage_record_id"),
            "source_dataset": row.get("source_dataset"),
            "source_split": row.get("source_split"),
            "benchmark_variant": row.get("benchmark_variant"),
            "length_family": row.get("length_family"),
            "bucket_label": row.get("bucket_label"),
            "success": bool(request_result.get("success")),
            "http_status": request_result.get("http_status"),
            "error_type": request_result.get("error_type"),
            "error_message": request_result.get("error_message"),
            "latency_ms": latency_ms,
            "ttft_ms": request_result.get("ttft_ms"),
            "prompt_tokens_est": budget.final_prompt_tokens,
            "prompt_tokens_original_est": budget.original_prompt_tokens,
            "completion_tokens_est": completion_tokens,
            "tokens_per_second_est": (completion_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else None,
            "trim_applied": budget.trim_applied,
            "trimmed_input_tokens": budget.trimmed_input_tokens,
            "trim_reason": budget.trim_reason,
            "desired_output_tokens": budget.desired_output_tokens,
            "final_output_tokens": budget.final_output_tokens,
            "json_recovery_used": recovery_used,
            "parse_error": parse_error,
            "schema_is_valid": validation.is_valid if validation is not None else False,
            "missing_fields": list(validation.missing_fields) if validation is not None else [],
            "unexpected_fields": list(validation.unexpected_fields) if validation is not None else [],
            "raw_output": request_result.get("text"),
            "parsed_payload": parsed_payload,
            "reference_payload": row.get("reference_payload"),
            "categorical_matches": categorical_matches,
            "model_name": target.request_model_name,
        }

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(execute_one, item) for item in enumerate(workload_rows)]
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed == 1 or completed == total_requests or completed % progress_step == 0:
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[benchmark] {experiment_id} "
                    f"{completed}/{total_requests} complete "
                    f"({rate:.2f} req/s elapsed={elapsed:.1f}s)",
                    flush=True,
                )
    finished = time.perf_counter()

    ordered_results = sorted(results, key=lambda row: row["request_id"])
    summary = summarize_benchmark_results(
        ordered_results,
        experiment_id=experiment_id,
        workload_name=workload_name,
        concurrency=concurrency,
        elapsed_seconds=finished - started,
        server_config_id=server_config_id,
    )
    return ordered_results, summary


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil((percentile / 100.0) * len(ordered)) - 1)))
    return float(ordered[index])


def summarize_benchmark_results(
    results: list[dict[str, Any]],
    *,
    experiment_id: str,
    workload_name: str,
    concurrency: int,
    elapsed_seconds: float,
    server_config_id: str,
) -> dict[str, Any]:
    """Build overall and bucketed percentile summaries."""

    latencies = [float(row["latency_ms"]) for row in results if row.get("success")]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in results:
        if row.get("success"):
            grouped[str(row.get("bucket_label") or "unknown")].append(float(row["latency_ms"]))
    throughput_rps = len(results) / elapsed_seconds if elapsed_seconds > 0 else 0.0
    success_count = sum(1 for row in results if row.get("success"))
    failure_count = len(results) - success_count
    summary = {
        "experiment_id": experiment_id,
        "workload_name": workload_name,
        "server_config_id": server_config_id,
        "concurrency": concurrency,
        "request_count": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": (success_count / len(results)) if results else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "throughput_rps": throughput_rps,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p90_ms": _percentile(latencies, 90),
        "latency_p99_ms": _percentile(latencies, 99),
        "tail_inflation_p99_over_p50": (
            (_percentile(latencies, 99) or 0.0) / (_percentile(latencies, 50) or 1.0)
            if latencies
            else None
        ),
        "bucket_latency_ms": {
            bucket: {
                "p50": _percentile(values, 50),
                "p90": _percentile(values, 90),
                "p99": _percentile(values, 99),
                "count": len(values),
            }
            for bucket, values in grouped.items()
        },
    }
    short_p99 = summary["bucket_latency_ms"].get("short", {}).get("p99")
    long_p99 = summary["bucket_latency_ms"].get("long", {}).get("p99")
    summary["short_to_long_p99_ratio"] = (
        (short_p99 / long_p99) if short_p99 and long_p99 else None
    )
    return summary


def build_correctness_summary(
    results: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
    experiment_id: str,
) -> dict[str, Any]:
    """Run a deterministic correctness spot-check over benchmark responses."""

    eligible = [row for row in results if row.get("success")]
    sample = _select_rows(eligible, seed=seed, count=min(sample_size, len(eligible)))
    if not sample:
        return {
            "experiment_id": experiment_id,
            "sample_size_requested": sample_size,
            "sample_size_actual": 0,
            "json_parse_pass_rate": 0.0,
            "schema_validation_pass_rate": 0.0,
            "categorical_exact_match": {field_name: None for field_name in CATEGORICAL_EXACT_MATCH_FIELDS},
            "warnings": ["No successful benchmark responses were available for correctness sampling."],
        }

    parse_pass = sum(1 for row in sample if row.get("parsed_payload") is not None)
    schema_pass = sum(1 for row in sample if row.get("schema_is_valid"))
    categorical_exact_match: dict[str, float | None] = {}
    warnings: list[str] = []
    for field_name in CATEGORICAL_EXACT_MATCH_FIELDS:
        values = [
            match
            for row in sample
            for key, match in (row.get("categorical_matches") or {}).items()
            if key == field_name and match is not None
        ]
        categorical_exact_match[field_name] = (sum(1 for item in values if item) / len(values)) if values else None

    if schema_pass < parse_pass:
        warnings.append("Some sampled outputs were parseable JSON but failed schema validation.")

    return {
        "experiment_id": experiment_id,
        "sample_size_requested": sample_size,
        "sample_size_actual": len(sample),
        "json_parse_pass_rate": parse_pass / len(sample),
        "schema_validation_pass_rate": schema_pass / len(sample),
        "categorical_exact_match": categorical_exact_match,
        "warnings": warnings,
    }


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write a list of dictionaries to CSV with the union of keys."""

    path.parent.mkdir(parents=True, exist_ok=True)
    field_names: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in field_names:
                field_names.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def snapshot_metrics_endpoint(api_base: str, output_path: Path) -> Path:
    """Persist the current `/metrics` endpoint response."""

    base = api_base.rstrip("/")
    req = request.Request(f"{base}/metrics", method="GET")
    with request.urlopen(req, timeout=20.0) as response:  # noqa: S310 - local server
        metrics_text = response.read().decode("utf-8")
    return write_text(output_path, metrics_text)
