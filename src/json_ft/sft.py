"""Reusable helpers for single-GPU QLoRA supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import shutil

from .artifacts import mirror_small_artifact
from .manifests import LatestModelManifest, save_latest_model_manifest
from .runtime import RuntimeContext, resolve_repo_artifact_targets
from .sampling import SampleSelectionMetadata, select_rows
from .stage_metadata import build_data_pipeline_metadata
from .token_cache import (
    build_token_cache_key,
    load_cached_token_payload,
    summarize_token_counts,
    write_cached_token_payload,
)
from .training_plots import PlotSpec, write_training_history_and_plots
from .utils import load_yaml, read_jsonl, write_json


PROFILE_ALIASES = {"colab_full": "full"}


def _compact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_repo_path(repo_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


@dataclass(frozen=True)
class SFTResolvedConfig:
    """Resolved config for a single SFT run profile."""

    config_path: Path
    profile_name: str
    model_name_or_path: str
    trust_remote_code: bool
    eos_token: str | None
    device_map: str | None
    torch_dtype: str | None
    train_manifest: Path
    eval_manifest: Path
    build_summary_path: Path | None
    composition_summary_path: Path | None
    dataset_format: str
    max_seq_length: int
    train_sample_limit: int | None
    eval_sample_limit: int | None
    train_sample_percent: float | None
    eval_sample_percent: float | None
    sample_seed: int
    token_cache: dict[str, Any]
    quantization: dict[str, Any]
    lora: dict[str, Any]
    training: dict[str, Any]
    artifacts: dict[str, Any]
    raw_config: dict[str, Any]


@dataclass(frozen=True)
class SFTOutputPaths:
    """Runtime file layout for one SFT run."""

    checkpoint_root: Path
    adapter_dir: Path
    logs_dir: Path
    trainer_state_path: Path
    summary_path: Path
    history_path: Path
    checkpoint_manifest_path: Path
    loss_curve_path: Path
    eval_loss_curve_path: Path
    learning_rate_curve_path: Path
    examples_seen_curve_path: Path
    tokens_seen_curve_path: Path


@dataclass(frozen=True)
class TrainerBundle:
    """Minimal objects returned by the trainer builder."""

    trainer: Any
    model: Any
    tokenizer: Any
    dataset_telemetry: dict[str, Any]


def load_sft_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML configuration for SFT."""

    resolved = Path(config_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"SFT config does not exist: {resolved}")
    return load_yaml(resolved)


def resolve_sft_config(
    *,
    config_path: str | Path,
    repo_root: str | Path,
    profile_name: str,
    training_overrides: dict[str, Any] | None = None,
    data_overrides: dict[str, Any] | None = None,
) -> SFTResolvedConfig:
    """Resolve config defaults plus one named profile into a concrete run config."""

    resolved_repo_root = Path(repo_root).resolve()
    resolved_config_path = _resolve_repo_path(resolved_repo_root, config_path)
    raw_config = load_sft_config(resolved_config_path)
    profiles = raw_config.get("profiles", {})
    requested_profile_name = profile_name
    profile_name = PROFILE_ALIASES.get(profile_name, profile_name)
    if profile_name not in profiles:
        raise ValueError(f"Unknown SFT profile: {profile_name}")

    profile_config = profiles[profile_name]
    merged = _deep_merge(raw_config, profile_config)
    model_config = dict(merged.get("model", {}))
    data_config = dict(merged.get("data", {}))
    training_config = dict(merged.get("training", {}))
    if data_overrides:
        data_config.update(_compact_mapping(data_overrides))
    if training_overrides:
        training_config.update(_compact_mapping(training_overrides))
    model_name_or_path = str(model_config.get("model_name_or_path", "")).strip()
    if not model_name_or_path:
        raise ValueError(f"SFT config is missing model.model_name_or_path: {resolved_config_path}")

    token_cache_config = dict(data_config.get("token_cache", {}))

    return SFTResolvedConfig(
        config_path=resolved_config_path,
        profile_name=profile_name,
        model_name_or_path=model_name_or_path,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        eos_token=model_config.get("eos_token"),
        device_map=model_config.get("device_map"),
        torch_dtype=model_config.get("torch_dtype", "auto"),
        train_manifest=_resolve_repo_path(
            resolved_repo_root,
            data_config.get("train_manifest", "data/manifests/support_tickets_sft_messages.jsonl"),
        ),
        eval_manifest=_resolve_repo_path(
            resolved_repo_root,
            data_config.get("eval_manifest", "data/manifests/support_tickets_eval_manifest.jsonl"),
        ),
        build_summary_path=_resolve_repo_path(
            resolved_repo_root,
            data_config.get("build_summary_path", "data/manifests/support_tickets_dataset_build_summary.json"),
        ),
        composition_summary_path=_resolve_repo_path(
            resolved_repo_root,
            data_config.get("composition_summary_path", "artifacts/metrics/support_tickets_dataset_composition.json"),
        ),
        dataset_format=str(data_config.get("dataset_format", "messages")),
        max_seq_length=int(data_config.get("max_seq_length", 1024)),
        train_sample_limit=data_config.get("train_sample_limit"),
        eval_sample_limit=data_config.get("eval_sample_limit"),
        train_sample_percent=data_config.get("train_sample_percent"),
        eval_sample_percent=data_config.get("eval_sample_percent"),
        sample_seed=int(data_config.get("sample_seed", 17)),
        token_cache=token_cache_config,
        quantization=dict(merged.get("quantization", {})),
        lora=dict(merged.get("lora", {})),
        training=training_config,
        artifacts=dict(merged.get("artifacts", {})),
        raw_config={
            **raw_config,
            "_requested_profile_name": requested_profile_name,
            "_resolved_profile_name": profile_name,
            "model": model_config,
            "data": data_config,
            "training": training_config,
        },
    )


def resolve_sft_output_paths(
    context: RuntimeContext,
    run_name: str,
    artifact_config: dict[str, Any] | None = None,
) -> SFTOutputPaths:
    """Build the runtime paths used by the SFT stage."""

    artifact_names = artifact_config or {}
    checkpoint_root = (context.checkpoints_dir / "sft" / run_name).resolve()
    adapter_dir = (checkpoint_root / "adapter").resolve()
    logs_dir = (context.logs_dir / "sft" / run_name).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return SFTOutputPaths(
        checkpoint_root=checkpoint_root,
        adapter_dir=adapter_dir,
        logs_dir=logs_dir,
        trainer_state_path=(logs_dir / "trainer_state.json").resolve(),
        summary_path=(
            context.metrics_dir
            / artifact_names.get("summary_filename", "{run_name}_sft_summary.json").format(run_name=run_name)
        ).resolve(),
        history_path=(
            context.metrics_dir
            / artifact_names.get("history_filename", "{run_name}_sft_history.json").format(run_name=run_name)
        ).resolve(),
        checkpoint_manifest_path=(
            checkpoint_root
            / artifact_names.get("checkpoint_manifest_filename", "{run_name}_adapter_manifest.json").format(
                run_name=run_name
            )
        ).resolve(),
        loss_curve_path=(
            context.plots_dir
            / artifact_names.get("loss_curve_filename", "{run_name}_sft_loss_curve.png").format(run_name=run_name)
        ).resolve(),
        eval_loss_curve_path=(
            context.plots_dir
            / artifact_names.get("eval_loss_curve_filename", "{run_name}_sft_eval_loss_curve.png").format(
                run_name=run_name
            )
        ).resolve(),
        learning_rate_curve_path=(
            context.plots_dir
            / artifact_names.get("learning_rate_curve_filename", "{run_name}_sft_learning_rate_curve.png").format(
                run_name=run_name
            )
        ).resolve(),
        examples_seen_curve_path=(
            context.plots_dir
            / artifact_names.get("examples_seen_curve_filename", "{run_name}_sft_examples_seen_curve.png").format(
                run_name=run_name
            )
        ).resolve(),
        tokens_seen_curve_path=(
            context.plots_dir
            / artifact_names.get("tokens_seen_curve_filename", "{run_name}_sft_tokens_seen_curve.png").format(
                run_name=run_name
            )
        ).resolve(),
    )


def _normalize_messages_row(record_id: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    if len(messages) < 2:
        raise ValueError(f"Record {record_id} does not contain enough messages for SFT.")
    completion = messages[-1]
    if str(completion.get("role", "")).strip() != "assistant":
        raise ValueError(f"Record {record_id} must end with an assistant message.")

    # We keep the repo manifests in messages form, but normalize to conversational
    # prompt-completion so TRL applies the model chat template and masks prompt loss.
    return {
        "record_id": record_id,
        "prompt": [dict(message) for message in messages[:-1]],
        "completion": [dict(completion)],
    }


def load_sft_training_records(config: SFTResolvedConfig) -> tuple[list[dict[str, Any]], SampleSelectionMetadata]:
    """Load the train manifest into TRL-ready rows."""

    selection = select_rows(
        read_jsonl(config.train_manifest),
        sample_limit=config.train_sample_limit,
        sample_percent=config.train_sample_percent,
        sample_seed=config.sample_seed,
    )
    rows = selection.rows
    if config.dataset_format == "messages":
        return [
            _normalize_messages_row(str(row.get("record_id", f"row-{index}")), list(row.get("messages", [])))
            for index, row in enumerate(rows, start=1)
        ], selection.metadata
    if config.dataset_format == "prompt_completion":
        return [
            {
                "record_id": str(row.get("record_id", f"row-{index}")),
                "prompt": str(row.get("prompt", "")),
                "completion": str(row.get("completion", "")),
            }
            for index, row in enumerate(rows, start=1)
        ], selection.metadata
    raise ValueError(f"Unsupported SFT dataset format: {config.dataset_format}")


def load_sft_eval_records(config: SFTResolvedConfig) -> tuple[list[dict[str, Any]], SampleSelectionMetadata]:
    """Load the eval manifest into TRL-ready rows."""

    selection = select_rows(
        read_jsonl(config.eval_manifest),
        sample_limit=config.eval_sample_limit,
        sample_percent=config.eval_sample_percent,
        sample_seed=config.sample_seed,
    )
    rows = selection.rows
    if config.dataset_format == "messages":
        return [
            _normalize_messages_row(str(row.get("record_id", f"eval-{index}")), list(row.get("messages", [])))
            for index, row in enumerate(rows, start=1)
        ], selection.metadata
    if config.dataset_format == "prompt_completion":
        return [
            {
                "record_id": str(row.get("record_id", f"eval-{index}")),
                "prompt": str(row.get("prompt", "")),
                "completion": str(row.get("reference_json", "")),
            }
            for index, row in enumerate(rows, start=1)
        ], selection.metadata
    raise ValueError(f"Unsupported SFT dataset format: {config.dataset_format}")


def _render_record_text(record: dict[str, Any], tokenizer: Any) -> str:
    if isinstance(record.get("prompt"), list) and isinstance(record.get("completion"), list):
        messages = [*record["prompt"], *record["completion"]]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if isinstance(rendered, str):
                    return rendered
            except Exception:
                pass
        return "\n".join(
            f"{message.get('role', 'unknown')}: {message.get('content', '')}" for message in messages
        )
    prompt_text = record.get("prompt", "")
    completion_text = record.get("completion", "")
    if isinstance(prompt_text, list):
        prompt_text = json.dumps(prompt_text, sort_keys=True, ensure_ascii=True)
    if isinstance(completion_text, list):
        completion_text = json.dumps(completion_text, sort_keys=True, ensure_ascii=True)
    return f"{prompt_text}\n{completion_text}".strip()


def _count_rendered_tokens(text: str, tokenizer: Any) -> int:
    if hasattr(tokenizer, "__call__"):
        try:
            tokens = tokenizer(text, add_special_tokens=False)
            input_ids = tokens.get("input_ids")
            if isinstance(input_ids, list):
                return len(input_ids)
        except Exception:
            pass
    return len(text.split())


def _build_token_cache_payload(
    *,
    cache_dir: Path,
    cache_key: str,
    manifest_path: Path,
    split_label: str,
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    cached = load_cached_token_payload(cache_dir)
    if cached is not None:
        return cached

    rendered_rows: list[dict[str, Any]] = []
    token_counts: list[int] = []
    for row in rows:
        rendered_text = _render_record_text(row, tokenizer)
        token_count = _count_rendered_tokens(rendered_text, tokenizer)
        token_counts.append(token_count)
        rendered_rows.append(
            {
                "record_id": str(row.get("record_id", "")),
                "rendered_text": rendered_text,
                "token_count": token_count,
            }
        )

    payload = {
        "cache_key": cache_key,
        "manifest_path": str(manifest_path),
        "split_label": split_label,
        "stats": summarize_token_counts(token_counts).to_dict(),
        "rows": rendered_rows,
    }
    write_cached_token_payload(cache_dir, payload)
    return payload


def _prepare_token_cache(
    *,
    config: SFTResolvedConfig,
    artifacts: SFTOutputPaths,
    tokenizer: Any,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    train_subset_metadata: SampleSelectionMetadata,
    eval_subset_metadata: SampleSelectionMetadata,
) -> dict[str, Any]:
    token_cache = dict(config.token_cache or {})
    if not token_cache.get("enabled", False):
        return {"enabled": False}

    configured_cache_root = token_cache.get("cache_root")
    if configured_cache_root in (None, ""):
        cache_root = (artifacts.checkpoint_root.parents[2] / "tokenized" / "sft").resolve()
    else:
        cache_root_path = Path(str(configured_cache_root))
        if cache_root_path.is_absolute():
            cache_root = cache_root_path.resolve()
        else:
            cache_root = (artifacts.checkpoint_root.parents[3] / cache_root_path).resolve()
    cache_mode = str(token_cache.get("mode", "rendered_messages"))
    completion_only_loss = bool(config.training.get("completion_only_loss", True))
    packing = bool(config.training.get("packing", False))

    train_key = build_token_cache_key(
        manifest_path=config.train_manifest,
        rows=train_records,
        model_name_or_path=config.model_name_or_path,
        max_seq_length=config.max_seq_length,
        packing=packing,
        completion_only_loss=completion_only_loss,
        mode=cache_mode,
        sample_percent=train_subset_metadata.sample_percent,
        sample_seed=train_subset_metadata.sample_seed,
    )
    eval_key = build_token_cache_key(
        manifest_path=config.eval_manifest,
        rows=eval_records,
        model_name_or_path=config.model_name_or_path,
        max_seq_length=config.max_seq_length,
        packing=packing,
        completion_only_loss=completion_only_loss,
        mode=cache_mode,
        sample_percent=eval_subset_metadata.sample_percent,
        sample_seed=eval_subset_metadata.sample_seed,
    )

    train_payload = _build_token_cache_payload(
        cache_dir=cache_root / "train" / train_key,
        cache_key=train_key,
        manifest_path=config.train_manifest,
        split_label="train",
        rows=train_records,
        tokenizer=tokenizer,
    )
    eval_payload = _build_token_cache_payload(
        cache_dir=cache_root / "eval" / eval_key,
        cache_key=eval_key,
        manifest_path=config.eval_manifest,
        split_label="eval",
        rows=eval_records,
        tokenizer=tokenizer,
    )
    return {
        "enabled": True,
        "mode": cache_mode,
        "cache_root": str(cache_root),
        "train": {
            "cache_key": train_key,
            "cache_dir": str((cache_root / "train" / train_key).resolve()),
            **train_payload,
        },
        "eval": {
            "cache_key": eval_key,
            "cache_dir": str((cache_root / "eval" / eval_key).resolve()),
            **eval_payload,
        },
    }


def _effective_batch_size(training: dict[str, Any]) -> int:
    return int(training.get("per_device_train_batch_size", 1)) * int(
        training.get("gradient_accumulation_steps", 1)
    )


def _build_history_telemetry(
    *,
    log_history: list[dict[str, Any]],
    dataset_telemetry: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, list[dict[str, float]]]:
    token_cache = dataset_telemetry.get("token_cache", {}) or {}
    train_stats = (token_cache.get("train") or {}).get("stats") or {}
    average_tokens = float(train_stats.get("avg_token_count") or 0.0)
    examples_per_step = float(_effective_batch_size(training))

    derived: dict[str, list[dict[str, float]]] = {
        "examples_seen": [],
        "tokens_seen": [],
    }
    for entry in log_history:
        step_value = entry.get("step")
        if step_value is None:
            continue
        try:
            step = float(step_value)
        except (TypeError, ValueError):
            continue
        epoch_value = entry.get("epoch")
        try:
            epoch = float(epoch_value) if epoch_value is not None else 0.0
        except (TypeError, ValueError):
            epoch = 0.0
        examples_seen = step * examples_per_step
        derived["examples_seen"].append(
            {"step": step, "epoch": epoch, "examples_seen": examples_seen}
        )
        if average_tokens > 0:
            derived["tokens_seen"].append(
                {
                    "step": step,
                    "epoch": epoch,
                    "tokens_seen": examples_seen * average_tokens,
                }
            )
    return derived


def _load_training_stack() -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required for SFT training. Run inside Colab after installing requirements-colab.txt."
        ) from exc

    try:
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TRL, PEFT, transformers, and datasets are required for SFT training. "
            "Use requirements-colab.txt for the Colab runtime."
        ) from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def _resolve_compute_dtype(torch_module: Any, value: str | None) -> Any:
    if value in (None, "", "auto"):
        if torch_module.cuda.is_available():
            if hasattr(torch_module.cuda, "is_bf16_supported") and torch_module.cuda.is_bf16_supported():
                return torch_module.bfloat16
            return torch_module.float16
        return torch_module.float32
    if not hasattr(torch_module, value):
        raise ValueError(f"Unsupported compute dtype: {value}")
    return getattr(torch_module, value)


def build_trainer_bundle(
    *,
    config: SFTResolvedConfig,
    artifacts: SFTOutputPaths,
    run_name: str,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    train_subset_metadata: SampleSelectionMetadata,
    eval_subset_metadata: SampleSelectionMetadata,
) -> TrainerBundle:
    """Instantiate the tokenizer, model, and TRL SFT trainer lazily."""

    modules = _load_training_stack()
    torch_module = modules["torch"]
    compute_dtype = _resolve_compute_dtype(torch_module, config.quantization.get("compute_dtype"))

    tokenizer = modules["AutoTokenizer"].from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if config.eos_token:
        tokenizer.eos_token = config.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    model_dtype = compute_dtype
    if config.quantization.get("enabled", False):
        # NF4 plus double quantization is the standard QLoRA recipe for fitting a
        # larger instruct model into a single Colab Pro GPU while preserving quality.
        quantization_config = modules["BitsAndBytesConfig"](
            load_in_4bit=bool(config.quantization.get("load_in_4bit", True)),
            bnb_4bit_quant_type=str(config.quantization.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(config.quantization.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    device_map = config.device_map
    if device_map is None and torch_module.cuda.is_available():
        device_map = "auto"

    model = modules["AutoModelForCausalLM"].from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=_resolve_compute_dtype(torch_module, config.torch_dtype)
        if config.torch_dtype not in (None, "", "auto")
        else model_dtype,
        device_map=device_map,
    )

    if config.training.get("gradient_checkpointing", False):
        if hasattr(model, "config"):
            model.config.use_cache = False
        # With batch size 1 on Colab, gradient accumulation carries throughput
        # while checkpointing keeps activation memory low enough for QLoRA.
        model = modules["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=True,
        )
    elif config.quantization.get("enabled", False):
        model = modules["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=False,
        )

    lora_config = modules["LoraConfig"](
        r=int(config.lora.get("r", 16)),
        lora_alpha=int(config.lora.get("alpha", 32)),
        lora_dropout=float(config.lora.get("dropout", 0.05)),
        bias=str(config.lora.get("bias", "none")),
        task_type=str(config.lora.get("task_type", "CAUSAL_LM")),
        target_modules=list(config.lora.get("target_modules", [])),
    )

    sft_args = modules["SFTConfig"](
        **_compact_mapping(
            {
                "output_dir": str(artifacts.checkpoint_root),
                "logging_dir": str(artifacts.logs_dir / "tensorboard"),
                "run_name": run_name,
                "max_length": config.max_seq_length,
                "learning_rate": float(config.training.get("learning_rate", 2e-4)),
                "lr_scheduler_type": config.training.get("lr_scheduler_type", "cosine"),
                "warmup_ratio": float(config.training.get("warmup_ratio", 0.05)),
                "weight_decay": float(config.training.get("weight_decay", 0.0)),
                "optim": config.training.get("optim", "paged_adamw_32bit"),
                "gradient_checkpointing": bool(config.training.get("gradient_checkpointing", True)),
                # Packing is intentionally disabled for v1 so the first training
                # stage stays easier to inspect and debug against individual prompts.
                "packing": bool(config.training.get("packing", False)),
                "per_device_train_batch_size": int(config.training.get("per_device_train_batch_size", 1)),
                "per_device_eval_batch_size": int(config.training.get("per_device_eval_batch_size", 1)),
                "gradient_accumulation_steps": int(config.training.get("gradient_accumulation_steps", 1)),
                "logging_steps": int(config.training.get("logging_steps", 1)),
                "save_total_limit": int(config.training.get("save_total_limit", 2)),
                "report_to": list(config.training.get("report_to", [])),
                "num_train_epochs": config.training.get("num_train_epochs"),
                "max_steps": config.training.get("max_steps"),
                "eval_strategy": config.training.get("eval_strategy"),
                "eval_steps": config.training.get("eval_steps"),
                "save_strategy": config.training.get("save_strategy"),
                "save_steps": config.training.get("save_steps"),
                "completion_only_loss": bool(config.training.get("completion_only_loss", True)),
                "bf16": compute_dtype == torch_module.bfloat16,
                "fp16": compute_dtype == torch_module.float16,
                "eos_token": config.eos_token,
            }
        )
    )

    train_dataset = modules["Dataset"].from_list(train_records)
    eval_dataset = modules["Dataset"].from_list(eval_records) if eval_records else None
    token_cache_payload = _prepare_token_cache(
        config=config,
        artifacts=artifacts,
        tokenizer=tokenizer,
        train_records=train_records,
        eval_records=eval_records,
        train_subset_metadata=train_subset_metadata,
        eval_subset_metadata=eval_subset_metadata,
    )

    trainer = modules["SFTTrainer"](
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    return TrainerBundle(
        trainer=trainer,
        model=model,
        tokenizer=tokenizer,
        dataset_telemetry={
            "token_cache": token_cache_payload,
            "effective_batch_size": _effective_batch_size(config.training),
            "subset_selection": {
                "train": train_subset_metadata.to_dict(),
                "eval": eval_subset_metadata.to_dict(),
            },
        },
    )


def write_checkpoint_manifest(
    *,
    config: SFTResolvedConfig,
    artifacts: SFTOutputPaths,
    run_name: str,
    status: str,
    train_record_count: int,
    eval_record_count: int,
    history_artifacts: dict[str, Any] | None = None,
    train_metrics: dict[str, Any] | None = None,
    dataset_telemetry: dict[str, Any] | None = None,
) -> Path:
    """Persist a small checkpoint manifest that points to the runtime adapter."""

    data_pipeline_metadata = build_data_pipeline_metadata(
        repo_root=config.config_path.parents[1],
        build_summary_path=config.build_summary_path,
        composition_summary_path=config.composition_summary_path,
    )
    payload = {
        "stage": "sft",
        "run_name": run_name,
        "status": status,
        "profile": config.profile_name,
        "config_path": str(config.config_path),
        "base_model": config.model_name_or_path,
        "adapter_path": str(artifacts.adapter_dir),
        "checkpoint_root": str(artifacts.checkpoint_root),
        "logs_dir": str(artifacts.logs_dir),
        "trainer_state_path": str(artifacts.trainer_state_path),
        "train_manifest": str(config.train_manifest),
        "eval_manifest": str(config.eval_manifest),
        "dataset_format": config.dataset_format,
        "train_record_count": train_record_count,
        "eval_record_count": eval_record_count,
        "max_seq_length": config.max_seq_length,
        "effective_batch_size": _effective_batch_size(config.training),
        "subset_selection": (dataset_telemetry or {}).get("subset_selection", {}),
        "quantization": config.quantization,
        "lora": config.lora,
        "training": config.training,
        "dataset_telemetry": dataset_telemetry or {},
        "data_pipeline": data_pipeline_metadata,
        "history_artifacts": history_artifacts or {},
        "train_metrics": train_metrics or {},
    }
    return write_json(artifacts.checkpoint_manifest_path, payload)


def write_sft_summary(
    *,
    context: RuntimeContext,
    config: SFTResolvedConfig,
    artifacts: SFTOutputPaths,
    run_name: str,
    status: str,
    train_record_count: int,
    eval_record_count: int,
    history_artifacts: dict[str, Any] | None = None,
    train_metrics: dict[str, Any] | None = None,
    dataset_telemetry: dict[str, Any] | None = None,
) -> Path:
    """Write the run summary used by the review notebook and docs."""

    data_pipeline_metadata = build_data_pipeline_metadata(
        repo_root=config.config_path.parents[1],
        build_summary_path=config.build_summary_path,
        composition_summary_path=config.composition_summary_path,
    )
    runtime_summary = {
        "train_runtime": (train_metrics or {}).get("train_runtime"),
        "train_samples_per_second": (train_metrics or {}).get("train_samples_per_second"),
        "train_steps_per_second": (train_metrics or {}).get("train_steps_per_second"),
        "total_flos": (train_metrics or {}).get("total_flos"),
    }
    payload = {
        "stage": "sft",
        "run_name": run_name,
        "status": status,
        "profile": config.profile_name,
        "config_path": str(config.config_path),
        "runtime_root": str(context.runtime_root),
        "base_model": config.model_name_or_path,
        "adapter_path": str(artifacts.adapter_dir),
        "checkpoint_root": str(artifacts.checkpoint_root),
        "logs_dir": str(artifacts.logs_dir),
        "summary_path": str(artifacts.summary_path),
        "history_path": str(artifacts.history_path),
        "loss_curve_path": str(artifacts.loss_curve_path),
        "eval_loss_curve_path": str(artifacts.eval_loss_curve_path),
        "train_manifest": str(config.train_manifest),
        "eval_manifest": str(config.eval_manifest),
        "dataset_format": config.dataset_format,
        "dataset_view": "conversational_prompt_completion"
        if config.dataset_format == "messages"
        else "prompt_completion",
        "train_record_count": train_record_count,
        "eval_record_count": eval_record_count,
        "max_seq_length": config.max_seq_length,
        "effective_batch_size": _effective_batch_size(config.training),
        "subset_selection": (dataset_telemetry or {}).get("subset_selection", {}),
        "runtime_summary": runtime_summary,
        "quantization": config.quantization,
        "lora": config.lora,
        "training": config.training,
        "dataset_telemetry": dataset_telemetry or {},
        "data_pipeline": data_pipeline_metadata,
        "history_artifacts": history_artifacts or {},
        "train_metrics": train_metrics or {},
    }
    return write_json(artifacts.summary_path, payload)


def save_trainer_state(trainer: Any, artifacts: SFTOutputPaths) -> Path | None:
    """Copy the trainer state JSON into the dedicated runtime logs directory."""

    if hasattr(trainer, "save_state"):
        trainer.save_state()

    source_path = artifacts.checkpoint_root / "trainer_state.json"
    if source_path.exists():
        artifacts.logs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, artifacts.trainer_state_path)
        return artifacts.trainer_state_path
    return None


def collect_log_history(trainer: Any) -> list[dict[str, Any]]:
    """Return the trainer log history in a stable list form."""

    state = getattr(trainer, "state", None)
    if state is None:
        return []
    return list(getattr(state, "log_history", []) or [])


def mirror_sft_artifacts(
    *,
    repo_root: str | Path,
    artifacts: SFTOutputPaths,
    mirror_metrics: bool,
    mirror_plots: bool,
    mirror_checkpoint_manifest: bool,
) -> dict[str, list[str]]:
    """Mirror selected small runtime artifacts into repo artifact paths."""

    targets = resolve_repo_artifact_targets(repo_root)
    mirrored: dict[str, list[str]] = {"metrics": [], "plots": [], "checkpoints": []}

    if mirror_metrics:
        for source_path in (artifacts.summary_path, artifacts.history_path):
            if source_path.exists():
                destination = targets["metrics"] / source_path.name
                mirrored_path = mirror_small_artifact(source_path, destination)
                mirrored["metrics"].append(str(mirrored_path))

    if mirror_plots:
        for source_path in (
            artifacts.loss_curve_path,
            artifacts.eval_loss_curve_path,
            artifacts.learning_rate_curve_path,
            artifacts.examples_seen_curve_path,
            artifacts.tokens_seen_curve_path,
        ):
            if source_path.exists():
                destination = targets["plots"] / source_path.name
                mirrored_path = mirror_small_artifact(source_path, destination)
                mirrored["plots"].append(str(mirrored_path))

    if mirror_checkpoint_manifest and artifacts.checkpoint_manifest_path.exists():
        destination = targets["checkpoints"] / artifacts.checkpoint_manifest_path.name
        mirrored_path = mirror_small_artifact(artifacts.checkpoint_manifest_path, destination)
        mirrored["checkpoints"].append(str(mirrored_path))

    return mirrored


def promote_latest_sft_model(
    *,
    repo_root: str | Path,
    config: SFTResolvedConfig,
    artifacts: SFTOutputPaths,
    mirrored_artifacts: dict[str, list[str]],
) -> Path:
    """Update the repo-side latest-model manifest for the completed SFT adapter."""

    metrics_paths = mirrored_artifacts.get("metrics") or [str(artifacts.summary_path), str(artifacts.history_path)]
    report_paths = mirrored_artifacts.get("checkpoints") or [str(artifacts.checkpoint_manifest_path)]
    manifest = LatestModelManifest(
        stage="sft",
        status="ready",
        base_model=config.model_name_or_path,
        adapter_path=str(artifacts.adapter_dir),
        schema_version="1.0.0",
        config_paths=[str(config.config_path)],
        metrics_paths=metrics_paths,
        report_paths=report_paths,
    )
    return save_latest_model_manifest(repo_root, manifest)


def save_training_artifacts(
    *,
    trainer: Any,
    config: SFTResolvedConfig,
    artifacts: SFTOutputPaths,
    run_name: str,
    context: RuntimeContext,
    train_record_count: int,
    eval_record_count: int,
    train_metrics: dict[str, Any] | None = None,
    dataset_telemetry: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Persist history JSON, plots, summary, and checkpoint metadata after training."""

    log_history = collect_log_history(trainer)
    history_metadata = {
        "stage": "sft",
        "run_name": run_name,
        "effective_batch_size": _effective_batch_size(config.training),
        "train_record_count": train_record_count,
        "eval_record_count": eval_record_count,
        "dataset_telemetry": dataset_telemetry or {},
    }
    extra_plot_specs = [
        PlotSpec(
            metric_key="learning_rate",
            output_path=artifacts.learning_rate_curve_path,
            title="SFT Learning Rate",
            color="#2ca02c",
        ),
        PlotSpec(
            metric_key="examples_seen",
            output_path=artifacts.examples_seen_curve_path,
            title="SFT Examples Seen",
            color="#9467bd",
        ),
        PlotSpec(
            metric_key="tokens_seen",
            output_path=artifacts.tokens_seen_curve_path,
            title="SFT Tokens Seen",
            color="#ff7f0e",
        ),
    ]
    history_artifacts = write_training_history_and_plots(
        log_history=log_history,
        history_path=artifacts.history_path,
        loss_curve_path=artifacts.loss_curve_path,
        eval_loss_curve_path=artifacts.eval_loss_curve_path,
        tracked_scalar_keys=["learning_rate"],
        extra_plot_specs=extra_plot_specs,
        derived_scalar_series=_build_history_telemetry(
            log_history=log_history,
            dataset_telemetry=dataset_telemetry or {},
            training=config.training,
        ),
        metadata=history_metadata,
    )
    merged_metrics = dict(train_metrics or {})
    if hasattr(trainer, "state") and getattr(trainer.state, "best_metric", None) is not None:
        merged_metrics["best_metric"] = trainer.state.best_metric
    merged_metrics["effective_batch_size"] = _effective_batch_size(config.training)

    summary_path = write_sft_summary(
        context=context,
        config=config,
        artifacts=artifacts,
        run_name=run_name,
        status="completed",
        train_record_count=train_record_count,
        eval_record_count=eval_record_count,
        history_artifacts=history_artifacts,
        train_metrics=merged_metrics,
        dataset_telemetry=dataset_telemetry,
    )
    checkpoint_manifest_path = write_checkpoint_manifest(
        config=config,
        artifacts=artifacts,
        run_name=run_name,
        status="completed",
        train_record_count=train_record_count,
        eval_record_count=eval_record_count,
        history_artifacts=history_artifacts,
        train_metrics=merged_metrics,
        dataset_telemetry=dataset_telemetry,
    )
    return history_artifacts, summary_path, checkpoint_manifest_path


def write_dry_run_artifacts(
    *,
    context: RuntimeContext,
    config: SFTResolvedConfig,
    artifacts: SFTOutputPaths,
    run_name: str,
    train_record_count: int,
    eval_record_count: int,
    train_subset_metadata: SampleSelectionMetadata,
    eval_subset_metadata: SampleSelectionMetadata,
) -> tuple[Path, Path]:
    """Persist the resolved run contract without importing the heavy training stack."""

    summary_path = write_sft_summary(
        context=context,
        config=config,
        artifacts=artifacts,
        run_name=run_name,
        status="dry_run_ready",
        train_record_count=train_record_count,
        eval_record_count=eval_record_count,
        history_artifacts={},
        train_metrics={"effective_batch_size": _effective_batch_size(config.training)},
        dataset_telemetry={
            "token_cache": {
                "enabled": bool(config.token_cache.get("enabled", False)),
                "cache_root": config.token_cache.get("cache_root"),
                "mode": config.token_cache.get("mode"),
            },
            "effective_batch_size": _effective_batch_size(config.training),
            "subset_selection": {
                "train": train_subset_metadata.to_dict(),
                "eval": eval_subset_metadata.to_dict(),
            },
        },
    )
    checkpoint_manifest_path = write_checkpoint_manifest(
        config=config,
        artifacts=artifacts,
        run_name=run_name,
        status="dry_run_ready",
        train_record_count=train_record_count,
        eval_record_count=eval_record_count,
        history_artifacts={},
        train_metrics={"effective_batch_size": _effective_batch_size(config.training)},
        dataset_telemetry={
            "token_cache": {
                "enabled": bool(config.token_cache.get("enabled", False)),
                "cache_root": config.token_cache.get("cache_root"),
                "mode": config.token_cache.get("mode"),
            },
            "effective_batch_size": _effective_batch_size(config.training),
            "subset_selection": {
                "train": train_subset_metadata.to_dict(),
                "eval": eval_subset_metadata.to_dict(),
            },
        },
    )
    return summary_path, checkpoint_manifest_path
