"""Reusable helpers for single-GPU QLoRA supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

from .artifacts import mirror_small_artifact
from .manifests import LatestModelManifest, save_latest_model_manifest
from .runtime import RuntimeContext, resolve_repo_artifact_targets
from .training_plots import write_training_history_and_plots
from .utils import load_yaml, read_jsonl, write_json


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
    train_manifest: Path
    eval_manifest: Path
    dataset_format: str
    max_seq_length: int
    train_sample_limit: int | None
    eval_sample_limit: int | None
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


@dataclass(frozen=True)
class TrainerBundle:
    """Minimal objects returned by the trainer builder."""

    trainer: Any
    model: Any
    tokenizer: Any


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
) -> SFTResolvedConfig:
    """Resolve config defaults plus one named profile into a concrete run config."""

    resolved_repo_root = Path(repo_root).resolve()
    resolved_config_path = _resolve_repo_path(resolved_repo_root, config_path)
    raw_config = load_sft_config(resolved_config_path)
    profiles = raw_config.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown SFT profile: {profile_name}")

    profile_config = profiles[profile_name]
    data_config = _deep_merge(raw_config.get("data", {}), profile_config.get("data", {}))
    training_config = _deep_merge(raw_config.get("training", {}), profile_config.get("training", {}))
    model_name_or_path = str(raw_config.get("model", {}).get("model_name_or_path", "")).strip()
    if not model_name_or_path:
        raise ValueError(f"SFT config is missing model.model_name_or_path: {resolved_config_path}")

    return SFTResolvedConfig(
        config_path=resolved_config_path,
        profile_name=profile_name,
        model_name_or_path=model_name_or_path,
        trust_remote_code=bool(raw_config.get("model", {}).get("trust_remote_code", False)),
        eos_token=raw_config.get("model", {}).get("eos_token"),
        train_manifest=_resolve_repo_path(
            resolved_repo_root,
            data_config.get("train_manifest", "data/manifests/support_tickets_sft_messages.jsonl"),
        ),
        eval_manifest=_resolve_repo_path(
            resolved_repo_root,
            data_config.get("eval_manifest", "data/manifests/support_tickets_eval_manifest.jsonl"),
        ),
        dataset_format=str(data_config.get("dataset_format", "messages")),
        max_seq_length=int(data_config.get("max_seq_length", 1024)),
        train_sample_limit=data_config.get("train_sample_limit"),
        eval_sample_limit=data_config.get("eval_sample_limit"),
        quantization=dict(raw_config.get("quantization", {})),
        lora=dict(raw_config.get("lora", {})),
        training=training_config,
        artifacts=dict(raw_config.get("artifacts", {})),
        raw_config=raw_config,
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
    )


def _sample_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    return rows[: int(limit)]


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


def load_sft_training_records(config: SFTResolvedConfig) -> list[dict[str, Any]]:
    """Load the train manifest into TRL-ready rows."""

    rows = _sample_rows(read_jsonl(config.train_manifest), config.train_sample_limit)
    if config.dataset_format == "messages":
        return [
            _normalize_messages_row(str(row.get("record_id", f"row-{index}")), list(row.get("messages", [])))
            for index, row in enumerate(rows, start=1)
        ]
    if config.dataset_format == "prompt_completion":
        return [
            {
                "record_id": str(row.get("record_id", f"row-{index}")),
                "prompt": str(row.get("prompt", "")),
                "completion": str(row.get("completion", "")),
            }
            for index, row in enumerate(rows, start=1)
        ]
    raise ValueError(f"Unsupported SFT dataset format: {config.dataset_format}")


def load_sft_eval_records(config: SFTResolvedConfig) -> list[dict[str, Any]]:
    """Load the eval manifest into TRL-ready rows."""

    rows = _sample_rows(read_jsonl(config.eval_manifest), config.eval_sample_limit)
    if config.dataset_format == "messages":
        return [
            _normalize_messages_row(str(row.get("record_id", f"eval-{index}")), list(row.get("messages", [])))
            for index, row in enumerate(rows, start=1)
        ]
    if config.dataset_format == "prompt_completion":
        return [
            {
                "record_id": str(row.get("record_id", f"eval-{index}")),
                "prompt": str(row.get("prompt", "")),
                "completion": str(row.get("reference_json", "")),
            }
            for index, row in enumerate(rows, start=1)
        ]
    raise ValueError(f"Unsupported SFT dataset format: {config.dataset_format}")


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

    device_map = config.raw_config.get("model", {}).get("device_map")
    if device_map is None and torch_module.cuda.is_available():
        device_map = "auto"

    model = modules["AutoModelForCausalLM"].from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=model_dtype,
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

    trainer = modules["SFTTrainer"](
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    return TrainerBundle(trainer=trainer, model=model, tokenizer=tokenizer)


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
) -> Path:
    """Persist a small checkpoint manifest that points to the runtime adapter."""

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
        "quantization": config.quantization,
        "lora": config.lora,
        "training": config.training,
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
) -> Path:
    """Write the run summary used by the review notebook and docs."""

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
        "quantization": config.quantization,
        "lora": config.lora,
        "training": config.training,
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
        for source_path in (artifacts.loss_curve_path, artifacts.eval_loss_curve_path):
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
) -> tuple[dict[str, Any], Path, Path]:
    """Persist history JSON, plots, summary, and checkpoint metadata after training."""

    history_artifacts = write_training_history_and_plots(
        log_history=collect_log_history(trainer),
        history_path=artifacts.history_path,
        loss_curve_path=artifacts.loss_curve_path,
        eval_loss_curve_path=artifacts.eval_loss_curve_path,
    )
    merged_metrics = dict(train_metrics or {})
    if hasattr(trainer, "state") and getattr(trainer.state, "best_metric", None) is not None:
        merged_metrics["best_metric"] = trainer.state.best_metric

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
        train_metrics={},
    )
    checkpoint_manifest_path = write_checkpoint_manifest(
        config=config,
        artifacts=artifacts,
        run_name=run_name,
        status="dry_run_ready",
        train_record_count=train_record_count,
        eval_record_count=eval_record_count,
        history_artifacts={},
        train_metrics={},
    )
    return summary_path, checkpoint_manifest_path
