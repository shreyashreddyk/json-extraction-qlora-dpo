"""Reusable helpers for single-GPU DPO training on top of the SFT stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

from .artifacts import mirror_small_artifact
from .manifests import LatestModelManifest, save_latest_model_manifest
from .runtime import RuntimeContext, resolve_repo_artifact_targets
from .training_plots import PlotSpec, write_training_history_and_plots
from .utils import load_yaml, read_json, read_jsonl, write_json

DPO_REWARD_METRIC_SPECS = (
    ("rewards/chosen", "DPO Reward: Chosen", "#2ca02c"),
    ("rewards/rejected", "DPO Reward: Rejected", "#d62728"),
    ("rewards/accuracies", "DPO Reward Accuracy", "#9467bd"),
    ("rewards/margins", "DPO Reward Margin", "#ff7f0e"),
)


def _compact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_repo_path(repo_root: Path, path_value: str | Path | None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return read_json(path)


def _load_latest_model_manifest_at_path(path: Path | None, repo_root: Path) -> LatestModelManifest | None:
    default_path = (repo_root / "artifacts" / "checkpoints" / "latest_model.json").resolve()
    if path is None or not path.exists():
        return None
    if path.resolve() == default_path:
        from .manifests import load_latest_model_manifest

        return load_latest_model_manifest(repo_root)
    return LatestModelManifest(**read_json(path))


def _coerce_report_manifest_path(
    repo_root: Path,
    latest_manifest: LatestModelManifest | None,
) -> Path | None:
    if latest_manifest is None:
        return None
    for candidate in latest_manifest.report_paths:
        resolved = _resolve_repo_path(repo_root, candidate)
        if resolved is not None:
            return resolved
    return None


@dataclass(frozen=True)
class DPOResolvedConfig:
    """Resolved config for one DPO run profile."""

    config_path: Path
    profile_name: str
    latest_model_manifest_path: Path | None
    latest_model_manifest: LatestModelManifest | None
    source_sft_manifest_path: Path | None
    source_sft_manifest: dict[str, Any] | None
    base_model: str
    model_name_or_path: str
    adapter_path: str | None
    reference_adapter_path: str | None
    merged_model_path: str | None
    reference_strategy: str
    revision: str | None
    trust_remote_code: bool
    torch_dtype: str | None
    device_map: str | None
    quantization: dict[str, Any]
    preference_manifest: Path | None
    eval_preference_manifest: Path | None
    train_sample_limit: int | None
    eval_sample_limit: int | None
    training: dict[str, Any]
    artifacts: dict[str, Any]
    raw_config: dict[str, Any]


@dataclass(frozen=True)
class DPOOutputPaths:
    """Runtime file layout for one DPO run."""

    checkpoint_root: Path
    adapter_dir: Path
    logs_dir: Path
    trainer_state_path: Path
    summary_path: Path
    history_path: Path
    checkpoint_manifest_path: Path
    loss_curve_path: Path
    eval_loss_curve_path: Path
    reward_plot_paths: dict[str, Path]


@dataclass(frozen=True)
class TrainerBundle:
    """Minimal objects returned by the DPO trainer builder."""

    trainer: Any
    model: Any
    ref_model: Any
    tokenizer: Any


def load_dpo_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML configuration for DPO."""

    resolved = Path(config_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"DPO config does not exist: {resolved}")
    return load_yaml(resolved)


def resolve_dpo_config(
    *,
    config_path: str | Path,
    repo_root: str | Path,
    profile_name: str,
    preference_manifest: str | Path | None = None,
    source_sft_manifest: str | Path | None = None,
    base_model: str | None = None,
    adapter_path: str | None = None,
    merged_model_path: str | None = None,
) -> DPOResolvedConfig:
    """Resolve config defaults plus one named profile into a concrete DPO config."""

    resolved_repo_root = Path(repo_root).resolve()
    resolved_config_path = _resolve_repo_path(resolved_repo_root, config_path)
    if resolved_config_path is None:
        raise FileNotFoundError(f"DPO config does not exist: {config_path}")
    raw_config = load_dpo_config(resolved_config_path)
    profiles = raw_config.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown DPO profile: {profile_name}")

    profile_config = profiles[profile_name]
    merged = _deep_merge(raw_config, profile_config)
    model_config = dict(merged.get("model", {}))
    training_config = dict(merged.get("training", {}))

    latest_model_manifest_path = _resolve_repo_path(
        resolved_repo_root,
        model_config.get("latest_model_manifest", "artifacts/checkpoints/latest_model.json"),
    )
    latest_manifest = _load_latest_model_manifest_at_path(latest_model_manifest_path, resolved_repo_root)

    source_sft_manifest_path = (
        _resolve_repo_path(resolved_repo_root, source_sft_manifest)
        or _resolve_repo_path(resolved_repo_root, model_config.get("source_sft_manifest"))
        or _coerce_report_manifest_path(resolved_repo_root, latest_manifest)
    )
    source_sft_manifest_payload = _load_optional_json(source_sft_manifest_path)

    source_base_model = None
    source_adapter_path = None
    source_merged_path = None
    source_quantization: dict[str, Any] = {}
    if source_sft_manifest_payload is not None:
        source_base_model = source_sft_manifest_payload.get("base_model")
        source_adapter_path = source_sft_manifest_payload.get("adapter_path")
        source_merged_path = source_sft_manifest_payload.get("merged_export_path")
        source_quantization = dict(source_sft_manifest_payload.get("quantization", {}) or {})

    reference_strategy = str(model_config.get("reference_strategy", "adapter")).strip() or "adapter"
    if reference_strategy not in {"adapter", "merged"}:
        raise ValueError("DPO model.reference_strategy must be 'adapter' or 'merged'.")

    resolved_base_model = (
        base_model
        or model_config.get("base_model")
        or source_base_model
        or (latest_manifest.base_model if latest_manifest is not None else None)
    )
    if not resolved_base_model:
        raise ValueError(
            "Could not resolve a base model for DPO. Provide model.base_model, "
            "an SFT manifest, or a latest-model manifest that points to SFT."
        )

    resolved_adapter_path = (
        adapter_path
        or model_config.get("adapter_path")
        or source_adapter_path
        or (latest_manifest.adapter_path if latest_manifest is not None else None)
    )
    resolved_merged_model_path = (
        merged_model_path
        or model_config.get("merged_model_path")
        or source_merged_path
        or (latest_manifest.merged_export_path if latest_manifest is not None else None)
    )

    quantization_config = _deep_merge(source_quantization, dict(merged.get("quantization", {})))
    if reference_strategy == "merged":
        if not resolved_merged_model_path:
            raise ValueError(
                "DPO merged mode requires a merged model path. Provide model.merged_model_path "
                "or promote an SFT manifest that includes merged_export_path."
            )
        if quantization_config.get("enabled", False):
            raise ValueError(
                "Merged-model DPO is only supported here without 4-bit quantization. "
                "Disable quantization or use the default adapter strategy."
            )
        model_name_or_path = str(_resolve_repo_path(resolved_repo_root, resolved_merged_model_path))
        resolved_adapter_path = None
        resolved_reference_adapter_path = None
    else:
        if not resolved_adapter_path:
            raise ValueError(
                "Adapter-first DPO requires an SFT adapter path. Provide model.adapter_path, "
                "a source SFT manifest, or a latest-model manifest that points to SFT."
            )
        model_name_or_path = resolved_base_model
        resolved_adapter_path = str(_resolve_repo_path(resolved_repo_root, resolved_adapter_path))
        resolved_reference_adapter_path = resolved_adapter_path

    resolved_preference_manifest = (
        _resolve_repo_path(resolved_repo_root, preference_manifest)
        or _resolve_repo_path(resolved_repo_root, training_config.get("preference_manifest"))
    )
    resolved_eval_preference_manifest = _resolve_repo_path(
        resolved_repo_root,
        training_config.get("eval_preference_manifest"),
    )

    return DPOResolvedConfig(
        config_path=resolved_config_path,
        profile_name=profile_name,
        latest_model_manifest_path=latest_model_manifest_path,
        latest_model_manifest=latest_manifest,
        source_sft_manifest_path=source_sft_manifest_path,
        source_sft_manifest=source_sft_manifest_payload,
        base_model=resolved_base_model,
        model_name_or_path=model_name_or_path,
        adapter_path=resolved_adapter_path,
        reference_adapter_path=resolved_reference_adapter_path if reference_strategy == "adapter" else None,
        merged_model_path=str(_resolve_repo_path(resolved_repo_root, resolved_merged_model_path))
        if resolved_merged_model_path
        else None,
        reference_strategy=reference_strategy,
        revision=model_config.get("revision"),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        torch_dtype=model_config.get("torch_dtype", "auto"),
        device_map=model_config.get("device_map"),
        quantization=quantization_config,
        preference_manifest=resolved_preference_manifest,
        eval_preference_manifest=resolved_eval_preference_manifest,
        train_sample_limit=training_config.get("train_sample_limit"),
        eval_sample_limit=training_config.get("eval_sample_limit"),
        training=training_config,
        artifacts=dict(merged.get("artifacts", {})),
        raw_config=raw_config,
    )


def resolve_dpo_output_paths(
    context: RuntimeContext,
    run_name: str,
    artifact_config: dict[str, Any] | None = None,
) -> DPOOutputPaths:
    """Build the runtime paths used by the DPO stage."""

    artifact_names = artifact_config or {}
    checkpoint_root = (context.checkpoints_dir / "dpo" / run_name).resolve()
    adapter_dir = (checkpoint_root / "adapter").resolve()
    logs_dir = (context.logs_dir / "dpo" / run_name).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    reward_plot_paths = {
        metric_key: (
            context.plots_dir
            / artifact_names.get(
                f"{metric_key.replace('/', '_')}_curve_filename",
                f"{{run_name}}_dpo_{metric_key.replace('/', '_')}_curve.png",
            ).format(run_name=run_name)
        ).resolve()
        for metric_key, _, _ in DPO_REWARD_METRIC_SPECS
    }

    return DPOOutputPaths(
        checkpoint_root=checkpoint_root,
        adapter_dir=adapter_dir,
        logs_dir=logs_dir,
        trainer_state_path=(logs_dir / "trainer_state.json").resolve(),
        summary_path=(
            context.metrics_dir
            / artifact_names.get("summary_filename", "{run_name}_dpo_summary.json").format(run_name=run_name)
        ).resolve(),
        history_path=(
            context.metrics_dir
            / artifact_names.get("history_filename", "{run_name}_dpo_history.json").format(run_name=run_name)
        ).resolve(),
        checkpoint_manifest_path=(
            checkpoint_root
            / artifact_names.get("checkpoint_manifest_filename", "{run_name}_dpo_manifest.json").format(
                run_name=run_name
            )
        ).resolve(),
        loss_curve_path=(
            context.plots_dir
            / artifact_names.get("loss_curve_filename", "{run_name}_dpo_loss_curve.png").format(run_name=run_name)
        ).resolve(),
        eval_loss_curve_path=(
            context.plots_dir
            / artifact_names.get("eval_loss_curve_filename", "{run_name}_dpo_eval_loss_curve.png").format(
                run_name=run_name
            )
        ).resolve(),
        reward_plot_paths=reward_plot_paths,
    )


def _sample_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    return rows[: int(limit)]


def load_dpo_preference_records(
    path: str | Path | None,
    *,
    sample_limit: int | None = None,
    required: bool = True,
) -> list[dict[str, str]]:
    """Load prompt/chosen/rejected preference rows for DPO."""

    if path is None:
        if required:
            raise ValueError(
                "DPO training requires a preference manifest. Provide training.preference_manifest "
                "or pass --preference-manifest."
            )
        return []

    resolved = Path(path).resolve()
    if not resolved.exists():
        if required:
            raise FileNotFoundError(f"DPO preference manifest does not exist: {resolved}")
        return []

    rows = _sample_rows(read_jsonl(resolved), sample_limit)
    normalized_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        prompt = str(row.get("prompt", ""))
        chosen = str(row.get("chosen", ""))
        rejected = str(row.get("rejected", ""))
        if not prompt or not chosen or not rejected:
            raise ValueError(
                f"DPO preference row {index} in {resolved} must include non-empty prompt, chosen, and rejected."
            )
        normalized_rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return normalized_rows


def _load_training_stack() -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required for DPO training. Run inside Colab after installing requirements-colab.txt."
        ) from exc

    try:
        from datasets import Dataset
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TRL, PEFT, transformers, and datasets are required for DPO training. "
            "Use requirements-colab.txt for the Colab runtime."
        ) from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "PeftModel": PeftModel,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DPOConfig": DPOConfig,
        "DPOTrainer": DPOTrainer,
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


def _build_quantization_config(config: DPOResolvedConfig, modules: dict[str, Any], compute_dtype: Any) -> Any | None:
    if not config.quantization.get("enabled", False):
        return None
    return modules["BitsAndBytesConfig"](
        load_in_4bit=bool(config.quantization.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(config.quantization.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(config.quantization.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _load_base_model(
    *,
    config: DPOResolvedConfig,
    modules: dict[str, Any],
    compute_dtype: Any,
    quantization_config: Any | None,
    prepare_for_training: bool,
) -> Any:
    torch_module = modules["torch"]
    device_map = config.device_map
    if device_map is None and torch_module.cuda.is_available():
        device_map = "auto"

    model = modules["AutoModelForCausalLM"].from_pretrained(
        config.model_name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )
    if prepare_for_training:
        if config.training.get("gradient_checkpointing", False) and hasattr(model, "config"):
            model.config.use_cache = False
        if quantization_config is not None:
            model = modules["prepare_model_for_kbit_training"](
                model,
                use_gradient_checkpointing=bool(config.training.get("gradient_checkpointing", True)),
            )
    return model


def _build_dpo_args_kwargs(
    *,
    config: DPOResolvedConfig,
    artifacts: DPOOutputPaths,
    run_name: str,
    modules: dict[str, Any],
    compute_dtype: Any,
) -> dict[str, Any]:
    torch_module = modules["torch"]
    max_prompt_length = int(config.training.get("max_prompt_length", 1024))
    max_completion_length = int(config.training.get("max_completion_length", 256))
    return _compact_mapping(
        {
            "output_dir": str(artifacts.checkpoint_root),
            "logging_dir": str(artifacts.logs_dir / "tensorboard"),
            "run_name": run_name,
            "learning_rate": float(config.training.get("learning_rate", 5.0e-6)),
            "lr_scheduler_type": config.training.get("lr_scheduler_type", "cosine"),
            "warmup_ratio": float(config.training.get("warmup_ratio", 0.05)),
            "weight_decay": float(config.training.get("weight_decay", 0.0)),
            "optim": config.training.get("optim", "paged_adamw_32bit"),
            "gradient_checkpointing": bool(config.training.get("gradient_checkpointing", True)),
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
            "max_length": max_prompt_length + max_completion_length,
            "precompute_ref_log_probs": bool(config.training.get("precompute_ref_log_probs", False)),
            "loss_type": [str(config.training.get("loss_type", "sigmoid"))],
            "beta": float(config.training.get("beta", 0.1)),
            "bf16": compute_dtype == torch_module.bfloat16,
            "fp16": compute_dtype == torch_module.float16,
        }
    )


def build_trainer_bundle(
    *,
    config: DPOResolvedConfig,
    artifacts: DPOOutputPaths,
    run_name: str,
    train_records: list[dict[str, str]],
    eval_records: list[dict[str, str]],
) -> TrainerBundle:
    """Instantiate the tokenizer, policy, reference model, and TRL DPO trainer."""

    modules = _load_training_stack()
    compute_dtype = _resolve_compute_dtype(modules["torch"], config.torch_dtype)
    quantization_config = _build_quantization_config(config, modules, compute_dtype)

    tokenizer = modules["AutoTokenizer"].from_pretrained(
        config.base_model if config.reference_strategy == "adapter" else config.model_name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if config.reference_strategy == "adapter":
        policy_base_model = _load_base_model(
            config=config,
            modules=modules,
            compute_dtype=compute_dtype,
            quantization_config=quantization_config,
            prepare_for_training=True,
        )
        policy_model = modules["PeftModel"].from_pretrained(
            policy_base_model,
            config.adapter_path,
            is_trainable=True,
        )

        ref_base_model = modules["AutoModelForCausalLM"].from_pretrained(
            config.base_model,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            device_map=config.device_map or ("auto" if modules["torch"].cuda.is_available() else None),
        )
        ref_model = modules["PeftModel"].from_pretrained(
            ref_base_model,
            config.reference_adapter_path,
            is_trainable=False,
        )
        ref_model.eval()
        if hasattr(policy_model, "config"):
            policy_model.config.use_cache = False
    else:
        policy_model = _load_base_model(
            config=config,
            modules=modules,
            compute_dtype=compute_dtype,
            quantization_config=None,
            prepare_for_training=False,
        )
        ref_model = modules["AutoModelForCausalLM"].from_pretrained(
            config.model_name_or_path,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=compute_dtype,
            device_map=config.device_map or ("auto" if modules["torch"].cuda.is_available() else None),
        )
        ref_model.eval()

    dpo_args = modules["DPOConfig"](
        **_build_dpo_args_kwargs(
            config=config,
            artifacts=artifacts,
            run_name=run_name,
            modules=modules,
            compute_dtype=compute_dtype,
        )
    )

    train_dataset = modules["Dataset"].from_list(train_records)
    eval_dataset = modules["Dataset"].from_list(eval_records) if eval_records else None
    trainer_kwargs = {
        "model": policy_model,
        "ref_model": ref_model,
        "args": dpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    try:
        trainer = modules["DPOTrainer"](
            processing_class=tokenizer,
            **trainer_kwargs,
        )
    except TypeError:
        trainer = modules["DPOTrainer"](
            tokenizer=tokenizer,
            **trainer_kwargs,
        )
    return TrainerBundle(
        trainer=trainer,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )


def write_checkpoint_manifest(
    *,
    config: DPOResolvedConfig,
    artifacts: DPOOutputPaths,
    run_name: str,
    status: str,
    train_record_count: int | None,
    eval_record_count: int | None,
    history_artifacts: dict[str, Any] | None = None,
    train_metrics: dict[str, Any] | None = None,
) -> Path:
    """Persist small checkpoint metadata for the completed DPO run."""

    payload = {
        "stage": "dpo",
        "run_name": run_name,
        "status": status,
        "profile": config.profile_name,
        "config_path": str(config.config_path),
        "source_sft_manifest_path": str(config.source_sft_manifest_path) if config.source_sft_manifest_path else None,
        "base_model": config.base_model,
        "model_name_or_path": config.model_name_or_path,
        "reference_strategy": config.reference_strategy,
        "source_adapter_path": config.adapter_path,
        "reference_adapter_path": config.reference_adapter_path,
        "merged_model_path": config.merged_model_path,
        "adapter_path": str(artifacts.adapter_dir),
        "checkpoint_root": str(artifacts.checkpoint_root),
        "logs_dir": str(artifacts.logs_dir),
        "trainer_state_path": str(artifacts.trainer_state_path),
        "preference_manifest": str(config.preference_manifest) if config.preference_manifest else None,
        "eval_preference_manifest": str(config.eval_preference_manifest) if config.eval_preference_manifest else None,
        "train_record_count": train_record_count,
        "eval_record_count": eval_record_count,
        "quantization": config.quantization,
        "training": config.training,
        "history_artifacts": history_artifacts or {},
        "train_metrics": train_metrics or {},
    }
    return write_json(artifacts.checkpoint_manifest_path, payload)


def write_dpo_summary(
    *,
    context: RuntimeContext,
    config: DPOResolvedConfig,
    artifacts: DPOOutputPaths,
    run_name: str,
    status: str,
    train_record_count: int | None,
    eval_record_count: int | None,
    history_artifacts: dict[str, Any] | None = None,
    train_metrics: dict[str, Any] | None = None,
) -> Path:
    """Write the DPO run summary used by the review notebook and docs."""

    history_payload = history_artifacts or {}
    resolved_loss_curve_path = history_payload.get("loss_curve_path")
    resolved_eval_loss_curve_path = history_payload.get("eval_loss_curve_path")
    if status != "completed":
        resolved_loss_curve_path = resolved_loss_curve_path or str(artifacts.loss_curve_path)
        resolved_eval_loss_curve_path = resolved_eval_loss_curve_path or str(artifacts.eval_loss_curve_path)

    payload = {
        "stage": "dpo",
        "run_name": run_name,
        "status": status,
        "profile": config.profile_name,
        "config_path": str(config.config_path),
        "runtime_root": str(context.runtime_root),
        "source_sft_manifest_path": str(config.source_sft_manifest_path) if config.source_sft_manifest_path else None,
        "base_model": config.base_model,
        "model_name_or_path": config.model_name_or_path,
        "reference_strategy": config.reference_strategy,
        "source_adapter_path": config.adapter_path,
        "reference_adapter_path": config.reference_adapter_path,
        "merged_model_path": config.merged_model_path,
        "adapter_path": str(artifacts.adapter_dir),
        "checkpoint_root": str(artifacts.checkpoint_root),
        "logs_dir": str(artifacts.logs_dir),
        "summary_path": str(artifacts.summary_path),
        "history_path": str(artifacts.history_path),
        "loss_curve_path": resolved_loss_curve_path,
        "eval_loss_curve_path": resolved_eval_loss_curve_path,
        "reward_plot_paths": {key: str(path) for key, path in artifacts.reward_plot_paths.items()},
        "preference_manifest": str(config.preference_manifest) if config.preference_manifest else None,
        "eval_preference_manifest": str(config.eval_preference_manifest) if config.eval_preference_manifest else None,
        "train_record_count": train_record_count,
        "eval_record_count": eval_record_count,
        "quantization": config.quantization,
        "training": config.training,
        "history_artifacts": history_payload,
        "train_metrics": train_metrics or {},
    }
    return write_json(artifacts.summary_path, payload)


def save_trainer_state(trainer: Any, artifacts: DPOOutputPaths) -> Path | None:
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


def save_training_artifacts(
    *,
    trainer: Any,
    config: DPOResolvedConfig,
    artifacts: DPOOutputPaths,
    run_name: str,
    context: RuntimeContext,
    train_record_count: int | None,
    eval_record_count: int | None,
    train_metrics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Persist history JSON, plots, summary, and checkpoint metadata after DPO."""

    extra_plot_specs = [
        PlotSpec(metric_key=metric_key, output_path=artifacts.reward_plot_paths[metric_key], title=title, color=color)
        for metric_key, title, color in DPO_REWARD_METRIC_SPECS
    ]
    history_artifacts = write_training_history_and_plots(
        log_history=collect_log_history(trainer),
        history_path=artifacts.history_path,
        loss_curve_path=artifacts.loss_curve_path,
        eval_loss_curve_path=artifacts.eval_loss_curve_path,
        tracked_scalar_keys=[metric_key for metric_key, _, _ in DPO_REWARD_METRIC_SPECS],
        extra_plot_specs=extra_plot_specs,
        loss_curve_title="DPO Training Loss",
        eval_loss_curve_title="DPO Evaluation Loss",
    )
    merged_metrics = dict(train_metrics or {})
    if hasattr(trainer, "state") and getattr(trainer.state, "best_metric", None) is not None:
        merged_metrics["best_metric"] = trainer.state.best_metric

    summary_path = write_dpo_summary(
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
    config: DPOResolvedConfig,
    artifacts: DPOOutputPaths,
    run_name: str,
    train_record_count: int | None,
    eval_record_count: int | None,
) -> tuple[Path, Path]:
    """Persist the resolved DPO run contract without importing the heavy training stack."""

    summary_path = write_dpo_summary(
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


def mirror_dpo_artifacts(
    *,
    repo_root: str | Path,
    artifacts: DPOOutputPaths,
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
        plot_sources = [
            artifacts.loss_curve_path,
            artifacts.eval_loss_curve_path,
            *artifacts.reward_plot_paths.values(),
        ]
        for source_path in plot_sources:
            if source_path.exists():
                destination = targets["plots"] / source_path.name
                mirrored_path = mirror_small_artifact(source_path, destination)
                mirrored["plots"].append(str(mirrored_path))

    if mirror_checkpoint_manifest and artifacts.checkpoint_manifest_path.exists():
        destination = targets["checkpoints"] / artifacts.checkpoint_manifest_path.name
        mirrored_path = mirror_small_artifact(artifacts.checkpoint_manifest_path, destination)
        mirrored["checkpoints"].append(str(mirrored_path))

    return mirrored


def promote_latest_dpo_model(
    *,
    repo_root: str | Path,
    config: DPOResolvedConfig,
    artifacts: DPOOutputPaths,
    mirrored_artifacts: dict[str, list[str]],
) -> Path:
    """Update the repo-side latest-model manifest for the completed DPO adapter."""

    metrics_paths = mirrored_artifacts.get("metrics") or [str(artifacts.summary_path), str(artifacts.history_path)]
    report_paths = mirrored_artifacts.get("checkpoints") or [str(artifacts.checkpoint_manifest_path)]
    manifest = LatestModelManifest(
        stage="dpo",
        status="ready",
        base_model=config.base_model,
        adapter_path=str(artifacts.adapter_dir),
        schema_version="1.0.0",
        config_paths=[str(config.config_path)],
        metrics_paths=metrics_paths,
        report_paths=report_paths,
    )
    return save_latest_model_manifest(repo_root, manifest)
