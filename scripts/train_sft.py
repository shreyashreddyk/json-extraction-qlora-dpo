"""Train the first supervised fine-tuning stage with QLoRA, TRL, and PEFT."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.runtime import (
    format_runtime_backend_summary,
    format_runtime_summary,
    resolve_runtime_context,
)
from json_ft.sft import (
    build_trainer_bundle,
    load_sft_eval_records,
    load_sft_training_records,
    mirror_sft_artifacts,
    promote_latest_sft_model,
    resolve_sft_config,
    resolve_sft_output_paths,
    save_training_artifacts,
    save_trainer_state,
    write_dry_run_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    parser.add_argument(
        "--profile",
        choices=("dev", "full", "colab_full", "large_gpu_full"),
        default="full",
    )
    parser.add_argument("--run-name", default="sft-qwen2.5-1.5b-qlora-v1")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--train-sample-percent", type=float, default=None)
    parser.add_argument("--eval-sample-percent", type=float, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mirror-metrics-to-repo", action="store_true")
    parser.add_argument("--mirror-plots-to-repo", action="store_true")
    parser.add_argument("--mirror-checkpoint-manifest-to-repo", action="store_true")
    parser.add_argument("--promote-latest", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    context = resolve_runtime_context(
        repo_root=repo_root,
        stage="sft",
        run_name=args.run_name,
        runtime_root=args.runtime_root,
    )
    config = resolve_sft_config(
        config_path=args.config,
        repo_root=repo_root,
        profile_name=args.profile,
        training_overrides={
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
        },
        data_overrides={
            "train_sample_percent": args.train_sample_percent,
            "eval_sample_percent": args.eval_sample_percent,
            "sample_seed": args.sample_seed,
        },
    )
    artifacts = resolve_sft_output_paths(context, args.run_name, artifact_config=config.artifacts)
    train_records, train_subset_metadata = load_sft_training_records(config)
    eval_records, eval_subset_metadata = load_sft_eval_records(config)

    print("SFT training")
    print(f"Config: {config.config_path}")
    print(f"Profile: {config.profile_name}")
    print(f"Run name: {args.run_name}")
    print(f"Base model: {config.model_name_or_path}")
    print(f"Train manifest: {config.train_manifest}")
    print(f"Eval manifest: {config.eval_manifest}")
    print(f"Dataset build summary: {config.build_summary_path}")
    print(f"Dataset composition summary: {config.composition_summary_path}")
    print(f"Train rows: {len(train_records)}")
    print(f"Eval rows: {len(eval_records)}")
    print(f"Train subset: {train_subset_metadata.to_dict()}")
    print(f"Eval subset: {eval_subset_metadata.to_dict()}")
    print(f"Adapter output: {artifacts.adapter_dir}")
    print(format_runtime_summary(context))
    print(
        format_runtime_backend_summary(
            explicit_device_map=config.raw_config.get("model", {}).get("device_map"),
            cuda_default="auto",
        )
    )

    if args.dry_run:
        summary_path, checkpoint_manifest_path = write_dry_run_artifacts(
            context=context,
            config=config,
            artifacts=artifacts,
            run_name=args.run_name,
            train_record_count=len(train_records),
            eval_record_count=len(eval_records),
            train_subset_metadata=train_subset_metadata,
            eval_subset_metadata=eval_subset_metadata,
        )
        print("Dry run complete. The trainer stack was not imported.")
        print(f"Summary artifact: {summary_path}")
        print(f"Checkpoint manifest: {checkpoint_manifest_path}")
        return 0

    # The heavy training stack is imported only after dry-run validation so local
    # boot checks stay fast and do not require Colab-only GPU dependencies.
    bundle = build_trainer_bundle(
        config=config,
        artifacts=artifacts,
        run_name=args.run_name,
        train_records=train_records,
        eval_records=eval_records,
        train_subset_metadata=train_subset_metadata,
        eval_subset_metadata=eval_subset_metadata,
    )
    trainer = bundle.trainer
    train_result = trainer.train()

    if hasattr(trainer, "save_model"):
        trainer.save_model(str(artifacts.adapter_dir))
    elif hasattr(bundle.model, "save_pretrained"):
        bundle.model.save_pretrained(str(artifacts.adapter_dir))
    if hasattr(bundle.tokenizer, "save_pretrained"):
        bundle.tokenizer.save_pretrained(str(artifacts.adapter_dir))

    save_trainer_state(trainer, artifacts)
    history_artifacts, summary_path, checkpoint_manifest_path = save_training_artifacts(
        trainer=trainer,
        config=config,
        artifacts=artifacts,
        run_name=args.run_name,
        context=context,
        train_record_count=len(train_records),
        eval_record_count=len(eval_records),
        train_metrics=dict(getattr(train_result, "metrics", {}) or {}),
        dataset_telemetry=bundle.dataset_telemetry,
    )

    # The full adapter checkpoints remain in runtime storage because they are too
    # large for repo mirroring. Only small metadata, logs, and plots come back.
    mirrored = mirror_sft_artifacts(
        repo_root=repo_root,
        artifacts=artifacts,
        mirror_metrics=args.mirror_metrics_to_repo,
        mirror_plots=args.mirror_plots_to_repo,
        mirror_checkpoint_manifest=args.mirror_checkpoint_manifest_to_repo,
    )

    print(f"Summary artifact: {summary_path}")
    print(f"Checkpoint manifest: {checkpoint_manifest_path}")
    if history_artifacts.get("loss_curve_path"):
        print(f"Loss curve: {history_artifacts['loss_curve_path']}")
    if history_artifacts.get("eval_loss_curve_path"):
        print(f"Eval loss curve: {history_artifacts['eval_loss_curve_path']}")
    extra_plot_paths = history_artifacts.get("extra_plot_paths", {})
    if extra_plot_paths:
        print(f"Additional SFT curves: {extra_plot_paths}")
    if any(mirrored.values()):
        print(f"Mirrored small artifacts: {mirrored}")

    if args.promote_latest:
        latest_path = promote_latest_sft_model(
            repo_root=repo_root,
            config=config,
            artifacts=artifacts,
            mirrored_artifacts=mirrored,
        )
        print(f"Updated latest-model manifest: {latest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
