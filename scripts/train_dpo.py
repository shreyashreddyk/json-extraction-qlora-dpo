"""Train the second-stage DPO run on top of the existing SFT stage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.dpo import (
    build_trainer_bundle,
    load_dpo_preference_records,
    mirror_dpo_artifacts,
    promote_latest_dpo_model,
    resolve_dpo_config,
    resolve_dpo_output_paths,
    save_trainer_state,
    save_training_artifacts,
    write_dry_run_artifacts,
)
from json_ft.runtime import (
    format_runtime_backend_summary,
    format_runtime_summary,
    resolve_runtime_context,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dpo.yaml"))
    parser.add_argument(
        "--profile",
        choices=("dev", "full", "colab_full", "large_gpu_full"),
        default="full",
    )
    parser.add_argument("--run-name", default="dpo-qwen2.5-1.5b-v1")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--preference-manifest", type=Path, default=None)
    parser.add_argument("--source-sft-manifest", type=Path, default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--merged-model-path", default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
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
        stage="dpo",
        run_name=args.run_name,
        runtime_root=args.runtime_root,
    )
    config = resolve_dpo_config(
        config_path=args.config,
        repo_root=repo_root,
        profile_name=args.profile,
        preference_manifest=args.preference_manifest,
        source_sft_manifest=args.source_sft_manifest,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        merged_model_path=args.merged_model_path,
        training_overrides={
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "train_sample_percent": args.train_sample_percent,
            "eval_sample_percent": args.eval_sample_percent,
            "sample_seed": args.sample_seed,
        },
    )
    artifacts = resolve_dpo_output_paths(context, args.run_name, artifact_config=config.artifacts)
    train_records, train_subset_metadata = load_dpo_preference_records(
        config.preference_manifest,
        sample_limit=config.train_sample_limit,
        sample_percent=config.train_sample_percent,
        sample_seed=config.sample_seed,
        required=not args.dry_run,
    )
    eval_records, eval_subset_metadata = load_dpo_preference_records(
        config.eval_preference_manifest,
        sample_limit=config.eval_sample_limit,
        sample_percent=config.eval_sample_percent,
        sample_seed=config.sample_seed,
        required=False,
    )

    print("DPO training")
    print(f"Config: {config.config_path}")
    print(f"Profile: {config.profile_name}")
    print(f"Run name: {args.run_name}")
    print(f"Base model: {config.base_model}")
    print(f"Policy init: {config.model_name_or_path}")
    print(f"Reference strategy: {config.reference_strategy}")
    print(f"Source adapter path: {config.adapter_path or '<none>'}")
    print(f"Preference manifest: {config.preference_manifest or '<unresolved>'}")
    print(f"Eval preference manifest: {config.eval_preference_manifest or '<none>'}")
    print(f"Dataset build summary: {config.build_summary_path}")
    print(f"Dataset composition summary: {config.composition_summary_path}")
    print(f"Pair quality gates: {config.quality_gates}")
    print(f"Train rows: {len(train_records) if train_records else 0}")
    print(f"Eval rows: {len(eval_records)}")
    print(f"Train subset: {train_subset_metadata.to_dict()}")
    print(f"Eval subset: {eval_subset_metadata.to_dict()}")
    print(f"Adapter output: {artifacts.adapter_dir}")
    print(format_runtime_summary(context))
    print(
        format_runtime_backend_summary(
            explicit_device_map=config.device_map,
            cuda_default="auto",
        )
    )

    if args.dry_run:
        summary_path, checkpoint_manifest_path = write_dry_run_artifacts(
            context=context,
            config=config,
            artifacts=artifacts,
            run_name=args.run_name,
            train_record_count=len(train_records) if train_records else None,
            eval_record_count=len(eval_records) if eval_records else 0,
            train_subset_metadata=train_subset_metadata,
            eval_subset_metadata=eval_subset_metadata,
        )
        print("Dry run complete. The trainer stack was not imported.")
        print(f"Summary artifact: {summary_path}")
        print(f"Checkpoint manifest: {checkpoint_manifest_path}")
        return 0

    bundle = build_trainer_bundle(
        config=config,
        artifacts=artifacts,
        run_name=args.run_name,
        train_records=train_records,
        eval_records=eval_records,
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
        train_subset_metadata=train_subset_metadata,
        eval_subset_metadata=eval_subset_metadata,
        train_metrics=dict(getattr(train_result, "metrics", {}) or {}),
    )

    mirrored = mirror_dpo_artifacts(
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
        print(f"Reward and auxiliary curves: {extra_plot_paths}")
    if any(mirrored.values()):
        print(f"Mirrored small artifacts: {mirrored}")

    if args.promote_latest:
        latest_path = promote_latest_dpo_model(
            repo_root=repo_root,
            config=config,
            artifacts=artifacts,
            mirrored_artifacts=mirrored,
        )
        print(f"Updated latest-model manifest: {latest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
