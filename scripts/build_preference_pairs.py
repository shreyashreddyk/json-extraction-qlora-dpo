"""Build task-specific chosen/rejected preference pairs for DPO training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.artifacts import mirror_small_artifact
from json_ft.inference import build_inference_backend
from json_ft.preference import (
    build_preference_run,
    load_preference_samples,
    resolve_preference_config,
    resolve_preference_output_paths,
    write_preference_artifacts,
)
from json_ft.runtime import (
    format_runtime_backend_summary,
    format_runtime_summary,
    resolve_runtime_context,
)
from json_ft.schemas import build_support_ticket_schema
from json_ft.utils import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dpo.yaml"))
    parser.add_argument(
        "--profile",
        choices=("dev", "full", "colab_full", "large_gpu_full"),
        default="full",
    )
    parser.add_argument("--run-name", default="pref-support-tickets-dpo-v1")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--source-format", choices=("json_extraction", "nemotron_sft"), default=None)
    parser.add_argument("--source-split", choices=("train", "eval"), default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--sample-percent", type=float, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mirror-pairs-to-repo", action="store_true")
    parser.add_argument("--mirror-audit-to-repo", action="store_true")
    parser.add_argument("--mirror-summary-to-repo", action="store_true")
    return parser


def _mirror_preference_artifacts(
    *,
    repo_root: Path,
    paths: dict[str, Path],
    mirror_pairs: bool,
    mirror_audit: bool,
    mirror_summary: bool,
) -> dict[str, str]:
    mirrored: dict[str, str] = {}
    if mirror_pairs:
        destination = repo_root / "artifacts" / "reports" / paths["pairs"].name
        mirrored["pairs"] = str(mirror_small_artifact(paths["pairs"], destination))
    if mirror_audit:
        destination = repo_root / "artifacts" / "reports" / paths["audit"].name
        mirrored["audit"] = str(mirror_small_artifact(paths["audit"], destination))
    if mirror_summary:
        destination = repo_root / "artifacts" / "metrics" / paths["summary"].name
        mirrored["summary"] = str(mirror_small_artifact(paths["summary"], destination))
    return mirrored


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    context = resolve_runtime_context(
        repo_root=repo_root,
        stage="preferences",
        run_name=args.run_name,
        runtime_root=args.runtime_root,
    )
    config = resolve_preference_config(
        config_path=args.config,
        repo_root=repo_root,
        profile_name=args.profile,
        input_path=args.input_path,
        source_format=args.source_format,
        source_split=args.source_split,
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        inference_batch_size=args.inference_batch_size,
        sample_percent=args.sample_percent,
        sample_seed=args.sample_seed,
    )
    output_paths = resolve_preference_output_paths(
        output_dir=context.run_dir,
        run_name=args.run_name,
        artifact_names=config.artifact_names,
    )
    samples, source_subset_metadata = load_preference_samples(
        input_path=config.input_path,
        source_format=config.source_format,
        source_split=config.source_split,
        sample_limit=config.sample_limit,
        sample_percent=config.sample_percent,
        sample_seed=config.sample_seed,
    )

    print("Preference pair generation")
    print(f"Config: {config.config_path}")
    print(f"Profile: {config.profile_name}")
    print(f"Run name: {args.run_name}")
    print(f"Input path: {config.input_path}")
    print(f"Source split: {config.source_split}")
    print(f"Model name or path: {config.model_name_or_path}")
    print(f"Adapter path: {config.adapter_path or '<none>'}")
    print(f"Prompt source: {config.prompt_source}")
    print(f"Dataset build summary: {config.build_summary_path}")
    print(f"Dataset composition summary: {config.composition_summary_path}")
    print(f"Quality gates: {config.quality_gates}")
    print(f"Source rows: {len(samples)}")
    print(f"Source subset: {source_subset_metadata.to_dict()}")
    print(f"Candidate count per prompt: {config.candidate_count}")
    print(f"Inference batch size: {config.inference_batch_size}")
    print(format_runtime_summary(context))
    print(
        format_runtime_backend_summary(
            explicit_device_map=config.device_map,
            cuda_default="cuda",
        )
    )

    if args.dry_run:
        summary_path = write_json(
            output_paths.summary_path,
            {
                "status": "dry_run_ready",
                "config_path": str(config.config_path),
                "profile": config.profile_name,
                "run_name": args.run_name,
                "input_path": str(config.input_path),
                "source_split": config.source_split,
                "source_row_count": len(samples),
                "model_name_or_path": config.model_name_or_path,
                "adapter_path": config.adapter_path,
                "candidate_count": config.candidate_count,
                "inference_batch_size": config.inference_batch_size,
                "subset_selection": source_subset_metadata.to_dict(),
                "prompt_source": config.prompt_source,
                "quality_gates": config.quality_gates,
                "output_dir": str(output_paths.output_dir),
                "pairs_path": str(output_paths.pairs_path),
                "audit_path": str(output_paths.audit_path),
                "summary_path": str(output_paths.summary_path),
                "diagnostics_path": str(output_paths.diagnostics_path),
            },
        )
        print("Dry run complete. The model backend was not imported.")
        print(f"Summary artifact: {summary_path}")
        return 0

    backend = build_inference_backend(
        backend="local-transformers",
        model_name_or_path=config.model_name_or_path,
        adapter_path=config.adapter_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        schema=build_support_ticket_schema(),
    )
    pair_rows, audit_rows, summary, diagnostics = build_preference_run(
        samples=samples,
        backend=backend,
        config=config,
        schema=build_support_ticket_schema(),
        source_subset_metadata=source_subset_metadata,
    )
    summary.update(
        {
            "run_name": args.run_name,
            "runtime_root": str(context.runtime_root),
            "output_dir": str(output_paths.output_dir),
        }
    )
    pairs_path, audit_path, summary_path, diagnostics_path, plot_paths = write_preference_artifacts(
        paths=output_paths,
        pair_rows=pair_rows,
        audit_rows=audit_rows,
        summary=summary,
        diagnostics=diagnostics,
    )
    mirrored = _mirror_preference_artifacts(
        repo_root=repo_root,
        paths={"pairs": pairs_path, "audit": audit_path, "summary": summary_path},
        mirror_pairs=args.mirror_pairs_to_repo,
        mirror_audit=args.mirror_audit_to_repo,
        mirror_summary=args.mirror_summary_to_repo,
    )

    print(f"DPO pairs: {pairs_path}")
    print(f"Audit log: {audit_path}")
    print(f"Summary: {summary_path}")
    print(f"Diagnostics: {diagnostics_path}")
    if plot_paths:
        print(f"Preference plots: {plot_paths}")
    print(f"Emitted pairs: {summary['pair_count']} / {summary['source_row_count']}")
    if mirrored:
        print(f"Mirrored small artifacts: {mirrored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
