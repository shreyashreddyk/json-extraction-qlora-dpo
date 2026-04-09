"""Scaffold CLI for DPO training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.manifests import LatestModelManifest, save_latest_model_manifest
from json_ft.runtime import format_runtime_summary, resolve_runtime_context
from json_ft.utils import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dpo.yaml"))
    parser.add_argument("--run-name", default="dpo-scaffold")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--base-model", default="placeholder-base-model")
    parser.add_argument("--schema-version", default="0.1.0")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--promote-latest", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    context = resolve_runtime_context(
        repo_root=repo_root,
        stage="dpo",
        run_name=args.run_name,
        runtime_root=args.runtime_root,
    )
    checkpoint_output = args.output_dir or (context.run_dir / "adapter")
    report_output = args.report_output or (context.reports_dir / f"{args.run_name}_dpo_summary.json")

    print("DPO training scaffold")
    print(f"Config: {args.config}")
    print(f"Run name: {args.run_name}")
    print(f"Output dir: {checkpoint_output}")
    print(format_runtime_summary(context))

    checkpoint_output.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(
        report_output,
        {
            "stage": "dpo",
            "run_name": args.run_name,
            "config": str(args.config),
            "runtime_root": str(context.runtime_root),
            "checkpoint_output": str(checkpoint_output),
            "base_model": args.base_model,
            "schema_version": args.schema_version,
            "status": "scaffold_ready",
            "note": "DPO training is not implemented yet. This file records the intended runtime contract.",
        },
    )
    print(f"Summary artifact: {summary_path}")

    if args.promote_latest:
        manifest = LatestModelManifest(
            stage="dpo",
            status="scaffold_ready",
            base_model=args.base_model,
            adapter_path=args.adapter_path or str(checkpoint_output),
            schema_version=args.schema_version,
            config_paths=[str(args.config)],
            report_paths=[str(summary_path)],
        )
        manifest_path = save_latest_model_manifest(repo_root, manifest)
        print(f"Updated latest-model manifest: {manifest_path}")

    print("TODO: implement preference dataset loading, trainer wiring, and checkpoint metadata capture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

