"""Build deterministic natural and stress benchmark promptsets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.benchmarking import build_benchmark_promptsets, resolve_serving_target
from json_ft.runtime import format_runtime_summary, resolve_runtime_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--run-name", default="vllm-benchmark-lab")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--target-kind", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--merged-model-path", default=None)
    parser.add_argument("--latest-model-manifest", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    target, config = resolve_serving_target(
        config_path=args.config,
        repo_root=repo_root,
        target_kind=args.target_kind,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        merged_model_path=args.merged_model_path,
        latest_model_manifest_path=args.latest_model_manifest,
    )
    context = resolve_runtime_context(
        repo_root=repo_root,
        stage="benchmark",
        run_name=args.run_name,
        runtime_root=args.runtime_root,
    )
    benchmark_config = dict(config.get("benchmark", {}) or {})
    promptset_config = dict(benchmark_config.get("promptsets", {}) or {})
    dataset_path = Path(
        promptset_config.get("dataset_path")
        or benchmark_config.get("dataset_path")
        or "data/manifests/support_tickets_eval_manifest.jsonl"
    )
    if not dataset_path.is_absolute():
        dataset_path = (repo_root / dataset_path).resolve()

    result = build_benchmark_promptsets(
        dataset_path=dataset_path,
        target=target,
        promptset_config=promptset_config,
        output_dir=context.run_dir / "promptsets",
    )
    print("Built benchmark promptsets")
    print(f"Config: {args.config}")
    print(f"Target kind: {target.target_kind}")
    print(f"Dataset path: {dataset_path}")
    print(format_runtime_summary(context))
    print(f"Promptset manifest: {result['manifest_path']}")
    print(f"Natural prompt rows: {len(result['natural_rows'])}")
    print(f"Stress prompt rows: {len(result['stress_rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
