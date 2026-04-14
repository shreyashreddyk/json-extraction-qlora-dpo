"""Run Colab-native vLLM serving benchmarks for the promoted extraction model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.artifacts import mirror_small_artifact
from json_ft.benchmark_reporting import render_benchmark_report, save_benchmark_bundle
from json_ft.benchmarking import (
    benchmark_checkpoint_paths,
    benchmark_paths,
    compute_benchmark_fingerprint,
    build_benchmark_promptsets,
    build_workload_mix_rows,
    launch_vllm_server,
    load_inference_config,
    load_benchmark_checkpoint_state,
    load_benchmark_step_checkpoints,
    load_checkpointed_benchmark_bundle,
    resolve_serving_target,
    run_benchmark_workload,
    snapshot_metrics_endpoint,
    save_benchmark_checkpoint_state,
    save_benchmark_step_checkpoint,
    _now_utc,
    validate_benchmark_checkpoint_resume,
    stop_vllm_server,
    wait_for_vllm_ready,
    write_csv_rows,
)
from json_ft.runtime import format_runtime_summary, resolve_repo_artifact_targets, resolve_runtime_context
from json_ft.utils import write_json, write_jsonl


def _log(message: str) -> None:
    timestamp = datetime.now(UTC).strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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
    parser.add_argument("--promptset-only", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--mirror-report-to-repo", action="store_true")
    parser.add_argument("--mirror-summary-to-repo", action="store_true")
    return parser


def _merge_server_config(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (overrides or {}).items():
        if value is not None:
            merged[key] = value
    return merged


def _select_tuned_config(rows: list[dict[str, Any]], floor_ratio: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("No config-search rows were produced for tuned config selection.")
    max_rps = max(float(row.get("throughput_rps") or 0.0) for row in rows)
    eligible = [
        row
        for row in rows
        if float(row.get("throughput_rps") or 0.0) >= max_rps * floor_ratio
    ]
    if not eligible:
        eligible = rows

    def short_p99(row: dict[str, Any]) -> float:
        return float(row.get("short_p99_ms") or 1e18)

    ranked = sorted(
        eligible,
        key=lambda row: (
            short_p99(row),
            -float(row.get("throughput_rps") or 0.0),
            int(row.get("max_num_batched_tokens") or 0),
            int(row.get("max_num_seqs") or 0),
        ),
    )
    return ranked[0]


def _experiment_summary_row(
    summary: dict[str, Any],
    *,
    experiment_family: str,
    target_kind: str,
    workload_name: str,
    server_settings: dict[str, Any],
) -> dict[str, Any]:
    buckets = summary.get("bucket_latency_ms", {}) or {}
    return {
        "experiment_family": experiment_family,
        "target_kind": target_kind,
        "workload_name": workload_name,
        "server_config_id": summary.get("server_config_id"),
        "concurrency": summary.get("concurrency"),
        "request_count": summary.get("request_count"),
        "success_rate": summary.get("success_rate"),
        "throughput_rps": summary.get("throughput_rps"),
        "latency_p50_ms": summary.get("latency_p50_ms"),
        "latency_p90_ms": summary.get("latency_p90_ms"),
        "latency_p99_ms": summary.get("latency_p99_ms"),
        "tail_inflation_p99_over_p50": summary.get("tail_inflation_p99_over_p50"),
        "short_p99_ms": (buckets.get("short") or {}).get("p99"),
        "medium_p99_ms": (buckets.get("medium") or {}).get("p99"),
        "long_p99_ms": (buckets.get("long") or {}).get("p99"),
        "max_num_batched_tokens": server_settings.get("max_num_batched_tokens"),
        "max_num_seqs": server_settings.get("max_num_seqs"),
        "gpu_memory_utilization": server_settings.get("gpu_memory_utilization"),
        "max_model_len": server_settings.get("max_model_len"),
        "bucket_latency_ms": summary.get("bucket_latency_ms"),
    }


def _apply_checkpointed_step_payload(
    *,
    payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    correctness_rows: list[dict[str, Any]],
    raw_request_rows: list[dict[str, Any]],
    config_search_rows: list[dict[str, Any]],
    checkpoint_state: dict[str, Any],
    checkpoint_state_path: Path,
) -> None:
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

    step_id = str(payload.get("step_id") or "")
    if step_id:
        completed_steps = checkpoint_state.setdefault("completed_steps", [])
        if step_id not in completed_steps:
            completed_steps.append(step_id)
        step_artifacts = checkpoint_state.setdefault("step_artifacts", {})
        checkpoint_path = payload.get("checkpoint_path")
        if checkpoint_path:
            step_artifacts[step_id] = str(checkpoint_path)
        checkpoint_state["updated_at_utc"] = _now_utc()
        save_benchmark_checkpoint_state(checkpoint_state_path, checkpoint_state)


def _run_server_backed_experiment_group(
    *,
    experiment_family: str,
    workload_plan: list[tuple[str, int, int]],
    target: Any,
    serving_config: dict[str, Any],
    server_dir: Path,
    runtime_root: Path,
    config_path: Path,
    promptsets: dict[str, Any],
    tokenizer: Any,
    generation_config: dict[str, Any],
    budgeting_config: dict[str, Any],
    request_seed: int,
    summary_rows: list[dict[str, Any]],
    raw_request_rows: list[dict[str, Any]],
    correctness_rows: list[dict[str, Any]],
    correctness_sample_size: int,
    snapshot_dir: Path,
    checkpoint_state: dict[str, Any],
    checkpoint_state_path: Path,
    checkpoint_steps_dir: Path,
    completed_step_payloads: dict[str, dict[str, Any]],
    config_search_rows: list[dict[str, Any]],
) -> None:
    from json_ft.benchmarking import build_correctness_summary

    config_id = str(serving_config.get("config_id") or experiment_family)
    process = None
    launch_result = None
    group_has_incomplete_steps = any(
        f"{experiment_family}-{config_id}-{workload_name}-c{concurrency}" not in completed_step_payloads
        for workload_name, _total_count, concurrency in workload_plan
    )
    if not group_has_incomplete_steps:
        _log(f"Skipping completed experiment group {experiment_family} / {config_id}")
        for workload_name, _total_count, concurrency in workload_plan:
            experiment_id = f"{experiment_family}-{config_id}-{workload_name}-c{concurrency}"
            payload = completed_step_payloads[experiment_id]
            _apply_checkpointed_step_payload(
                payload=payload,
                summary_rows=summary_rows,
                correctness_rows=correctness_rows,
                raw_request_rows=raw_request_rows,
                config_search_rows=config_search_rows,
                checkpoint_state=checkpoint_state,
                checkpoint_state_path=checkpoint_state_path,
            )
        return
    try:
        _log(f"Launching vLLM server for {experiment_family} / {config_id}")
        process, launch_result = launch_vllm_server(
            target=target,
            serving_config=serving_config,
            server_dir=server_dir,
            config_id=config_id,
            runtime_root=runtime_root,
            config_path=config_path,
        )
        _log(f"Waiting for server readiness at {launch_result.api_base} expecting {target.request_model_name}")
        health = wait_for_vllm_ready(
            launch_result.api_base,
            expected_model_name=target.request_model_name,
            timeout_seconds=float(serving_config.get("health_timeout_seconds", 240.0)),
        )
        _log(
            "Server ready: "
            f"health={health.get('health_ok')} "
            f"models={health.get('models_ok')} "
            f"metrics={health.get('metrics_ok')}"
        )
        write_json(snapshot_dir / f"{config_id}_health.json", health)
        snapshot_metrics_endpoint(launch_result.api_base, snapshot_dir / f"{config_id}_metrics_before.prom")
        for workload_name, total_count, concurrency in workload_plan:
            experiment_id = f"{experiment_family}-{config_id}-{workload_name}-c{concurrency}"
            checkpointed_payload = completed_step_payloads.get(experiment_id)
            if checkpointed_payload is not None:
                _log(f"Reusing checkpointed workload {experiment_id}")
                _apply_checkpointed_step_payload(
                    payload=checkpointed_payload,
                    summary_rows=summary_rows,
                    correctness_rows=correctness_rows,
                    raw_request_rows=raw_request_rows,
                    config_search_rows=config_search_rows,
                    checkpoint_state=checkpoint_state,
                    checkpoint_state_path=checkpoint_state_path,
                )
                continue
            _log(
                f"Starting workload {experiment_family} / {config_id} / {workload_name} "
                f"(requests={total_count}, concurrency={concurrency})"
            )
            workload_rows = build_workload_mix_rows(
                natural_rows=promptsets["natural_rows"],
                stress_rows=promptsets["stress_rows"],
                mix_name=workload_name,
                total_count=total_count,
                seed=request_seed + concurrency,
            )
            experiment_id = f"{experiment_family}-{config_id}-{workload_name}-c{concurrency}"
            raw_rows, summary = run_benchmark_workload(
                workload_rows=workload_rows,
                target=target,
                api_base=launch_result.api_base,
                tokenizer=tokenizer,
                generation_config=generation_config,
                budgeting_config=budgeting_config,
                concurrency=concurrency,
                experiment_id=experiment_id,
                server_config_id=config_id,
                workload_name=workload_name,
                request_seed=request_seed,
            )
            _log(
                f"Finished workload {experiment_id}: "
                f"success_rate={summary.get('success_rate'):.3f}, "
                f"throughput_rps={summary.get('throughput_rps'):.2f}, "
                f"latency_p99_ms={summary.get('latency_p99_ms')}"
            )
            summary_row = _experiment_summary_row(
                summary,
                experiment_family=experiment_family,
                target_kind=target.target_kind,
                workload_name=workload_name,
                server_settings=serving_config,
            )
            correctness_row = build_correctness_summary(
                raw_rows,
                sample_size=correctness_sample_size,
                seed=request_seed,
                experiment_id=experiment_id,
            )
            step_payload = {
                "checkpoint_version": 1,
                "completed_at_utc": _now_utc(),
                "experiment_family": experiment_family,
                "workload_name": workload_name,
                "server_config_id": config_id,
                "concurrency": concurrency,
                "step_id": experiment_id,
                "summary_row": summary_row,
                "correctness_row": correctness_row,
                "raw_rows": raw_rows,
            }
            checkpoint_path = save_benchmark_step_checkpoint(
                checkpoint_steps_dir,
                step_id=experiment_id,
                payload=step_payload,
            )
            step_payload["checkpoint_path"] = str(checkpoint_path)
            completed_step_payloads[experiment_id] = step_payload
            _apply_checkpointed_step_payload(
                payload=step_payload,
                summary_rows=summary_rows,
                correctness_rows=correctness_rows,
                raw_request_rows=raw_request_rows,
                config_search_rows=config_search_rows,
                checkpoint_state=checkpoint_state,
                checkpoint_state_path=checkpoint_state_path,
            )
        snapshot_metrics_endpoint(launch_result.api_base, snapshot_dir / f"{config_id}_metrics_after.prom")
        _log(f"Completed experiment group {experiment_family} / {config_id}")
    finally:
        if launch_result is not None:
            stop_vllm_server(process, Path(launch_result.pid_path))
        elif process is not None:
            stop_vllm_server(process)


def main(argv: Sequence[str] | None = None) -> int:
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
    paths = benchmark_paths(context.run_dir)
    repo_targets = resolve_repo_artifact_targets(repo_root)

    benchmark_config = dict(config.get("benchmark", {}) or {})
    serving_config = dict(config.get("serving", {}) or {})
    promptset_config = dict(benchmark_config.get("promptsets", {}) or {})
    generation_config = dict(benchmark_config.get("generation", {}) or {})
    budgeting_config = dict(benchmark_config.get("budgeting", {}) or {})
    correctness_config = dict(benchmark_config.get("correctness", {}) or {})
    experiments_config = dict(benchmark_config.get("experiments", {}) or {})
    seed = int(benchmark_config.get("seed", promptset_config.get("seed", 17)))

    dataset_path = Path(
        promptset_config.get("dataset_path")
        or benchmark_config.get("dataset_path")
        or "data/manifests/support_tickets_eval_manifest.jsonl"
    )
    if not dataset_path.is_absolute():
        dataset_path = (repo_root / dataset_path).resolve()

    checkpoint_paths = benchmark_checkpoint_paths(paths.run_dir)
    checkpoint_state_path = checkpoint_paths["state_path"]
    checkpoint_steps_dir = checkpoint_paths["steps_dir"]
    bundle_path = checkpoint_paths["bundle_path"]
    fingerprint = compute_benchmark_fingerprint(
        target=target,
        config=config,
        dataset_path=dataset_path,
    )
    existing_bundle = load_checkpointed_benchmark_bundle(paths.run_dir) if bundle_path.exists() else None
    existing_state = load_benchmark_checkpoint_state(checkpoint_state_path)
    if existing_bundle is not None and existing_bundle.get("fingerprint") == fingerprint:
        _log(f"Found completed bundle for run {args.run_name}; reusing {bundle_path}")
        _log(f"Bundle: {bundle_path}")
        return 0
    validate_benchmark_checkpoint_resume(existing_state, fingerprint)

    checkpoint_state = existing_state or {
        "checkpoint_version": 1,
        "run_name": args.run_name,
        "fingerprint": fingerprint,
        "config_path": str(args.config.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "target": asdict(target),
        "promptset_manifest": None,
        "created_at_utc": _now_utc(),
        "updated_at_utc": _now_utc(),
        "completed_steps": [],
        "step_artifacts": {},
        "complete": False,
        "bundle_path": str(bundle_path.resolve()),
        "checkpoint_dir": str(checkpoint_paths["checkpoint_dir"].resolve()),
        "checkpoint_steps_dir": str(checkpoint_steps_dir.resolve()),
    }
    checkpoint_state.setdefault("fingerprint", fingerprint)
    checkpoint_state.setdefault("completed_steps", [])
    checkpoint_state.setdefault("step_artifacts", {})
    save_benchmark_checkpoint_state(checkpoint_state_path, checkpoint_state)

    _log("Running vLLM benchmark lab")
    _log(f"Config: {args.config}")
    _log(f"Run name: {args.run_name}")
    _log(f"Target kind: {target.target_kind}")
    _log(f"Request model: {target.request_model_name}")
    _log(f"Served model path: {target.served_model_name_or_path}")
    _log(format_runtime_summary(context))

    _log(f"Building promptsets from {dataset_path}")
    promptsets = build_benchmark_promptsets(
        dataset_path=dataset_path,
        target=target,
        promptset_config=promptset_config,
        output_dir=paths.promptsets_dir,
    )
    _log(
        f"Promptsets ready: natural={len(promptsets['natural_rows'])}, stress={len(promptsets['stress_rows'])}"
    )
    checkpoint_state["promptset_manifest"] = promptsets["manifest"]
    checkpoint_state["promptset_manifest_path"] = str(promptsets["manifest_path"])
    checkpoint_state["updated_at_utc"] = _now_utc()
    save_benchmark_checkpoint_state(checkpoint_state_path, checkpoint_state)

    mix_names = {
        "smoke_stratified",
        "natural_short_only",
        "natural_medium_only",
        "natural_long_only",
        "natural_mix_90_10_short_long",
        "natural_mix_70_30_short_long",
        "natural_mix_50_50_short_long",
        "stress_long_only",
        "stress_mix_90_10_natural_short_stress_long",
        "stress_mix_70_30_natural_short_stress_long",
        "stress_mix_50_50_natural_short_stress_long",
    }
    from json_ft.benchmarking import write_workload_mix_artifacts, _load_tokenizer

    write_workload_mix_artifacts(
        mix_names=sorted(mix_names),
        total_count=int(promptset_config.get("workload_sample_size", 240)),
        seed=seed,
        natural_rows=promptsets["natural_rows"],
        stress_rows=promptsets["stress_rows"],
        output_dir=paths.promptsets_dir,
    )

    if args.promptset_only:
        _log(f"Promptsets written to {paths.promptsets_dir}")
        return 0

    tokenizer_name = str(
        promptset_config.get("tokenizer_name_or_path")
        or target.base_model
        or target.served_model_name_or_path
    )
    tokenizer = _load_tokenizer(tokenizer_name)

    summary_rows: list[dict[str, Any]] = []
    correctness_rows: list[dict[str, Any]] = []
    config_search_rows: list[dict[str, Any]] = []
    raw_request_rows: list[dict[str, Any]] = []
    completed_step_payloads = {
        str(payload.get("step_id")): payload
        for payload in load_benchmark_step_checkpoints(checkpoint_steps_dir)
        if payload.get("step_id")
    }

    smoke_config = dict(experiments_config.get("smoke_single_stage", {}) or {})
    if smoke_config.get("enabled", True):
        _log("Starting smoke_single_stage experiment family")
        smoke_plan = [
            ("smoke_stratified", int(smoke_config.get("sample_size", 32)), int(concurrency))
            for concurrency in smoke_config.get("concurrency_levels", [1, 4])
        ]
        _run_server_backed_experiment_group(
            experiment_family="smoke_single_stage",
            workload_plan=smoke_plan,
            target=target,
            serving_config=_merge_server_config(serving_config, {"config_id": "default"}),
            server_dir=paths.server_dir,
            runtime_root=context.runtime_root,
            config_path=args.config,
            promptsets=promptsets,
            tokenizer=tokenizer,
            generation_config=generation_config,
            budgeting_config=budgeting_config,
            request_seed=seed,
            summary_rows=summary_rows,
            raw_request_rows=raw_request_rows,
            correctness_rows=correctness_rows,
            correctness_sample_size=int(correctness_config.get("sample_size", 128)),
            snapshot_dir=paths.server_dir,
            checkpoint_state=checkpoint_state,
            checkpoint_state_path=checkpoint_state_path,
            checkpoint_steps_dir=checkpoint_steps_dir,
            completed_step_payloads=completed_step_payloads,
            config_search_rows=config_search_rows,
        )

    baseline_config = dict(experiments_config.get("mixed_workload_baseline_sweep", {}) or {})
    if baseline_config.get("enabled", True):
        _log("Starting mixed_workload_baseline_sweep experiment family")
        baseline_mixes = baseline_config.get(
            "mixes",
            [
                "natural_mix_90_10_short_long",
                "natural_mix_70_30_short_long",
                "natural_mix_50_50_short_long",
                "stress_mix_50_50_natural_short_stress_long",
            ],
        )
        baseline_plan = [
            (mix_name, int(baseline_config.get("requests_per_run", 240)), int(concurrency))
            for mix_name in baseline_mixes
            for concurrency in baseline_config.get("concurrency_levels", [1, 2, 4, 8, 16, 24, 32])
        ]
        _run_server_backed_experiment_group(
            experiment_family="mixed_workload_baseline_sweep",
            workload_plan=baseline_plan,
            target=target,
            serving_config=_merge_server_config(serving_config, {"config_id": "default"}),
            server_dir=paths.server_dir,
            runtime_root=context.runtime_root,
            config_path=args.config,
            promptsets=promptsets,
            tokenizer=tokenizer,
            generation_config=generation_config,
            budgeting_config=budgeting_config,
            request_seed=seed + 17,
            summary_rows=summary_rows,
            raw_request_rows=raw_request_rows,
            correctness_rows=correctness_rows,
            correctness_sample_size=int(correctness_config.get("sample_size", 128)),
            snapshot_dir=paths.server_dir,
            checkpoint_state=checkpoint_state,
            checkpoint_state_path=checkpoint_state_path,
            checkpoint_steps_dir=checkpoint_steps_dir,
            completed_step_payloads=completed_step_payloads,
            config_search_rows=config_search_rows,
        )

    comparison_config = dict(experiments_config.get("bad_vs_tuned_server_config", {}) or {})
    if comparison_config.get("enabled", True):
        _log("Starting bad_vs_tuned_server_config experiment family")
        tuned_search = dict(comparison_config.get("tuned_search", {}) or {})
        search_mix = str(tuned_search.get("workload_mix", "stress_mix_50_50_natural_short_stress_long"))
        for batched_tokens in tuned_search.get("max_num_batched_tokens_values", [512, 768, 1024, 1536, 2048, 3072]):
            for max_num_seqs in tuned_search.get("max_num_seqs_values", [8, 16, 24, 32, 48]):
                search_serving_config = _merge_server_config(
                    serving_config,
                    {
                        "config_id": f"search-btok{batched_tokens}-seq{max_num_seqs}",
                        "max_num_batched_tokens": batched_tokens,
                        "max_num_seqs": max_num_seqs,
                        "gpu_memory_utilization": tuned_search.get("gpu_memory_utilization", 0.92),
                    },
                )
                search_summary_rows: list[dict[str, Any]] = []
                search_raw_rows: list[dict[str, Any]] = []
                search_correctness_rows: list[dict[str, Any]] = []
                _run_server_backed_experiment_group(
                    experiment_family="config_search",
                    workload_plan=[
                        (
                            search_mix,
                            int(tuned_search.get("requests_per_run", 120)),
                            int(tuned_search.get("concurrency", 16)),
                        )
                    ],
                    target=target,
                    serving_config=search_serving_config,
                    server_dir=paths.server_dir,
                    runtime_root=context.runtime_root,
                    config_path=args.config,
                    promptsets=promptsets,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    budgeting_config=budgeting_config,
                    request_seed=seed + 101,
                    summary_rows=search_summary_rows,
                    raw_request_rows=search_raw_rows,
                    correctness_rows=search_correctness_rows,
                    correctness_sample_size=int(correctness_config.get("sample_size", 128)),
                    snapshot_dir=paths.server_dir,
                    checkpoint_state=checkpoint_state,
                    checkpoint_state_path=checkpoint_state_path,
                    checkpoint_steps_dir=checkpoint_steps_dir,
                    completed_step_payloads=completed_step_payloads,
                    config_search_rows=config_search_rows,
                )
                if search_summary_rows:
                    summary_rows.extend(search_summary_rows)
                    raw_request_rows.extend(search_raw_rows)
                    correctness_rows.extend(search_correctness_rows)

        chosen = _select_tuned_config(
            config_search_rows,
            floor_ratio=float(tuned_search.get("rps_floor_ratio", 0.9)),
        )
        bad_serving_config = _merge_server_config(
            serving_config,
            {
                "config_id": "bad-config",
                **dict(comparison_config.get("bad_config", {}) or {}),
            },
        )
        tuned_serving_config = _merge_server_config(
            serving_config,
            {
                "config_id": "tuned-config",
                "max_num_batched_tokens": chosen.get("max_num_batched_tokens"),
                "max_num_seqs": chosen.get("max_num_seqs"),
                "gpu_memory_utilization": chosen.get("gpu_memory_utilization") or tuned_search.get("gpu_memory_utilization", 0.92),
            },
        )
        final_workload_name = str(comparison_config.get("workload_mix", "stress_mix_50_50_natural_short_stress_long"))
        final_requests = int(comparison_config.get("requests_per_run", 240))
        concurrencies = comparison_config.get("concurrency_levels", [1, 2, 4, 8, 16, 24, 32])
        for config_payload in (bad_serving_config, tuned_serving_config):
            _run_server_backed_experiment_group(
                experiment_family="bad_vs_tuned_server_config",
                workload_plan=[
                    (final_workload_name, final_requests, int(concurrency))
                    for concurrency in concurrencies
                ],
                target=target,
                serving_config=config_payload,
                server_dir=paths.server_dir,
                runtime_root=context.runtime_root,
                config_path=args.config,
                promptsets=promptsets,
                tokenizer=tokenizer,
                generation_config=generation_config,
                budgeting_config=budgeting_config,
                request_seed=seed + 303,
                summary_rows=summary_rows,
                raw_request_rows=raw_request_rows,
                correctness_rows=correctness_rows,
                correctness_sample_size=int(correctness_config.get("sample_size", 128)),
                snapshot_dir=paths.server_dir,
                checkpoint_state=checkpoint_state,
                checkpoint_state_path=checkpoint_state_path,
                checkpoint_steps_dir=checkpoint_steps_dir,
                completed_step_payloads=completed_step_payloads,
                config_search_rows=config_search_rows,
            )

    adapter_vs_merged_config = dict(experiments_config.get("adapter_vs_merged_optional", {}) or {})
    if (
        adapter_vs_merged_config.get("enabled", True)
        and target.target_kind == "base_plus_lora"
        and config.get("model_resolution", {}).get("fallback", {}).get("merged_model_path")
    ):
        _log("Starting adapter_vs_merged_optional experiment family")
        merged_target, _ = resolve_serving_target(
            config_path=args.config,
            repo_root=repo_root,
            target_kind="merged_model",
            merged_model_path=config.get("model_resolution", {}).get("fallback", {}).get("merged_model_path"),
        )
        workload_name = str(adapter_vs_merged_config.get("workload_mix", "natural_mix_50_50_short_long"))
        requests_per_run = int(adapter_vs_merged_config.get("requests_per_run", 120))
        concurrencies = adapter_vs_merged_config.get("concurrency_levels", [1, 8, 16])
        for comparison_target, config_id in ((target, "adapter-serving"), (merged_target, "merged-serving")):
            _run_server_backed_experiment_group(
                experiment_family="adapter_vs_merged_optional",
                workload_plan=[
                    (workload_name, requests_per_run, int(concurrency))
                    for concurrency in concurrencies
                ],
                target=comparison_target,
                serving_config=_merge_server_config(serving_config, {"config_id": config_id}),
                server_dir=paths.server_dir,
                runtime_root=context.runtime_root,
                config_path=args.config,
                promptsets=promptsets,
                tokenizer=tokenizer,
                generation_config=generation_config,
                budgeting_config=budgeting_config,
                request_seed=seed + 707,
                summary_rows=summary_rows,
                raw_request_rows=raw_request_rows,
                correctness_rows=correctness_rows,
                correctness_sample_size=int(adapter_vs_merged_config.get("correctness_sample_size", 128)),
                snapshot_dir=paths.server_dir,
                checkpoint_state=checkpoint_state,
                checkpoint_state_path=checkpoint_state_path,
                checkpoint_steps_dir=checkpoint_steps_dir,
                completed_step_payloads=completed_step_payloads,
                config_search_rows=config_search_rows,
            )

    raw_requests_path = write_jsonl(paths.raw_dir / "request_results.jsonl", raw_request_rows)
    summary_csv_path = write_csv_rows(paths.tables_dir / "summary_rows.csv", summary_rows)
    summary_json_path = write_json(paths.tables_dir / "summary_rows.json", {"rows": summary_rows})
    correctness_csv_path = write_csv_rows(paths.correctness_dir / "correctness_rows.csv", correctness_rows)
    correctness_json_path = write_json(paths.correctness_dir / "correctness_rows.json", {"rows": correctness_rows})
    config_search_csv_path = write_csv_rows(paths.tables_dir / "config_search_rows.csv", config_search_rows)
    config_search_json_path = write_json(paths.tables_dir / "config_search_rows.json", {"rows": config_search_rows})

    bundle = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_name": args.run_name,
        "fingerprint": fingerprint,
        "config_path": str(args.config.resolve()),
        "run_dir": str(paths.run_dir.resolve()),
        "checkpoint_state_path": str(checkpoint_state_path.resolve()),
        "checkpoint_dir": str(checkpoint_paths["checkpoint_dir"].resolve()),
        "checkpoint_steps_dir": str(checkpoint_steps_dir.resolve()),
        "target": target.__dict__,
        "promptset_manifest": promptsets["manifest"],
        "summary_rows": summary_rows,
        "correctness_rows": correctness_rows,
        "config_search_rows": config_search_rows,
        "raw_requests_path": str(raw_requests_path),
        "summary_csv_path": str(summary_csv_path),
        "summary_json_path": str(summary_json_path),
        "correctness_csv_path": str(correctness_csv_path),
        "correctness_json_path": str(correctness_json_path),
        "config_search_csv_path": str(config_search_csv_path),
        "config_search_json_path": str(config_search_json_path),
    }
    bundle_path = save_benchmark_bundle(bundle, paths.run_dir / "bundle.json")

    rendered: dict[str, str] = {}
    if not args.skip_render:
        rendered = render_benchmark_report(bundle, paths.reports_dir)

    checkpoint_state["complete"] = True
    checkpoint_state["bundle_path"] = str(bundle_path.resolve())
    checkpoint_state["report_path"] = str(rendered.get("report_path")) if rendered.get("report_path") else None
    checkpoint_state["updated_at_utc"] = _now_utc()
    save_benchmark_checkpoint_state(checkpoint_state_path, checkpoint_state)

    if args.mirror_summary_to_repo:
        mirror_small_artifact(summary_csv_path, repo_targets["metrics"] / f"{args.run_name}_benchmark_summary.csv")
        mirror_small_artifact(correctness_csv_path, repo_targets["metrics"] / f"{args.run_name}_benchmark_correctness.csv")
        mirror_small_artifact(config_search_csv_path, repo_targets["metrics"] / f"{args.run_name}_benchmark_config_search.csv")
        mirror_small_artifact(bundle_path, repo_targets["metrics"] / f"{args.run_name}_benchmark_bundle.json")
    if args.mirror_report_to_repo and rendered.get("report_path"):
        mirror_small_artifact(rendered["report_path"], repo_targets["reports"] / f"{args.run_name}_benchmark_report.md")

    _log(f"Promptsets dir: {paths.promptsets_dir}")
    _log(f"Raw requests: {raw_requests_path}")
    _log(f"Summary CSV: {summary_csv_path}")
    _log(f"Correctness CSV: {correctness_csv_path}")
    _log(f"Bundle: {bundle_path}")
    if rendered:
        _log(f"Rendered report: {rendered.get('report_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
