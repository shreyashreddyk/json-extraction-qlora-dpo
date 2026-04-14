"""Plotting and markdown reporting for vLLM benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .benchmarking import write_csv_rows
from .utils import read_json, write_json, write_text


def _load_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency-backed
        raise RuntimeError(
            "matplotlib is required to render vLLM benchmark plots."
        ) from exc
    return plt


def _write_figure(figure: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    return output_path


def _line_plot(
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    y_keys: list[tuple[str, str]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    sorted_rows = sorted(rows, key=lambda row: float(row.get(x_key, 0)))
    x_values = [float(row.get(x_key, 0)) for row in sorted_rows]
    for key, label in y_keys:
        y_values = [float(row.get(key) or 0.0) for row in sorted_rows]
        axis.plot(x_values, y_values, marker="o", label=label)
    axis.set_title(title)
    axis.set_xlabel(x_key.replace("_", " ").title())
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    if len(y_keys) > 1:
        axis.legend()
    return _write_figure(figure, output_path)


def _scatter_plot(
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    output_path: Path,
) -> Path:
    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.scatter(
        [float(row.get(x_key) or 0.0) for row in rows],
        [float(row.get(y_key) or 0.0) for row in rows],
        c="#1f77b4",
        alpha=0.8,
    )
    axis.set_title(title)
    axis.set_xlabel(x_key.replace("_", " ").title())
    axis.set_ylabel(y_key.replace("_", " ").title())
    axis.grid(True, alpha=0.3)
    return _write_figure(figure, output_path)


def _heatmap_plot(
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    value_key: str,
    title: str,
    output_path: Path,
) -> Path:
    plt = _load_pyplot()
    x_values = sorted({int(float(row.get(x_key) or 0)) for row in rows})
    y_values = sorted({int(float(row.get(y_key) or 0)) for row in rows})
    matrix = []
    for y_value in y_values:
        matrix_row = []
        for x_value in x_values:
            matching = [
                float(row.get(value_key) or 0.0)
                for row in rows
                if int(float(row.get(x_key) or 0)) == x_value and int(float(row.get(y_key) or 0)) == y_value
            ]
            matrix_row.append(matching[0] if matching else 0.0)
        matrix.append(matrix_row)
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_title(title)
    axis.set_xticks(range(len(x_values)))
    axis.set_xticklabels(x_values)
    axis.set_yticks(range(len(y_values)))
    axis.set_yticklabels(y_values)
    axis.set_xlabel(x_key.replace("_", " ").title())
    axis.set_ylabel(y_key.replace("_", " ").title())
    figure.colorbar(image, ax=axis)
    return _write_figure(figure, output_path)


def _comparison_table_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No comparison rows were produced."
    headers = ["server_config_id", "concurrency", "throughput_rps", "latency_p50_ms", "latency_p90_ms", "latency_p99_ms", "tail_inflation_p99_over_p50"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def render_benchmark_report(bundle: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Render plots and a markdown report from the benchmark bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = list(bundle.get("summary_rows", []))
    correctness_rows = list(bundle.get("correctness_rows", []))
    config_search_rows = list(bundle.get("config_search_rows", []))
    smoke_rows = [row for row in summary_rows if str(row.get("experiment_family")) == "smoke_single_stage"]
    baseline_rows = [row for row in summary_rows if str(row.get("experiment_family")) == "mixed_workload_baseline_sweep"]
    comparison_rows = [row for row in summary_rows if str(row.get("experiment_family")) == "bad_vs_tuned_server_config"]

    plot_paths: dict[str, str] = {}
    if baseline_rows:
        plot_paths["throughput_vs_concurrency"] = str(
            _line_plot(
                rows=baseline_rows,
                x_key="concurrency",
                y_keys=[("throughput_rps", "Throughput RPS")],
                title="Throughput vs Concurrency",
                ylabel="RPS",
                output_path=output_dir / "throughput_vs_concurrency.png",
            )
        )
        plot_paths["overall_latency_percentiles"] = str(
            _line_plot(
                rows=baseline_rows,
                x_key="concurrency",
                y_keys=[
                    ("latency_p50_ms", "p50"),
                    ("latency_p90_ms", "p90"),
                    ("latency_p99_ms", "p99"),
                ],
                title="Overall Latency Percentiles vs Concurrency",
                ylabel="Latency (ms)",
                output_path=output_dir / "overall_latency_percentiles_vs_concurrency.png",
            )
        )
        bucket_rows: list[dict[str, Any]] = []
        for row in baseline_rows:
            buckets = row.get("bucket_latency_ms", {}) or {}
            bucket_rows.append(
                {
                    "concurrency": row.get("concurrency"),
                    "short_p99_ms": (buckets.get("short") or {}).get("p99") or 0.0,
                    "medium_p99_ms": (buckets.get("medium") or {}).get("p99") or 0.0,
                    "long_p99_ms": (buckets.get("long") or {}).get("p99") or 0.0,
                }
            )
        plot_paths["bucketed_p99"] = str(
            _line_plot(
                rows=bucket_rows,
                x_key="concurrency",
                y_keys=[
                    ("short_p99_ms", "Short p99"),
                    ("medium_p99_ms", "Medium p99"),
                    ("long_p99_ms", "Long p99"),
                ],
                title="Short / Medium / Long p99 vs Concurrency",
                ylabel="Latency (ms)",
                output_path=output_dir / "bucketed_p99_vs_concurrency.png",
            )
        )
        plot_paths["p99_vs_rps"] = str(
            _scatter_plot(
                rows=baseline_rows,
                x_key="throughput_rps",
                y_key="latency_p99_ms",
                title="p99 vs Throughput (RPS)",
                output_path=output_dir / "p99_vs_rps.png",
            )
        )
        plot_paths["tail_inflation"] = str(
            _line_plot(
                rows=baseline_rows,
                x_key="concurrency",
                y_keys=[("tail_inflation_p99_over_p50", "Tail inflation")],
                title="Tail Inflation vs Concurrency",
                ylabel="p99 / p50",
                output_path=output_dir / "tail_inflation_vs_concurrency.png",
            )
        )

    if config_search_rows:
        plot_paths["config_search_rps_heatmap"] = str(
            _heatmap_plot(
                rows=config_search_rows,
                x_key="max_num_batched_tokens",
                y_key="max_num_seqs",
                value_key="throughput_rps",
                title="Concurrency x Config Sweep: Throughput",
                output_path=output_dir / "config_search_rps_heatmap.png",
            )
        )
        plot_paths["config_search_short_p99_heatmap"] = str(
            _heatmap_plot(
                rows=config_search_rows,
                x_key="max_num_batched_tokens",
                y_key="max_num_seqs",
                value_key="short_p99_ms",
                title="Concurrency x Config Sweep: Short p99",
                output_path=output_dir / "config_search_short_p99_heatmap.png",
            )
        )

    summary_table_path = write_csv_rows(output_dir / "summary_rows.csv", summary_rows)
    correctness_table_path = write_csv_rows(output_dir / "correctness_rows.csv", correctness_rows)
    config_search_table_path = write_csv_rows(output_dir / "config_search_rows.csv", config_search_rows)

    markdown_lines = [
        "# vLLM Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Generated at: `{bundle.get('generated_at_utc')}`",
        f"- Run name: `{bundle.get('run_name')}`",
        f"- Promoted target kind: `{bundle.get('target', {}).get('target_kind')}`",
        f"- Request model name: `{bundle.get('target', {}).get('request_model_name')}`",
        "",
        "## Rendered artifacts",
        "",
        f"- Summary CSV: `{summary_table_path}`",
        f"- Correctness CSV: `{correctness_table_path}`",
        f"- Config-search CSV: `{config_search_table_path}`",
    ]
    if plot_paths:
        markdown_lines.extend(
            [
                "",
                "## Plot inventory",
                "",
            ]
        )
        markdown_lines.extend([f"- {name}: `{path}`" for name, path in sorted(plot_paths.items())])

    if smoke_rows:
        markdown_lines.extend(
            [
                "",
                "## Smoke benchmark",
                "",
                _comparison_table_markdown(smoke_rows),
            ]
        )
    if baseline_rows:
        markdown_lines.extend(
            [
                "",
                "## Mixed-workload baseline sweep",
                "",
                _comparison_table_markdown(baseline_rows[:12]),
            ]
        )
    if comparison_rows:
        markdown_lines.extend(
            [
                "",
                "## Bad vs tuned config comparison",
                "",
                _comparison_table_markdown(comparison_rows),
            ]
        )
    if correctness_rows:
        markdown_lines.extend(
            [
                "",
                "## Correctness spot checks",
                "",
                _comparison_table_markdown(
                    [
                        {
                            "server_config_id": row.get("experiment_id"),
                            "concurrency": row.get("sample_size_actual"),
                            "throughput_rps": row.get("json_parse_pass_rate"),
                            "latency_p50_ms": row.get("schema_validation_pass_rate"),
                            "latency_p90_ms": row.get("categorical_exact_match", {}).get("issue_category"),
                            "latency_p99_ms": row.get("categorical_exact_match", {}).get("priority"),
                            "tail_inflation_p99_over_p50": row.get("categorical_exact_match", {}).get("product_area"),
                        }
                        for row in correctness_rows
                    ]
                ),
            ]
        )

    report_path = write_text(output_dir / "benchmark_report.md", "\n".join(markdown_lines))
    index_payload = {
        "generated_at_utc": bundle.get("generated_at_utc"),
        "plot_paths": plot_paths,
        "summary_table_path": str(summary_table_path),
        "correctness_table_path": str(correctness_table_path),
        "config_search_table_path": str(config_search_table_path),
        "report_path": str(report_path),
    }
    index_path = write_json(output_dir / "render_index.json", index_payload)
    return {
        "report_path": str(report_path),
        "index_path": str(index_path),
        **plot_paths,
    }


def load_benchmark_bundle(path: Path) -> dict[str, Any]:
    """Load a saved benchmark bundle JSON."""

    return read_json(path)


def save_benchmark_bundle(bundle: dict[str, Any], path: Path) -> Path:
    """Persist a benchmark bundle JSON."""

    return write_json(path, bundle)

