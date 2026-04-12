"""Matplotlib-based plotting helpers for the final report layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from collections import Counter
import math

from .loaders import ReportingBundle
from .tables import build_field_level_table


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to render report plots. "
            "Install train dependencies or use requirements-colab.txt."
        ) from exc
    return plt


def _write_figure(figure: Any, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    return output_path


def _empty_plot(plt: Any, title: str, message: str, output_path: Path) -> tuple[Any, Any, Path]:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    axis.axis("off")
    return figure, axis, _write_figure(figure, output_path)


def _row_prompt_text(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )
    input_text = row.get("input_text")
    return str(input_text or "")


def _compute_length_series(rows: list[dict[str, Any]] | None) -> list[int]:
    if not rows:
        return []
    return [len(_row_prompt_text(row)) for row in rows if _row_prompt_text(row)]


def _compute_token_length_series(rows: list[dict[str, Any]] | None, model_name: str | None) -> tuple[list[int], str]:
    prompts = [_row_prompt_text(row) for row in (rows or []) if _row_prompt_text(row)]
    if not prompts:
        return [], "no prompt rows available"
    if not model_name:
        return [len(prompt) for prompt in prompts], "char_length_fallback"
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        return [len(prompt) for prompt in prompts], "char_length_fallback"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return [len(prompt) for prompt in prompts], "char_length_fallback"
    lengths = [len(tokenizer(prompt, add_special_tokens=True)["input_ids"]) for prompt in prompts]
    return lengths, "tokenizer"


def plot_label_distribution(bundle: ReportingBundle, output_path: Path) -> tuple[Any, Any, Path]:
    """Plot saved label counts from the composition summary."""

    plt = _load_pyplot()
    summary = (bundle.composition_summary or {}).get("summary", {})
    labels = ["issue_category", "priority", "product_area", "sentiment"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, label in zip(axes.flat, labels, strict=True):
        counts = summary.get(f"{label}_counts", {})
        axis.bar(list(counts.keys()), list(counts.values()), color="#1f77b4")
        axis.set_title(label.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=35)
    return figure, axes, _write_figure(figure, output_path)


def plot_prompt_length_distribution(bundle: ReportingBundle, output_path: Path) -> tuple[Any, Any, Path]:
    """Plot prompt length distribution using saved manifests."""

    plt = _load_pyplot()
    lengths = _compute_length_series(bundle.sft_manifest_rows) or _compute_length_series(bundle.eval_manifest_rows)
    if not lengths:
        return _empty_plot(plt, "Prompt Length Distribution", "Prompt manifests are not available.", output_path)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(lengths, bins=min(40, max(10, int(math.sqrt(len(lengths))))), color="#4c78a8", alpha=0.85)
    axis.set_title("Prompt Length Distribution (Characters)")
    axis.set_xlabel("Prompt length")
    axis.set_ylabel("Row count")
    return figure, axis, _write_figure(figure, output_path)


def plot_token_length_distribution(bundle: ReportingBundle, output_path: Path) -> tuple[Any, Any, Path]:
    """Plot prompt token length distribution with a char-length fallback."""

    plt = _load_pyplot()
    model_name = (bundle.sft.metrics or bundle.baseline.metrics or {}).get("model_name_or_path")
    lengths, mode = _compute_token_length_series(bundle.sft_manifest_rows, model_name)
    if not lengths:
        return _empty_plot(plt, "Prompt Token Distribution", "Prompt manifests are not available.", output_path)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(lengths, bins=min(40, max(10, int(math.sqrt(len(lengths))))), color="#f58518", alpha=0.85)
    title = "Prompt Token Distribution" if mode == "tokenizer" else "Prompt Length Proxy Distribution (Characters)"
    axis.set_title(title)
    axis.set_xlabel("Tokens" if mode == "tokenizer" else "Characters")
    axis.set_ylabel("Row count")
    return figure, axis, _write_figure(figure, output_path)


def plot_training_curves(history_payload: dict[str, Any] | None, output_path: Path, title: str) -> tuple[Any, Any, Path]:
    """Plot train/eval/lr curves from saved training history JSON."""

    plt = _load_pyplot()
    if not history_payload:
        return _empty_plot(plt, title, "History artifact is not available.", output_path)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plotted = False
    for axis, key, color in zip(
        axes,
        ("train_loss", "eval_loss", "learning_rate"),
        ("#1f77b4", "#d62728", "#2ca02c"),
        strict=True,
    ):
        series = history_payload.get(key, [])
        if series:
            value_key = "loss" if key == "train_loss" else key
            axis.plot([point["step"] for point in series], [point[value_key] for point in series], color=color)
            plotted = True
        axis.set_title(key.replace("_", " ").title())
        axis.set_xlabel("Step")
        axis.grid(True, alpha=0.3)
    if not plotted:
        axes[1].text(0.5, 0.5, "No scalar series available.", ha="center", va="center")
    figure.suptitle(title)
    return figure, axes, _write_figure(figure, output_path)


def plot_dpo_reward_curves(history_payload: dict[str, Any] | None, output_path: Path) -> tuple[Any, Any, Path]:
    """Plot saved DPO reward traces."""

    plt = _load_pyplot()
    if not history_payload:
        return _empty_plot(plt, "DPO Reward Curves", "DPO history artifact is not available.", output_path)
    scalar_series = history_payload.get("scalar_series", {})
    reward_keys = ["rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    plotted = False
    for axis, key in zip(axes.flat, reward_keys, strict=True):
        series = scalar_series.get(key, [])
        if series:
            axis.plot([point["step"] for point in series], [point[key] for point in series], color="#9467bd")
            plotted = True
        axis.set_title(key.replace("/", " ").title())
        axis.set_xlabel("Step")
        axis.grid(True, alpha=0.3)
    if not plotted:
        axes[0][0].text(0.5, 0.5, "No reward traces available.", ha="center", va="center")
    figure.suptitle("DPO Reward Traces")
    return figure, axes, _write_figure(figure, output_path)


def plot_stage_comparison(bundle: ReportingBundle, output_path: Path) -> tuple[Any, Any, Path]:
    """Plot grouped stage comparison bars for syntax and semantic metrics."""

    plt = _load_pyplot()
    metrics = {
        "baseline": bundle.baseline.metrics or {},
        "sft": bundle.sft.metrics or {},
        "dpo": bundle.dpo.metrics or {},
    }
    stages = list(metrics.keys())
    labels = ["json_validity_rate", "schema_validation_pass_rate", "field_micro_f1", "field_macro_f1"]
    values = {
        "baseline": [
            metrics["baseline"].get("json_validity_rate", 0.0),
            metrics["baseline"].get("schema_validation_pass_rate", 0.0),
            metrics["baseline"].get("field_level", {}).get("micro", {}).get("f1", 0.0),
            metrics["baseline"].get("field_level", {}).get("macro", {}).get("f1", 0.0),
        ],
        "sft": [
            metrics["sft"].get("json_validity_rate", 0.0),
            metrics["sft"].get("schema_validation_pass_rate", 0.0),
            metrics["sft"].get("field_level", {}).get("micro", {}).get("f1", 0.0),
            metrics["sft"].get("field_level", {}).get("macro", {}).get("f1", 0.0),
        ],
        "dpo": [
            metrics["dpo"].get("json_validity_rate", 0.0),
            metrics["dpo"].get("schema_validation_pass_rate", 0.0),
            metrics["dpo"].get("field_level", {}).get("micro", {}).get("f1", 0.0),
            metrics["dpo"].get("field_level", {}).get("macro", {}).get("f1", 0.0),
        ],
    }
    figure, axis = plt.subplots(figsize=(10, 5))
    x_positions = list(range(len(labels)))
    width = 0.23
    for offset, stage in zip((-width, 0.0, width), stages, strict=True):
        axis.bar([x + offset for x in x_positions], values[stage], width=width, label=stage)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([label.replace("_", " ") for label in labels], rotation=15)
    axis.set_ylim(0, 1.05)
    axis.set_title("Stage Comparison")
    axis.legend()
    return figure, axis, _write_figure(figure, output_path)


def plot_field_level_metric(bundle: ReportingBundle, output_path: Path, metric: str = "f1") -> tuple[Any, Any, Path]:
    """Plot grouped per-field metrics."""

    plt = _load_pyplot()
    rows = build_field_level_table(bundle, metric=metric)
    if not rows:
        return _empty_plot(plt, "Field-Level Metrics", "Field-level metrics are not available.", output_path)
    labels = [row["field"] for row in rows]
    baseline_values = [row["baseline"] or 0.0 for row in rows]
    sft_values = [row["sft"] or 0.0 for row in rows]
    dpo_values = [row["dpo"] or 0.0 for row in rows]
    figure, axis = plt.subplots(figsize=(12, 6))
    x_positions = list(range(len(labels)))
    width = 0.25
    axis.bar([x - width for x in x_positions], baseline_values, width=width, label="baseline")
    axis.bar(x_positions, sft_values, width=width, label="sft")
    axis.bar([x + width for x in x_positions], dpo_values, width=width, label="dpo")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_title(f"Field-Level {metric.replace('_', ' ').title()}")
    axis.legend()
    return figure, axis, _write_figure(figure, output_path)


def plot_syntax_semantic_deltas(bundle: ReportingBundle, output_path: Path) -> tuple[Any, Any, Path]:
    """Plot syntax vs semantic deltas for SFT and DPO."""

    plt = _load_pyplot()
    comparison = bundle.comparison_summary or {}
    deltas = comparison.get("deltas", {})
    labels = ["sft_vs_baseline", "dpo_vs_sft", "dpo_vs_baseline"]
    syntax_metric = "schema_validation_pass_rate"
    semantic_metric = "field_level_micro_f1"
    syntax_values = [deltas.get(label, {}).get("syntax", {}).get(syntax_metric, 0.0) for label in labels]
    semantic_values = [deltas.get(label, {}).get("semantic", {}).get(semantic_metric, 0.0) for label in labels]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    x_positions = list(range(len(labels)))
    width = 0.35
    axis.bar([x - width / 2 for x in x_positions], syntax_values, width=width, label="schema pass delta")
    axis.bar([x + width / 2 for x in x_positions], semantic_values, width=width, label="micro F1 delta")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=20)
    axis.set_title("Syntax vs Semantic Deltas")
    axis.legend()
    return figure, axis, _write_figure(figure, output_path)


def plot_failure_buckets(bundle: ReportingBundle, output_path: Path) -> tuple[Any, Any, Path]:
    """Plot grouped failure bucket counts by stage."""

    plt = _load_pyplot()
    bucket_counts = {
        "baseline": (bundle.baseline.diagnostics or {}).get("bucket_counts", {}),
        "sft": (bundle.sft.diagnostics or {}).get("bucket_counts", {}),
        "dpo": (bundle.dpo.diagnostics or {}).get("bucket_counts", {}),
    }
    labels = sorted({*bucket_counts["baseline"], *bucket_counts["sft"], *bucket_counts["dpo"]})
    if not labels:
        return _empty_plot(plt, "Failure Buckets", "Failure bucket diagnostics are not available.", output_path)
    figure, axis = plt.subplots(figsize=(11, 5))
    x_positions = list(range(len(labels)))
    width = 0.25
    for offset, stage in zip((-width, 0.0, width), ("baseline", "sft", "dpo"), strict=True):
        axis.bar(
            [x + offset for x in x_positions],
            [bucket_counts[stage].get(label, 0) for label in labels],
            width=width,
            label=stage,
        )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_title("Failure Bucket Summary")
    axis.legend()
    return figure, axis, _write_figure(figure, output_path)


def plot_preference_diagnostics(bundle: ReportingBundle, output_path: Path) -> tuple[Any, Any, Path]:
    """Plot preference skip reasons and pair-emission data when available."""

    plt = _load_pyplot()
    diagnostics = bundle.preference_diagnostics or {}
    summary = bundle.preference_summary or {}
    if not diagnostics and not summary:
        return _empty_plot(
            plt,
            "Preference Pair Diagnostics",
            "Preference audit artifacts are not available in the repo or runtime mirror.",
            output_path,
        )
    skipped = diagnostics.get("skipped_counts", {}) or summary.get("skipped_counts", {})
    emitted = int(summary.get("pair_count", 0))
    skipped_total = int(summary.get("skipped_count", 0))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(["emitted", "skipped"], [emitted, skipped_total], color=["#2ca02c", "#d62728"])
    axes[0].set_title("Pair Emission")
    if skipped:
        axes[1].bar(list(skipped.keys()), list(skipped.values()), color="#ff9da7")
        axes[1].tick_params(axis="x", rotation=30)
    else:
        axes[1].text(0.5, 0.5, "No skipped counts available.", ha="center", va="center")
        axes[1].axis("off")
    axes[1].set_title("Skipped Reasons")
    return figure, axes, _write_figure(figure, output_path)


def generate_report_plots(bundle: ReportingBundle, output_dir: Path) -> dict[str, str]:
    """Generate the standard plot set for the final report."""

    plotters: list[tuple[str, Any]] = [
        ("dataset_label_distribution", lambda path: plot_label_distribution(bundle, path)),
        ("prompt_length_distribution", lambda path: plot_prompt_length_distribution(bundle, path)),
        ("token_length_distribution", lambda path: plot_token_length_distribution(bundle, path)),
        ("sft_training_curves", lambda path: plot_training_curves(bundle.sft.history, path, "SFT Training Curves")),
        ("dpo_training_curves", lambda path: plot_training_curves(bundle.dpo.history, path, "DPO Training Curves")),
        ("dpo_reward_curves", lambda path: plot_dpo_reward_curves(bundle.dpo.history, path)),
        ("stage_comparison", lambda path: plot_stage_comparison(bundle, path)),
        ("field_level_f1", lambda path: plot_field_level_metric(bundle, path, metric="f1")),
        ("syntax_semantic_deltas", lambda path: plot_syntax_semantic_deltas(bundle, path)),
        ("failure_buckets", lambda path: plot_failure_buckets(bundle, path)),
        ("preference_diagnostics", lambda path: plot_preference_diagnostics(bundle, path)),
    ]
    generated: dict[str, str] = {}
    for label, plotter in plotters:
        _, _, rendered = plotter(output_dir / f"{label}.png")
        generated[label] = str(rendered)
    return generated
