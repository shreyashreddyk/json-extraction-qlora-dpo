"""Markdown export helpers for the final project report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cases import CaseStudy
from .loaders import ReportingBundle
from .tables import (
    build_dataset_composition_table,
    build_failure_bucket_table,
    build_pair_quality_table,
    build_stage_delta_table,
    build_stage_metrics_table,
)
from ..utils import write_text


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_Not available._"
    headers = list(rows[0].keys())
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join([header_row, separator, *body])


def _plot_link(report_path: Path, plot_paths: dict[str, str] | None, key: str) -> str:
    if not plot_paths or key not in plot_paths:
        return ""
    relative = Path(plot_paths[key]).resolve().relative_to(report_path.resolve().parents[2])
    return f"![{key}]({relative.as_posix()})"


def _case_study_markdown(case_studies: dict[str, list[CaseStudy]]) -> str:
    sections: list[str] = []
    for label, title in (
        ("baseline_bad_to_sft_good", "Baseline Bad -> SFT Good"),
        ("sft_good_to_dpo_better", "SFT Good -> DPO Better"),
        ("sft_good_to_dpo_worse", "SFT Good -> DPO Worse"),
        ("syntax_cleaned_up_semantics_unchanged", "Syntax Cleaned Up But Semantics Unchanged"),
        ("unchanged_hard_failures", "Unchanged Hard Failures"),
    ):
        sections.append(f"### {title}")
        rows = case_studies.get(label, [])
        if not rows:
            sections.append("_No saved row-level example was available for this category in the current artifact set._")
            sections.append("")
            continue
        for case in rows:
            sections.extend(
                [
                    f"#### `{case.record_id}`",
                    f"- Why it matters: {case.why_it_matters}",
                    f"- Syntax tuples: baseline={case.baseline_syntax}, sft={case.sft_syntax}, dpo={case.dpo_syntax}",
                    (
                        f"- Semantic scores: baseline={case.baseline_semantics['semantic_score']}, "
                        f"sft={case.sft_semantics['semantic_score']}, dpo={case.dpo_semantics['semantic_score']}"
                    ),
                    "",
                ]
            )
        sections.append("")
    return "\n".join(sections)


def render_final_markdown_report(
    bundle: ReportingBundle,
    case_studies: dict[str, list[CaseStudy]],
    output_path: str | Path,
    plot_paths: dict[str, str] | None = None,
) -> Path:
    """Render a GitHub-readable final project report from saved artifacts."""

    resolved_output_path = Path(output_path).resolve()
    stage_metrics = build_stage_metrics_table(bundle)
    stage_deltas = build_stage_delta_table(bundle)
    dataset_table = build_dataset_composition_table(bundle)
    failure_table = build_failure_bucket_table(bundle)
    pair_quality_table = build_pair_quality_table(bundle)
    summary = (bundle.composition_summary or {}).get("summary", {})

    lines = [
        "# Final Project Report",
        "",
        "## Project Summary",
        "",
        "This report summarizes a schema-constrained support-ticket JSON extraction project across three saved stages: baseline, SFT, and DPO.",
        "The narrative keeps syntax quality separate from semantic quality so the repo stays honest about what improved, what regressed, and where the gains are mixed.",
        "",
        "## Dataset Upgrade Summary",
        "",
        f"- Total rows: `{summary.get('total_rows', 'n/a')}`",
        f"- Split counts: `{summary.get('split_counts', {})}`",
        f"- Synthetic row rate: `{summary.get('synthetic_row_rate', 'n/a')}`",
        f"- Leakage clean: `{(summary.get('leakage_checks') or {}).get('is_lineage_clean', 'n/a')}`",
        "",
        _markdown_table(dataset_table),
        "",
        _plot_link(resolved_output_path, plot_paths, "dataset_label_distribution"),
        "",
        _plot_link(resolved_output_path, plot_paths, "prompt_length_distribution"),
        "",
        _plot_link(resolved_output_path, plot_paths, "token_length_distribution"),
        "",
        "## Key Metrics",
        "",
        _markdown_table(stage_metrics),
        "",
        "## Syntax vs Semantic Takeaways",
        "",
        "- Baseline: very strong surface formatting and schema compliance, but weak task understanding and field correctness.",
        "- SFT: the major semantic jump happened here, but it also introduced a real schema-discipline regression.",
        "- DPO: mostly recovered schema discipline and added a smaller semantic gain over SFT, but with slower inference and non-trivial row-level regressions.",
        "",
        _markdown_table(stage_deltas),
        "",
        _plot_link(resolved_output_path, plot_paths, "stage_comparison"),
        "",
        _plot_link(resolved_output_path, plot_paths, "syntax_semantic_deltas"),
        "",
        _plot_link(resolved_output_path, plot_paths, "field_level_f1"),
        "",
        "## Pair-Quality Summary",
        "",
        (
            _markdown_table(pair_quality_table)
            if pair_quality_table
            else "_Preference-pair artifacts were not available in the current repo/runtime mirror, so this section is intentionally limited._"
        ),
        "",
        _plot_link(resolved_output_path, plot_paths, "preference_diagnostics"),
        "",
        "## Failure Analysis Highlights",
        "",
        _markdown_table(failure_table),
        "",
        _plot_link(resolved_output_path, plot_paths, "failure_buckets"),
        "",
        "## Case Studies",
        "",
        _case_study_markdown(case_studies),
        "",
        "## Honest Conclusion",
        "",
        "SFT delivered the main semantic improvement on this task. DPO appears most useful here as a syntax and schema-discipline repair layer with selective semantic gains rather than as a universal quality win.",
        "The saved artifacts support a nuanced story: DPO improved the aggregate comparison, but it also slowed inference and still hurt some rows. That makes the repo more credible, not less, because the reporting layer shows both the wins and the remaining failure modes.",
        "",
        "## Next Step",
        "",
        "The next practical step is a serving and benchmarking lab that packages the best checkpoint for vLLM-first inference and measures the quality-latency tradeoff explicitly.",
        "",
    ]
    return write_text(resolved_output_path, "\n".join(lines))
