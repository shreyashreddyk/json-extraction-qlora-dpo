"""Build one consolidated baseline vs SFT vs DPO comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.artifacts import mirror_small_artifact
from json_ft.metrics import CATEGORICAL_EXACT_MATCH_FIELDS, LIST_PRF_FIELDS, STRUCTURED_PRF_FIELDS
from json_ft.runtime import resolve_repo_artifact_targets, resolve_runtime_context
from json_ft.utils import read_json, read_jsonl, write_json, write_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="baseline-sft-dpo-comparison")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--sft-metrics", type=Path, required=True)
    parser.add_argument("--sft-predictions", type=Path, required=True)
    parser.add_argument("--dpo-metrics", type=Path, required=True)
    parser.add_argument("--dpo-predictions", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--mirror-summary-to-repo", action="store_true")
    parser.add_argument("--mirror-report-to-repo", action="store_true")
    return parser


def _nested_value(payload: dict[str, Any] | None, field_path: str) -> Any:
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _list_tokens(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _semantic_breakdown(row: dict[str, Any]) -> dict[str, Any]:
    predicted = row.get("parsed_payload")
    reference = row.get("reference_payload") or {}
    structured_matches = {
        field_name: _nested_value(predicted, field_name) == _nested_value(reference, field_name)
        for field_name in STRUCTURED_PRF_FIELDS
    }
    structured_match_count = sum(1 for matched in structured_matches.values() if matched)
    nullable_hallucination_fields = ("customer.name", "customer.account_id", "customer.plan_tier")
    nullable_hallucination_count = 0
    for field_name in nullable_hallucination_fields:
        predicted_value = _nested_value(predicted, field_name)
        reference_value = _nested_value(reference, field_name)
        if reference_value is None and predicted_value not in (None, "", []):
            nullable_hallucination_count += 1
    predicted_actions = _list_tokens(_nested_value(predicted, LIST_PRF_FIELDS[0]))
    reference_actions = _list_tokens(_nested_value(reference, LIST_PRF_FIELDS[0]))
    overlap = predicted_actions & reference_actions
    tp = len(overlap)
    fp = len(predicted_actions - overlap)
    fn = len(reference_actions - overlap)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    actions_f1 = 0.0 if precision + recall == 0 else (2 * precision * recall / (precision + recall))
    semantic_score = (
        structured_match_count + actions_f1 - nullable_hallucination_count
    ) / (len(STRUCTURED_PRF_FIELDS) + 1)
    return {
        "structured_match_count": structured_match_count,
        "structured_total": len(STRUCTURED_PRF_FIELDS),
        "structured_matches": structured_matches,
        "actions_f1": actions_f1,
        "nullable_hallucination_count": nullable_hallucination_count,
        "semantic_score": semantic_score,
    }


def _syntax_tuple(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("parsed_payload") is not None),
        int(bool(row.get("schema_is_valid", False))),
        int(not row.get("unexpected_fields")),
    )


def _prediction_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["record_id"]): row for row in rows}


def _stage_summary(metrics: dict[str, Any], metrics_path: Path, predictions_path: Path) -> dict[str, Any]:
    return {
        "stage_label": metrics.get("stage", "unknown"),
        "run_name": metrics.get("run_name"),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "syntax": {
            "json_validity_rate": metrics["json_validity_rate"],
            "schema_validation_pass_rate": metrics["schema_validation_pass_rate"],
            "hallucinated_field_rate": metrics["hallucinated_field_rate"],
            "json_recovery_rate": metrics["json_recovery_rate"],
        },
        "semantic": {
            "field_level_micro_f1": metrics["field_level"]["micro"]["f1"],
            "field_level_macro_f1": metrics["field_level"]["macro"]["f1"],
            "categorical_exact_match": metrics["categorical_exact_match"],
        },
        "latency_ms": metrics["latency_ms"],
        "model": {
            "model_name_or_path": metrics.get("model_name_or_path"),
            "base_model": metrics.get("base_model"),
            "adapter_path": metrics.get("adapter_path"),
            "merged_model_path": metrics.get("merged_model_path"),
            "model_manifest_path": metrics.get("model_manifest_path"),
        },
    }


def _delta(before: float, after: float) -> float:
    return after - before


def _build_evidence_example(
    *,
    record_id: str,
    reason: str,
    baseline_row: dict[str, Any],
    sft_row: dict[str, Any],
    dpo_row: dict[str, Any],
) -> dict[str, Any]:
    baseline_semantics = _semantic_breakdown(baseline_row)
    sft_semantics = _semantic_breakdown(sft_row)
    dpo_semantics = _semantic_breakdown(dpo_row)
    return {
        "record_id": record_id,
        "reason": reason,
        "baseline": {
            "syntax_tuple": _syntax_tuple(baseline_row),
            "semantic": baseline_semantics,
            "raw_output": baseline_row.get("raw_output"),
            "parsed_payload": baseline_row.get("parsed_payload"),
        },
        "sft": {
            "syntax_tuple": _syntax_tuple(sft_row),
            "semantic": sft_semantics,
            "raw_output": sft_row.get("raw_output"),
            "parsed_payload": sft_row.get("parsed_payload"),
        },
        "dpo": {
            "syntax_tuple": _syntax_tuple(dpo_row),
            "semantic": dpo_semantics,
            "raw_output": dpo_row.get("raw_output"),
            "parsed_payload": dpo_row.get("parsed_payload"),
        },
        "reference_payload": baseline_row.get("reference_payload"),
    }


def _collect_row_evidence(
    baseline_rows: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    dpo_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    baseline_index = _prediction_index(baseline_rows)
    sft_index = _prediction_index(sft_rows)
    dpo_index = _prediction_index(dpo_rows)
    shared_record_ids = sorted(set(baseline_index) & set(sft_index) & set(dpo_index))

    semantic_gain: list[dict[str, Any]] = []
    syntax_gain_only: list[dict[str, Any]] = []
    semantic_regression: list[dict[str, Any]] = []
    mixed_result: list[dict[str, Any]] = []

    for record_id in shared_record_ids:
        baseline_row = baseline_index[record_id]
        sft_row = sft_index[record_id]
        dpo_row = dpo_index[record_id]
        sft_semantics = _semantic_breakdown(sft_row)
        dpo_semantics = _semantic_breakdown(dpo_row)
        sft_syntax = _syntax_tuple(sft_row)
        dpo_syntax = _syntax_tuple(dpo_row)

        semantic_improved = dpo_semantics["semantic_score"] > sft_semantics["semantic_score"] + 1e-9
        semantic_regressed = dpo_semantics["semantic_score"] + 1e-9 < sft_semantics["semantic_score"]
        syntax_improved = dpo_syntax > sft_syntax
        syntax_regressed = dpo_syntax < sft_syntax

        if semantic_improved and not syntax_regressed:
            semantic_gain.append(
                _build_evidence_example(
                    record_id=record_id,
                    reason=(
                        "DPO improved semantic fidelity versus SFT while keeping syntax at least as strong. "
                        f"Semantic score {sft_semantics['semantic_score']:.4f} -> {dpo_semantics['semantic_score']:.4f}."
                    ),
                    baseline_row=baseline_row,
                    sft_row=sft_row,
                    dpo_row=dpo_row,
                )
            )
            continue

        if syntax_improved and not semantic_improved and not semantic_regressed:
            syntax_gain_only.append(
                _build_evidence_example(
                    record_id=record_id,
                    reason=(
                        "DPO improved syntax or validation, but did not improve semantic correctness over SFT. "
                        f"Syntax {sft_syntax} -> {dpo_syntax}; semantic score {sft_semantics['semantic_score']:.4f} -> "
                        f"{dpo_semantics['semantic_score']:.4f}."
                    ),
                    baseline_row=baseline_row,
                    sft_row=sft_row,
                    dpo_row=dpo_row,
                )
            )
            continue

        if semantic_regressed:
            semantic_regression.append(
                _build_evidence_example(
                    record_id=record_id,
                    reason=(
                        "DPO regressed semantically relative to SFT. "
                        f"Syntax {sft_syntax} -> {dpo_syntax}; semantic score {sft_semantics['semantic_score']:.4f} -> "
                        f"{dpo_semantics['semantic_score']:.4f}."
                    ),
                    baseline_row=baseline_row,
                    sft_row=sft_row,
                    dpo_row=dpo_row,
                )
            )
            continue

        if syntax_regressed or dpo_row.get("raw_output") != sft_row.get("raw_output"):
            mixed_result.append(
                _build_evidence_example(
                    record_id=record_id,
                    reason=(
                        "DPO changed behavior with mixed evidence: syntax, semantics, or output text moved without a clean win. "
                        f"Syntax {sft_syntax} -> {dpo_syntax}; semantic score {sft_semantics['semantic_score']:.4f} -> "
                        f"{dpo_semantics['semantic_score']:.4f}."
                    ),
                    baseline_row=baseline_row,
                    sft_row=sft_row,
                    dpo_row=dpo_row,
                )
            )

    sort_key = lambda item: item["dpo"]["semantic"]["semantic_score"] - item["sft"]["semantic"]["semantic_score"]
    semantic_gain = sorted(semantic_gain, key=sort_key, reverse=True)[:3]
    syntax_gain_only = sorted(
        syntax_gain_only,
        key=lambda item: (item["dpo"]["syntax_tuple"], -item["dpo"]["semantic"]["semantic_score"]),
        reverse=True,
    )[:3]
    semantic_regression = sorted(semantic_regression, key=sort_key)[:3]
    mixed_result = sorted(
        mixed_result,
        key=lambda item: (
            item["dpo"]["semantic"]["semantic_score"] - item["sft"]["semantic"]["semantic_score"],
            item["dpo"]["syntax_tuple"],
        ),
    )[:3]
    return {
        "semantic_gain": semantic_gain,
        "syntax_gain_only": syntax_gain_only,
        "semantic_regression": semantic_regression,
        "mixed_result": mixed_result,
    }


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _render_stage_metrics(title: str, summary: dict[str, Any]) -> list[str]:
    syntax = summary["syntax"]
    semantic = summary["semantic"]
    latency = summary["latency_ms"]
    lines = [
        f"### {title}",
        "",
        f"- Model: `{summary['model']['model_name_or_path']}`",
        f"- Base model: `{summary['model']['base_model'] or 'n/a'}`",
        f"- Adapter path: `{summary['model']['adapter_path'] or 'n/a'}`",
        f"- JSON validity rate: `{_format_metric(syntax['json_validity_rate'])}`",
        f"- Schema validation pass rate: `{_format_metric(syntax['schema_validation_pass_rate'])}`",
        f"- Hallucinated field rate: `{_format_metric(syntax['hallucinated_field_rate'])}`",
        f"- JSON recovery rate: `{_format_metric(syntax['json_recovery_rate'])}`",
        f"- Field-level micro F1: `{_format_metric(semantic['field_level_micro_f1'])}`",
        f"- Field-level macro F1: `{_format_metric(semantic['field_level_macro_f1'])}`",
        f"- Mean latency (ms): `{_format_metric(latency['mean'])}`",
        "",
        "Exact match by categorical field:",
    ]
    for field_name in CATEGORICAL_EXACT_MATCH_FIELDS:
        lines.append(
            f"- `{field_name}`: `{_format_metric(semantic['categorical_exact_match'][field_name])}`"
        )
    lines.append("")
    return lines


def _render_example_group(title: str, examples: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not examples:
        lines.append("No examples were found in this category.")
        lines.append("")
        return lines

    for example in examples:
        lines.extend(
            [
                f"### `{example['record_id']}`",
                "",
                example["reason"],
                "",
                "Reference payload:",
                "",
                "```json",
                json.dumps(example["reference_payload"], indent=2, sort_keys=True, ensure_ascii=True),
                "```",
                "",
                "Baseline output:",
                "",
                "```text",
                example["baseline"]["raw_output"] or "",
                "```",
                "",
                "SFT output:",
                "",
                "```text",
                example["sft"]["raw_output"] or "",
                "```",
                "",
                "DPO output:",
                "",
                "```text",
                example["dpo"]["raw_output"] or "",
                "```",
                "",
            ]
        )
    return lines


def render_comparison_report(summary: dict[str, Any]) -> str:
    """Render one consolidated markdown report."""

    baseline = summary["stages"]["baseline"]
    sft = summary["stages"]["sft"]
    dpo = summary["stages"]["dpo"]
    deltas = summary["deltas"]
    evidence = summary["row_evidence"]

    lines = [
        f"# Consolidated Comparison Report: {summary['run_name']}",
        "",
        "## Comparison Rules",
        "",
        "- Syntax metrics are reported separately from semantic metrics.",
        "- Semantic example ranking uses structured exact-match count plus `actions_requested` F1 as a row-level inspection aid.",
        "- This row-level ranking is diagnostic only; the headline comparison remains the saved aggregate metrics.",
        "- Row-level labels classify DPO relative to SFT as `syntax_gain_only`, `semantic_gain`, `semantic_regression`, or `mixed_result`.",
        "",
        "## Stage Metrics",
        "",
    ]
    lines.extend(_render_stage_metrics("Baseline", baseline))
    lines.extend(_render_stage_metrics("SFT", sft))
    lines.extend(_render_stage_metrics("DPO", dpo))
    lines.extend(
        [
            "## Deltas",
            "",
            "### DPO vs SFT",
            "",
            f"- JSON validity delta: `{_format_metric(deltas['dpo_vs_sft']['syntax']['json_validity_rate'])}`",
            f"- Schema pass delta: `{_format_metric(deltas['dpo_vs_sft']['syntax']['schema_validation_pass_rate'])}`",
            f"- Hallucinated field delta: `{_format_metric(deltas['dpo_vs_sft']['syntax']['hallucinated_field_rate'])}`",
            f"- Micro F1 delta: `{_format_metric(deltas['dpo_vs_sft']['semantic']['field_level_micro_f1'])}`",
            f"- Macro F1 delta: `{_format_metric(deltas['dpo_vs_sft']['semantic']['field_level_macro_f1'])}`",
            "",
            "### SFT vs Baseline",
            "",
            f"- JSON validity delta: `{_format_metric(deltas['sft_vs_baseline']['syntax']['json_validity_rate'])}`",
            f"- Schema pass delta: `{_format_metric(deltas['sft_vs_baseline']['syntax']['schema_validation_pass_rate'])}`",
            f"- Hallucinated field delta: `{_format_metric(deltas['sft_vs_baseline']['syntax']['hallucinated_field_rate'])}`",
            f"- Micro F1 delta: `{_format_metric(deltas['sft_vs_baseline']['semantic']['field_level_micro_f1'])}`",
            f"- Macro F1 delta: `{_format_metric(deltas['sft_vs_baseline']['semantic']['field_level_macro_f1'])}`",
            "",
            "### DPO vs Baseline",
            "",
            f"- JSON validity delta: `{_format_metric(deltas['dpo_vs_baseline']['syntax']['json_validity_rate'])}`",
            f"- Schema pass delta: `{_format_metric(deltas['dpo_vs_baseline']['syntax']['schema_validation_pass_rate'])}`",
            f"- Hallucinated field delta: `{_format_metric(deltas['dpo_vs_baseline']['syntax']['hallucinated_field_rate'])}`",
            f"- Micro F1 delta: `{_format_metric(deltas['dpo_vs_baseline']['semantic']['field_level_micro_f1'])}`",
            f"- Macro F1 delta: `{_format_metric(deltas['dpo_vs_baseline']['semantic']['field_level_macro_f1'])}`",
            "",
        ]
    )
    lines.extend(_render_example_group("Semantic Gain", evidence["semantic_gain"]))
    lines.extend(_render_example_group("Syntax Gain Only", evidence["syntax_gain_only"]))
    lines.extend(_render_example_group("Semantic Regression", evidence["semantic_regression"]))
    lines.extend(_render_example_group("Mixed Result", evidence["mixed_result"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    context = resolve_runtime_context(
        repo_root=repo_root,
        stage="reports",
        run_name=args.run_name,
        runtime_root=args.runtime_root,
    )
    summary_output = args.summary_output or (context.metrics_dir / f"{args.run_name}_comparison_summary.json")
    report_output = args.report_output or (context.reports_dir / f"{args.run_name}_comparison_report.md")

    baseline_metrics = read_json(args.baseline_metrics)
    baseline_predictions = read_jsonl(args.baseline_predictions)
    sft_metrics = read_json(args.sft_metrics)
    sft_predictions = read_jsonl(args.sft_predictions)
    dpo_metrics = read_json(args.dpo_metrics)
    dpo_predictions = read_jsonl(args.dpo_predictions)

    stages = {
        "baseline": _stage_summary(baseline_metrics, args.baseline_metrics, args.baseline_predictions),
        "sft": _stage_summary(sft_metrics, args.sft_metrics, args.sft_predictions),
        "dpo": _stage_summary(dpo_metrics, args.dpo_metrics, args.dpo_predictions),
    }
    deltas = {
        "sft_vs_baseline": {
            "syntax": {
                key: _delta(stages["baseline"]["syntax"][key], stages["sft"]["syntax"][key])
                for key in stages["baseline"]["syntax"]
            },
            "semantic": {
                "field_level_micro_f1": _delta(
                    stages["baseline"]["semantic"]["field_level_micro_f1"],
                    stages["sft"]["semantic"]["field_level_micro_f1"],
                ),
                "field_level_macro_f1": _delta(
                    stages["baseline"]["semantic"]["field_level_macro_f1"],
                    stages["sft"]["semantic"]["field_level_macro_f1"],
                ),
            },
        },
        "dpo_vs_sft": {
            "syntax": {
                key: _delta(stages["sft"]["syntax"][key], stages["dpo"]["syntax"][key])
                for key in stages["sft"]["syntax"]
            },
            "semantic": {
                "field_level_micro_f1": _delta(
                    stages["sft"]["semantic"]["field_level_micro_f1"],
                    stages["dpo"]["semantic"]["field_level_micro_f1"],
                ),
                "field_level_macro_f1": _delta(
                    stages["sft"]["semantic"]["field_level_macro_f1"],
                    stages["dpo"]["semantic"]["field_level_macro_f1"],
                ),
            },
        },
        "dpo_vs_baseline": {
            "syntax": {
                key: _delta(stages["baseline"]["syntax"][key], stages["dpo"]["syntax"][key])
                for key in stages["baseline"]["syntax"]
            },
            "semantic": {
                "field_level_micro_f1": _delta(
                    stages["baseline"]["semantic"]["field_level_micro_f1"],
                    stages["dpo"]["semantic"]["field_level_micro_f1"],
                ),
                "field_level_macro_f1": _delta(
                    stages["baseline"]["semantic"]["field_level_macro_f1"],
                    stages["dpo"]["semantic"]["field_level_macro_f1"],
                ),
            },
        },
    }
    row_evidence = _collect_row_evidence(
        baseline_predictions,
        sft_predictions,
        dpo_predictions,
    )
    summary = {
        "run_name": args.run_name,
        "classification_basis": "DPO relative to SFT",
        "stages": stages,
        "deltas": deltas,
        "row_evidence": row_evidence,
    }
    report_text = render_comparison_report(summary)

    summary_path = write_json(summary_output, summary)
    report_path = write_text(report_output, report_text)

    print(f"Comparison summary: {summary_path}")
    print(f"Comparison report: {report_path}")
    repo_targets = resolve_repo_artifact_targets(repo_root)
    if args.mirror_summary_to_repo:
        mirrored = mirror_small_artifact(summary_path, repo_targets["metrics"] / summary_path.name)
        print(f"Mirrored comparison summary: {mirrored}")
    if args.mirror_report_to_repo:
        mirrored = mirror_small_artifact(report_path, repo_targets["reports"] / report_path.name)
        print(f"Mirrored comparison report: {mirrored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
