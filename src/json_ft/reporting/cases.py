"""Case-study extraction helpers for the final project report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .loaders import ReportingBundle

STRUCTURED_PRF_FIELDS = (
    "issue_category",
    "priority",
    "product_area",
    "customer.name",
    "customer.account_id",
    "customer.plan_tier",
    "sentiment",
    "requires_human_followup",
)

LIST_PRF_FIELDS = ("actions_requested",)


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


def _syntax_tuple(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("parsed_payload") is not None),
        int(bool(row.get("schema_is_valid", False))),
        int(not row.get("unexpected_fields")),
    )


def _semantic_score(row: dict[str, Any]) -> float:
    predicted = row.get("parsed_payload")
    reference = row.get("reference_payload") or {}
    structured_match_count = sum(
        1
        for field_name in STRUCTURED_PRF_FIELDS
        if _nested_value(predicted, field_name) == _nested_value(reference, field_name)
    )
    predicted_actions = _list_tokens(_nested_value(predicted, LIST_PRF_FIELDS[0]))
    reference_actions = _list_tokens(_nested_value(reference, LIST_PRF_FIELDS[0]))
    overlap = predicted_actions & reference_actions
    tp = len(overlap)
    fp = len(predicted_actions - overlap)
    fn = len(reference_actions - overlap)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    actions_f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return (structured_match_count + actions_f1) / (len(STRUCTURED_PRF_FIELDS) + 1)


def _is_exact_match(row: dict[str, Any]) -> bool:
    return bool(
        row.get("parsed_payload") == row.get("reference_payload")
        and row.get("schema_is_valid")
        and not row.get("unexpected_fields")
        and not row.get("parse_error")
    )


def _semantic_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_match": _is_exact_match(row),
        "semantic_score": round(_semantic_score(row), 4),
    }


@dataclass(frozen=True)
class CaseStudy:
    """One cross-stage row-level example used in the final report."""

    category: str
    record_id: str
    input_text: str
    reference_payload: dict[str, Any]
    baseline_output: str
    sft_output: str
    dpo_output: str
    baseline_syntax: tuple[int, int, int]
    sft_syntax: tuple[int, int, int]
    dpo_syntax: tuple[int, int, int]
    baseline_semantics: dict[str, Any]
    sft_semantics: dict[str, Any]
    dpo_semantics: dict[str, Any]
    why_it_matters: str


def _prediction_index(rows: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {str(row["record_id"]): row for row in (rows or [])}


def _make_case_study(category: str, baseline_row: dict[str, Any], sft_row: dict[str, Any], dpo_row: dict[str, Any], why: str) -> CaseStudy:
    return CaseStudy(
        category=category,
        record_id=str(baseline_row["record_id"]),
        input_text=str(baseline_row.get("input_text", "")),
        reference_payload=baseline_row.get("reference_payload") or {},
        baseline_output=str(baseline_row.get("raw_output", "")),
        sft_output=str(sft_row.get("raw_output", "")),
        dpo_output=str(dpo_row.get("raw_output", "")),
        baseline_syntax=_syntax_tuple(baseline_row),
        sft_syntax=_syntax_tuple(sft_row),
        dpo_syntax=_syntax_tuple(dpo_row),
        baseline_semantics=_semantic_summary(baseline_row),
        sft_semantics=_semantic_summary(sft_row),
        dpo_semantics=_semantic_summary(dpo_row),
        why_it_matters=why,
    )


def extract_case_studies(bundle: ReportingBundle, max_per_category: int = 3) -> dict[str, list[CaseStudy]]:
    """Extract deterministic case studies from saved cross-stage predictions."""

    baseline_index = _prediction_index(bundle.baseline.predictions)
    sft_index = _prediction_index(bundle.sft.predictions)
    dpo_index = _prediction_index(bundle.dpo.predictions)
    shared_record_ids = sorted(set(baseline_index) & set(sft_index) & set(dpo_index))

    categories: dict[str, list[tuple[float, CaseStudy]]] = {
        "baseline_bad_to_sft_good": [],
        "sft_good_to_dpo_better": [],
        "sft_good_to_dpo_worse": [],
        "syntax_cleaned_up_semantics_unchanged": [],
        "unchanged_hard_failures": [],
    }

    for record_id in shared_record_ids:
        baseline_row = baseline_index[record_id]
        sft_row = sft_index[record_id]
        dpo_row = dpo_index[record_id]

        baseline_exact = _is_exact_match(baseline_row)
        sft_exact = _is_exact_match(sft_row)
        dpo_exact = _is_exact_match(dpo_row)
        baseline_score = _semantic_score(baseline_row)
        sft_score = _semantic_score(sft_row)
        dpo_score = _semantic_score(dpo_row)
        baseline_syntax = _syntax_tuple(baseline_row)
        sft_syntax = _syntax_tuple(sft_row)
        dpo_syntax = _syntax_tuple(dpo_row)

        if not baseline_exact and sft_exact:
            improvement = sft_score - baseline_score
            categories["baseline_bad_to_sft_good"].append(
                (
                    improvement,
                    _make_case_study(
                        "baseline_bad_to_sft_good",
                        baseline_row,
                        sft_row,
                        dpo_row,
                        (
                            "SFT converted a weak baseline row into a clean structured extraction. "
                            f"Semantic score moved {baseline_score:.4f} -> {sft_score:.4f}."
                        ),
                    ),
                )
            )

        if sft_syntax >= (1, 1, 1) and sft_score >= 0.75 and dpo_score > sft_score + 1e-9 and dpo_syntax >= sft_syntax:
            categories["sft_good_to_dpo_better"].append(
                (
                    dpo_score - sft_score,
                    _make_case_study(
                        "sft_good_to_dpo_better",
                        baseline_row,
                        sft_row,
                        dpo_row,
                        (
                            "DPO improved an already-good SFT row without giving up syntax discipline. "
                            f"Semantic score moved {sft_score:.4f} -> {dpo_score:.4f}."
                        ),
                    ),
                )
            )

        if sft_syntax >= (1, 1, 1) and sft_score >= 0.75 and (dpo_score + 1e-9 < sft_score or dpo_syntax < sft_syntax):
            categories["sft_good_to_dpo_worse"].append(
                (
                    sft_score - dpo_score + float(dpo_syntax < sft_syntax),
                    _make_case_study(
                        "sft_good_to_dpo_worse",
                        baseline_row,
                        sft_row,
                        dpo_row,
                        (
                            "DPO changed a strong SFT row in the wrong direction. "
                            f"Syntax {sft_syntax} -> {dpo_syntax}; semantic score {sft_score:.4f} -> {dpo_score:.4f}."
                        ),
                    ),
                )
            )

        if dpo_syntax > sft_syntax and abs(dpo_score - sft_score) <= 1e-9:
            categories["syntax_cleaned_up_semantics_unchanged"].append(
                (
                    float(dpo_syntax > sft_syntax),
                    _make_case_study(
                        "syntax_cleaned_up_semantics_unchanged",
                        baseline_row,
                        sft_row,
                        dpo_row,
                        (
                            "DPO cleaned up parse or schema behavior, but the semantic content stayed effectively unchanged. "
                            f"Syntax {sft_syntax} -> {dpo_syntax}; semantic score stayed {sft_score:.4f}."
                        ),
                    ),
                )
            )

        if (
            baseline_score < 0.5
            and sft_score < 0.5
            and dpo_score < 0.5
            and not baseline_exact
            and not sft_exact
            and not dpo_exact
        ):
            categories["unchanged_hard_failures"].append(
                (
                    1.5 - max(baseline_score, sft_score, dpo_score),
                    _make_case_study(
                        "unchanged_hard_failures",
                        baseline_row,
                        sft_row,
                        dpo_row,
                        (
                            "This row remained difficult across all three stages, which makes it a good failure-analysis anchor. "
                            f"Scores stayed baseline={baseline_score:.4f}, sft={sft_score:.4f}, dpo={dpo_score:.4f}."
                        ),
                    ),
                )
            )

    extracted: dict[str, list[CaseStudy]] = {}
    for label, candidates in categories.items():
        extracted[label] = [case for _, case in sorted(candidates, key=lambda item: item[0], reverse=True)[:max_per_category]]
    return extracted
