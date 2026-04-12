"""Deterministic scoring helpers for preference-pair ranking and audit trails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import re

from .schemas import SchemaConstraint, ValidationResult, format_support_ticket_json, validate_extraction_payload

STRUCTURED_PREFERENCE_FIELDS = (
    "issue_category",
    "priority",
    "product_area",
    "customer.name",
    "customer.account_id",
    "customer.plan_tier",
    "sentiment",
    "requires_human_followup",
)
NULLABLE_PREFERENCE_FIELDS = (
    "customer.name",
    "customer.account_id",
    "customer.plan_tier",
)


def _nested_value(payload: dict[str, Any] | None, field_path: str) -> Any:
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _token_f1(predicted: list[str], reference: list[str]) -> tuple[float, int, int, int]:
    predicted_counts: dict[str, int] = {}
    reference_counts: dict[str, int] = {}
    for token in predicted:
        predicted_counts[token] = predicted_counts.get(token, 0) + 1
    for token in reference:
        reference_counts[token] = reference_counts.get(token, 0) + 1

    overlap = 0
    for token, count in predicted_counts.items():
        overlap += min(count, reference_counts.get(token, 0))

    predicted_total = len(predicted)
    reference_total = len(reference)
    precision = overlap / predicted_total if predicted_total else 0.0
    recall = overlap / reference_total if reference_total else 0.0
    if precision + recall == 0:
        return 0.0, overlap, predicted_total - overlap, reference_total - overlap
    return 2 * precision * recall / (precision + recall), overlap, predicted_total - overlap, reference_total - overlap


def _action_f1(predicted_actions: Any, reference_actions: Any) -> tuple[float, int, int, int]:
    predicted_set = {str(item).strip() for item in predicted_actions or [] if str(item).strip()}
    reference_set = {str(item).strip() for item in reference_actions or [] if str(item).strip()}
    overlap = predicted_set & reference_set
    tp = len(overlap)
    fp = len(predicted_set - overlap)
    fn = len(reference_set - overlap)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0, tp, fp, fn
    return 2 * precision * recall / (precision + recall), tp, fp, fn


def _canonicalize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)


@dataclass(frozen=True)
class CandidateScorecard:
    """Detailed deterministic rubric used to rank sampled candidates."""

    parses_json: bool
    schema_valid: bool
    hallucinated_paths: tuple[str, ...]
    structured_field_matches: int
    structured_field_total: int
    structured_field_accuracy: float
    field_matches: dict[str, bool]
    actions_f1: float
    actions_tp: int
    actions_fp: int
    actions_fn: int
    summary_faithfulness_proxy: float
    summary_word_count: int
    summary_overlap: int
    summary_fp: int
    summary_fn: int
    null_handling_mistake_count: int
    concision_score: float
    dominant_failure_mode: str
    numeric_score: float
    stable_text_key: str

    @property
    def hallucinated_key_count(self) -> int:
        return len(self.hallucinated_paths)

    @property
    def no_hallucinated_keys(self) -> bool:
        return self.hallucinated_key_count == 0

    @property
    def ranking_key(self) -> tuple[Any, ...]:
        """Ascending sort key implementing the lexicographic preference rubric."""

        return (
            -int(self.parses_json),
            -int(self.schema_valid),
            -int(self.no_hallucinated_keys),
            self.hallucinated_key_count,
            -self.structured_field_matches,
            -self.actions_f1,
            -self.summary_faithfulness_proxy,
            self.null_handling_mistake_count,
            self.summary_word_count,
            -self.concision_score,
            self.stable_text_key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parses_json": self.parses_json,
            "schema_valid": self.schema_valid,
            "hallucinated_paths": list(self.hallucinated_paths),
            "hallucinated_key_count": self.hallucinated_key_count,
            "structured_field_matches": self.structured_field_matches,
            "structured_field_total": self.structured_field_total,
            "structured_field_accuracy": self.structured_field_accuracy,
            "field_matches": dict(self.field_matches),
            "actions_f1": self.actions_f1,
            "actions_tp": self.actions_tp,
            "actions_fp": self.actions_fp,
            "actions_fn": self.actions_fn,
            "summary_faithfulness_proxy": self.summary_faithfulness_proxy,
            "summary_word_count": self.summary_word_count,
            "summary_overlap": self.summary_overlap,
            "summary_fp": self.summary_fp,
            "summary_fn": self.summary_fn,
            "null_handling_mistake_count": self.null_handling_mistake_count,
            "concision_score": self.concision_score,
            "dominant_failure_mode": self.dominant_failure_mode,
            "numeric_score": self.numeric_score,
            "ranking_key": list(self.ranking_key),
        }


@dataclass(frozen=True)
class RankedCandidate:
    """Candidate completion plus parsed audit data and ranking details."""

    candidate_index: int
    raw_text: str
    parsed_payload: dict[str, Any] | None
    parse_error: str | None
    validation: ValidationResult | None
    normalized_completion: str | None
    dedupe_key: str
    scorecard: CandidateScorecard
    rank: int | None = None

    def to_audit_dict(self) -> dict[str, Any]:
        validation = self.validation
        return {
            "candidate_index": self.candidate_index,
            "rank": self.rank,
            "raw_text": self.raw_text,
            "parsed_payload": self.parsed_payload,
            "parse_error": self.parse_error,
            "normalized_completion": self.normalized_completion,
            "dedupe_key": self.dedupe_key,
            "schema_is_valid": validation.is_valid if validation is not None else False,
            "missing_fields": list(validation.missing_fields) if validation is not None else [],
            "unexpected_fields": list(validation.unexpected_fields) if validation is not None else [],
            "validation_issues": [
                {
                    "path": list(issue.path),
                    "issue_type": issue.issue_type,
                    "message": issue.message,
                }
                for issue in (validation.issues if validation is not None else ())
            ],
            "scorecard": self.scorecard.to_dict(),
        }


def score_payload_against_schema(payload: dict, schema: SchemaConstraint) -> int:
    """Backward-compatible schema-only score retained for older callers/tests."""

    validation = validate_extraction_payload(payload, schema)
    score = 100
    score -= len(validation.missing_fields) * 25
    score -= len(validation.unexpected_fields) * 10
    other_issues = len(validation.issues) - len(validation.missing_fields) - len(
        validation.unexpected_fields
    )
    score -= max(other_issues, 0) * 15
    return max(score, 0)


def choose_better_payload(chosen: dict, rejected: dict, schema: SchemaConstraint) -> bool:
    """Return True when the chosen payload outranks the rejected payload."""

    return score_payload_against_schema(chosen, schema) >= score_payload_against_schema(
        rejected,
        schema,
    )


def build_candidate_scorecard(
    *,
    parsed_payload: dict[str, Any] | None,
    validation: ValidationResult | None,
    reference_payload: dict[str, Any],
    stable_text_key: str,
) -> CandidateScorecard:
    """Score a single parsed candidate against the gold task labels."""

    field_matches = {
        field_name: _nested_value(parsed_payload, field_name) == _nested_value(reference_payload, field_name)
        for field_name in STRUCTURED_PREFERENCE_FIELDS
    }
    structured_field_matches = sum(1 for value in field_matches.values() if value)
    structured_field_total = len(STRUCTURED_PREFERENCE_FIELDS)
    structured_field_accuracy = (
        structured_field_matches / structured_field_total if structured_field_total else 0.0
    )

    actions_f1, actions_tp, actions_fp, actions_fn = _action_f1(
        _nested_value(parsed_payload, "actions_requested"),
        _nested_value(reference_payload, "actions_requested"),
    )
    predicted_summary = str(_nested_value(parsed_payload, "summary") or "")
    reference_summary = str(_nested_value(reference_payload, "summary") or "")
    summary_tokens = _normalized_tokens(predicted_summary)
    reference_tokens = _normalized_tokens(reference_summary)
    summary_faithfulness_proxy, summary_overlap, summary_fp, summary_fn = _token_f1(summary_tokens, reference_tokens)

    hallucinated_paths = tuple(sorted(validation.unexpected_fields)) if validation is not None else ()
    null_handling_mistake_count = sum(
        1
        for field_name in NULLABLE_PREFERENCE_FIELDS
        if _nested_value(reference_payload, field_name) is None
        and _nested_value(parsed_payload, field_name) not in (None, "", [])
    )
    if parsed_payload is None:
        dominant_failure_mode = "parse_failure"
    elif validation is None or not validation.is_valid:
        dominant_failure_mode = "schema_failure"
    elif hallucinated_paths:
        dominant_failure_mode = "hallucinated_keys"
    elif null_handling_mistake_count:
        dominant_failure_mode = "null_handling_mistake"
    elif structured_field_matches < structured_field_total or actions_f1 < 1.0:
        dominant_failure_mode = "semantic_mismatch"
    else:
        dominant_failure_mode = "clean"
    concision_score = 1.0 if len(summary_tokens) <= 32 else 32.0 / len(summary_tokens)
    numeric_score = (
        float(int(parsed_payload is not None)) * 5.0
        + float(int(bool(validation.is_valid) if validation is not None else False)) * 5.0
        - float(len(hallucinated_paths)) * 2.0
        + float(structured_field_matches) * 1.5
        + actions_f1 * 2.0
        + summary_faithfulness_proxy * 1.5
        - float(null_handling_mistake_count) * 1.5
        + concision_score * 0.5
    )
    return CandidateScorecard(
        parses_json=parsed_payload is not None,
        schema_valid=bool(validation.is_valid) if validation is not None else False,
        hallucinated_paths=hallucinated_paths,
        structured_field_matches=structured_field_matches,
        structured_field_total=structured_field_total,
        structured_field_accuracy=structured_field_accuracy,
        field_matches=field_matches,
        actions_f1=actions_f1,
        actions_tp=actions_tp,
        actions_fp=actions_fp,
        actions_fn=actions_fn,
        summary_faithfulness_proxy=summary_faithfulness_proxy,
        summary_word_count=len(summary_tokens),
        summary_overlap=summary_overlap,
        summary_fp=summary_fp,
        summary_fn=summary_fn,
        null_handling_mistake_count=null_handling_mistake_count,
        concision_score=concision_score,
        dominant_failure_mode=dominant_failure_mode,
        numeric_score=numeric_score,
        stable_text_key=stable_text_key,
    )


def build_ranked_candidate(
    *,
    candidate_index: int,
    raw_text: str,
    parsed_payload: dict[str, Any] | None,
    parse_error: str | None,
    validation: ValidationResult | None,
    reference_payload: dict[str, Any],
) -> RankedCandidate:
    """Construct a scored candidate from raw model output and parsed metadata."""

    stripped_text = raw_text.strip()
    normalized_completion = _canonicalize_json(parsed_payload) if parsed_payload is not None else None
    dedupe_key = normalized_completion if normalized_completion is not None else stripped_text
    scorecard = build_candidate_scorecard(
        parsed_payload=parsed_payload,
        validation=validation,
        reference_payload=reference_payload,
        stable_text_key=normalized_completion or stripped_text,
    )
    return RankedCandidate(
        candidate_index=candidate_index,
        raw_text=raw_text,
        parsed_payload=parsed_payload,
        parse_error=parse_error,
        validation=validation,
        normalized_completion=normalized_completion,
        dedupe_key=dedupe_key,
        scorecard=scorecard,
    )


def dedupe_ranked_candidates(candidates: list[RankedCandidate]) -> list[RankedCandidate]:
    """Drop duplicate candidates while preserving first occurrence order."""

    deduped: list[RankedCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.dedupe_key in seen:
            continue
        seen.add(candidate.dedupe_key)
        deduped.append(candidate)
    return deduped


def rank_preference_candidates(candidates: list[RankedCandidate]) -> list[RankedCandidate]:
    """Sort candidates from best to worst using the deterministic rubric."""

    ranked_candidates: list[RankedCandidate] = []
    for rank, candidate in enumerate(sorted(candidates, key=lambda item: item.scorecard.ranking_key), start=1):
        ranked_candidates.append(
            RankedCandidate(
                candidate_index=candidate.candidate_index,
                raw_text=candidate.raw_text,
                parsed_payload=candidate.parsed_payload,
                parse_error=candidate.parse_error,
                validation=candidate.validation,
                normalized_completion=candidate.normalized_completion,
                dedupe_key=candidate.dedupe_key,
                scorecard=candidate.scorecard,
                rank=rank,
            )
        )
    return ranked_candidates


def ranking_gap_is_strict(better: RankedCandidate, worse: RankedCandidate) -> bool:
    """Return True when the rubric produces a strict preference between candidates."""

    return better.scorecard.ranking_key != worse.scorecard.ranking_key


def select_rejected_candidate(ranked_candidates: list[RankedCandidate]) -> RankedCandidate | None:
    """Return the worst remaining candidate, preferring parseable JSON when available."""

    remaining = ranked_candidates[1:]
    if not remaining:
        return None
    parseable = [candidate for candidate in remaining if candidate.scorecard.parses_json]
    if parseable:
        return parseable[-1]
    return remaining[-1]


def explain_preference_decision(chosen: RankedCandidate, rejected: RankedCandidate) -> str:
    """Describe the first decisive ranking difference between chosen and rejected."""

    chosen_card = chosen.scorecard
    rejected_card = rejected.scorecard
    if chosen_card.parses_json and not rejected_card.parses_json:
        return "Chosen parsed into a JSON object while rejected did not."
    if chosen_card.schema_valid and not rejected_card.schema_valid:
        return "Chosen passed full schema validation while rejected did not."
    if chosen_card.hallucinated_key_count < rejected_card.hallucinated_key_count:
        return (
            "Chosen contained fewer hallucinated keys than rejected "
            f"({chosen_card.hallucinated_key_count} vs {rejected_card.hallucinated_key_count})."
        )
    if chosen_card.structured_field_matches > rejected_card.structured_field_matches:
        return (
            "Chosen matched more structured gold fields than rejected "
            f"({chosen_card.structured_field_matches} vs {rejected_card.structured_field_matches})."
        )
    if chosen_card.actions_f1 > rejected_card.actions_f1:
        return (
            "Chosen matched the requested actions more accurately than rejected "
            f"({chosen_card.actions_f1:.4f} vs {rejected_card.actions_f1:.4f})."
        )
    if chosen_card.null_handling_mistake_count < rejected_card.null_handling_mistake_count:
        return (
            "Chosen respected nullable customer fields better than rejected "
            f"({chosen_card.null_handling_mistake_count} vs {rejected_card.null_handling_mistake_count} mistakes)."
        )
    if chosen_card.summary_faithfulness_proxy > rejected_card.summary_faithfulness_proxy:
        return (
            "Chosen summary stayed closer to the gold summary than rejected "
            f"({chosen_card.summary_faithfulness_proxy:.4f} vs {rejected_card.summary_faithfulness_proxy:.4f})."
        )
    if chosen_card.summary_word_count < rejected_card.summary_word_count:
        return (
            "Chosen summary was more concise than rejected on an otherwise-tied semantic score "
            f"({chosen_card.summary_word_count} vs {rejected_card.summary_word_count} words)."
        )
    return "Chosen won the final stable text tie-break after all higher-priority rubric checks tied."


def pair_selection_skip_reason(ranked_candidates: list[RankedCandidate]) -> str | None:
    """Return a skip reason when ranked candidates cannot form a reliable pair."""

    if len(ranked_candidates) < 2:
        return "insufficient_distinct_candidates"

    top_candidate = ranked_candidates[0]
    if not top_candidate.scorecard.schema_valid:
        return "no_schema_valid_candidate"

    if len(ranked_candidates) > 1 and not ranking_gap_is_strict(top_candidate, ranked_candidates[1]):
        return "non_unique_top_candidate"

    rejected_candidate = select_rejected_candidate(ranked_candidates)
    if rejected_candidate is None:
        return "insufficient_distinct_candidates"
    if not ranking_gap_is_strict(top_candidate, rejected_candidate):
        return "no_strict_score_gap"
    return None


def rejected_completion_text(candidate: RankedCandidate) -> str:
    """Return the persisted rejected text using canonical JSON when possible."""

    return candidate.normalized_completion or candidate.raw_text


def chosen_completion_text(candidate: RankedCandidate, schema: SchemaConstraint) -> str:
    """Return the persisted chosen text using schema-normalized JSON."""

    if candidate.parsed_payload is None:
        raise ValueError("Chosen candidate must have a parsed payload.")
    return format_support_ticket_json(candidate.parsed_payload, schema)
