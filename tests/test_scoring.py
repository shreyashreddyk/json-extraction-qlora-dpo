import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.scoring import (
    CandidateScorecard,
    RankedCandidate,
    build_ranked_candidate,
    rank_preference_candidates,
)
from json_ft.schemas import build_support_ticket_schema, validate_extraction_payload


def reference_payload() -> dict:
    return {
        "summary": "Business admin is locked out after an MFA reset and needs urgent account restoration before a rollout.",
        "issue_category": "account_access",
        "priority": "urgent",
        "product_area": "account_portal",
        "customer": {
            "name": "Ben Ortiz",
            "account_id": "BIZ-204",
            "plan_tier": "business",
        },
        "sentiment": "negative",
        "requires_human_followup": True,
        "actions_requested": ["Restore admin access", "Reset MFA enrollment"],
    }


class PreferenceScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = build_support_ticket_schema()
        self.gold = reference_payload()

    def _candidate(self, payload: dict | None, raw_text: str = "raw", candidate_index: int = 0) -> RankedCandidate:
        validation = validate_extraction_payload(payload, self.schema) if payload is not None else None
        return build_ranked_candidate(
            candidate_index=candidate_index,
            raw_text=raw_text,
            parsed_payload=payload,
            parse_error=None if payload is not None else "parse failed",
            validation=validation,
            reference_payload=self.gold,
        )

    def test_valid_json_beats_invalid_output(self) -> None:
        valid_candidate = self._candidate(self.gold, raw_text='{"ok": true}', candidate_index=0)
        invalid_candidate = self._candidate(None, raw_text="not-json", candidate_index=1)

        ranked = rank_preference_candidates([invalid_candidate, valid_candidate])

        self.assertEqual(ranked[0].candidate_index, 0)

    def test_schema_valid_beats_schema_invalid_output(self) -> None:
        invalid_payload = dict(self.gold)
        invalid_payload.pop("priority")
        valid_candidate = self._candidate(self.gold, candidate_index=0)
        invalid_candidate = self._candidate(invalid_payload, candidate_index=1)

        ranked = rank_preference_candidates([invalid_candidate, valid_candidate])

        self.assertEqual(ranked[0].candidate_index, 0)

    def test_zero_hallucinated_keys_beats_extra_key_output(self) -> None:
        hallucinated_payload = dict(self.gold)
        hallucinated_payload["unexpected"] = "extra"
        clean_candidate = self._candidate(self.gold, candidate_index=0)
        hallucinated_candidate = self._candidate(hallucinated_payload, candidate_index=1)

        ranked = rank_preference_candidates([hallucinated_candidate, clean_candidate])

        self.assertEqual(ranked[0].candidate_index, 0)

    def test_structured_field_match_count_is_recorded(self) -> None:
        partial_payload = dict(self.gold)
        partial_payload["priority"] = "high"
        candidate = self._candidate(partial_payload, candidate_index=0)

        self.assertEqual(candidate.scorecard.structured_field_matches, 7)
        self.assertFalse(candidate.scorecard.field_matches["priority"])

    def test_actions_requested_f1_is_scored_deterministically(self) -> None:
        partial_payload = dict(self.gold)
        partial_payload["actions_requested"] = ["Restore admin access"]
        candidate = self._candidate(partial_payload, candidate_index=0)

        self.assertAlmostEqual(candidate.scorecard.actions_f1, 2 / 3)
        self.assertEqual(candidate.scorecard.actions_tp, 1)
        self.assertEqual(candidate.scorecard.actions_fn, 1)

    def test_summary_faithfulness_beats_less_aligned_summary(self) -> None:
        faithful_payload = dict(self.gold)
        loose_payload = dict(self.gold)
        loose_payload["summary"] = "Customer has a general support problem and wants help soon."
        faithful_candidate = self._candidate(faithful_payload, candidate_index=0)
        loose_candidate = self._candidate(loose_payload, candidate_index=1)

        ranked = rank_preference_candidates([loose_candidate, faithful_candidate])

        self.assertEqual(ranked[0].candidate_index, 0)
        self.assertGreater(
            faithful_candidate.scorecard.summary_faithfulness_proxy,
            loose_candidate.scorecard.summary_faithfulness_proxy,
        )

    def test_more_concise_summary_breaks_ties(self) -> None:
        concise_card = CandidateScorecard(
            parses_json=True,
            schema_valid=True,
            hallucinated_paths=(),
            structured_field_matches=8,
            structured_field_total=8,
            structured_field_accuracy=1.0,
            field_matches={field_name: True for field_name in self.gold if field_name != "actions_requested" and field_name != "summary"},
            actions_f1=1.0,
            actions_tp=2,
            actions_fp=0,
            actions_fn=0,
            summary_faithfulness_proxy=0.8,
            summary_word_count=10,
            summary_overlap=8,
            summary_fp=2,
            summary_fn=2,
            null_handling_mistake_count=0,
            concision_score=1.0,
            dominant_failure_mode="clean",
            numeric_score=20.0,
            stable_text_key="concise",
        )
        verbose_card = CandidateScorecard(
            parses_json=True,
            schema_valid=True,
            hallucinated_paths=(),
            structured_field_matches=8,
            structured_field_total=8,
            structured_field_accuracy=1.0,
            field_matches={field_name: True for field_name in self.gold if field_name != "actions_requested" and field_name != "summary"},
            actions_f1=1.0,
            actions_tp=2,
            actions_fp=0,
            actions_fn=0,
            summary_faithfulness_proxy=0.8,
            summary_word_count=14,
            summary_overlap=8,
            summary_fp=6,
            summary_fn=2,
            null_handling_mistake_count=0,
            concision_score=0.7,
            dominant_failure_mode="clean",
            numeric_score=19.5,
            stable_text_key="verbose",
        )
        concise_candidate = RankedCandidate(
            candidate_index=0,
            raw_text="concise",
            parsed_payload=self.gold,
            parse_error=None,
            validation=validate_extraction_payload(self.gold, self.schema),
            normalized_completion="concise",
            dedupe_key="concise",
            scorecard=concise_card,
        )
        verbose_candidate = RankedCandidate(
            candidate_index=1,
            raw_text="verbose",
            parsed_payload=self.gold,
            parse_error=None,
            validation=validate_extraction_payload(self.gold, self.schema),
            normalized_completion="verbose",
            dedupe_key="verbose",
            scorecard=verbose_card,
        )

        ranked = rank_preference_candidates([verbose_candidate, concise_candidate])

        self.assertEqual(ranked[0].candidate_index, 0)

    def test_stable_text_tiebreak_is_deterministic(self) -> None:
        scorecard_a = CandidateScorecard(
            parses_json=False,
            schema_valid=False,
            hallucinated_paths=(),
            structured_field_matches=0,
            structured_field_total=8,
            structured_field_accuracy=0.0,
            field_matches={},
            actions_f1=0.0,
            actions_tp=0,
            actions_fp=0,
            actions_fn=0,
            summary_faithfulness_proxy=0.0,
            summary_word_count=0,
            summary_overlap=0,
            summary_fp=0,
            summary_fn=0,
            null_handling_mistake_count=0,
            concision_score=0.0,
            dominant_failure_mode="parse_failure",
            numeric_score=0.0,
            stable_text_key="a",
        )
        scorecard_b = CandidateScorecard(
            parses_json=False,
            schema_valid=False,
            hallucinated_paths=(),
            structured_field_matches=0,
            structured_field_total=8,
            structured_field_accuracy=0.0,
            field_matches={},
            actions_f1=0.0,
            actions_tp=0,
            actions_fp=0,
            actions_fn=0,
            summary_faithfulness_proxy=0.0,
            summary_word_count=0,
            summary_overlap=0,
            summary_fp=0,
            summary_fn=0,
            null_handling_mistake_count=0,
            concision_score=0.0,
            dominant_failure_mode="parse_failure",
            numeric_score=0.0,
            stable_text_key="b",
        )
        candidate_a = RankedCandidate(
            candidate_index=0,
            raw_text="A",
            parsed_payload=None,
            parse_error="parse failed",
            validation=None,
            normalized_completion=None,
            dedupe_key="A",
            scorecard=scorecard_a,
        )
        candidate_b = RankedCandidate(
            candidate_index=1,
            raw_text="B",
            parsed_payload=None,
            parse_error="parse failed",
            validation=None,
            normalized_completion=None,
            dedupe_key="B",
            scorecard=scorecard_b,
        )

        ranked = rank_preference_candidates([candidate_b, candidate_a])

        self.assertEqual(ranked[0].candidate_index, 0)


if __name__ == "__main__":
    unittest.main()
