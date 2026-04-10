"""Task-specific preference-pair generation helpers for DPO preparation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset_adapters import DatasetSplit, adapt_source_record, build_messages_sft_example, build_sft_example
from .inference import InferenceBackend, InferenceRequest
from .manifests import LatestModelManifest, load_latest_model_manifest
from .scoring import (
    RankedCandidate,
    build_ranked_candidate,
    chosen_completion_text,
    dedupe_ranked_candidates,
    explain_preference_decision,
    pair_selection_skip_reason,
    rank_preference_candidates,
    rejected_completion_text,
    select_rejected_candidate,
)
from .schemas import SchemaConstraint, build_support_ticket_schema, dump_support_ticket_payload
from .utils import load_yaml, read_json, read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class PreferenceBuildConfig:
    """Resolved config for task-specific preference-pair generation."""

    config_path: Path
    profile_name: str
    latest_model_manifest_path: Path
    model_name_or_path: str
    adapter_path: str | None
    revision: str | None
    trust_remote_code: bool
    torch_dtype: str | None
    device_map: str | None
    input_path: Path
    source_format: str
    source_split: str
    prompt_source: str
    candidate_count: int
    sample_limit: int | None
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    base_seed: int
    artifact_names: dict[str, str]
    training: dict[str, Any]


@dataclass(frozen=True)
class PreferenceOutputPaths:
    """Runtime artifact paths for one preference-pair build run."""

    output_dir: Path
    pairs_path: Path
    audit_path: Path
    summary_path: Path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(repo_root: Path, value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _load_latest_model_manifest_at_path(path: Path, repo_root: Path) -> LatestModelManifest | None:
    default_path = (repo_root / "artifacts" / "checkpoints" / "latest_model.json").resolve()
    if not path.exists():
        return None
    if path.resolve() == default_path:
        return load_latest_model_manifest(repo_root)
    return LatestModelManifest(**read_json(path))


def resolve_preference_config(
    *,
    config_path: Path,
    repo_root: Path,
    profile_name: str,
    input_path: Path | None = None,
    source_format: str | None = None,
    source_split: str | None = None,
    model_name_or_path: str | None = None,
    adapter_path: str | None = None,
) -> PreferenceBuildConfig:
    """Load the DPO config and resolve the active preference-pair settings."""

    resolved_config_path = _resolve_path(repo_root, config_path)
    if resolved_config_path is None or not resolved_config_path.exists():
        raise FileNotFoundError(f"DPO config does not exist: {config_path}")

    config = load_yaml(resolved_config_path)
    profile_overrides = config.get("profiles", {}).get(profile_name, {})
    merged = _deep_merge(config, profile_overrides)

    model_config = merged.get("model", {})
    pair_generation_config = merged.get("pair_generation", {})
    generation_config = pair_generation_config.get("generation", {})
    artifact_config = merged.get("artifacts", {})

    latest_model_manifest_path = _resolve_path(
        repo_root,
        model_config.get("latest_model_manifest", "artifacts/checkpoints/latest_model.json"),
    )
    manifest = (
        _load_latest_model_manifest_at_path(latest_model_manifest_path, repo_root)
        if latest_model_manifest_path is not None
        else None
    )
    if manifest is not None and manifest.stage != "sft":
        raise ValueError(
            "Latest model manifest must point to an SFT adapter before building preference pairs."
        )

    resolved_model_name = model_name_or_path or model_config.get("base_model") or (
        manifest.base_model if manifest is not None else None
    )
    resolved_adapter_path = adapter_path or model_config.get("adapter_path") or (
        manifest.adapter_path if manifest is not None else None
    )
    if not resolved_model_name:
        raise ValueError(
            "Could not resolve a base model. Provide model.base_model in configs/dpo.yaml "
            "or promote an SFT adapter into artifacts/checkpoints/latest_model.json."
        )

    resolved_input_path = _resolve_path(
        repo_root,
        input_path or pair_generation_config.get("input_path", "data/fixtures/support_tickets.jsonl"),
    )
    if resolved_input_path is None or not resolved_input_path.exists():
        raise FileNotFoundError(f"Preference source data does not exist: {resolved_input_path}")

    return PreferenceBuildConfig(
        config_path=resolved_config_path,
        profile_name=profile_name,
        latest_model_manifest_path=latest_model_manifest_path or (repo_root / "artifacts" / "checkpoints" / "latest_model.json"),
        model_name_or_path=resolved_model_name,
        adapter_path=str(_resolve_path(repo_root, resolved_adapter_path)) if resolved_adapter_path else None,
        revision=model_config.get("revision"),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        torch_dtype=model_config.get("torch_dtype", "auto"),
        device_map=model_config.get("device_map"),
        input_path=resolved_input_path,
        source_format=source_format or pair_generation_config.get("source_format", "json_extraction"),
        source_split=source_split or pair_generation_config.get("source_split", "train"),
        prompt_source=pair_generation_config.get("prompt_source", "messages"),
        candidate_count=int(pair_generation_config.get("candidate_count", 6)),
        sample_limit=pair_generation_config.get("sample_limit"),
        max_new_tokens=int(generation_config.get("max_new_tokens", 256)),
        temperature=float(generation_config.get("temperature", 0.8)),
        top_p=float(generation_config.get("top_p", 0.95)),
        do_sample=bool(generation_config.get("do_sample", True)),
        base_seed=int(generation_config.get("base_seed", 17)),
        artifact_names={
            "pairs_filename": artifact_config.get("pairs_filename", "{run_name}_dpo_pairs.jsonl"),
            "audit_filename": artifact_config.get("audit_filename", "{run_name}_preference_audit.jsonl"),
            "summary_filename": artifact_config.get("summary_filename", "{run_name}_preference_summary.json"),
        },
        training=dict(merged.get("training", {})),
    )


def resolve_preference_output_paths(
    output_dir: Path,
    run_name: str,
    artifact_names: dict[str, str],
) -> PreferenceOutputPaths:
    """Resolve runtime output artifact paths for one preference build."""

    output_dir.mkdir(parents=True, exist_ok=True)
    return PreferenceOutputPaths(
        output_dir=output_dir,
        pairs_path=(output_dir / artifact_names["pairs_filename"].format(run_name=run_name)).resolve(),
        audit_path=(output_dir / artifact_names["audit_filename"].format(run_name=run_name)).resolve(),
        summary_path=(output_dir / artifact_names["summary_filename"].format(run_name=run_name)).resolve(),
    )


def load_preference_samples(
    *,
    input_path: Path,
    source_format: str,
    source_split: str,
    sample_limit: int | None = None,
) -> list[Any]:
    """Load canonical task rows and filter them to the selected preference split."""

    requested_split = DatasetSplit(source_split)
    samples = [adapt_source_record(row, source_format) for row in read_jsonl(input_path)]
    filtered_samples = [sample for sample in samples if sample.split == requested_split]
    if sample_limit is not None:
        filtered_samples = filtered_samples[:sample_limit]
    return filtered_samples


def _build_request_bundle(sample: Any, config: PreferenceBuildConfig, candidate_index: int) -> tuple[str, list[dict[str, str]], InferenceRequest]:
    prompt_example = build_sft_example(sample)
    messages_example = build_messages_sft_example(sample)
    request = InferenceRequest(
        prompt=prompt_example.prompt,
        messages=messages_example.messages if config.prompt_source == "messages" else None,
        record_id=sample.record_id,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=config.do_sample,
        prompt_source=config.prompt_source,
        seed=config.base_seed + candidate_index,
    )
    return prompt_example.prompt, messages_example.messages, request


def _score_gap(chosen: RankedCandidate, rejected: RankedCandidate) -> dict[str, float]:
    chosen_card = chosen.scorecard
    rejected_card = rejected.scorecard
    return {
        "parses_json_gap": float(int(chosen_card.parses_json) - int(rejected_card.parses_json)),
        "schema_valid_gap": float(int(chosen_card.schema_valid) - int(rejected_card.schema_valid)),
        "hallucinated_key_reduction": float(
            rejected_card.hallucinated_key_count - chosen_card.hallucinated_key_count
        ),
        "structured_field_match_gap": float(
            chosen_card.structured_field_matches - rejected_card.structured_field_matches
        ),
        "actions_f1_gap": chosen_card.actions_f1 - rejected_card.actions_f1,
        "summary_f1_gap": chosen_card.summary_f1 - rejected_card.summary_f1,
        "summary_word_reduction": float(
            rejected_card.summary_word_count - chosen_card.summary_word_count
        ),
    }


def _average_gap(gaps: list[dict[str, float]]) -> dict[str, float]:
    if not gaps:
        return {
            "parses_json_gap": 0.0,
            "schema_valid_gap": 0.0,
            "hallucinated_key_reduction": 0.0,
            "structured_field_match_gap": 0.0,
            "actions_f1_gap": 0.0,
            "summary_f1_gap": 0.0,
            "summary_word_reduction": 0.0,
        }
    keys = gaps[0].keys()
    return {
        key: sum(gap[key] for gap in gaps) / len(gaps)
        for key in keys
    }


def build_preference_run(
    *,
    samples: list[Any],
    backend: InferenceBackend,
    config: PreferenceBuildConfig,
    schema: SchemaConstraint | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    """Generate candidates, rank them, and assemble DPO-ready preference pairs."""

    active_schema = schema or build_support_ticket_schema()
    pair_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, Any]] = []
    skipped_counts: Counter[str] = Counter()
    total_candidates = 0
    parseable_candidates = 0
    schema_valid_candidates = 0
    chosen_schema_valid_count = 0
    rejected_schema_valid_count = 0
    score_gaps: list[dict[str, float]] = []

    for sample in samples:
        prompt_text, message_prompt, _ = _build_request_bundle(sample, config, 0)
        gold_payload = dump_support_ticket_payload(sample.target, active_schema)
        raw_candidates: list[RankedCandidate] = []

        for candidate_index in range(config.candidate_count):
            _, _, request = _build_request_bundle(sample, config, candidate_index)
            response = backend.generate(request)
            candidate = build_ranked_candidate(
                candidate_index=candidate_index,
                raw_text=response.text,
                parsed_payload=response.parsed_payload,
                parse_error=response.parse_error,
                validation=response.validation,
                reference_payload=gold_payload,
            )
            raw_candidates.append(candidate)
            total_candidates += 1
            if candidate.scorecard.parses_json:
                parseable_candidates += 1
            if candidate.scorecard.schema_valid:
                schema_valid_candidates += 1

        deduped_candidates = dedupe_ranked_candidates(raw_candidates)
        ranked_candidates = rank_preference_candidates(deduped_candidates)
        skip_reason = pair_selection_skip_reason(ranked_candidates)
        chosen_candidate = ranked_candidates[0] if ranked_candidates else None
        rejected_candidate = select_rejected_candidate(ranked_candidates) if ranked_candidates else None

        audit_row: dict[str, Any] = {
            "record_id": sample.record_id,
            "source_dataset": sample.source_dataset,
            "split": sample.split.value,
            "metadata": sample.metadata,
            "input_text": sample.input_text,
            "prompt": prompt_text,
            "prompt_source": config.prompt_source,
            "prompt_messages": message_prompt,
            "gold_payload": gold_payload,
            "candidate_count_requested": config.candidate_count,
            "candidate_count_generated": len(raw_candidates),
            "candidate_count_after_dedup": len(ranked_candidates),
            "generation": {
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "do_sample": config.do_sample,
                "base_seed": config.base_seed,
            },
            "candidates": [candidate.to_audit_dict() for candidate in ranked_candidates],
            "chosen_index": chosen_candidate.candidate_index if chosen_candidate is not None and skip_reason is None else None,
            "rejected_index": rejected_candidate.candidate_index if rejected_candidate is not None and skip_reason is None else None,
            "decision_rationale": None,
            "skip_reason": skip_reason,
            "score_gap": None,
        }

        if skip_reason is not None:
            skipped_counts[skip_reason] += 1
            audit_row["decision_rationale"] = f"Skipped because {skip_reason.replace('_', ' ')}."
            audit_rows.append(audit_row)
            continue

        assert chosen_candidate is not None
        assert rejected_candidate is not None
        rationale = explain_preference_decision(chosen_candidate, rejected_candidate)
        gap = _score_gap(chosen_candidate, rejected_candidate)
        audit_row["decision_rationale"] = rationale
        audit_row["score_gap"] = gap
        audit_rows.append(audit_row)
        pair_rows.append(
            {
                "prompt": prompt_text,
                "chosen": chosen_completion_text(chosen_candidate, active_schema),
                "rejected": rejected_completion_text(rejected_candidate),
            }
        )
        chosen_schema_valid_count += int(chosen_candidate.scorecard.schema_valid)
        rejected_schema_valid_count += int(rejected_candidate.scorecard.schema_valid)
        score_gaps.append(gap)

    pair_count = len(pair_rows)
    summary = {
        "config_path": str(config.config_path),
        "profile": config.profile_name,
        "input_path": str(config.input_path),
        "source_format": config.source_format,
        "source_split": config.source_split,
        "model_name_or_path": config.model_name_or_path,
        "adapter_path": config.adapter_path,
        "prompt_source": config.prompt_source,
        "source_row_count": len(samples),
        "pair_count": pair_count,
        "pair_emission_rate": pair_count / len(samples) if samples else 0.0,
        "skipped_count": sum(skipped_counts.values()),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "candidate_count_total": total_candidates,
        "candidate_json_valid_rate": parseable_candidates / total_candidates if total_candidates else 0.0,
        "candidate_schema_pass_rate": schema_valid_candidates / total_candidates if total_candidates else 0.0,
        "chosen_schema_valid_rate": chosen_schema_valid_count / pair_count if pair_count else 0.0,
        "rejected_schema_valid_rate": rejected_schema_valid_count / pair_count if pair_count else 0.0,
        "average_chosen_vs_rejected_score_gap": _average_gap(score_gaps),
    }
    return pair_rows, audit_rows, summary


def write_preference_artifacts(
    *,
    paths: PreferenceOutputPaths,
    pair_rows: list[dict[str, str]],
    audit_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Persist the DPO pair dataset, audit log, and summary report."""

    pairs_path = write_jsonl(paths.pairs_path, pair_rows)
    audit_path = write_jsonl(paths.audit_path, audit_rows)
    summary_path = write_json(paths.summary_path, summary)
    return pairs_path, audit_path, summary_path
