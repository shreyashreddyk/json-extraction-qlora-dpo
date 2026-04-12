"""Task-specific preference-pair generation helpers for DPO preparation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher

from .dataset_adapters import DatasetSplit, adapt_source_record, build_messages_sft_example, build_sft_example
from .inference import InferenceBackend, InferenceRequest
from .manifests import LatestModelManifest, load_latest_model_manifest
from .sampling import SampleSelectionMetadata, select_rows
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
from .stage_metadata import build_data_pipeline_metadata
from .utils import load_yaml, read_json, read_jsonl, write_json, write_jsonl


PROFILE_ALIASES = {"colab_full": "full"}


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
    build_summary_path: Path | None
    composition_summary_path: Path | None
    source_format: str
    source_split: str
    prompt_source: str
    candidate_count: int
    inference_batch_size: int
    sample_limit: int | None
    sample_percent: float | None
    sample_seed: int
    quality_gates: dict[str, Any]
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
    diagnostics_path: Path
    plot_paths: dict[str, Path]


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


def _resolve_mirrored_artifact_path(repo_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.exists():
        return path
    for subdir in ("checkpoints", "metrics", "reports"):
        candidate = (repo_root / "artifacts" / subdir / path.name).resolve()
        if candidate.exists():
            return candidate
    return path


def _load_optional_json(repo_root: Path, path: Path | None) -> dict[str, Any] | None:
    resolved_path = _resolve_mirrored_artifact_path(repo_root, path)
    if resolved_path is None or not resolved_path.exists():
        return None
    payload = read_json(resolved_path)
    return payload if isinstance(payload, dict) else None


def _resolve_sft_source_from_latest_manifest(
    repo_root: Path,
    manifest: LatestModelManifest | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    if manifest is None or manifest.stage != "dpo":
        return None, None

    for candidate in manifest.report_paths:
        candidate_path = _resolve_mirrored_artifact_path(repo_root, _resolve_path(repo_root, candidate))
        payload = _load_optional_json(repo_root, candidate_path)
        if not payload:
            continue
        if payload.get("stage") != "dpo":
            continue

        source_manifest_path = _resolve_mirrored_artifact_path(
            repo_root,
            _resolve_path(repo_root, payload.get("source_sft_manifest_path")),
        )
        source_manifest_payload = _load_optional_json(repo_root, source_manifest_path)
        if source_manifest_payload is not None:
            return source_manifest_path, source_manifest_payload

        source_adapter_path = payload.get("source_adapter_path")
        if source_adapter_path:
            return candidate_path, {
                "stage": "sft",
                "base_model": payload.get("base_model"),
                "adapter_path": source_adapter_path,
            }

    return None, None


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
    inference_batch_size: int | None = None,
    sample_percent: float | None = None,
    sample_seed: int | None = None,
) -> PreferenceBuildConfig:
    """Load the DPO config and resolve the active preference-pair settings."""

    resolved_config_path = _resolve_path(repo_root, config_path)
    if resolved_config_path is None or not resolved_config_path.exists():
        raise FileNotFoundError(f"DPO config does not exist: {config_path}")

    config = load_yaml(resolved_config_path)
    requested_profile_name = profile_name
    profile_name = PROFILE_ALIASES.get(profile_name, profile_name)
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
    source_sft_manifest_path = _resolve_path(repo_root, model_config.get("source_sft_manifest"))
    source_sft_manifest_payload = _load_optional_json(repo_root, source_sft_manifest_path)
    fallback_sft_manifest_path, fallback_sft_manifest_payload = _resolve_sft_source_from_latest_manifest(
        repo_root,
        manifest,
    )

    if source_sft_manifest_payload is None:
        source_sft_manifest_path = fallback_sft_manifest_path
        source_sft_manifest_payload = fallback_sft_manifest_payload

    if manifest is not None and manifest.stage not in {"sft", "dpo"}:
        raise ValueError(
            "Latest model manifest must point to an SFT or DPO adapter before building preference pairs."
        )

    resolved_model_name = (
        model_name_or_path
        or model_config.get("base_model")
        or (source_sft_manifest_payload.get("base_model") if source_sft_manifest_payload is not None else None)
        or (manifest.base_model if manifest is not None else None)
    )
    resolved_adapter_path = (
        adapter_path
        or model_config.get("adapter_path")
        or (source_sft_manifest_payload.get("adapter_path") if source_sft_manifest_payload is not None else None)
        or (manifest.adapter_path if manifest is not None and manifest.stage == "sft" else None)
    )
    if not resolved_model_name:
        raise ValueError(
            "Could not resolve a base model. Provide model.base_model in configs/dpo.yaml "
            "or promote an SFT adapter into artifacts/checkpoints/latest_model.json."
        )
    if manifest is not None and manifest.stage == "dpo" and not resolved_adapter_path:
        raise ValueError(
            "Latest model manifest points to DPO, but the source SFT adapter could not be resolved. "
            "Set model.source_sft_manifest or model.adapter_path in configs/dpo.yaml, or keep the DPO manifest "
            "linked to its source SFT manifest."
        )

    resolved_input_path = _resolve_path(
        repo_root,
        input_path or pair_generation_config.get("input_path", "data/manifests/support_tickets_canonical.jsonl"),
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
        build_summary_path=_resolve_path(
            repo_root,
            pair_generation_config.get("build_summary_path", "data/manifests/support_tickets_dataset_build_summary.json"),
        ),
        composition_summary_path=_resolve_path(
            repo_root,
            pair_generation_config.get(
                "composition_summary_path",
                "artifacts/metrics/support_tickets_dataset_composition.json",
            ),
        ),
        source_format=source_format or pair_generation_config.get("source_format", "json_extraction"),
        source_split=source_split or pair_generation_config.get("source_split", "train"),
        prompt_source=pair_generation_config.get("prompt_source", "messages"),
        candidate_count=int(pair_generation_config.get("candidate_count", 6)),
        inference_batch_size=int(
            inference_batch_size
            if inference_batch_size is not None
            else pair_generation_config.get("inference_batch_size", 1)
        ),
        sample_limit=pair_generation_config.get("sample_limit"),
        sample_percent=sample_percent
        if sample_percent is not None
        else pair_generation_config.get("sample_percent"),
        sample_seed=int(
            sample_seed if sample_seed is not None else pair_generation_config.get("sample_seed", 17)
        ),
        quality_gates=dict(pair_generation_config.get("quality_gates", {})),
        max_new_tokens=int(generation_config.get("max_new_tokens", 256)),
        temperature=float(generation_config.get("temperature", 0.8)),
        top_p=float(generation_config.get("top_p", 0.95)),
        do_sample=bool(generation_config.get("do_sample", True)),
        base_seed=int(generation_config.get("base_seed", 17)),
        artifact_names={
            "pairs_filename": artifact_config.get("pairs_filename", "{run_name}_dpo_pairs.jsonl"),
            "audit_filename": artifact_config.get("audit_filename", "{run_name}_preference_audit.jsonl"),
            "summary_filename": artifact_config.get(
                "preference_summary_filename",
                artifact_config.get("summary_filename", "{run_name}_preference_summary.json"),
            ),
            "diagnostics_filename": artifact_config.get("diagnostics_filename", "{run_name}_preference_diagnostics.json"),
            "pair_emission_curve_filename": artifact_config.get(
                "pair_emission_curve_filename",
                "{run_name}_preference_pair_emission.png",
            ),
            "skipped_reasons_curve_filename": artifact_config.get(
                "skipped_reasons_curve_filename",
                "{run_name}_preference_skipped_reasons.png",
            ),
            "score_gap_curve_filename": artifact_config.get(
                "score_gap_curve_filename",
                "{run_name}_preference_score_gap.png",
            ),
            "source_quality_curve_filename": artifact_config.get(
                "source_quality_curve_filename",
                "{run_name}_preference_source_quality.png",
            ),
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
        diagnostics_path=(output_dir / artifact_names["diagnostics_filename"].format(run_name=run_name)).resolve(),
        plot_paths={
            "pair_emission": (
                output_dir / artifact_names["pair_emission_curve_filename"].format(run_name=run_name)
            ).resolve(),
            "skipped_reasons": (
                output_dir / artifact_names["skipped_reasons_curve_filename"].format(run_name=run_name)
            ).resolve(),
            "score_gap": (
                output_dir / artifact_names["score_gap_curve_filename"].format(run_name=run_name)
            ).resolve(),
            "source_quality": (
                output_dir / artifact_names["source_quality_curve_filename"].format(run_name=run_name)
            ).resolve(),
        },
    )


def load_preference_samples(
    *,
    input_path: Path,
    source_format: str,
    source_split: str,
    sample_limit: int | None = None,
    sample_percent: float | None = None,
    sample_seed: int = 17,
) -> tuple[list[Any], SampleSelectionMetadata]:
    """Load canonical task rows and filter them to the selected preference split."""

    requested_split = DatasetSplit(source_split)
    samples = [adapt_source_record(row, source_format) for row in read_jsonl(input_path)]
    filtered_samples = [sample for sample in samples if sample.split == requested_split]
    selection = select_rows(
        filtered_samples,
        sample_limit=sample_limit,
        sample_percent=sample_percent,
        sample_seed=sample_seed,
    )
    return selection.rows, selection.metadata


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


def _chunk_items(items: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0:
        raise ValueError("inference_batch_size must be greater than zero.")
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _score_gap(chosen: RankedCandidate, rejected: RankedCandidate) -> dict[str, float]:
    chosen_card = chosen.scorecard
    rejected_card = rejected.scorecard
    return {
        "numeric_score_gap": chosen_card.numeric_score - rejected_card.numeric_score,
        "parses_json_gap": float(int(chosen_card.parses_json) - int(rejected_card.parses_json)),
        "schema_valid_gap": float(int(chosen_card.schema_valid) - int(rejected_card.schema_valid)),
        "hallucinated_key_reduction": float(
            rejected_card.hallucinated_key_count - chosen_card.hallucinated_key_count
        ),
        "structured_field_match_gap": float(
            chosen_card.structured_field_matches - rejected_card.structured_field_matches
        ),
        "actions_f1_gap": chosen_card.actions_f1 - rejected_card.actions_f1,
        "summary_faithfulness_gap": (
            chosen_card.summary_faithfulness_proxy - rejected_card.summary_faithfulness_proxy
        ),
        "null_handling_gap": float(
            rejected_card.null_handling_mistake_count - chosen_card.null_handling_mistake_count
        ),
        "concision_gap": chosen_card.concision_score - rejected_card.concision_score,
        "summary_word_reduction": float(
            rejected_card.summary_word_count - chosen_card.summary_word_count
        ),
    }


def _average_gap(gaps: list[dict[str, float]]) -> dict[str, float]:
    if not gaps:
        return {
            "numeric_score_gap": 0.0,
            "parses_json_gap": 0.0,
            "schema_valid_gap": 0.0,
            "hallucinated_key_reduction": 0.0,
            "structured_field_match_gap": 0.0,
            "actions_f1_gap": 0.0,
            "summary_faithfulness_gap": 0.0,
            "null_handling_gap": 0.0,
            "concision_gap": 0.0,
            "summary_word_reduction": 0.0,
        }
    keys = gaps[0].keys()
    return {
        key: sum(gap[key] for gap in gaps) / len(gaps)
        for key in keys
    }


def _similarity_ratio(chosen: RankedCandidate, rejected: RankedCandidate) -> float:
    chosen_text = chosen.normalized_completion or chosen.raw_text
    rejected_text = rejected.normalized_completion or rejected.raw_text
    return SequenceMatcher(a=chosen_text, b=rejected_text).ratio()


def _skip_reason_for_quality_gates(
    *,
    chosen_candidate: RankedCandidate | None,
    rejected_candidate: RankedCandidate | None,
    config: PreferenceBuildConfig,
) -> str | None:
    if chosen_candidate is None or rejected_candidate is None:
        return None

    minimum_score_gap = float(config.quality_gates.get("minimum_score_gap", 0.0))
    if chosen_candidate.scorecard.numeric_score - rejected_candidate.scorecard.numeric_score < minimum_score_gap:
        return "score_gap_below_threshold"

    max_similarity_ratio = float(config.quality_gates.get("max_similarity_ratio", 1.0))
    similarity_ratio = _similarity_ratio(chosen_candidate, rejected_candidate)
    if similarity_ratio >= max_similarity_ratio:
        return "chosen_rejected_too_similar"

    if bool(config.quality_gates.get("reject_same_failure_mode", True)):
        chosen_mode = chosen_candidate.scorecard.dominant_failure_mode
        rejected_mode = rejected_candidate.scorecard.dominant_failure_mode
        if chosen_mode == rejected_mode and chosen_mode != "clean":
            return "same_failure_mode"

    if bool(config.quality_gates.get("require_chosen_schema_valid", True)) and not chosen_candidate.scorecard.schema_valid:
        return "chosen_not_schema_valid"

    return None


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in live environments
        raise RuntimeError(
            "matplotlib is required to render preference diagnostic plots."
        ) from exc
    return plt


def _render_bar_plot(
    *,
    pyplot: Any,
    values: dict[str, float],
    output_path: Path,
    title: str,
    ylabel: str,
) -> str | None:
    if not values:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = pyplot.figure(figsize=(8, 4.5))
    labels = list(values.keys())
    series = [values[label] for label in labels]
    pyplot.bar(labels, series, color="#1f77b4")
    pyplot.xticks(rotation=20, ha="right")
    pyplot.title(title)
    pyplot.ylabel(ylabel)
    pyplot.tight_layout()
    figure.savefig(output_path, dpi=160)
    pyplot.close(figure)
    return str(output_path)


def _render_histogram_plot(
    *,
    pyplot: Any,
    values: list[float],
    output_path: Path,
    title: str,
    xlabel: str,
) -> str | None:
    if not values:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = pyplot.figure(figsize=(8, 4.5))
    pyplot.hist(values, bins=min(10, max(3, len(values))), color="#ff7f0e", edgecolor="black")
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel("Count")
    pyplot.tight_layout()
    figure.savefig(output_path, dpi=160)
    pyplot.close(figure)
    return str(output_path)


def _render_preference_plots(
    *,
    paths: PreferenceOutputPaths,
    summary: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, str]:
    try:
        pyplot = _load_pyplot()
    except RuntimeError:
        return {}
    plot_outputs: dict[str, str] = {}
    pair_emission_values = {
        "emitted": float(summary.get("pair_count", 0)),
        "skipped": float(summary.get("skipped_count", 0)),
    }
    rendered = _render_bar_plot(
        pyplot=pyplot,
        values=pair_emission_values,
        output_path=paths.plot_paths["pair_emission"],
        title="Preference Pair Emission",
        ylabel="Rows",
    )
    if rendered:
        plot_outputs["pair_emission"] = rendered

    rendered = _render_bar_plot(
        pyplot=pyplot,
        values={key: float(value) for key, value in summary.get("skipped_counts", {}).items()},
        output_path=paths.plot_paths["skipped_reasons"],
        title="Skipped Rows by Reason",
        ylabel="Rows",
    )
    if rendered:
        plot_outputs["skipped_reasons"] = rendered

    rendered = _render_histogram_plot(
        pyplot=pyplot,
        values=[float(value) for value in diagnostics.get("score_gap_distribution", [])],
        output_path=paths.plot_paths["score_gap"],
        title="Preference Score Gap Distribution",
        xlabel="Chosen - Rejected Numeric Score",
    )
    if rendered:
        plot_outputs["score_gap"] = rendered

    rendered = _render_bar_plot(
        pyplot=pyplot,
        values={
            dataset: float(source_metrics.get("pair_emission_rate", 0.0))
            for dataset, source_metrics in diagnostics.get("pair_quality_by_source_dataset", {}).items()
        },
        output_path=paths.plot_paths["source_quality"],
        title="Pair Quality by Source Dataset",
        ylabel="Emission Rate",
    )
    if rendered:
        plot_outputs["source_quality"] = rendered
    return plot_outputs


def build_preference_run(
    *,
    samples: list[Any],
    backend: InferenceBackend,
    config: PreferenceBuildConfig,
    schema: SchemaConstraint | None = None,
    source_subset_metadata: SampleSelectionMetadata | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Generate candidates, rank them, and assemble DPO-ready preference pairs."""

    active_schema = schema or build_support_ticket_schema()
    pair_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    skipped_counts: Counter[str] = Counter()
    total_candidates = 0
    parseable_candidates = 0
    schema_valid_candidates = 0
    chosen_schema_valid_count = 0
    rejected_schema_valid_count = 0
    score_gaps: list[dict[str, float]] = []
    score_gap_distribution: list[float] = []
    source_pair_counts: Counter[str] = Counter()
    source_row_counts: Counter[str] = Counter()
    source_skip_counts: dict[str, Counter[str]] = {}
    candidate_buckets: dict[str, list[RankedCandidate]] = {sample.record_id: [] for sample in samples}

    for sample in samples:
        source_row_counts[str(sample.source_dataset)] += 1
        source_skip_counts.setdefault(str(sample.source_dataset), Counter())
    prompt_bundle_by_record: dict[str, tuple[str, list[dict[str, str]], dict[str, Any]]] = {}
    for sample in samples:
        prompt_text, message_prompt, _ = _build_request_bundle(sample, config, 0)
        gold_payload = dump_support_ticket_payload(sample.target, active_schema)
        prompt_bundle_by_record[sample.record_id] = (prompt_text, message_prompt, gold_payload)

    for candidate_index in range(config.candidate_count):
        print(f"Candidate round {candidate_index + 1}/{config.candidate_count}")
        request_batch: list[tuple[Any, InferenceRequest]] = []
        for sample in samples:
            _, _, request = _build_request_bundle(sample, config, candidate_index)
            request_batch.append((sample, request))

        for batch_index, batch in enumerate(_chunk_items(request_batch, config.inference_batch_size), start=1):
            batch_start = (batch_index - 1) * config.inference_batch_size + 1
            batch_end = batch_start + len(batch) - 1
            print(
                f"Generating rows {batch_start}-{batch_end}/{len(request_batch)} "
                f"for candidate {candidate_index + 1}"
            )
            requests = [request for _, request in batch]
            if config.inference_batch_size > 1 and hasattr(backend, "generate_batch"):
                responses = backend.generate_batch(requests)
            else:
                responses = [backend.generate(request) for request in requests]

            for (sample, _request), response in zip(batch, responses, strict=True):
                gold_payload = prompt_bundle_by_record[sample.record_id][2]
                candidate = build_ranked_candidate(
                    candidate_index=candidate_index,
                    raw_text=response.text,
                    parsed_payload=response.parsed_payload,
                    parse_error=response.parse_error,
                    validation=response.validation,
                    reference_payload=gold_payload,
                )
                candidate_buckets[sample.record_id].append(candidate)
                total_candidates += 1
                if candidate.scorecard.parses_json:
                    parseable_candidates += 1
                if candidate.scorecard.schema_valid:
                    schema_valid_candidates += 1

    for sample in samples:
        prompt_text, message_prompt, _gold_payload = prompt_bundle_by_record[sample.record_id]
        raw_candidates = candidate_buckets[sample.record_id]
        deduped_candidates = dedupe_ranked_candidates(raw_candidates)
        ranked_candidates = rank_preference_candidates(deduped_candidates)
        skip_reason = pair_selection_skip_reason(ranked_candidates)
        chosen_candidate = ranked_candidates[0] if ranked_candidates else None
        rejected_candidate = select_rejected_candidate(ranked_candidates) if ranked_candidates else None
        if skip_reason is None:
            skip_reason = _skip_reason_for_quality_gates(
                chosen_candidate=chosen_candidate,
                rejected_candidate=rejected_candidate,
                config=config,
            )

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
            "candidate_diagnostics": {
                "parseable_json_count": sum(1 for candidate in ranked_candidates if candidate.scorecard.parses_json),
                "schema_valid_count": sum(1 for candidate in ranked_candidates if candidate.scorecard.schema_valid),
                "dominant_failure_modes": [candidate.scorecard.dominant_failure_mode for candidate in ranked_candidates],
            },
        }

        if skip_reason is not None:
            skipped_counts[skip_reason] += 1
            source_skip_counts[str(sample.source_dataset)][skip_reason] += 1
            audit_row["decision_rationale"] = f"Skipped because {skip_reason.replace('_', ' ')}."
            audit_rows.append(audit_row)
            continue

        assert chosen_candidate is not None
        assert rejected_candidate is not None
        rationale = explain_preference_decision(chosen_candidate, rejected_candidate)
        gap = _score_gap(chosen_candidate, rejected_candidate)
        similarity_ratio = _similarity_ratio(chosen_candidate, rejected_candidate)
        audit_row["decision_rationale"] = rationale
        audit_row["score_gap"] = gap
        audit_row["similarity_ratio"] = similarity_ratio
        audit_rows.append(audit_row)
        pair_rows.append(
            {
                "record_id": sample.record_id,
                "source_dataset": sample.source_dataset,
                "prompt": prompt_text,
                "chosen": chosen_completion_text(chosen_candidate, active_schema),
                "rejected": rejected_completion_text(rejected_candidate),
                "score_gap": gap,
                "similarity_ratio": similarity_ratio,
                "decision_rationale": rationale,
            }
        )
        source_pair_counts[str(sample.source_dataset)] += 1
        chosen_schema_valid_count += int(chosen_candidate.scorecard.schema_valid)
        rejected_schema_valid_count += int(rejected_candidate.scorecard.schema_valid)
        score_gaps.append(gap)
        score_gap_distribution.append(float(gap["numeric_score_gap"]))

    pair_count = len(pair_rows)
    pair_quality_by_source_dataset = {
        dataset: {
            "source_row_count": source_row_counts[dataset],
            "pair_count": source_pair_counts[dataset],
            "pair_emission_rate": (
                source_pair_counts[dataset] / source_row_counts[dataset] if source_row_counts[dataset] else 0.0
            ),
            "skipped_counts": dict(sorted(source_skip_counts.get(dataset, Counter()).items())),
        }
        for dataset in sorted(source_row_counts)
    }
    summary = {
        "config_path": str(config.config_path),
        "profile": config.profile_name,
        "input_path": str(config.input_path),
        "build_summary_path": str(config.build_summary_path) if config.build_summary_path else None,
        "composition_summary_path": (
            str(config.composition_summary_path) if config.composition_summary_path else None
        ),
        "source_format": config.source_format,
        "source_split": config.source_split,
        "model_name_or_path": config.model_name_or_path,
        "adapter_path": config.adapter_path,
        "prompt_source": config.prompt_source,
        "inference_batch_size": config.inference_batch_size,
        "quality_gates": config.quality_gates,
        "source_row_count": len(samples),
        "subset_selection": source_subset_metadata.to_dict() if source_subset_metadata else {},
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
        "score_gap_distribution": score_gap_distribution,
        "pair_quality_by_source_dataset": pair_quality_by_source_dataset,
    }
    diagnostics = {
        "data_pipeline": build_data_pipeline_metadata(
            repo_root=config.config_path.parents[1],
            build_summary_path=config.build_summary_path,
            composition_summary_path=config.composition_summary_path,
        ),
        "pair_quality_by_source_dataset": pair_quality_by_source_dataset,
        "subset_selection": summary["subset_selection"],
        "inference_batch_size": config.inference_batch_size,
        "score_gap_distribution": score_gap_distribution,
        "candidate_count_total": total_candidates,
        "candidate_json_valid_rate": summary["candidate_json_valid_rate"],
        "candidate_schema_pass_rate": summary["candidate_schema_pass_rate"],
        "chosen_schema_valid_rate": summary["chosen_schema_valid_rate"],
        "rejected_schema_valid_rate": summary["rejected_schema_valid_rate"],
        "skipped_counts": summary["skipped_counts"],
    }
    return pair_rows, audit_rows, summary, diagnostics


def write_preference_artifacts(
    *,
    paths: PreferenceOutputPaths,
    pair_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    diagnostics: dict[str, Any],
) -> tuple[Path, Path, Path, Path, dict[str, str]]:
    """Persist the DPO pair dataset, audit log, and summary report."""

    pairs_path = write_jsonl(paths.pairs_path, pair_rows)
    audit_path = write_jsonl(paths.audit_path, audit_rows)
    summary_path = write_json(paths.summary_path, summary)
    diagnostics_path = write_json(paths.diagnostics_path, diagnostics)
    plot_paths = _render_preference_plots(paths=paths, summary=summary, diagnostics=diagnostics)
    return pairs_path, audit_path, summary_path, diagnostics_path, plot_paths
