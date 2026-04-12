"""Shared dataset build pipeline for multi-source manifest generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import math
import random
import urllib.request
import zipfile

from .augmentations import generate_augmentations
from .data_registry import DatasetSource, SourceGroup, SourceType, load_dataset_registry
from .dataset_adapters import (
    DatasetSplit,
    JsonExtractionSample,
    adapt_source_record,
    eval_manifest_record,
    messages_record,
    prompt_completion_record,
)
from .schemas import build_support_ticket_schema
from .source_adapters import AdapterReject, MAPPING_VERSION, adapt_source_row, reject_row
from .utils import load_yaml, read_jsonl, write_json, write_jsonl, write_text


@dataclass(frozen=True)
class BuildOutputs:
    canonical_output: Path
    prompt_completion_output: Path
    messages_output: Path
    eval_output: Path
    summary_output: Path
    composition_json_output: Path
    composition_csv_output: Path
    composition_markdown_output: Path


@dataclass(frozen=True)
class BuildProfile:
    profile_name: str
    seed: int
    raw_root: Path
    runtime_root: Path | None
    eval_ratio: float
    train_target_count: int | None
    eval_target_count: int | None
    max_source_share: float
    max_synthetic_share: float
    include_sources: tuple[str, ...]
    exclude_sources: tuple[str, ...]
    include_groups: tuple[str, ...]
    source_group_weights: dict[str, float]
    source_weight_overrides: dict[str, float]
    outputs: BuildOutputs
    eval_allow_synthetic: bool
    augmentation_enabled: bool
    synthetic_source_name: str
    schema_discipline_enabled: bool
    prefer_local_fixtures: bool
    allow_fixture_fallback: bool


@dataclass(frozen=True)
class LoadedSourceRows:
    source: DatasetSource
    records: list[dict[str, Any]]
    resolved_source: str
    used_local_fixture: bool = False


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_output(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def load_build_profile(
    *,
    config_path: Path,
    repo_root: Path,
    profile_name: str,
    seed_override: int | None = None,
    raw_root: Path | None = None,
    runtime_root: Path | None = None,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    include_groups: list[str] | None = None,
) -> BuildProfile:
    """Resolve the active data-build profile."""

    config = load_yaml(config_path)
    profile_overrides = config.get("profiles", {}).get(profile_name, {})
    merged = _deep_merge(config, profile_overrides)
    outputs = merged.get("outputs", {})

    resolved_raw_root = raw_root or _resolve_output(repo_root, merged.get("raw_root", "data/fixtures/source_adapter_samples"))
    resolved_runtime_root = runtime_root or (
        _resolve_output(repo_root, merged["runtime_root"]) if merged.get("runtime_root") else None
    )

    return BuildProfile(
        profile_name=profile_name,
        seed=int(seed_override if seed_override is not None else merged.get("seed", 17)),
        raw_root=resolved_raw_root,
        runtime_root=resolved_runtime_root,
        eval_ratio=float(merged.get("eval_ratio", 0.2)),
        train_target_count=merged.get("train_target_count"),
        eval_target_count=merged.get("eval_target_count"),
        max_source_share=float(merged.get("max_source_share", 0.45)),
        max_synthetic_share=float(merged.get("max_synthetic_share", 0.3)),
        include_sources=tuple(include_sources if include_sources is not None else merged.get("include_sources", [])),
        exclude_sources=tuple(exclude_sources if exclude_sources is not None else merged.get("exclude_sources", [])),
        include_groups=tuple(include_groups if include_groups is not None else merged.get("include_groups", [])),
        source_group_weights={str(key): float(value) for key, value in merged.get("source_group_weights", {}).items()},
        source_weight_overrides={str(key): float(value) for key, value in merged.get("source_weight_overrides", {}).items()},
        outputs=BuildOutputs(
            canonical_output=_resolve_output(repo_root, outputs["canonical_output"]),
            prompt_completion_output=_resolve_output(repo_root, outputs["prompt_completion_output"]),
            messages_output=_resolve_output(repo_root, outputs["messages_output"]),
            eval_output=_resolve_output(repo_root, outputs["eval_output"]),
            summary_output=_resolve_output(repo_root, outputs["summary_output"]),
            composition_json_output=_resolve_output(repo_root, outputs["composition_json_output"]),
            composition_csv_output=_resolve_output(repo_root, outputs["composition_csv_output"]),
            composition_markdown_output=_resolve_output(repo_root, outputs["composition_markdown_output"]),
        ),
        eval_allow_synthetic=bool(merged.get("eval_allow_synthetic", False)),
        augmentation_enabled=bool(merged.get("augmentation", {}).get("enabled", True)),
        synthetic_source_name=str(merged.get("augmentation", {}).get("synthetic_source_name", "synthetic_hardening_v1")),
        schema_discipline_enabled=bool(merged.get("schema_discipline_enabled", False)),
        prefer_local_fixtures=bool(merged.get("prefer_local_fixtures", True)),
        allow_fixture_fallback=bool(merged.get("allow_fixture_fallback", True)),
    )


def _select_active_sources(sources: list[DatasetSource], profile: BuildProfile) -> list[DatasetSource]:
    included_sources = set(profile.include_sources)
    included_groups = set(profile.include_groups)
    excluded_sources = set(profile.exclude_sources)

    active: list[DatasetSource] = []
    for source in sources:
        if source.dataset_name in excluded_sources:
            continue
        if source.source_group == SourceGroup.SCHEMA_DISCIPLINE_DATA and not profile.schema_discipline_enabled:
            if source.dataset_name not in included_sources and source.source_group.value not in included_groups:
                continue
        should_include = source.enabled_by_default
        if source.dataset_name in included_sources or source.source_group.value in included_groups:
            should_include = True
        if should_include:
            active.append(source)
    return active


def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _load_csv_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def _download_missing_source(*, source: DatasetSource, destination: Path) -> Path | None:
    """Download a missing local source when the registry provides a public fetch URL."""

    download_url = source.extra.get("download_url")
    if not download_url:
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_member_name = source.extra.get("archive_member_name")
    if archive_member_name or str(download_url).endswith(".zip"):
        archive_path = destination.with_suffix(destination.suffix + ".zip")
        urllib.request.urlretrieve(str(download_url), archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            member_name = str(archive_member_name) if archive_member_name else None
            if member_name is None:
                csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if not csv_members:
                    raise FileNotFoundError(
                        f"Downloaded archive for {source.dataset_name} did not contain a CSV file: {archive_path}"
                    )
                member_name = csv_members[0]
            with archive.open(member_name) as extracted, destination.open("wb") as output_handle:
                output_handle.write(extracted.read())
        archive_path.unlink(missing_ok=True)
        return destination

    urllib.request.urlretrieve(str(download_url), destination)
    return destination


def _load_huggingface_dataset(
    source: DatasetSource,
    repo_root: Path,
    raw_root: Path,
    *,
    prefer_local_fixtures: bool,
) -> tuple[list[dict[str, Any]], str, bool]:
    fixture_path = source.resolve_local_fixture_path(repo_root)
    if prefer_local_fixtures and fixture_path is not None and fixture_path.exists():
        if fixture_path.suffix == ".csv":
            return _load_csv_file(fixture_path), str(fixture_path.resolve()), True
        return _load_jsonl_file(fixture_path), str(fixture_path.resolve()), True

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in live environments
        raise RuntimeError(
            "datasets is required to load Hugging Face sources. Install the train dependencies or use local fixtures."
        ) from exc

    split = source.hf_split or "train"
    cache_dir = (raw_root / "huggingface_cache").resolve()
    dataset = load_dataset(source.source_uri_or_path, split=split, cache_dir=str(cache_dir))
    return [dict(row) for row in dataset], source.source_uri_or_path, False


def load_source_rows(
    *,
    source: DatasetSource,
    repo_root: Path,
    raw_root: Path,
    prefer_local_fixtures: bool,
    allow_fixture_fallback: bool,
) -> LoadedSourceRows:
    """Load raw rows for one registry source."""

    if source.source_type == SourceType.GENERATED:
        return LoadedSourceRows(source=source, records=[], resolved_source=source.source_uri_or_path)

    if source.source_type == SourceType.HUGGINGFACE:
        rows, resolved_source, used_local_fixture = _load_huggingface_dataset(
            source,
            repo_root,
            raw_root,
            prefer_local_fixtures=prefer_local_fixtures,
        )
        return LoadedSourceRows(
            source=source,
            records=rows,
            resolved_source=resolved_source,
            used_local_fixture=used_local_fixture,
        )

    path = source.resolve_source_path(repo_root, raw_root)
    if path is None:
        raise FileNotFoundError(f"Could not resolve source path for {source.dataset_name}")
    if not path.exists():
        downloaded_path = _download_missing_source(source=source, destination=path)
        if downloaded_path is not None:
            path = downloaded_path
        fixture_path = source.resolve_local_fixture_path(repo_root)
        if not path.exists() and allow_fixture_fallback and fixture_path is not None and fixture_path.exists():
            path = fixture_path
        elif not path.exists():
            raise FileNotFoundError(f"Source data does not exist for {source.dataset_name}: {path}")
    if source.source_type == SourceType.LOCAL_CSV or path.suffix == ".csv":
        rows = _load_csv_file(path)
    else:
        rows = _load_jsonl_file(path)
    fixture_path = source.resolve_local_fixture_path(repo_root)
    return LoadedSourceRows(
        source=source,
        records=rows,
        resolved_source=str(path.resolve()),
        used_local_fixture=(
            fixture_path is not None and fixture_path.exists() and fixture_path.resolve() == path.resolve()
        ),
    )


def _hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _normalize_split(split_hint: str | None) -> str | None:
    if not split_hint:
        return None
    lowered = split_hint.strip().lower()
    if lowered in {"train", "training"}:
        return DatasetSplit.TRAIN.value
    if lowered in {"eval", "evaluation", "test", "validation"}:
        return DatasetSplit.EVAL.value
    return None


def assign_split(
    *,
    source_dataset: str,
    source_record_id: str,
    split_hint: str | None,
    eval_ratio: float,
) -> DatasetSplit:
    """Assign a stable split, honoring explicit source splits when present."""

    normalized = _normalize_split(split_hint)
    if normalized is not None:
        return DatasetSplit(normalized)
    digest = int(_hash_text(f"{source_dataset}:{source_record_id}")[:8], 16)
    threshold = int(eval_ratio * 1000)
    bucket = digest % 1000
    return DatasetSplit.EVAL if bucket < threshold else DatasetSplit.TRAIN


def _source_weight(source: DatasetSource, profile: BuildProfile) -> float:
    group_weight = profile.source_group_weights.get(source.source_group.value, 1.0)
    override = profile.source_weight_overrides.get(source.dataset_name)
    return float(override if override is not None else source.default_inclusion_weight * group_weight)


def adapt_loaded_rows(
    *,
    loaded: LoadedSourceRows,
    profile: BuildProfile,
    repo_root: Path,
) -> tuple[list[JsonExtractionSample], list[AdapterReject]]:
    """Adapt raw source rows into canonical samples."""

    accepted: list[JsonExtractionSample] = []
    rejected: list[AdapterReject] = []
    resolved_source = loaded.resolved_source
    ingested_at = datetime.now(UTC).isoformat()

    for raw in loaded.records:
        try:
            draft = adapt_source_row(loaded.source.adapter_name, raw)
            split = assign_split(
                source_dataset=loaded.source.dataset_name,
                source_record_id=draft.record_id,
                split_hint=draft.split_hint,
                eval_ratio=profile.eval_ratio,
            )
            raw_hash = _hash_text(json.dumps(raw, sort_keys=True, ensure_ascii=True))
            metadata = dict(draft.metadata)
            metadata.update(
                {
                    "source_group": loaded.source.source_group.value,
                    "source_type": loaded.source.source_type.value,
                    "source_record_id": draft.record_id,
                    "source_uri_or_path": resolved_source,
                    "adapter_name": loaded.source.adapter_name,
                    "license_note": loaded.source.license_note,
                    "synthetic": bool(metadata.get("synthetic", False)),
                    "augmentation_kind": metadata.get("augmentation_kind"),
                    "parent_record_id": metadata.get("parent_record_id"),
                    "lineage_root_id": metadata.get("lineage_root_id", draft.record_id),
                    "split_origin": _normalize_split(draft.split_hint) or "stable_hash",
                    "mapping_version": MAPPING_VERSION,
                    "ingested_at_utc": ingested_at,
                    "raw_hash": raw_hash,
                }
            )
            accepted.append(
                adapt_source_record(
                    {
                        "record_id": draft.record_id,
                        "split": split.value,
                        "source_dataset": loaded.source.dataset_name,
                        "input_text": draft.input_text,
                        "target": draft.target.model_dump(),
                        "metadata": metadata,
                    },
                    "json_extraction",
                )
            )
        except Exception as exc:
            rejected.append(reject_row(loaded.source.adapter_name, raw, str(exc)))
    return accepted, rejected


def _deterministic_shuffle(samples: list[JsonExtractionSample], seed: int) -> list[JsonExtractionSample]:
    rnd = random.Random(seed)
    copied = list(samples)
    rnd.shuffle(copied)
    return copied


def _sample_rows(
    grouped_samples: dict[str, list[JsonExtractionSample]],
    *,
    target_count: int | None,
    weights: dict[str, float],
    max_source_share: float | None,
    seed: int,
) -> list[JsonExtractionSample]:
    """Deterministically sample rows with simple source-aware balancing."""

    if target_count is None:
        merged = []
        for sample_rows in grouped_samples.values():
            merged.extend(sample_rows)
        return sorted(merged, key=lambda sample: (sample.split.value, sample.source_dataset, sample.record_id))

    shuffled = {source: _deterministic_shuffle(rows, seed + index) for index, (source, rows) in enumerate(sorted(grouped_samples.items()))}
    total_weight = sum(weights.get(source, 1.0) for source in shuffled) or 1.0
    provisional_counts: dict[str, int] = {}
    for source, rows in shuffled.items():
        desired = max(1, round(target_count * weights.get(source, 1.0) / total_weight))
        provisional_counts[source] = min(len(rows), desired)

    selected_total = sum(provisional_counts.values())
    if selected_total < target_count:
        for source, rows in shuffled.items():
            while provisional_counts[source] < len(rows) and selected_total < target_count:
                provisional_counts[source] += 1
                selected_total += 1
    if selected_total > target_count:
        for source in sorted(provisional_counts, key=lambda name: provisional_counts[name], reverse=True):
            while provisional_counts[source] > 0 and selected_total > target_count:
                provisional_counts[source] -= 1
                selected_total -= 1

    if max_source_share is not None:
        max_rows_per_source = max(1, math.floor(target_count * max_source_share))
        for source in provisional_counts:
            provisional_counts[source] = min(provisional_counts[source], max_rows_per_source)

    selected: list[JsonExtractionSample] = []
    for source, rows in shuffled.items():
        selected.extend(rows[: provisional_counts[source]])
    return sorted(selected, key=lambda sample: (sample.source_dataset, sample.record_id))


def _enforce_eval_policy(samples: list[JsonExtractionSample], profile: BuildProfile) -> list[JsonExtractionSample]:
    filtered: list[JsonExtractionSample] = []
    for sample in samples:
        synthetic = bool(sample.metadata.get("synthetic", False))
        if sample.split == DatasetSplit.EVAL and synthetic and not profile.eval_allow_synthetic:
            continue
        filtered.append(sample)
    return filtered


def _cap_synthetic_rows(train_samples: list[JsonExtractionSample], profile: BuildProfile) -> list[JsonExtractionSample]:
    synthetic_rows = [sample for sample in train_samples if bool(sample.metadata.get("synthetic", False))]
    real_rows = [sample for sample in train_samples if not bool(sample.metadata.get("synthetic", False))]
    if not synthetic_rows:
        return train_samples
    max_synthetic = math.floor(len(real_rows) * profile.max_synthetic_share / max(1.0 - profile.max_synthetic_share, 0.01))
    synthetic_groups: dict[str, list[JsonExtractionSample]] = defaultdict(list)
    for sample in synthetic_rows:
        synthetic_groups[sample.source_dataset].append(sample)
    synthetic_rows = _sample_rows(
        synthetic_groups,
        target_count=max_synthetic,
        weights={source_dataset: 1.0 for source_dataset in synthetic_groups},
        max_source_share=None,
        seed=profile.seed + 200,
    )
    return sorted(real_rows + synthetic_rows, key=lambda sample: (sample.source_dataset, sample.record_id))


def _leakage_summary(samples: list[JsonExtractionSample]) -> dict[str, Any]:
    eval_roots = {
        sample.metadata.get("lineage_root_id", sample.record_id)
        for sample in samples
        if sample.split == DatasetSplit.EVAL
    }
    leaked_train_rows = [
        sample.record_id
        for sample in samples
        if sample.split == DatasetSplit.TRAIN and sample.metadata.get("lineage_root_id", sample.record_id) in eval_roots
    ]
    return {
        "eval_root_count": len(eval_roots),
        "leaked_train_row_count": len(leaked_train_rows),
        "leaked_train_rows": leaked_train_rows[:10],
        "is_lineage_clean": not leaked_train_rows,
    }


def _length_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "avg": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 2),
    }


def summarize_dataset(
    *,
    samples: list[JsonExtractionSample],
    rejects: list[AdapterReject],
    profile: BuildProfile,
    registry: list[DatasetSource],
) -> dict[str, Any]:
    """Build a detailed dataset summary for review."""

    schema = build_support_ticket_schema()
    split_counts = Counter(sample.split.value for sample in samples)
    source_counts = Counter(sample.source_dataset for sample in samples)
    source_group_counts = Counter(sample.metadata.get("source_group", "unknown") for sample in samples)
    issue_category_counts = Counter(sample.target.issue_category.value for sample in samples)
    priority_counts = Counter(sample.target.priority.value for sample in samples)
    product_area_counts = Counter(sample.target.product_area.value for sample in samples)
    sentiment_counts = Counter(sample.target.sentiment.value for sample in samples)
    synthetic_count = sum(1 for sample in samples if bool(sample.metadata.get("synthetic", False)))
    prompt_lengths = [len(prompt_completion_record(sample)["prompt"]) for sample in samples if sample.split == DatasetSplit.TRAIN]
    summary_lengths = [len(sample.target.summary) for sample in samples]
    reject_counts = Counter(reject.reason for reject in rejects)
    reject_counts_by_source = Counter(reject.adapter_name for reject in rejects)
    total = len(samples) or 1
    null_rates = {
        "customer.name": sum(1 for sample in samples if sample.target.customer.name is None) / total,
        "customer.account_id": sum(1 for sample in samples if sample.target.customer.account_id is None) / total,
        "customer.plan_tier": sum(1 for sample in samples if sample.target.customer.plan_tier is None) / total,
    }
    dominance = {
        source: round(count / total, 4)
        for source, count in sorted(source_counts.items())
    }
    return {
        "schema": {"name": schema.name, "version": schema.version},
        "profile": profile.profile_name,
        "seed": profile.seed,
        "raw_root": str(profile.raw_root),
        "runtime_root": str(profile.runtime_root) if profile.runtime_root is not None else None,
        "prefer_local_fixtures": profile.prefer_local_fixtures,
        "allow_fixture_fallback": profile.allow_fixture_fallback,
        "registry_sources": [source.dataset_name for source in registry],
        "total_rows": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "source_group_counts": dict(sorted(source_group_counts.items())),
        "issue_category_counts": dict(sorted(issue_category_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
        "product_area_counts": dict(sorted(product_area_counts.items())),
        "sentiment_counts": dict(sorted(sentiment_counts.items())),
        "synthetic_row_count": synthetic_count,
        "synthetic_row_rate": round(synthetic_count / total, 4),
        "nullable_field_null_rates": {key: round(value, 4) for key, value in sorted(null_rates.items())},
        "source_dominance_share": dominance,
        "prompt_length_chars": _length_stats(prompt_lengths),
        "summary_length_chars": _length_stats(summary_lengths),
        "adapter_reject_count": len(rejects),
        "adapter_reject_counts": dict(sorted(reject_counts.items())),
        "adapter_reject_counts_by_source": dict(sorted(reject_counts_by_source.items())),
        "adapter_reject_examples": [asdict(reject) for reject in rejects[:10]],
        "leakage_checks": _leakage_summary(samples),
    }


def _composition_rows(samples: list[JsonExtractionSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[JsonExtractionSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.source_dataset, sample.split.value)].append(sample)
    for (source_dataset, split), sample_rows in sorted(grouped.items()):
        total = len(sample_rows)
        synthetic = sum(1 for sample in sample_rows if bool(sample.metadata.get("synthetic", False)))
        rows.append(
            {
                "source_dataset": source_dataset,
                "split": split,
                "row_count": total,
                "synthetic_row_count": synthetic,
                "synthetic_row_rate": round(synthetic / total, 4) if total else 0.0,
                "issue_category_counts": dict(sorted(Counter(sample.target.issue_category.value for sample in sample_rows).items())),
                "priority_counts": dict(sorted(Counter(sample.target.priority.value for sample in sample_rows).items())),
            }
        )
    return rows


def _composition_markdown(summary: dict[str, Any], composition_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Support Tickets Dataset Composition",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Total rows: `{summary['total_rows']}`",
        f"- Split counts: `{summary['split_counts']}`",
        f"- Synthetic row rate: `{summary['synthetic_row_rate']}`",
        f"- Leakage clean: `{summary['leakage_checks']['is_lineage_clean']}`",
        "",
        "## Per-Source Split Counts",
        "",
        "| Source | Split | Rows | Synthetic Rows | Synthetic Rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in composition_rows:
        lines.append(
            f"| {row['source_dataset']} | {row['split']} | {row['row_count']} | {row['synthetic_row_count']} | {row['synthetic_row_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Review Gates",
            "",
            f"- Source dominance share: `{summary['source_dominance_share']}`",
            f"- Nullable field null rates: `{summary['nullable_field_null_rates']}`",
            f"- Adapter reject counts: `{summary['adapter_reject_counts']}`",
            f"- Prompt length chars: `{summary['prompt_length_chars']}`",
            f"- Summary length chars: `{summary['summary_length_chars']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_composition_artifacts(
    *,
    summary: dict[str, Any],
    composition_rows: list[dict[str, Any]],
    outputs: BuildOutputs,
) -> tuple[Path, Path, Path]:
    composition_json = write_json(outputs.composition_json_output, {"summary": summary, "rows": composition_rows})
    csv_lines = ["source_dataset,split,row_count,synthetic_row_count,synthetic_row_rate"]
    for row in composition_rows:
        csv_lines.append(
            ",".join(
                [
                    row["source_dataset"],
                    row["split"],
                    str(row["row_count"]),
                    str(row["synthetic_row_count"]),
                    str(row["synthetic_row_rate"]),
                ]
            )
        )
    composition_csv = write_text(outputs.composition_csv_output, "\n".join(csv_lines) + "\n")
    composition_md = write_text(outputs.composition_markdown_output, _composition_markdown(summary, composition_rows))
    return composition_json, composition_csv, composition_md


def export_samples(
    *,
    samples: list[JsonExtractionSample],
    outputs: BuildOutputs,
) -> dict[str, Path]:
    """Write canonical, SFT, and eval manifests from canonical samples."""

    canonical_rows = [sample.model_dump(mode="json") for sample in samples]
    train_samples = [sample for sample in samples if sample.split == DatasetSplit.TRAIN]
    eval_samples = [sample for sample in samples if sample.split == DatasetSplit.EVAL]
    prompt_rows = [prompt_completion_record(sample) for sample in train_samples]
    message_rows = [messages_record(sample) for sample in train_samples]
    eval_rows = [eval_manifest_record(sample) for sample in eval_samples]
    return {
        "canonical_output": write_jsonl(outputs.canonical_output, canonical_rows),
        "prompt_completion_output": write_jsonl(outputs.prompt_completion_output, prompt_rows),
        "messages_output": write_jsonl(outputs.messages_output, message_rows),
        "eval_output": write_jsonl(outputs.eval_output, eval_rows),
    }


def build_dataset_manifests(
    *,
    repo_root: Path,
    registry_config_path: Path,
    build_config_path: Path,
    profile_name: str,
    split_filter: str = "all",
    seed_override: int | None = None,
    raw_root: Path | None = None,
    runtime_root: Path | None = None,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    include_groups: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full multi-source data build and write artifacts."""

    registry = load_dataset_registry(registry_config_path)
    profile = load_build_profile(
        config_path=build_config_path,
        repo_root=repo_root,
        profile_name=profile_name,
        seed_override=seed_override,
        raw_root=raw_root,
        runtime_root=runtime_root,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        include_groups=include_groups,
    )
    active_sources = _select_active_sources(registry, profile)

    accepted_samples: list[JsonExtractionSample] = []
    rejects: list[AdapterReject] = []
    generated_source: DatasetSource | None = None
    source_load_details: dict[str, dict[str, Any]] = {}

    for source in active_sources:
        if source.source_type == SourceType.GENERATED:
            generated_source = source
            source_load_details[source.dataset_name] = {
                "resolved_source": source.source_uri_or_path,
                "used_local_fixture": False,
                "record_count": 0,
            }
            continue
        loaded = load_source_rows(
            source=source,
            repo_root=repo_root,
            raw_root=profile.raw_root,
            prefer_local_fixtures=profile.prefer_local_fixtures,
            allow_fixture_fallback=profile.allow_fixture_fallback,
        )
        source_load_details[source.dataset_name] = {
            "resolved_source": loaded.resolved_source,
            "used_local_fixture": loaded.used_local_fixture,
            "record_count": len(loaded.records),
        }
        source_samples, source_rejects = adapt_loaded_rows(loaded=loaded, profile=profile, repo_root=repo_root)
        accepted_samples.extend(source_samples)
        rejects.extend(source_rejects)

    accepted_samples = _enforce_eval_policy(accepted_samples, profile)
    train_groups: dict[str, list[JsonExtractionSample]] = defaultdict(list)
    eval_groups: dict[str, list[JsonExtractionSample]] = defaultdict(list)
    for sample in accepted_samples:
        if sample.split == DatasetSplit.TRAIN:
            train_groups[sample.source_dataset].append(sample)
        else:
            eval_groups[sample.source_dataset].append(sample)

    source_weights = {
        source.dataset_name: _source_weight(source, profile)
        for source in active_sources
        if source.source_type != SourceType.GENERATED
    }
    selected_train = _sample_rows(
        train_groups,
        target_count=profile.train_target_count,
        weights=source_weights,
        max_source_share=profile.max_source_share,
        seed=profile.seed,
    )
    selected_eval = _sample_rows(
        eval_groups,
        target_count=profile.eval_target_count,
        weights=source_weights,
        max_source_share=None,
        seed=profile.seed + 100,
    )

    if profile.augmentation_enabled and generated_source is not None:
        real_train_rows = [sample for sample in selected_train if not bool(sample.metadata.get("synthetic", False))]
        synthetic_cap = math.floor(
            len(real_train_rows) * profile.max_synthetic_share / max(1.0 - profile.max_synthetic_share, 0.01)
        )
        generated_rows = generate_augmentations(real_train_rows, max_generated_rows=synthetic_cap)
        selected_train = _cap_synthetic_rows(selected_train + generated_rows, profile)

    all_samples = sorted(selected_train + selected_eval, key=lambda sample: (sample.split.value, sample.source_dataset, sample.record_id))
    if split_filter in {DatasetSplit.TRAIN.value, DatasetSplit.EVAL.value}:
        all_samples = [sample for sample in all_samples if sample.split.value == split_filter]
    leakage = _leakage_summary(all_samples)
    if not leakage["is_lineage_clean"]:
        leaked_roots = set(leakage["leaked_train_rows"])
        all_samples = [sample for sample in all_samples if sample.record_id not in leaked_roots]

    export_paths = export_samples(samples=all_samples, outputs=profile.outputs)
    summary = summarize_dataset(samples=all_samples, rejects=rejects, profile=profile, registry=active_sources)
    summary.update(
        {
            "registry_config_path": str(registry_config_path),
            "build_config_path": str(build_config_path),
            "active_sources": [source.dataset_name for source in active_sources],
            "resolved_source_locations": {
                source_name: details["resolved_source"]
                for source_name, details in sorted(source_load_details.items())
            },
            "fixture_usage_by_source": {
                source_name: details["used_local_fixture"]
                for source_name, details in sorted(source_load_details.items())
            },
            "fixture_sources": sorted(
                source_name for source_name, details in source_load_details.items() if details["used_local_fixture"]
            ),
            "loaded_record_counts_by_source": {
                source_name: details["record_count"]
                for source_name, details in sorted(source_load_details.items())
            },
            "artifact_outputs": {key: str(value) for key, value in export_paths.items()},
        }
    )
    summary_path = write_json(profile.outputs.summary_output, summary)
    composition_rows = _composition_rows(all_samples)
    composition_paths = write_composition_artifacts(summary=summary, composition_rows=composition_rows, outputs=profile.outputs)
    return {
        "samples": all_samples,
        "rejects": rejects,
        "summary": summary,
        "summary_path": summary_path,
        "export_paths": export_paths,
        "composition_paths": composition_paths,
        "profile": profile,
    }
