"""Artifact loading helpers for the final analysis and reporting layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

from ..utils import read_json, read_jsonl


def _read_json_or_none(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return read_json(path)


def _read_jsonl_or_none(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None or not path.exists():
        return None
    return read_jsonl(path)


def _resolve_optional_runtime_root(
    repo_root: Path,
    runtime_root: str | Path | None,
) -> Path:
    if runtime_root is not None:
        return Path(runtime_root).resolve()
    return (repo_root / "runtime").resolve()


def _as_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).resolve()


def _candidate_paths(
    *,
    explicit_path: str | Path | None,
    filename: str | None,
    repo_root: Path,
    source_root: Path,
    runtime_root: Path,
    repo_subdir: str | None,
    runtime_subdir: str | None,
) -> list[Path]:
    candidates: list[Path] = []
    explicit = _as_path(explicit_path)
    if explicit is not None:
        candidates.append(explicit)
        if repo_subdir is not None:
            candidates.append((repo_root / repo_subdir / explicit.name).resolve())
            candidates.append((source_root / repo_subdir / explicit.name).resolve())
        if runtime_subdir is not None:
            candidates.append((runtime_root / runtime_subdir / explicit.name).resolve())
    if filename:
        if repo_subdir is not None:
            candidates.append((repo_root / repo_subdir / filename).resolve())
            candidates.append((source_root / repo_subdir / filename).resolve())
        if runtime_subdir is not None:
            candidates.append((runtime_root / runtime_subdir / filename).resolve())
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _resolve_existing_path(
    *,
    explicit_path: str | Path | None,
    filename: str | None,
    repo_root: Path,
    source_root: Path,
    runtime_root: Path,
    repo_subdir: str | None,
    runtime_subdir: str | None,
) -> Path | None:
    for candidate in _candidate_paths(
        explicit_path=explicit_path,
        filename=filename,
        repo_root=repo_root,
        source_root=source_root,
        runtime_root=runtime_root,
        repo_subdir=repo_subdir,
        runtime_subdir=runtime_subdir,
    ):
        if candidate.exists():
            return candidate
    return None


def _artifact_from_stage_summary(
    summary: dict[str, Any] | None,
    *keys: str,
) -> str | None:
    if not summary:
        return None
    current: Any = summary
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return str(current) if isinstance(current, str) and current else None


def _stage_metrics_filename(stage_name: str) -> str:
    mapping = {
        "baseline": "baseline-qwen2.5-1.5b_metrics.json",
        "sft": "sft-full-colab_eval_metrics.json",
        "dpo": "dpo-full-colab_eval_metrics.json",
    }
    return mapping[stage_name]


def _stage_diagnostics_filename(stage_name: str) -> str:
    mapping = {
        "baseline": "baseline-qwen2.5-1.5b_diagnostics.json",
        "sft": "sft-full-colab_eval_diagnostics.json",
        "dpo": "dpo-full-colab_eval_diagnostics.json",
    }
    return mapping[stage_name]


def _stage_predictions_filename(stage_name: str) -> str:
    mapping = {
        "baseline": "baseline-qwen2.5-1.5b_predictions.jsonl",
        "sft": "sft-full-colab_eval_predictions.jsonl",
        "dpo": "dpo-full-colab_eval_predictions.jsonl",
    }
    return mapping[stage_name]


def _stage_report_filename(stage_name: str) -> str:
    mapping = {
        "baseline": "baseline-qwen2.5-1.5b_report.md",
        "sft": "sft-full-colab_eval_report.md",
        "dpo": "dpo-full-colab_eval_report.md",
    }
    return mapping[stage_name]


@dataclass(frozen=True)
class StageArtifacts:
    """Loaded saved artifacts for one stage."""

    stage_name: str
    metrics_path: Path | None
    metrics: dict[str, Any] | None
    diagnostics_path: Path | None
    diagnostics: dict[str, Any] | None
    report_path: Path | None
    summary_path: Path | None = None
    summary: dict[str, Any] | None = None
    history_path: Path | None = None
    history: dict[str, Any] | None = None
    predictions_path: Path | None = None
    predictions: list[dict[str, Any]] | None = None

    @property
    def has_predictions(self) -> bool:
        return bool(self.predictions)


@dataclass(frozen=True)
class ReportingBundle:
    """Resolved saved artifacts used by the final report notebook and CLI."""

    repo_root: Path
    source_root: Path
    runtime_root: Path
    build_summary_path: Path | None
    build_summary: dict[str, Any] | None
    composition_summary_path: Path | None
    composition_summary: dict[str, Any] | None
    comparison_summary_path: Path | None
    comparison_summary: dict[str, Any] | None
    canonical_manifest_path: Path | None
    canonical_manifest_rows: list[dict[str, Any]] | None
    eval_manifest_path: Path | None
    eval_manifest_rows: list[dict[str, Any]] | None
    sft_manifest_path: Path | None
    sft_manifest_rows: list[dict[str, Any]] | None
    baseline: StageArtifacts
    sft: StageArtifacts
    dpo: StageArtifacts
    preference_summary_path: Path | None = None
    preference_summary: dict[str, Any] | None = None
    preference_diagnostics_path: Path | None = None
    preference_diagnostics: dict[str, Any] | None = None
    preference_audit_path: Path | None = None
    preference_audit_rows: list[dict[str, Any]] | None = None
    availability: dict[str, bool] = field(default_factory=dict)

    def inventory_lines(self) -> list[str]:
        lines = [
            f"repo_root={self.repo_root}",
            f"source_root={self.source_root}",
            f"runtime_root={self.runtime_root}",
        ]
        for label, available in sorted(self.availability.items()):
            lines.append(f"{label}={'available' if available else 'missing'}")
        return lines


def _load_stage_artifacts(
    *,
    stage_name: str,
    repo_root: Path,
    source_root: Path,
    runtime_root: Path,
    comparison_summary: dict[str, Any] | None,
    summary_filename: str | None = None,
    history_filename: str | None = None,
) -> StageArtifacts:
    comparison_stage = (comparison_summary or {}).get("stages", {}).get(stage_name, {})
    metrics_path = _resolve_existing_path(
        explicit_path=comparison_stage.get("metrics_path"),
        filename=_stage_metrics_filename(stage_name),
        repo_root=repo_root,
        source_root=source_root,
        runtime_root=runtime_root,
        repo_subdir="artifacts/metrics",
        runtime_subdir="persistent/metrics",
    )
    diagnostics_path = _resolve_existing_path(
        explicit_path=None,
        filename=_stage_diagnostics_filename(stage_name),
        repo_root=repo_root,
        source_root=source_root,
        runtime_root=runtime_root,
        repo_subdir="artifacts/metrics",
        runtime_subdir="persistent/metrics",
    )
    report_path = _resolve_existing_path(
        explicit_path=None,
        filename=_stage_report_filename(stage_name),
        repo_root=repo_root,
        source_root=source_root,
        runtime_root=runtime_root,
        repo_subdir="artifacts/reports",
        runtime_subdir="persistent/reports",
    )
    predictions_path = _resolve_existing_path(
        explicit_path=comparison_stage.get("predictions_path"),
        filename=_stage_predictions_filename(stage_name),
        repo_root=repo_root,
        source_root=source_root,
        runtime_root=runtime_root,
        repo_subdir="artifacts/reports",
        runtime_subdir="persistent/reports",
    )
    summary_path = None
    history_path = None
    if summary_filename is not None:
        summary_path = _resolve_existing_path(
            explicit_path=None,
            filename=summary_filename,
            repo_root=repo_root,
            source_root=source_root,
            runtime_root=runtime_root,
            repo_subdir="artifacts/metrics",
            runtime_subdir="persistent/metrics",
        )
    if history_filename is not None:
        history_path = _resolve_existing_path(
            explicit_path=None,
            filename=history_filename,
            repo_root=repo_root,
            source_root=source_root,
            runtime_root=runtime_root,
            repo_subdir="artifacts/metrics",
            runtime_subdir="persistent/metrics",
        )
    return StageArtifacts(
        stage_name=stage_name,
        metrics_path=metrics_path,
        metrics=_read_json_or_none(metrics_path),
        diagnostics_path=diagnostics_path,
        diagnostics=_read_json_or_none(diagnostics_path),
        report_path=report_path,
        summary_path=summary_path,
        summary=_read_json_or_none(summary_path),
        history_path=history_path,
        history=_read_json_or_none(history_path),
        predictions_path=predictions_path,
        predictions=_read_jsonl_or_none(predictions_path),
    )


def load_reporting_bundle(
    repo_root: str | Path,
    source_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
    preference_run_name: str | None = None,
) -> ReportingBundle:
    """Load the saved artifacts needed for the final project report."""

    resolved_repo_root = Path(repo_root).resolve()
    resolved_source_root = (
        Path(source_root).resolve() if source_root is not None else resolved_repo_root
    )
    resolved_runtime_root = _resolve_optional_runtime_root(resolved_repo_root, runtime_root)

    composition_summary_path = _resolve_existing_path(
        explicit_path=None,
        filename="support_tickets_dataset_composition.json",
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        repo_subdir="artifacts/metrics",
        runtime_subdir="persistent/metrics",
    )
    composition_summary = _read_json_or_none(composition_summary_path)
    build_summary_path = _resolve_existing_path(
        explicit_path=_artifact_from_stage_summary(composition_summary, "summary", "artifact_outputs", "build_summary_json"),
        filename="support_tickets_dataset_build_summary.json",
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        repo_subdir="data/manifests",
        runtime_subdir=None,
    )
    build_summary = _read_json_or_none(build_summary_path)
    comparison_summary_path = _resolve_existing_path(
        explicit_path=None,
        filename="dpo-full-colab_comparison_comparison_summary.json",
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        repo_subdir="artifacts/metrics",
        runtime_subdir="persistent/metrics",
    )
    comparison_summary = _read_json_or_none(comparison_summary_path)

    canonical_manifest_path = _resolve_existing_path(
        explicit_path=None,
        filename="support_tickets_canonical.jsonl",
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        repo_subdir="data/manifests",
        runtime_subdir=None,
    )
    eval_manifest_path = _resolve_existing_path(
        explicit_path=None,
        filename="support_tickets_eval_manifest.jsonl",
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        repo_subdir="data/manifests",
        runtime_subdir=None,
    )
    sft_manifest_path = _resolve_existing_path(
        explicit_path=None,
        filename="support_tickets_sft_messages.jsonl",
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        repo_subdir="data/manifests",
        runtime_subdir=None,
    )

    baseline = _load_stage_artifacts(
        stage_name="baseline",
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        comparison_summary=comparison_summary,
    )
    sft = _load_stage_artifacts(
        stage_name="sft",
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        comparison_summary=comparison_summary,
        summary_filename="sft-full-colab_sft_summary.json",
        history_filename="sft-full-colab_sft_history.json",
    )
    dpo = _load_stage_artifacts(
        stage_name="dpo",
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        comparison_summary=comparison_summary,
        summary_filename="dpo-full-colab_dpo_summary.json",
        history_filename="dpo-full-colab_dpo_history.json",
    )

    resolved_preference_run_name = preference_run_name or "pref-full-colab"
    preference_summary_filename = f"{resolved_preference_run_name}_preference_summary.json"
    preference_diagnostics_filename = f"{resolved_preference_run_name}_preference_diagnostics.json"
    preference_audit_filename = f"{resolved_preference_run_name}_preference_audit.jsonl"
    preference_summary_path = _resolve_existing_path(
        explicit_path=None,
        filename=preference_summary_filename,
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        repo_subdir="artifacts/metrics",
        runtime_subdir=f"persistent/preferences/{resolved_preference_run_name}",
    )
    preference_diagnostics_path = _resolve_existing_path(
        explicit_path=None,
        filename=preference_diagnostics_filename,
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        repo_subdir="artifacts/metrics",
        runtime_subdir=f"persistent/preferences/{resolved_preference_run_name}",
    )
    preference_audit_path = _resolve_existing_path(
        explicit_path=None,
        filename=preference_audit_filename,
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        repo_subdir="artifacts/reports",
        runtime_subdir=f"persistent/preferences/{resolved_preference_run_name}",
    )

    availability = {
        "build_summary": build_summary is not None,
        "composition_summary": composition_summary is not None,
        "comparison_summary": comparison_summary is not None,
        "canonical_manifest": canonical_manifest_path is not None and canonical_manifest_path.exists(),
        "eval_manifest": eval_manifest_path is not None and eval_manifest_path.exists(),
        "sft_manifest": sft_manifest_path is not None and sft_manifest_path.exists(),
        "baseline_predictions": baseline.has_predictions,
        "sft_predictions": sft.has_predictions,
        "dpo_predictions": dpo.has_predictions,
        "preference_summary": preference_summary_path is not None and preference_summary_path.exists(),
        "preference_diagnostics": preference_diagnostics_path is not None and preference_diagnostics_path.exists(),
        "preference_audit": preference_audit_path is not None and preference_audit_path.exists(),
    }

    return ReportingBundle(
        repo_root=resolved_repo_root,
        source_root=resolved_source_root,
        runtime_root=resolved_runtime_root,
        build_summary_path=build_summary_path,
        build_summary=build_summary,
        composition_summary_path=composition_summary_path,
        composition_summary=composition_summary,
        comparison_summary_path=comparison_summary_path,
        comparison_summary=comparison_summary,
        canonical_manifest_path=canonical_manifest_path,
        canonical_manifest_rows=_read_jsonl_or_none(canonical_manifest_path),
        eval_manifest_path=eval_manifest_path,
        eval_manifest_rows=_read_jsonl_or_none(eval_manifest_path),
        sft_manifest_path=sft_manifest_path,
        sft_manifest_rows=_read_jsonl_or_none(sft_manifest_path),
        baseline=baseline,
        sft=sft,
        dpo=dpo,
        preference_summary_path=preference_summary_path,
        preference_summary=_read_json_or_none(preference_summary_path),
        preference_diagnostics_path=preference_diagnostics_path,
        preference_diagnostics=_read_json_or_none(preference_diagnostics_path),
        preference_audit_path=preference_audit_path,
        preference_audit_rows=_read_jsonl_or_none(preference_audit_path),
        availability=availability,
    )
