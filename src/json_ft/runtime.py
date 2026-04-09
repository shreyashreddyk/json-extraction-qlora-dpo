"""Runtime and path helpers for local development and Colab execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import os


def detect_colab() -> bool:
    """Return True when code is executing inside a Colab runtime."""

    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


@dataclass(frozen=True)
class RuntimeContext:
    """Resolved local and runtime paths for a single execution environment."""

    repo_root: Path
    runtime_root: Path
    persistent_root: Path
    scratch_root: Path
    stage: str
    run_name: str
    is_colab: bool

    def ensure_directories(self) -> None:
        """Create the standard runtime directories."""

        for path in (
            self.runtime_root,
            self.persistent_root,
            self.scratch_root,
            self.checkpoints_dir,
            self.metrics_dir,
            self.reports_dir,
            self.logs_dir,
            self.exports_dir,
            self.run_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoints_dir(self) -> Path:
        return self.persistent_root / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.persistent_root / "metrics"

    @property
    def reports_dir(self) -> Path:
        return self.persistent_root / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.persistent_root / "logs"

    @property
    def exports_dir(self) -> Path:
        return self.persistent_root / "exports"

    @property
    def run_dir(self) -> Path:
        return self.persistent_root / self.stage / self.run_name

    def summary_lines(self) -> list[str]:
        """Render the resolved runtime structure for notebook and CLI output."""

        return [
            f"is_colab={self.is_colab}",
            f"repo_root={self.repo_root}",
            f"runtime_root={self.runtime_root}",
            f"persistent_root={self.persistent_root}",
            f"scratch_root={self.scratch_root}",
            f"stage={self.stage}",
            f"run_name={self.run_name}",
            f"run_dir={self.run_dir}",
        ]


def resolve_runtime_context(
    repo_root: str | Path,
    stage: str,
    run_name: str,
    runtime_root: str | Path | None = None,
) -> RuntimeContext:
    """Resolve the runtime context for local or Colab execution.

    The default Colab-friendly runtime root is `/content/drive/MyDrive/json-ft-runs`
    when running inside Colab. Outside Colab, the repo-local `runtime/` directory is
    used so scripts remain executable during development.
    """

    resolved_repo_root = Path(repo_root).resolve()
    if runtime_root is not None:
        resolved_runtime_root = Path(runtime_root).resolve()
    elif detect_colab():
        resolved_runtime_root = Path("/content/drive/MyDrive/json-ft-runs").resolve()
    else:
        resolved_runtime_root = (resolved_repo_root / "runtime").resolve()

    context = RuntimeContext(
        repo_root=resolved_repo_root,
        runtime_root=resolved_runtime_root,
        persistent_root=resolved_runtime_root / "persistent",
        scratch_root=resolved_runtime_root / "scratch",
        stage=stage,
        run_name=run_name,
        is_colab=detect_colab(),
    )
    context.ensure_directories()
    return context


def format_runtime_summary(context: RuntimeContext) -> str:
    """Return a human-readable summary for logs and notebooks."""

    return "\n".join(context.summary_lines())


def resolve_repo_artifact_targets(repo_root: str | Path) -> dict[str, Path]:
    """Return the small repo-side artifact paths used for mirrored results."""

    root = Path(repo_root).resolve()
    return {
        "metrics": root / "artifacts" / "metrics",
        "reports": root / "artifacts" / "reports",
        "checkpoints": root / "artifacts" / "checkpoints",
    }


def ensure_paths(paths: Iterable[Path]) -> None:
    """Create a list of directories."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

