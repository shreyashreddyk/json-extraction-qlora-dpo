"""Helpers for syncing execution-relevant repo content into a Colab runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


SYNC_PATHS = ("src", "scripts", "configs")
SKIP_PATTERNS = (
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
    "artifacts/checkpoints",
    "runtime",
)


@dataclass(frozen=True)
class SyncResult:
    """Summary of what was copied into the runtime workspace."""

    destination_root: Path
    copied_paths: tuple[Path, ...]

    def summary(self) -> str:
        copied = ", ".join(str(path) for path in self.copied_paths) or "none"
        return f"destination_root={self.destination_root}\ncopied_paths={copied}"


def _should_skip(path: Path) -> bool:
    path_text = path.as_posix()
    return any(pattern in path_text for pattern in SKIP_PATTERNS)


def sync_repo_to_runtime(repo_root: str | Path, runtime_workspace: str | Path) -> SyncResult:
    """Copy the minimal execution-relevant repo content into the runtime workspace."""

    source_root = Path(repo_root).resolve()
    destination_root = Path(runtime_workspace).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    copied_paths: list[Path] = []

    for relative in SYNC_PATHS:
        source_path = source_root / relative
        destination_path = destination_root / relative
        if not source_path.exists():
            continue
        if destination_path.exists():
            shutil.rmtree(destination_path)
        shutil.copytree(
            source_path,
            destination_path,
            ignore=shutil.ignore_patterns(*SKIP_PATTERNS),
        )
        copied_paths.append(destination_path)

    return SyncResult(destination_root=destination_root, copied_paths=tuple(copied_paths))

