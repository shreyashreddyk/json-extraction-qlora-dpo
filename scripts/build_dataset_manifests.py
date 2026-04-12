"""Build the canonical dataset plus SFT and eval manifests from the registry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.data_build import build_dataset_manifests
from json_ft.utils import repo_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-config", type=Path, default=Path("configs/data_sources.yaml"))
    parser.add_argument("--build-config", type=Path, default=Path("configs/data_build.yaml"))
    parser.add_argument("--profile", choices=("dev", "full"), default="dev")
    parser.add_argument("--include-source", action="append", default=None)
    parser.add_argument("--exclude-source", action="append", default=None)
    parser.add_argument("--include-group", action="append", default=None)
    parser.add_argument("--split", choices=("all", "train", "eval"), default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    return parser


def _resolve_repo_path(root: Path, path: Path) -> Path:
    """Resolve CLI paths relative to the repo root instead of the caller cwd."""

    return path if path.is_absolute() else (root / path)


def main() -> int:
    args = build_parser().parse_args()
    root = repo_root()
    result = build_dataset_manifests(
        repo_root=root,
        registry_config_path=_resolve_repo_path(root, args.registry_config),
        build_config_path=_resolve_repo_path(root, args.build_config),
        profile_name=args.profile,
        split_filter=args.split,
        seed_override=args.seed,
        raw_root=args.raw_root,
        runtime_root=args.runtime_root,
        include_sources=args.include_source,
        exclude_sources=args.exclude_source,
        include_groups=args.include_group,
    )

    summary = result["summary"]
    print("Built dataset manifests")
    print(f"Profile: {summary['profile']}")
    print(f"Total rows: {summary['total_rows']}")
    print(f"Split counts: {summary['split_counts']}")
    print(f"Source counts: {summary['source_counts']}")
    print(f"Active sources: {summary.get('active_sources', [])}")
    print(f"Fixture sources: {summary.get('fixture_sources', [])}")
    print(f"Synthetic row rate: {summary['synthetic_row_rate']}")
    print(f"Leakage clean: {summary['leakage_checks']['is_lineage_clean']}")
    print(f"Canonical dataset: {result['export_paths']['canonical_output']}")
    print(f"SFT prompt-completion manifest: {result['export_paths']['prompt_completion_output']}")
    print(f"SFT messages manifest: {result['export_paths']['messages_output']}")
    print(f"Eval manifest: {result['export_paths']['eval_output']}")
    print(f"Build summary: {result['summary_path']}")
    print(f"Composition report: {result['composition_paths'][2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
