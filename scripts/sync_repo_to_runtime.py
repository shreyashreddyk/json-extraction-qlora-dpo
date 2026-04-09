"""CLI for syncing the execution-relevant repo content into a Colab runtime workspace."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.sync import sync_repo_to_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runtime-workspace", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = sync_repo_to_runtime(args.repo_root, args.runtime_workspace)
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
