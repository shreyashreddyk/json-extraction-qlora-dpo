"""Scaffold CLI for preparing held-out evaluation data."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--input-manifest", type=Path, default=Path("data/manifests/eval_manifest.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/eval/heldout_eval.jsonl"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print("Evaluation data preparation scaffold")
    print(f"Config: {args.config}")
    print(f"Input manifest: {args.input_manifest}")
    print(f"Output path: {args.output_path}")
    print("TODO: implement stable eval split creation and provenance recording.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

