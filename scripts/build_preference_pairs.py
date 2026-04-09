"""Scaffold CLI for building preference pairs for DPO."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dpo.yaml"))
    parser.add_argument("--input-path", type=Path, default=Path("data/eval/baseline_predictions.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/interim/preference_pairs.jsonl"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print("Preference pair builder scaffold")
    print(f"Config: {args.config}")
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print("TODO: implement chosen/rejected ranking, schema validation, and audit logging.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

