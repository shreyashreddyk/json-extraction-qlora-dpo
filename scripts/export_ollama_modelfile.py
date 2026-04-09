"""Scaffold CLI for exporting an Ollama Modelfile."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--adapter-path", type=Path, default=Path("artifacts/checkpoints/dpo"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/reports/Modelfile"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print("Ollama export scaffold")
    print(f"Config: {args.config}")
    print(f"Adapter path: {args.adapter_path}")
    print(f"Output path: {args.output_path}")
    print("TODO: implement a deterministic Modelfile export path for demos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

