"""Check local vLLM server health, model inventory, and metrics availability."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.benchmarking import check_vllm_health
from json_ft.utils import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-model-name", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    payload = check_vllm_health(
        args.api_base,
        expected_model_name=args.expected_model_name,
        timeout_seconds=args.timeout_seconds,
    )
    if args.output_path is not None:
        write_json(args.output_path, payload)
    print(f"API base: {payload['api_base']}")
    print(f"Health OK: {payload['health_ok']}")
    print(f"Models OK: {payload['models_ok']}")
    print(f"Metrics OK: {payload['metrics_ok']}")
    print(f"Served models: {payload.get('served_models', [])}")
    if payload.get("errors"):
        print(f"Errors: {payload['errors']}")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
