"""Render plots and a markdown report from a saved benchmark bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.benchmark_reporting import load_benchmark_bundle, render_benchmark_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    bundle = load_benchmark_bundle(args.bundle_path)
    output_dir = args.output_dir or (args.bundle_path.parent / "reports")
    rendered = render_benchmark_report(bundle, output_dir)
    print(f"Bundle path: {args.bundle_path}")
    print(f"Report path: {rendered.get('report_path')}")
    print(f"Render index: {rendered.get('index_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
