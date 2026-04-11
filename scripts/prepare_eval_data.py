"""Prepare a validated held-out evaluation manifest for support-ticket extraction."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.dataset_adapters import DatasetSplit, adapt_source_record, eval_manifest_record
from json_ft.schemas import build_support_ticket_schema
from json_ft.utils import read_jsonl, write_json, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument(
        "--input-path",
        "--input-manifest",
        dest="input_path",
        type=Path,
        default=Path("data/manifests/support_tickets_canonical.jsonl"),
    )
    parser.add_argument(
        "--source-format",
        choices=("json_extraction", "nemotron_sft"),
        default="json_extraction",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/manifests/support_tickets_eval_manifest.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/manifests/support_tickets_eval_summary.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    samples = [adapt_source_record(record, args.source_format) for record in read_jsonl(args.input_path)]
    eval_samples = [sample for sample in samples if sample.split == DatasetSplit.EVAL]
    eval_rows = [eval_manifest_record(sample) for sample in eval_samples]

    issue_category_counts = Counter(sample.target.issue_category.value for sample in eval_samples)
    priority_counts = Counter(sample.target.priority.value for sample in eval_samples)
    summary = {
        "schema": {
            "name": build_support_ticket_schema().name,
            "version": build_support_ticket_schema().version,
        },
        "config_path": str(args.config),
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "source_format": args.source_format,
        "eval_record_count": len(eval_rows),
        "issue_category_counts": dict(sorted(issue_category_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
    }

    manifest_path = write_jsonl(args.output_path, eval_rows)
    summary_path = write_json(args.summary_output, summary)

    print("Prepared evaluation manifest")
    print(f"Input path: {args.input_path}")
    print(f"Source format: {args.source_format}")
    print(f"Eval records: {len(eval_rows)}")
    print(f"Eval manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
