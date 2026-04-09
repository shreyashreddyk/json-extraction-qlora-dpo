"""Prepare validated SFT manifests for support-ticket JSON extraction."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.dataset_adapters import DatasetSplit, adapt_source_record, messages_record, prompt_completion_record
from json_ft.schemas import build_support_ticket_schema
from json_ft.utils import read_jsonl, write_json, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    parser.add_argument(
        "--input-path",
        "--input-manifest",
        dest="input_path",
        type=Path,
        default=Path("data/fixtures/support_tickets.jsonl"),
    )
    parser.add_argument(
        "--source-format",
        choices=("json_extraction", "nemotron_sft"),
        default="json_extraction",
    )
    parser.add_argument(
        "--prompt-completion-output",
        type=Path,
        default=Path("data/manifests/support_tickets_sft_prompt_completion.jsonl"),
    )
    parser.add_argument(
        "--messages-output",
        type=Path,
        default=Path("data/manifests/support_tickets_sft_messages.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/manifests/support_tickets_sft_summary.json"),
    )
    return parser


def load_samples(input_path: Path, source_format: str):
    """Read and validate source rows into canonical samples."""

    return [adapt_source_record(record, source_format) for record in read_jsonl(input_path)]


def summarize_samples(samples: list) -> dict:
    """Build a compact educational summary for the prepared dataset."""

    split_counts = Counter(sample.split.value for sample in samples)
    issue_category_counts = Counter(sample.target.issue_category.value for sample in samples)
    priority_counts = Counter(sample.target.priority.value for sample in samples)
    return {
        "schema": {
            "name": build_support_ticket_schema().name,
            "version": build_support_ticket_schema().version,
        },
        "total_records": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "issue_category_counts": dict(sorted(issue_category_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
    }


def main() -> int:
    args = build_parser().parse_args()
    samples = load_samples(args.input_path, args.source_format)
    train_samples = [sample for sample in samples if sample.split == DatasetSplit.TRAIN]

    prompt_rows = [prompt_completion_record(sample) for sample in train_samples]
    message_rows = [messages_record(sample) for sample in train_samples]
    summary = summarize_samples(samples)
    summary.update(
        {
            "config_path": str(args.config),
            "input_path": str(args.input_path),
            "source_format": args.source_format,
            "train_record_count": len(train_samples),
            "prompt_completion_output": str(args.prompt_completion_output),
            "messages_output": str(args.messages_output),
        }
    )

    prompt_path = write_jsonl(args.prompt_completion_output, prompt_rows)
    messages_path = write_jsonl(args.messages_output, message_rows)
    summary_path = write_json(args.summary_output, summary)

    print("Prepared SFT manifests")
    print(f"Input path: {args.input_path}")
    print(f"Source format: {args.source_format}")
    print(f"Train records: {len(train_samples)}")
    print(f"Prompt-completion manifest: {prompt_path}")
    print(f"Messages manifest: {messages_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
