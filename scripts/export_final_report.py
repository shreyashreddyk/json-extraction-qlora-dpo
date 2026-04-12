"""Export the final notebook-backed markdown report from saved artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from json_ft.reporting import extract_case_studies, generate_report_plots, load_reporting_bundle, render_final_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--preference-run-name", default=None)
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("artifacts/reports/final_project_report.md"),
    )
    parser.add_argument(
        "--output-plot-dir",
        type=Path,
        default=Path("artifacts/plots/final_report"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    bundle = load_reporting_bundle(
        repo_root=repo_root,
        source_root=args.source_root,
        runtime_root=args.runtime_root,
        preference_run_name=args.preference_run_name,
    )

    print("Final report export")
    for line in bundle.inventory_lines():
        print(line)

    plot_paths: dict[str, str] | None = None
    try:
        plot_paths = generate_report_plots(bundle, (repo_root / args.output_plot_dir).resolve())
        print(f"Generated plots: {plot_paths}")
    except RuntimeError as exc:
        print(f"Skipping plot generation: {exc}")

    case_studies = extract_case_studies(bundle)
    report_path = render_final_markdown_report(
        bundle=bundle,
        case_studies=case_studies,
        output_path=(repo_root / args.output_markdown).resolve(),
        plot_paths=plot_paths,
    )
    print(f"Markdown report: {report_path}")

    skipped_sections = [
        label
        for label, available in bundle.availability.items()
        if not available and label in {"sft_predictions", "dpo_predictions", "preference_summary", "preference_diagnostics", "preference_audit"}
    ]
    if skipped_sections:
        print(f"Optional sections limited by missing artifacts: {skipped_sections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
