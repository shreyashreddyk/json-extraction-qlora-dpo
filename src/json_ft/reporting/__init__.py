"""Public interfaces for the final analysis and reporting layer."""

from .cases import CaseStudy, extract_case_studies
from .export import render_final_markdown_report
from .loaders import ReportingBundle, StageArtifacts, load_reporting_bundle
from .plots import generate_report_plots
from .tables import (
    build_dataset_composition_table,
    build_failure_bucket_table,
    build_field_level_table,
    build_pair_quality_table,
    build_stage_delta_table,
    build_stage_metrics_table,
)

__all__ = [
    "CaseStudy",
    "ReportingBundle",
    "StageArtifacts",
    "build_dataset_composition_table",
    "build_failure_bucket_table",
    "build_field_level_table",
    "build_pair_quality_table",
    "build_stage_delta_table",
    "build_stage_metrics_table",
    "extract_case_studies",
    "generate_report_plots",
    "load_reporting_bundle",
    "render_final_markdown_report",
]
