# JSON Extraction with QLoRA + DPO

Production-style scaffold for a schema-constrained JSON extraction project.
The repository is designed as both a portfolio artifact and a learning trail for
modern post-training workflows on a narrow structured-output task.

## Project Goal

Build a measurable fine-tuning workflow for extracting schema-valid JSON from
natural language inputs and comparing three stages honestly:

1. Baseline untuned model evaluation
2. Supervised fine-tuning with LoRA or QLoRA
3. Preference tuning with DPO on the same schema contract

This repository now includes a Colab-native execution layer for notebook-driven
GPU work. The runtime model treats Colab as disposable compute, Google Drive as
the durable transport and runtime storage layer, and the local Git repo as the
source of truth for code, configs, docs, and tracked artifacts.

## Current Problem Definition

The current task is narrow on purpose:

- input: support-ticket style natural language text
- output: schema-constrained JSON under the `support_ticket_extraction` contract
- evaluation focus:
  - syntax quality
  - schema discipline
  - semantic field correctness
  - hallucination control

This repo treats syntax quality and semantic quality as separate signals. A
stage is not considered "better" just because its JSON looks cleaner.

## Current Results Snapshot

The checked-in saved artifacts currently support this story:

| Stage | JSON Validity | Schema Pass | Micro F1 | Macro F1 | Mean Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.9998 | 0.9876 | 0.3030 | 0.3008 | 10760.63 |
| SFT | 0.9990 | 0.8492 | 0.7334 | 0.6519 | 10773.13 |
| DPO | 0.9984 | 0.9982 | 0.7730 | 0.6871 | 16000.42 |

Interpretation:

- Baseline: very strong surface formatting, weak task understanding
- SFT: the main semantic improvement stage
- DPO: strong schema-discipline repair plus a smaller semantic gain over SFT,
  with slower inference and some row-level regressions

This is the main honest headline for the repo today:

1. SFT produced the big semantic jump.
2. DPO helped most with schema discipline and selective semantic cleanup.
3. DPO is not a blanket win on every row.

## Intended Workflow

### Phase 0: Project framing

- Define the schema contract, prompt format, and evaluation contract.
- Record assumptions, tradeoffs, and open questions in `docs/`.

### Phase 1: Baseline evaluation

- Prepare a held-out evaluation slice for schema-constrained extraction.
- Measure JSON validity, schema pass rate, field correctness, hallucination
  rate, and latency for the untuned model.

### Phase 2: SFT with LoRA or QLoRA

- Prepare instruction-style training examples aligned to the same schema.
- Train a compact open model with parameter-efficient fine-tuning.
- Compare baseline vs SFT on the same held-out evaluation contract.

### Phase 3: Preference data and DPO

- Curate or generate chosen vs rejected outputs for the same task.
- Train a DPO stage on top of the SFT checkpoint or adapter.
- Compare baseline vs SFT vs DPO with aggregate metrics and failure examples.

### Phase 4: Inference and benchmarking

- Serve the resulting model through vLLM.
- Benchmark latency and throughput on representative prompts.
- Optionally export an Ollama packaging path for local demos.

## Final Review Artifacts

The repo now has one final reporting layer that should be the primary review
path for a recruiter, hiring manager, or future-you coming back to the project.

Read these in order:

1. [`artifacts/reports/final_project_report.md`](artifacts/reports/final_project_report.md)
2. [`notebooks/06_final_report.ipynb`](notebooks/06_final_report.ipynb)
3. [`artifacts/reports/dpo-full-colab_comparison_comparison_report.md`](artifacts/reports/dpo-full-colab_comparison_comparison_report.md)
4. [`artifacts/reports/support_tickets_dataset_composition.md`](artifacts/reports/support_tickets_dataset_composition.md)

The final notebook and markdown report are designed to answer:

- why the original baseline underperformed semantically
- how the data got better
- what changed in SFT
- what changed in DPO
- what improved
- what did not improve
- which failures stayed hard

Key artifact locations:

- dataset composition:
  - `artifacts/metrics/support_tickets_dataset_composition.json`
  - `artifacts/reports/support_tickets_dataset_composition.md`
- baseline report:
  - `artifacts/metrics/baseline-qwen2.5-1.5b_metrics.json`
  - `artifacts/reports/baseline-qwen2.5-1.5b_report.md`
- SFT eval and training artifacts:
  - `artifacts/metrics/sft-full-colab_eval_metrics.json`
  - `artifacts/metrics/sft-full-colab_sft_summary.json`
  - `artifacts/metrics/sft-full-colab_sft_history.json`
  - `artifacts/plots/sft-full-colab_*`
- DPO eval and training artifacts:
  - `artifacts/metrics/dpo-full-colab_eval_metrics.json`
  - `artifacts/metrics/dpo-full-colab_dpo_summary.json`
  - `artifacts/metrics/dpo-full-colab_dpo_history.json`
  - `artifacts/plots/dpo-full-colab_*`
- consolidated comparison:
  - `artifacts/metrics/dpo-full-colab_comparison_comparison_summary.json`
  - `artifacts/reports/dpo-full-colab_comparison_comparison_report.md`

## What The Final Notebook Shows

[`notebooks/06_final_report.ipynb`](notebooks/06_final_report.ipynb) is the
project's portfolio-grade walkthrough. It:

- loads saved dataset summaries instead of recomputing them
- loads saved baseline, SFT, and DPO metrics
- loads saved comparison and row-level artifacts when available
- renders richer tables and matplotlib plots
- surfaces case studies when cross-stage prediction files are available
- exports the GitHub-readable markdown report

The notebook is intentionally read-mostly. It should never trigger retraining
or model inference.

## Repository Layout

```text
json-extraction-qlora-dpo/
├── configs/              # Placeholder YAML configs for model, SFT, DPO, eval, inference
├── scripts/              # Thin CLIs that orchestrate reusable code in src/
├── src/json_ft/          # Reusable Python package for schemas, prompts, metrics, inference
├── tests/                # Small deterministic tests for scaffolded helpers
├── notebooks/            # Colab-oriented control notebooks, no hidden business logic
├── data/                 # Manifests, fixtures, interim data, evaluation subsets
├── artifacts/            # Metrics, plots, reports, checkpoint metadata
├── docs/                 # Local learning notes and project documents
├── pyproject.toml        # Package metadata and dependency groups
└── Makefile              # Small developer workflows
```

## Quickstart

### 1. Local development environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Use the local environment for:

- editing code
- running fast tests and compile checks
- reviewing generated small artifacts before commit

### 2. Google Drive source + Colab runtime

The final Colab workflow uses two Google Drive folders:

- `json-ft-source`: mirrored execution copy of the repo subset needed by Colab
- `json-ft-runs`: persistent runtime outputs such as metrics, reports, logs,
  checkpoints, and scratch data

The default local Drive desktop root for this repo is:

```text
/Users/shreyashreddy/Library/CloudStorage/GoogleDrive-kshreyashreddy@gmail.com/My Drive
```

Initialize those folders locally:

```bash
make drive-init
```

For a clean-slate rewrite of both Drive folders:

```bash
make drive-rewrite-colab
```

Or preview and push the execution subset into Drive without resetting runtime
data:

```bash
make drive-push-source-dry-run
make drive-push-source
```

Then from a Colab notebook connected to the runtime:

1. Run [`notebooks/00_colab_setup.ipynb`](notebooks/00_colab_setup.ipynb).
2. Build the task manifests in Colab with `scripts/build_dataset_manifests.py --profile full`.
3. Run the phase notebook you need for baseline eval, SFT, preference generation, DPO, or vLLM benchmarking.

The setup notebook will:

- mount Google Drive
- verify the mirrored Drive source tree
- recreate any missing runtime or artifact mirror directories
- install pinned `requirements-colab.txt` from `json-ft-source`
- resolve `SOURCE_ROOT=/content/drive/MyDrive/json-ft-source`
- resolve `RUNTIME_ROOT=/content/drive/MyDrive/json-ft-runs`
- print resolved runtime paths

### 3. Inspect the repo and data contract

```bash
make tree
make validate-scaffold
./.venv/bin/python scripts/build_dataset_manifests.py --profile dev
```

### 4. Review the generated artifacts

The registry-driven data build now generates:

- `data/manifests/support_tickets_canonical.jsonl`
- `data/manifests/support_tickets_sft_prompt_completion.jsonl`
- `data/manifests/support_tickets_sft_messages.jsonl`
- `data/manifests/support_tickets_eval_manifest.jsonl`
- `data/manifests/support_tickets_dataset_build_summary.json`
- `artifacts/metrics/support_tickets_dataset_composition.json`
- `artifacts/reports/support_tickets_dataset_composition.csv`
- `artifacts/reports/support_tickets_dataset_composition.md`

Open [`notebooks/00_data_audit.ipynb`](notebooks/00_data_audit.ipynb) to inspect
the schema, source composition, null behavior, and example exports.

When the staged artifacts already exist, use
[`notebooks/06_final_report.ipynb`](notebooks/06_final_report.ipynb) for the
final narrative rather than scanning raw JSON files by hand.

## Design Principles

- Keep notebooks thin and push reusable logic into `src/`.
- Preserve clear boundaries between data prep, training, evaluation, and inference.
- Prefer explicit, readable modules over hidden framework abstractions.
- Treat documentation as part of the system, not an afterthought.
- Do not claim improvements without saved metrics and before/after comparisons.

## Local vs Colab Workflow

The recommended operating model is:

1. edit code locally in VS Code
2. validate locally with fast non-GPU checks
3. mirror the execution subset into `json-ft-source` with `make drive-push-source`
4. run Colab notebooks directly from the Drive-backed source tree
5. build `data_build:full` in Colab for the real training corpus
6. persist runtime outputs in `json-ft-runs`
7. pull mirrored small artifacts back into the repo with `make drive-pull-artifacts`

This repository intentionally separates:

- `git repo`: source of truth for code, configs, docs, manifests, and reviewed small artifacts
- `json-ft-source`: mirrored execution copy for Colab, refreshed intentionally from the repo
- `json-ft-runs`: persistent runtime outputs outside Git

## Why This Is Production-Grade With Colab

Using Colab as the available GPU resource does not lower the engineering bar.
The workflow remains production-oriented because:

- Colab is treated as a disposable execution plane, not the source of truth.
- The repo remains the controlled system of record.
- Drive is used only as a durable transport and runtime storage layer.
- Colab dependencies are pinned and aligned to the shared runtime image so setup stays reproducible without extra environment management.
- Notebooks stay thin and orchestration-only.
- Reusable logic remains in `src/` and `scripts/`.
- Config-driven execution and deterministic artifact paths preserve reproducibility.
- Outputs are saved as metrics and reports that can be reviewed, versioned, and compared.
- Baseline, SFT, and DPO stages stay aligned to one schema and eval contract.

The practical separation of concerns is:

1. author locally
2. sync source intentionally
3. run on ephemeral GPU
4. persist outputs durably
5. pull final artifacts back into the repo

This keeps the system disciplined even though the compute layer is Colab-native.

## Current DPO Status

The saved artifact trail supports a careful conclusion:

- DPO helped this project
- but not in the naive "everything improved" sense

What the current saved results suggest:

- DPO substantially repaired the SFT schema-validation regression
- DPO improved aggregate micro and macro F1 over SFT
- DPO increased latency materially
- DPO still produced some row-level regressions, so it should be described as a
  useful second-stage optimization pass, not a guaranteed improvement layer

That is exactly the kind of honest tradeoff this repo is meant to show.

## Next Step

The next lab for this repository is serving and benchmarking:

- serve the best current checkpoint with vLLM first
- benchmark latency and throughput under the same task contract
- compare quality versus serving cost before packaging an optional Ollama demo

## Standard Commands

Local machine:

```bash
make drive-init
make drive-rewrite-colab
make drive-push-source-dry-run
make drive-push-source
make drive-pull-artifacts
```

Colab setup:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
from pathlib import Path
import sys

SOURCE_ROOT = Path("/content/drive/MyDrive/json-ft-source")
RUNTIME_ROOT = Path("/content/drive/MyDrive/json-ft-runs")
sys.path.insert(0, str(SOURCE_ROOT / "src"))
```

```bash
python -m pip install --upgrade pip
python -m pip install -r /content/drive/MyDrive/json-ft-source/requirements-colab.txt
```

Baseline evaluation:

```bash
python /content/drive/MyDrive/json-ft-source/scripts/eval_model.py \
  --config /content/drive/MyDrive/json-ft-source/configs/eval.yaml \
  --stage-label baseline \
  --run-name baseline-qwen2.5-1.5b \
  --runtime-root /content/drive/MyDrive/json-ft-runs \
  --dataset-path /content/drive/MyDrive/json-ft-source/data/manifests/support_tickets_eval_manifest.jsonl \
  --mirror-metrics-to-repo \
  --mirror-report-to-repo \
  --mirror-predictions-to-repo
```

Dataset build:

```bash
python /content/drive/MyDrive/json-ft-source/scripts/build_dataset_manifests.py \
  --registry-config /content/drive/MyDrive/json-ft-source/configs/data_sources.yaml \
  --build-config /content/drive/MyDrive/json-ft-source/configs/data_build.yaml \
  --profile full \
  --raw-root /content/drive/MyDrive/json-ft-runs/raw-data
```

## Exact Staged Workflow

The intended production-style sequence is now:

1. Rewrite or initialize the Drive folders with `make drive-rewrite-colab` or
   `make drive-init`.
2. Sync the latest source tree with `make drive-push-source`.
3. Run `notebooks/00_colab_setup.ipynb`.
4. Build the task manifests with
   `scripts/build_dataset_manifests.py --profile full --raw-root /content/drive/MyDrive/json-ft-runs/raw-data`.
5. Run `notebooks/01_baseline_eval.ipynb` to produce the baseline metrics,
   report, and prediction artifact under `json-ft-source/artifacts/`.
6. Run `notebooks/02_sft_review.ipynb` to train the SFT adapter and mirror the
   SFT summary, checkpoint manifest, and plots back into
   `json-ft-source/artifacts/`.
7. Run `notebooks/03_preference_pair_audit.ipynb` to build and review the DPO
   pair dataset under `json-ft-runs/persistent/preferences/<run_name>/`.
8. Run `notebooks/04_dpo_review.ipynb` to:
   - train the DPO adapter
   - inspect DPO loss and reward traces
   - evaluate SFT and DPO on the held-out manifest
   - build a consolidated baseline vs SFT vs DPO comparison report
9. Pull the mirrored small artifacts back into the local repo with
   `make drive-pull-artifacts`.

### DPO training command

`notebooks/04_dpo_review.ipynb` now embeds the authoritative DPO command. It
uses:

- `scripts/train_dpo.py`
- `configs/dpo.yaml`
- the generated preference-pair JSONL
- the explicit SFT checkpoint manifest
- repo-mirroring flags for metrics, plots, and checkpoint metadata

### Three-stage comparison contract

The consolidated comparison is built from saved stage artifacts, not notebook
memory:

- baseline metrics + predictions
- SFT eval metrics + predictions
- DPO eval metrics + predictions

`scripts/compare_stages.py` produces:

- `<run_name>_comparison_summary.json`
- `<run_name>_comparison_report.md`

The report keeps these layers separate:

- syntax gains:
  - JSON validity
  - schema pass rate
  - hallucinated-field rate
  - parse recovery rate
- semantic gains:
  - categorical exact-match fields
  - field-level micro F1
  - field-level macro F1
- latency

It also includes row-level evidence for:

- cases where DPO helps semantically
- cases where DPO mostly helps syntax
- cases where DPO hurts relative to SFT

The final promoted model is tracked through
[`artifacts/checkpoints/latest_model.json`](artifacts/checkpoints/latest_model.json).
That manifest points to the promoted adapter or merged export path outside Git.

## Current Status

This repository currently provides:

- a production-style Python package layout
- a strict support-ticket extraction schema implemented with Pydantic
- a registry-driven multi-source data build layer with provenance-aware
  canonical manifests
- validated dataset adapters for canonical, prompt-completion, conversational,
  Nemotron-style, and eval manifest formats
- script entrypoints for dataset build, SFT, eval, preference generation, DPO,
  and comparison reporting
- generated canonical, SFT, eval, and composition artifacts under
  `data/manifests/` and `artifacts/`
- Colab runtime helpers for path resolution and latest-model tracking
- runnable SFT and DPO training CLIs with dry-run validation
- Colab-oriented notebooks for setup, data audit, baseline review, SFT review,
  preference auditing, DPO review, and benchmarking
- consolidated three-stage comparison reporting across baseline, SFT, and DPO
- local documentation for the data contract and evaluation plan
- deterministic tests for schema validation, formatting, preference building,
  eval reporting, DPO training, and CLI smoke paths

It intentionally does not yet provide:

- model benchmarking results
- a real external raw-data integration
