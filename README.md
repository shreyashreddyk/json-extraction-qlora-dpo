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
GPU work. The final runtime model treats Colab as disposable compute, Google
Drive as the durable transport and runtime storage layer, and the local Git
repo as the source of truth for code, configs, docs, and tracked artifacts.

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

Preview and push the execution subset into Drive:

```bash
make drive-push-source-dry-run
make drive-push-source
```

Then from a Colab notebook connected to the runtime:

1. Run [`notebooks/00_colab_setup.ipynb`](notebooks/00_colab_setup.ipynb).
2. Execute the phase notebook you need for baseline eval, SFT, DPO, or vLLM benchmarking.

The setup notebook will:

- mount Google Drive
- verify the mirrored Drive source tree
- install pinned `requirements-colab.txt` from `json-ft-source`
- resolve `SOURCE_ROOT=/content/drive/MyDrive/json-ft-source`
- resolve `RUNTIME_ROOT=/content/drive/MyDrive/json-ft-runs`
- print resolved runtime paths

### 3. Inspect the repo and data contract

```bash
make tree
make validate-scaffold
./.venv/bin/python scripts/prepare_sft_data.py --input-path data/fixtures/support_tickets.jsonl
./.venv/bin/python scripts/prepare_eval_data.py --input-path data/fixtures/support_tickets.jsonl
```

### 4. Review the generated artifacts

The first real data-prep version now generates:

- `data/manifests/support_tickets_sft_prompt_completion.jsonl`
- `data/manifests/support_tickets_sft_messages.jsonl`
- `data/manifests/support_tickets_eval_manifest.jsonl`
- summary JSON files for SFT and eval prep

Open [`notebooks/00_data_audit.ipynb`](notebooks/00_data_audit.ipynb) to inspect
the schema, class balance, and example exports.

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
5. persist runtime outputs in `json-ft-runs`
6. pull mirrored small artifacts back into the repo with `make drive-pull-artifacts`

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

## Standard Commands

Local machine:

```bash
make drive-init
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
  --run-name baseline-qwen2.5-1.5b \
  --runtime-root /content/drive/MyDrive/json-ft-runs \
  --dataset-path /content/drive/MyDrive/json-ft-source/data/manifests/support_tickets_eval_manifest.jsonl \
  --mirror-metrics-to-repo \
  --mirror-report-to-repo \
  --mirror-predictions-to-repo
```

The final promoted model is tracked through
[`artifacts/checkpoints/latest_model.json`](artifacts/checkpoints/latest_model.json).
That manifest points to the promoted adapter or merged export path outside Git.

## Current Status

This repository currently provides:

- a production-style Python package layout
- a strict support-ticket extraction schema implemented with Pydantic
- validated dataset adapters for canonical, prompt-completion, conversational,
  Nemotron-style, and eval manifest formats
- script entrypoints for generating SFT and eval manifests
- generated synthetic fixture data and repo-side manifests under `data/manifests/`
- Colab runtime helpers for path resolution and latest-model tracking
- Colab-oriented notebooks for setup, data audit, eval, training review, and benchmarking
- local documentation for the data contract and evaluation plan
- deterministic tests for schema validation, formatting, preference placeholders,
  and CLI smoke paths

It intentionally does not yet provide:

- runnable SFT or DPO training logic
- model benchmarking results
- a real external raw-data integration
