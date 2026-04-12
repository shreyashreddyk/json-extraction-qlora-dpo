# Consolidated Comparison Report: dpo-full-colab_comparison

## Comparison Rules

- Syntax metrics are reported separately from semantic metrics.
- Semantic example ranking uses structured exact-match count plus `actions_requested` F1 as a row-level inspection aid.
- This row-level ranking is diagnostic only; the headline comparison remains the saved aggregate metrics.
- Row-level labels classify DPO relative to SFT as `syntax_gain_only`, `semantic_gain`, `semantic_regression`, or `mixed_result`.

## Stage Metrics

### Baseline

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Base model: `n/a`
- Adapter path: `n/a`
- JSON validity rate: `0.9998`
- Schema validation pass rate: `0.9876`
- Hallucinated field rate: `0.0033`
- JSON recovery rate: `0.0000`
- Field-level micro F1: `0.3030`
- Field-level macro F1: `0.3008`
- Mean latency (ms): `10760.6296`

Exact match by categorical field:
- `issue_category`: `0.3022`
- `priority`: `0.1695`
- `product_area`: `0.2238`
- `sentiment`: `0.2991`
- `requires_human_followup`: `0.9579`
- `customer.plan_tier`: `0.2569`

### SFT

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter path: `/content/drive/MyDrive/json-ft-runs/persistent/checkpoints/sft/sft-full-colab/adapter`
- JSON validity rate: `0.9990`
- Schema validation pass rate: `0.8492`
- Hallucinated field rate: `0.0000`
- JSON recovery rate: `0.0000`
- Field-level micro F1: `0.7334`
- Field-level macro F1: `0.6519`
- Mean latency (ms): `10773.1347`

Exact match by categorical field:
- `issue_category`: `0.4455`
- `priority`: `0.4524`
- `product_area`: `0.6086`
- `sentiment`: `0.8394`
- `requires_human_followup`: `0.9807`
- `customer.plan_tier`: `0.5636`

### DPO

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter path: `/content/drive/MyDrive/json-ft-runs/persistent/checkpoints/dpo/dpo-full-colab/adapter`
- JSON validity rate: `0.9984`
- Schema validation pass rate: `0.9982`
- Hallucinated field rate: `0.0000`
- JSON recovery rate: `0.0000`
- Field-level micro F1: `0.7730`
- Field-level macro F1: `0.6871`
- Mean latency (ms): `16000.4178`

Exact match by categorical field:
- `issue_category`: `0.5031`
- `priority`: `0.4601`
- `product_area`: `0.6324`
- `sentiment`: `0.8613`
- `requires_human_followup`: `0.9802`
- `customer.plan_tier`: `0.7515`

## Deltas

### DPO vs SFT

- JSON validity delta: `-0.0006`
- Schema pass delta: `0.1490`
- Hallucinated field delta: `0.0000`
- Micro F1 delta: `0.0396`
- Macro F1 delta: `0.0352`

### SFT vs Baseline

- JSON validity delta: `-0.0009`
- Schema pass delta: `-0.1384`
- Hallucinated field delta: `-0.0033`
- Micro F1 delta: `0.4305`
- Macro F1 delta: `0.3511`

### DPO vs Baseline

- JSON validity delta: `-0.0014`
- Schema pass delta: `0.0106`
- Hallucinated field delta: `-0.0033`
- Micro F1 delta: `0.4700`
- Macro F1 delta: `0.3863`

## Semantic Gain

No examples were found in this category.

## Syntax Gain Only

No examples were found in this category.

## Semantic Regression

No examples were found in this category.

## Mixed Result

No examples were found in this category.

