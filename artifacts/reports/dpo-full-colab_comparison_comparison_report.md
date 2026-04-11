# Consolidated Comparison Report: dpo-full-colab_comparison

## Comparison Rules

- Syntax metrics are reported separately from semantic metrics.
- Semantic example ranking uses structured exact-match count plus `actions_requested` F1 as a row-level inspection aid.
- This row-level ranking is diagnostic only; the headline comparison remains the saved aggregate metrics.

## Stage Metrics

### Baseline

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Base model: `n/a`
- Adapter path: `n/a`
- JSON validity rate: `1.0000`
- Schema validation pass rate: `1.0000`
- Hallucinated field rate: `0.0000`
- JSON recovery rate: `0.0000`
- Field-level micro F1: `0.5965`
- Field-level macro F1: `0.6296`
- Mean latency (ms): `4033.6954`

Exact match by categorical field:
- `issue_category`: `0.6667`
- `priority`: `0.6667`
- `product_area`: `0.6667`
- `sentiment`: `1.0000`
- `requires_human_followup`: `0.6667`
- `customer.plan_tier`: `0.6667`

### SFT

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter path: `/content/drive/MyDrive/json-ft-runs/persistent/checkpoints/sft/sft-qwen2.5-1.5b-qlora-v1/adapter`
- JSON validity rate: `1.0000`
- Schema validation pass rate: `1.0000`
- Hallucinated field rate: `0.0000`
- JSON recovery rate: `0.0000`
- Field-level micro F1: `0.6207`
- Field-level macro F1: `0.6667`
- Mean latency (ms): `10344.3455`

Exact match by categorical field:
- `issue_category`: `1.0000`
- `priority`: `0.6667`
- `product_area`: `0.6667`
- `sentiment`: `1.0000`
- `requires_human_followup`: `0.6667`
- `customer.plan_tier`: `0.6667`

### DPO

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter path: `/content/drive/MyDrive/json-ft-runs/persistent/checkpoints/dpo/dpo-full-colab/adapter`
- JSON validity rate: `1.0000`
- Schema validation pass rate: `1.0000`
- Hallucinated field rate: `0.0000`
- JSON recovery rate: `0.0000`
- Field-level micro F1: `0.6207`
- Field-level macro F1: `0.6667`
- Mean latency (ms): `10266.5931`

Exact match by categorical field:
- `issue_category`: `1.0000`
- `priority`: `0.6667`
- `product_area`: `0.6667`
- `sentiment`: `1.0000`
- `requires_human_followup`: `0.6667`
- `customer.plan_tier`: `0.6667`

## Deltas

### DPO vs SFT

- JSON validity delta: `0.0000`
- Schema pass delta: `0.0000`
- Hallucinated field delta: `0.0000`
- Micro F1 delta: `0.0000`
- Macro F1 delta: `0.0000`

### SFT vs Baseline

- JSON validity delta: `0.0000`
- Schema pass delta: `0.0000`
- Hallucinated field delta: `0.0000`
- Micro F1 delta: `0.0242`
- Macro F1 delta: `0.0370`

### DPO vs Baseline

- JSON validity delta: `0.0000`
- Schema pass delta: `0.0000`
- Hallucinated field delta: `0.0000`
- Micro F1 delta: `0.0242`
- Macro F1 delta: `0.0370`

## Where DPO Helped Semantically

No examples were found in this category.

## Where DPO Helped Mostly on Syntax

No examples were found in this category.

## Where DPO Hurt

### `support-eval-003`

DPO changed the output without improving syntax or semantic score relative to SFT. Treat this as a regression-risk example because behavior moved with no measured gain.

Reference payload:

```json
{
  "actions_requested": [
    "Provide documentation about regional data residency availability"
  ],
  "customer": {
    "account_id": null,
    "name": null,
    "plan_tier": null
  },
  "issue_category": "general_question",
  "priority": "low",
  "product_area": "unknown",
  "requires_human_followup": false,
  "sentiment": "neutral",
  "summary": "Prospective customer asks whether regional data residency is available on starter plans and would accept documentation."
}
```

Baseline output:

```text
{
  "summary": "Regional Data Residency Support Question",
  "issue_category": "general_question",
  "priority": "medium",
  "product_area": "billing_portal",
  "customer": {
    "name": "Customer Name",
    "account_id": "Account ID",
    "plan_tier": "free"
  },
  "sentiment": "neutral",
  "requires_human_followup": true,
  "actions_requested": [
    "Provide documentation on regional data residency support in the starter tier."
  ]
}
```

SFT output:

```text
{
  "summary": "Customer inquiry about regional data residency support in the starter tier",
  "issue_category": "general_question",
  "priority": "medium",
  "product_area": "billing_portal",
  "customer": {
    "name": "Not provided",
    "account_id": "Not provided",
    "plan_tier": "free"
  },
  "sentiment": "neutral",
  "requires_human_followup": true,
  "actions_requested": [
    "Provide information about regional data residency support in the starter tier.",
    "If no documentation exists, provide a link to a relevant document."
  ]
}
```

DPO output:

```text
{
  "summary": "Customer inquiry about regional data residency support in the starter tier",
  "issue_category": "general_question",
  "priority": "medium",
  "product_area": "web_app",
  "customer": {
    "name": "Not provided",
    "account_id": "Not provided",
    "plan_tier": "free"
  },
  "sentiment": "neutral",
  "requires_human_followup": true,
  "actions_requested": [
    "Provide information about regional data residency support in the starter tier.",
    "If no documentation exists, provide a link to a relevant document."
  ]
}
```

