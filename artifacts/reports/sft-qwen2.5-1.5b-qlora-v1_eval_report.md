# Sft Evaluation Report: sft-qwen2.5-1.5b-qlora-v1_eval

## Run Summary

- Stage label: `sft`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter path: `/content/drive/MyDrive/json-ft-runs/persistent/checkpoints/sft/sft-qwen2.5-1.5b-qlora-v1/adapter`
- Merged model path: `n/a`
- Backend: `local-transformers`
- Prompt source: `messages`
- Dataset path: `/content/drive/MyDrive/json-ft-source/data/manifests/support_tickets_eval_manifest.jsonl`
- Config path: `/content/drive/MyDrive/json-ft-source/configs/eval.yaml`
- Model manifest path: `/content/drive/MyDrive/json-ft-source/artifacts/checkpoints/sft-qwen2.5-1.5b-qlora-v1_adapter_manifest.json`
- Evaluated records: `3`

## Syntax Metrics

- JSON validity rate: `1.0000`
- Schema validation pass rate: `1.0000`
- Hallucinated field rate: `0.0000`
- JSON recovery rate: `0.0000`

## Semantic Metrics

- Field-level micro F1: `0.6207`
- Field-level macro F1: `0.6667`

## Latency

- Mean latency (ms): `10344.3455`
- P95 latency (ms): `10627.7545`

## Exact Match by Field

- `issue_category`: `1.0000`
- `priority`: `0.6667`
- `product_area`: `0.6667`
- `sentiment`: `1.0000`
- `requires_human_followup`: `0.6667`
- `customer.plan_tier`: `0.6667`

## Field-Level Precision / Recall / F1

- `issue_category`: P=`1.0000`, R=`1.0000`, F1=`1.0000`, support=`3`
- `priority`: P=`0.6667`, R=`0.6667`, F1=`0.6667`, support=`3`
- `product_area`: P=`0.6667`, R=`0.6667`, F1=`0.6667`, support=`3`
- `customer.name`: P=`0.6667`, R=`0.6667`, F1=`0.6667`, support=`3`
- `customer.account_id`: P=`0.6667`, R=`0.6667`, F1=`0.6667`, support=`3`
- `customer.plan_tier`: P=`0.6667`, R=`0.6667`, F1=`0.6667`, support=`3`
- `sentiment`: P=`1.0000`, R=`1.0000`, F1=`1.0000`, support=`3`
- `requires_human_followup`: P=`0.6667`, R=`0.6667`, F1=`0.6667`, support=`3`
- `actions_requested`: P=`0.0000`, R=`0.0000`, F1=`0.0000`, support=`5`

## Failure Summary

- Parse failures: `0`
- Schema failures: `0`
- Hallucinated predictions: `0`
- Rows with semantic mismatch after parsing: `3`

## Example Failures

### `support-eval-001`

- Parse error: `none`
- Schema valid: `True`
- Unexpected fields: `none`

Input:

```text
This is Mei Tan from ENT-990. The tax exemption flag disappeared on our latest invoice, so the billing portal generated the wrong total. Please correct the invoice and send the updated PDF to finance before month-end close tomorrow.
```

Model output:

```text
{
  "summary": "Invoice Tax Exemption Flag Disappeared",
  "issue_category": "billing",
  "priority": "urgent",
  "product_area": "billing_portal",
  "customer": {
    "name": "Mei Tan",
    "account_id": "ENT-990",
    "plan_tier": "enterprise"
  },
  "sentiment": "negative",
  "requires_human_followup": true,
  "actions_requested": [
    "Correct the invoice and send the updated PDF to finance before month-end close tomorrow."
  ]
}
```

Reference payload:

```json
{
  "actions_requested": [
    "Correct the invoice total",
    "Send an updated invoice PDF to finance"
  ],
  "customer": {
    "account_id": "ENT-990",
    "name": "Mei Tan",
    "plan_tier": "enterprise"
  },
  "issue_category": "billing",
  "priority": "urgent",
  "product_area": "billing_portal",
  "requires_human_followup": true,
  "sentiment": "negative",
  "summary": "Enterprise customer reports a missing tax exemption on an invoice and needs a corrected PDF before close."
}
```

Parsed payload:

```json
{
  "actions_requested": [
    "Correct the invoice and send the updated PDF to finance before month-end close tomorrow."
  ],
  "customer": {
    "account_id": "ENT-990",
    "name": "Mei Tan",
    "plan_tier": "enterprise"
  },
  "issue_category": "billing",
  "priority": "urgent",
  "product_area": "billing_portal",
  "requires_human_followup": true,
  "sentiment": "negative",
  "summary": "Invoice Tax Exemption Flag Disappeared"
}
```

### `support-eval-002`

- Parse error: `none`
- Schema valid: `True`
- Unexpected fields: `none`

Input:

```text
Our API started returning intermittent 500 responses on the /v1/usage endpoint after today's deploy. I'm Noah Kim from account API-321 on the Business plan. Please investigate and share an ETA. We can retry some calls, but our usage dashboards are currently broken.
```

Model output:

```text
{
  "summary": "API returns intermittent 500 responses",
  "issue_category": "technical_bug",
  "priority": "urgent",
  "product_area": "api",
  "customer": {
    "name": "Noah Kim",
    "account_id": "API-321",
    "plan_tier": "business"
  },
  "sentiment": "negative",
  "requires_human_followup": true,
  "actions_requested": [
    "Investigate the intermittent 500 responses on the /v1/usage endpoint.",
    "Share an estimated time to address this issue."
  ]
}
```

Reference payload:

```json
{
  "actions_requested": [
    "Investigate intermittent 500 responses on /v1/usage",
    "Share an ETA for a fix"
  ],
  "customer": {
    "account_id": "API-321",
    "name": "Noah Kim",
    "plan_tier": "business"
  },
  "issue_category": "technical_bug",
  "priority": "urgent",
  "product_area": "api",
  "requires_human_followup": true,
  "sentiment": "negative",
  "summary": "Business customer reports intermittent API failures on the usage endpoint and requests investigation plus ETA."
}
```

Parsed payload:

```json
{
  "actions_requested": [
    "Investigate the intermittent 500 responses on the /v1/usage endpoint.",
    "Share an estimated time to address this issue."
  ],
  "customer": {
    "account_id": "API-321",
    "name": "Noah Kim",
    "plan_tier": "business"
  },
  "issue_category": "technical_bug",
  "priority": "urgent",
  "product_area": "api",
  "requires_human_followup": true,
  "sentiment": "negative",
  "summary": "API returns intermittent 500 responses"
}
```

### `support-eval-003`

- Parse error: `none`
- Schema valid: `True`
- Unexpected fields: `none`

Input:

```text
Hi there. We are comparing a few vendors and I can't tell whether your platform supports regional data residency in the starter tier or only on custom contracts. I do not have an account yet. A documentation link is enough if one exists.
```

Model output:

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

Parsed payload:

```json
{
  "actions_requested": [
    "Provide information about regional data residency support in the starter tier.",
    "If no documentation exists, provide a link to a relevant document."
  ],
  "customer": {
    "account_id": "Not provided",
    "name": "Not provided",
    "plan_tier": "free"
  },
  "issue_category": "general_question",
  "priority": "medium",
  "product_area": "billing_portal",
  "requires_human_followup": true,
  "sentiment": "neutral",
  "summary": "Customer inquiry about regional data residency support in the starter tier"
}
```

