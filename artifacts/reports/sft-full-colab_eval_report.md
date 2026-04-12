# Sft Evaluation Report: sft-full-colab_eval

## Run Summary

- Stage label: `sft`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter path: `/content/drive/MyDrive/json-ft-runs/persistent/checkpoints/sft/sft-full-colab/adapter`
- Merged model path: `n/a`
- Backend: `local-transformers`
- Prompt source: `messages`
- Dataset path: `/content/drive/MyDrive/json-ft-source/data/manifests/support_tickets_eval_manifest.jsonl`
- Config path: `/content/drive/MyDrive/json-ft-source/configs/eval.yaml`
- Model manifest path: `/content/drive/MyDrive/json-ft-source/artifacts/checkpoints/sft-full-colab_adapter_manifest.json`
- Evaluated records: `12450`

## Syntax Metrics

- JSON validity rate: `0.9990`
- Schema validation pass rate: `0.8492`
- Hallucinated field rate: `0.0000`
- JSON recovery rate: `0.0000`

## Semantic Metrics

- Field-level micro F1: `0.7334`
- Field-level macro F1: `0.6519`

## Latency

- Mean latency (ms): `10773.1347`
- P95 latency (ms): `12375.8968`

## Exact Match by Field

- `issue_category`: `0.4455`
- `priority`: `0.4524`
- `product_area`: `0.6086`
- `sentiment`: `0.8394`
- `requires_human_followup`: `0.9807`
- `customer.plan_tier`: `0.5636`

## Field-Level Precision / Recall / F1

- `issue_category`: P=`0.4459`, R=`0.4455`, F1=`0.4457`, support=`12450`
- `priority`: P=`0.4529`, R=`0.4524`, F1=`0.4527`, support=`12450`
- `product_area`: P=`0.6092`, R=`0.6086`, F1=`0.6089`, support=`12450`
- `customer.name`: P=`0.9762`, R=`0.9752`, F1=`0.9757`, support=`12450`
- `customer.account_id`: P=`1.0000`, R=`0.9990`, F1=`0.9995`, support=`12450`
- `customer.plan_tier`: P=`0.5642`, R=`0.5636`, F1=`0.5639`, support=`12450`
- `sentiment`: P=`0.8402`, R=`0.8394`, F1=`0.8398`, support=`12450`
- `requires_human_followup`: P=`0.9817`, R=`0.9807`, F1=`0.9812`, support=`12450`
- `actions_requested`: P=`0.0000`, R=`0.0000`, F1=`0.0000`, support=`0`

## Failure Summary

- Parse failures: `13`
- Schema failures: `1878`
- Hallucinated predictions: `0`
- Rows with semantic mismatch after parsing: `11771`
- Null-handling mistakes: `677`
- Partial action extraction rows: `0`

## Diagnostics

- Qualitative summary note: Use the example buckets and per-field diagnostics for qualitative review. The free-text summary field remains qualitative and is not scored as an aggregate metric.
- Prior-stage regressions tracked: `0`

## Example Failures

### `080jm0cxc`

- Parse error: `none`
- Schema valid: `False`
- Unexpected fields: `none`

Input:

```text
Subject: Request for Zoom Room Access

Hey IT! I hope you're doing well. I'm trying to set up a meeting in one of our Zoom Rooms but realized I don't have the necessary access permissions. Could you assist me in getting access? Thanks a ton! Best, J -- Jordan Smith | Project Coordinator Acme Co. (https://acme.co/) • New York, NY CONFIDENTIALITY NOTICE: The contents of this email message and any attachments are intended solely for the addressee(s) and may contain confidential and/or privileged information and may be legally protected from disclosure. If you are not the intended recipient of this message or their agent, or if this message has been addressed to you in error, please immediately alert the sender by reply email and then delete this message and any attachments. If you are not the intended recipient, you are hereby notified that any use, dissemination, copying, or storage of this message or its attachments is strictly prohibited. DISCLAIMER: Email Does Not Constitute a Binding Agreement. This email does not constitute an agreement to conduct transactions by electronic means and does not create any legally binding contract or enforceable obligation in the absence of a fully signed written contract. Any contract terms contained in this email are subject to approval by Acme Co. and are not binding until Acme Co. provides such approval in writing.
```

Model output:

```text
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": "Jordan Smith",
    "plan_tier": null
  },
  "issue_category": "account_access",
  "priority": "medium",
  "product_area": "account_portal",
  "requires_human_followup": true,
  "sentiment": "neutral"
}
```

Reference payload:

```json
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": "Jordan Smith",
    "plan_tier": "pro"
  },
  "issue_category": "account_access",
  "priority": "medium",
  "product_area": "web_app",
  "requires_human_followup": true,
  "sentiment": "positive",
  "summary": "Request for Zoom Room Access"
}
```

Parsed payload:

```json
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": "Jordan Smith",
    "plan_tier": null
  },
  "issue_category": "account_access",
  "priority": "medium",
  "product_area": "account_portal",
  "requires_human_followup": true,
  "sentiment": "neutral"
}
```

### `09fqvnu90`

- Parse error: `none`
- Schema valid: `True`
- Unexpected fields: `none`

Input:

```text
Subject: Access Request for GDrive and GSheets Tools

Hi IT Team, I need access to GDrive and GSheets for project collaboration. Can you please assist? Thanks!
```

Model output:

```text
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": null,
    "plan_tier": "pro"
  },
  "issue_category": "account_access",
  "priority": "medium",
  "product_area": "other",
  "requires_human_followup": true,
  "sentiment": "positive",
  "summary": "Access Request for GDrive and GSheets Tools"
}
```

Reference payload:

```json
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": "Jane Doe",
    "plan_tier": "pro"
  },
  "issue_category": "account_access",
  "priority": "medium",
  "product_area": "account_portal",
  "requires_human_followup": true,
  "sentiment": "positive",
  "summary": "Access Request for GDrive and GSheets Tools"
}
```

Parsed payload:

```json
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": null,
    "plan_tier": "pro"
  },
  "issue_category": "account_access",
  "priority": "medium",
  "product_area": "other",
  "requires_human_followup": true,
  "sentiment": "positive",
  "summary": "Access Request for GDrive and GSheets Tools"
}
```

### `0k8ro1kdx`

- Parse error: `none`
- Schema valid: `True`
- Unexpected fields: `none`

Input:

```text
Subject: Virtual Backgrounds Not Loading in Video Conferencing Software

I'm having persistent issues with my video conferencing software where the virtual backgrounds fail to load correctly. It seems to require a detailed investigation and manual adjustment of the software settings. Could you also check if there are any updates needed for my graphics drivers?
```

Model output:

```text
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": null,
    "plan_tier": null
  },
  "issue_category": "technical_bug",
  "priority": "high",
  "product_area": "other",
  "requires_human_followup": true,
  "sentiment": "neutral",
  "summary": "Virtual Backgrounds Not Loading in Video Conferencing Software"
}
```

Reference payload:

```json
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": "User123",
    "plan_tier": null
  },
  "issue_category": "other",
  "priority": "medium",
  "product_area": "other",
  "requires_human_followup": true,
  "sentiment": "neutral",
  "summary": "Virtual Backgrounds Not Loading in Video Conferencing Software"
}
```

Parsed payload:

```json
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": null,
    "plan_tier": null
  },
  "issue_category": "technical_bug",
  "priority": "high",
  "product_area": "other",
  "requires_human_followup": true,
  "sentiment": "neutral",
  "summary": "Virtual Backgrounds Not Loading in Video Conferencing Software"
}
```

### `0mgy5bnab`

- Parse error: `none`
- Schema valid: `True`
- Unexpected fields: `none`

Input:

```text
Subject: Email Synchronization Issues

Hello,\n\nI'm experiencing persistent issues with our enterprise cloud-based email client. Specific email folders intermittently fail to synchronize across my devices. This requires a detailed investigation and manual adjustment of synchronization settings. Could you also inspect the server-side folder mapping configurations?\n\nThanks,
```

Model output:

```text
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": null,
    "plan_tier": "enterprise"
  },
  "issue_category": "technical_bug",
  "priority": "high",
  "product_area": "integrations",
  "requires_human_followup": true,
  "sentiment": "neutral",
  "summary": "Email Synchronization Issues"
}
```

Reference payload:

```json
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": "John Doe",
    "plan_tier": "enterprise"
  },
  "issue_category": "technical_bug",
  "priority": "medium",
  "product_area": "web_app",
  "requires_human_followup": true,
  "sentiment": "positive",
  "summary": "Email Synchronization Issues"
}
```

Parsed payload:

```json
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": null,
    "plan_tier": "enterprise"
  },
  "issue_category": "technical_bug",
  "priority": "high",
  "product_area": "integrations",
  "requires_human_followup": true,
  "sentiment": "neutral",
  "summary": "Email Synchronization Issues"
}
```

### `0q84tpmtp`

- Parse error: `none`
- Schema valid: `False`
- Unexpected fields: `none`

Input:

```text
Subject: Email Filter Configuration

Request for Instructions - Configuring Email Filters to Report Phishing Emails

Name: John Doe
Reports to: Jane Smith
Department: IT Support
Title: IT Specialist
Request date: 10/15/2023
Expected completion: 10/20/2023
```

Model output:

```text
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": "John Doe",
    "plan_tier": null
  },
  "issue_category": "other",
  "priority": "medium",
  "product_area": "other",
  "requires_human_followup": true,
  "sentiment": "neutral"
}
```

Reference payload:

```json
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": "Johndoe",
    "plan_tier": null
  },
  "issue_category": "feature_request",
  "priority": "medium",
  "product_area": "other",
  "requires_human_followup": true,
  "sentiment": "neutral",
  "summary": "Email Filter Configuration"
}
```

Parsed payload:

```json
{
  "actions_requested": [],
  "customer": {
    "account_id": null,
    "name": "John Doe",
    "plan_tier": null
  },
  "issue_category": "other",
  "priority": "medium",
  "product_area": "other",
  "requires_human_followup": true,
  "sentiment": "neutral"
}
```

