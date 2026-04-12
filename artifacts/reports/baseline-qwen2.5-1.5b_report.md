# Baseline Evaluation Report: baseline-qwen2.5-1.5b

## Run Summary

- Stage label: `baseline`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Base model: `n/a`
- Adapter path: `n/a`
- Merged model path: `n/a`
- Backend: `local-transformers`
- Prompt source: `messages`
- Dataset path: `/content/drive/MyDrive/json-ft-source/data/manifests/support_tickets_eval_manifest.jsonl`
- Config path: `/content/drive/MyDrive/json-ft-source/configs/eval.yaml`
- Model manifest path: `n/a`
- Evaluated records: `12450`

## Syntax Metrics

- JSON validity rate: `0.9998`
- Schema validation pass rate: `0.9876`
- Hallucinated field rate: `0.0033`
- JSON recovery rate: `0.0000`

## Semantic Metrics

- Field-level micro F1: `0.3030`
- Field-level macro F1: `0.3008`

## Latency

- Mean latency (ms): `10760.6296`
- P95 latency (ms): `13152.7604`

## Exact Match by Field

- `issue_category`: `0.3022`
- `priority`: `0.1695`
- `product_area`: `0.2238`
- `sentiment`: `0.2991`
- `requires_human_followup`: `0.9579`
- `customer.plan_tier`: `0.2569`

## Field-Level Precision / Recall / F1

- `issue_category`: P=`0.3022`, R=`0.3022`, F1=`0.3022`, support=`12450`
- `priority`: P=`0.1695`, R=`0.1695`, F1=`0.1695`, support=`12450`
- `product_area`: P=`0.2238`, R=`0.2238`, F1=`0.2238`, support=`12450`
- `customer.name`: P=`0.0030`, R=`0.0030`, F1=`0.0030`, support=`12450`
- `customer.account_id`: P=`0.4949`, R=`0.4948`, F1=`0.4948`, support=`12450`
- `customer.plan_tier`: P=`0.2570`, R=`0.2569`, F1=`0.2570`, support=`12450`
- `sentiment`: P=`0.2992`, R=`0.2991`, F1=`0.2991`, support=`12450`
- `requires_human_followup`: P=`0.9581`, R=`0.9579`, F1=`0.9580`, support=`12450`
- `actions_requested`: P=`0.0000`, R=`0.0000`, F1=`0.0000`, support=`0`

## Failure Summary

- Parse failures: `2`
- Schema failures: `155`
- Hallucinated predictions: `41`
- Rows with semantic mismatch after parsing: `12450`
- Null-handling mistakes: `11658`
- Partial action extraction rows: `0`

## Diagnostics

- Qualitative summary note: Use the example buckets and per-field diagnostics for qualitative review. The free-text summary field remains qualitative and is not scored as an aggregate metric.
- Prior-stage regressions tracked: `0`

## Example Failures

### `080jm0cxc`

- Parse error: `none`
- Schema valid: `True`
- Unexpected fields: `none`

Input:

```text
Subject: Request for Zoom Room Access

Hey IT! I hope you're doing well. I'm trying to set up a meeting in one of our Zoom Rooms but realized I don't have the necessary access permissions. Could you assist me in getting access? Thanks a ton! Best, J -- Jordan Smith | Project Coordinator Acme Co. (https://acme.co/) • New York, NY CONFIDENTIALITY NOTICE: The contents of this email message and any attachments are intended solely for the addressee(s) and may contain confidential and/or privileged information and may be legally protected from disclosure. If you are not the intended recipient of this message or their agent, or if this message has been addressed to you in error, please immediately alert the sender by reply email and then delete this message and any attachments. If you are not the intended recipient, you are hereby notified that any use, dissemination, copying, or storage of this message or its attachments is strictly prohibited. DISCLAIMER: Email Does Not Constitute a Binding Agreement. This email does not constitute an agreement to conduct transactions by electronic means and does not create any legally binding contract or enforceable obligation in the absence of a fully signed written contract. Any contract terms contained in this email are subject to approval by Acme Co. and are not binding until Acme Co. provides such approval in writing.
```

Model output:

```text
{
  "summary": "Zoom Room Access Request",
  "issue_category": "account_access",
  "priority": "medium",
  "product_area": "billing_portal",
  "customer": {
    "name": "Jordan Smith",
    "account_id": "ACME_ACCOUNT_ID",
    "plan_tier": "free"
  },
  "sentiment": "neutral",
  "requires_human_followup": true,
  "actions_requested": [
    "Request Zoom Room Access"
  ]
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
  "actions_requested": [
    "Request Zoom Room Access"
  ],
  "customer": {
    "account_id": "ACME_ACCOUNT_ID",
    "name": "Jordan Smith",
    "plan_tier": "free"
  },
  "issue_category": "account_access",
  "priority": "medium",
  "product_area": "billing_portal",
  "requires_human_followup": true,
  "sentiment": "neutral",
  "summary": "Zoom Room Access Request"
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
  "summary": "Access Request for GDrive and GSheets",
  "issue_category": "account_access",
  "priority": "medium",
  "product_area": "mobile_app",
  "customer": {
    "name": "Customer Name",
    "account_id": "Account ID",
    "plan_tier": "pro"
  },
  "sentiment": "neutral",
  "requires_human_followup": true,
  "actions_requested": [
    "Provide access to GDrive and GSheets tools for project collaboration."
  ]
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
  "actions_requested": [
    "Provide access to GDrive and GSheets tools for project collaboration."
  ],
  "customer": {
    "account_id": "Account ID",
    "name": "Customer Name",
    "plan_tier": "pro"
  },
  "issue_category": "account_access",
  "priority": "medium",
  "product_area": "mobile_app",
  "requires_human_followup": true,
  "sentiment": "neutral",
  "summary": "Access Request for GDrive and GSheets"
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
  "summary": "Virtual Backgrounds Not Loading",
  "issue_category": "technical_bug",
  "priority": "urgent",
  "product_area": "web_app",
  "customer": {
    "name": "John Doe",
    "account_id": "A123456789",
    "plan_tier": "pro"
  },
  "sentiment": "negative",
  "requires_human_followup": true,
  "actions_requested": [
    "Investigate virtual background loading issues",
    "Check for necessary updates to graphics drivers"
  ]
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
  "actions_requested": [
    "Investigate virtual background loading issues",
    "Check for necessary updates to graphics drivers"
  ],
  "customer": {
    "account_id": "A123456789",
    "name": "John Doe",
    "plan_tier": "pro"
  },
  "issue_category": "technical_bug",
  "priority": "urgent",
  "product_area": "web_app",
  "requires_human_followup": true,
  "sentiment": "negative",
  "summary": "Virtual Backgrounds Not Loading"
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
  "summary": "Email Synchronization Issues",
  "issue_category": "technical_bug",
  "priority": "urgent",
  "product_area": "api",
  "customer": {
    "name": "Customer Name",
    "account_id": "Account ID",
    "plan_tier": "enterprise"
  },
  "sentiment": "negative",
  "requires_human_followup": true,
  "actions_requested": [
    "Inspection of server-side folder mapping configurations"
  ]
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
  "actions_requested": [
    "Inspection of server-side folder mapping configurations"
  ],
  "customer": {
    "account_id": "Account ID",
    "name": "Customer Name",
    "plan_tier": "enterprise"
  },
  "issue_category": "technical_bug",
  "priority": "urgent",
  "product_area": "api",
  "requires_human_followup": true,
  "sentiment": "negative",
  "summary": "Email Synchronization Issues"
}
```

### `0q84tpmtp`

- Parse error: `none`
- Schema valid: `False`
- Unexpected fields: `customer.plan_tier`

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
  "summary": "Email Filter Configuration",
  "issue_category": "technical_bug",
  "priority": "urgent",
  "product_area": "billing_portal",
  "customer": {
    "name": "John Doe",
    "account_id": null,
    "plan_tier": null
  },
  "customer.plan_tier": null,
  "sentiment": "neutral",
  "requires_human_followup": true,
  "actions_requested": [
    "Configure email filters to report phishing emails"
  ]
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
  "actions_requested": [
    "Configure email filters to report phishing emails"
  ],
  "customer": {
    "account_id": null,
    "name": "John Doe",
    "plan_tier": null
  },
  "customer.plan_tier": null,
  "issue_category": "technical_bug",
  "priority": "urgent",
  "product_area": "billing_portal",
  "requires_human_followup": true,
  "sentiment": "neutral",
  "summary": "Email Filter Configuration"
}
```

