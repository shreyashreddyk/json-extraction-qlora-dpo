# Support Tickets Dataset Composition

- Profile: `full`
- Total rows: `83611`
- Split counts: `{'eval': 12450, 'train': 71161}`
- Synthetic row rate: `0.2553`
- Leakage clean: `True`

## Per-Source Split Counts

| Source | Split | Rows | Synthetic Rows | Synthetic Rate |
| --- | --- | ---: | ---: | ---: |
| console_ai_it_helpdesk_synthetic_tickets | eval | 108 | 0 | 0.00% |
| console_ai_it_helpdesk_synthetic_tickets | train | 392 | 0 | 0.00% |
| prady06_customer_support_tickets | eval | 12342 | 0 | 0.00% |
| prady06_customer_support_tickets | train | 49421 | 0 | 0.00% |
| synthetic_hardening_v1 | train | 21341 | 21341 | 100.00% |
| synthetic_support_tickets_v1 | train | 7 | 7 | 100.00% |

## Review Gates

- Source dominance share: `{'console_ai_it_helpdesk_synthetic_tickets': 0.006, 'prady06_customer_support_tickets': 0.7387, 'synthetic_hardening_v1': 0.2552, 'synthetic_support_tickets_v1': 0.0001}`
- Nullable field null rates: `{'customer.account_id': 1.0, 'customer.name': 0.9714, 'customer.plan_tier': 0.3018}`
- Adapter reject counts: `{'missing_body': 2, 'missing_consumer_complaint_narrative': 14380839}`
- Prompt length chars: `{'count': 71161, 'min': 968, 'max': 4740, 'avg': 1483.14}`
- Summary length chars: `{'count': 83611, 'min': 3, 'max': 140, 'avg': 54.63}`
