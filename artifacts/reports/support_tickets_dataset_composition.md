# Support Tickets Dataset Composition

- Profile: `full`
- Total rows: `16`
- Split counts: `{'eval': 2, 'train': 14}`
- Synthetic row rate: `0.25`
- Leakage clean: `True`

## Per-Source Split Counts

| Source | Split | Rows | Synthetic Rows | Synthetic Rate |
| --- | --- | ---: | ---: | ---: |
| cfpb_consumer_complaints | eval | 1 | 0 | 0.00% |
| cfpb_consumer_complaints | train | 3 | 0 | 0.00% |
| console_ai_it_helpdesk_synthetic_tickets | eval | 1 | 0 | 0.00% |
| console_ai_it_helpdesk_synthetic_tickets | train | 3 | 0 | 0.00% |
| prady06_customer_support_tickets | train | 4 | 0 | 0.00% |
| synthetic_hardening_v1 | train | 2 | 2 | 100.00% |
| synthetic_support_tickets_v1 | train | 2 | 2 | 100.00% |

## Review Gates

- Source dominance share: `{'cfpb_consumer_complaints': 0.25, 'console_ai_it_helpdesk_synthetic_tickets': 0.25, 'prady06_customer_support_tickets': 0.25, 'synthetic_hardening_v1': 0.125, 'synthetic_support_tickets_v1': 0.125}`
- Nullable field null rates: `{'customer.account_id': 0.875, 'customer.name': 0.8125, 'customer.plan_tier': 0.875}`
- Adapter reject counts: `{}`
- Prompt length chars: `{'count': 14, 'min': 1138, 'max': 1379, 'avg': 1206.86}`
- Summary length chars: `{'count': 16, 'min': 29, 'max': 97, 'avg': 52.06}`
