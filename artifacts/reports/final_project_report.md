# Final Project Report

## Project Summary

This report summarizes a schema-constrained support-ticket JSON extraction project across three saved stages: baseline, SFT, and DPO.
The narrative keeps syntax quality separate from semantic quality so the repo stays honest about what improved, what regressed, and where the gains are mixed.

## Dataset Upgrade Summary

- Total rows: `83611`
- Split counts: `{'eval': 12450, 'train': 71161}`
- Synthetic row rate: `0.2553`
- Leakage clean: `True`

| source_dataset | split | row_count | synthetic_row_count | synthetic_row_rate |
| --- | --- | --- | --- | --- |
| console_ai_it_helpdesk_synthetic_tickets | eval | 108 | 0 | 0.0 |
| console_ai_it_helpdesk_synthetic_tickets | train | 392 | 0 | 0.0 |
| prady06_customer_support_tickets | eval | 12342 | 0 | 0.0 |
| prady06_customer_support_tickets | train | 49421 | 0 | 0.0 |
| synthetic_hardening_v1 | train | 21341 | 21341 | 1.0 |
| synthetic_support_tickets_v1 | train | 7 | 7 | 1.0 |







## Key Metrics

| stage | json_validity_rate | schema_validation_pass_rate | hallucinated_field_rate | json_recovery_rate | field_level_micro_f1 | field_level_macro_f1 | latency_mean_ms | latency_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.9998 | 0.9876 | 0.0033 | 0.0 | 0.303 | 0.3008 | 10760.63 | 13152.76 |
| sft | 0.999 | 0.8492 | 0.0 | 0.0 | 0.7334 | 0.6519 | 10773.13 | 12375.9 |
| dpo | 0.9984 | 0.9982 | 0.0 | 0.0 | 0.773 | 0.6871 | 16000.42 | 20388.68 |

## Syntax vs Semantic Takeaways

- Baseline: very strong surface formatting and schema compliance, but weak task understanding and field correctness.
- SFT: the major semantic jump happened here, but it also introduced a real schema-discipline regression.
- DPO: mostly recovered schema discipline and added a smaller semantic gain over SFT, but with slower inference and non-trivial row-level regressions.

| comparison | hallucinated_field_rate | json_recovery_rate | json_validity_rate | schema_validation_pass_rate | field_level_macro_f1 | field_level_micro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| sft_vs_baseline | -0.0033 | 0.0 | -0.0009 | -0.1384 | 0.3511 | 0.4305 |
| dpo_vs_sft | 0.0 | 0.0 | -0.0006 | 0.149 | 0.0352 | 0.0396 |
| dpo_vs_baseline | -0.0033 | 0.0 | -0.0014 | 0.0106 | 0.3863 | 0.47 |







## Pair-Quality Summary

_Preference-pair artifacts were not available in the current repo/runtime mirror, so this section is intentionally limited._



## Failure Analysis Highlights

| bucket | baseline | sft | dpo |
| --- | --- | --- | --- |
| hallucinated_keys | 41 | 0 | 0 |
| null_handling_mistakes | 11658 | 677 | 2998 |
| semantic_failures | 12448 | 11758 | 11197 |
| syntax_failures | 155 | 1878 | 23 |



## Case Studies

### Baseline Bad -> SFT Good
_No saved row-level example was available for this category in the current artifact set._

### SFT Good -> DPO Better
_No saved row-level example was available for this category in the current artifact set._

### SFT Good -> DPO Worse
_No saved row-level example was available for this category in the current artifact set._

### Syntax Cleaned Up But Semantics Unchanged
_No saved row-level example was available for this category in the current artifact set._

### Unchanged Hard Failures
_No saved row-level example was available for this category in the current artifact set._


## Honest Conclusion

SFT delivered the main semantic improvement on this task. DPO appears most useful here as a syntax and schema-discipline repair layer with selective semantic gains rather than as a universal quality win.
The saved artifacts support a nuanced story: DPO improved the aggregate comparison, but it also slowed inference and still hurt some rows. That makes the repo more credible, not less, because the reporting layer shows both the wins and the remaining failure modes.

## Next Step

The next practical step is a serving and benchmarking lab that packages the best checkpoint for vLLM-first inference and measures the quality-latency tradeoff explicitly.
