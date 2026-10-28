### Response lengths (tokens) vs the `max_new_tokens` cap

A cap is free only if (almost) no response reaches it. While `% truncated` is near 0, p99 and p99.9 show where a lower cap would start cutting responses off.

| Method | phase | responses | mean | p50 | p90 | p99 | p99.9 | max | cap | % truncated | % of truncated invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct (zero-shot) | eval | 6493 | 88 | 74 | 116 | 493 | 512 | 512 | 512 | 0.99 | 100 |
| GRPO | train | 9475 | 92 | 80 | 144 | 290 | 500 | 512 | 512 | 0.09 | 89 |
| GRPO | train, last 5 it | 3950 | 92 | 80 | 145 | 294 | 427 | 512 | 512 | 0.03 | 100 |
| GRPO | eval | 6549 | 90 | 76 | 118 | 512 | 512 | 512 | 512 | 1.11 | 100 |
| GRPO + EMPG | train | 9445 | 92 | 80 | 145 | 284 | 512 | 512 | 512 | 0.15 | 86 |
| GRPO + EMPG | train, last 5 it | 3886 | 93 | 81 | 145 | 288 | 512 | 512 | 512 | 0.18 | 100 |
| GRPO + EMPG | eval | 6479 | 89 | 75 | 118 | 512 | 512 | 512 | 512 | 1.06 | 100 |
