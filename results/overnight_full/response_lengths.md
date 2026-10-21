### Response lengths (tokens) vs the `max_new_tokens` cap

A cap is free only if (almost) no response reaches it. While `% truncated` is near 0, p99 and p99.9 show where a lower cap would start cutting responses off.

| Method | phase | responses | mean | p50 | p90 | p99 | p99.9 | max | cap | % truncated | % of truncated invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct (zero-shot) | eval | 2377 | 87 | 75 | 120 | 256 | 256 | 256 | 256 | 3.11 | 99 |
| GRPO | train | 3858 | 89 | 78 | 143 | 256 | 256 | 256 | 256 | 1.50 | 93 |
| GRPO | train, last 5 it | 3858 | 89 | 78 | 143 | 256 | 256 | 256 | 256 | 1.50 | 93 |
| GRPO | eval | 2298 | 86 | 74 | 115 | 256 | 256 | 256 | 256 | 3.09 | 100 |
| GRPO + EMPG | train | 3946 | 89 | 78 | 143 | 256 | 256 | 256 | 256 | 1.39 | 96 |
| GRPO + EMPG | train, last 5 it | 3946 | 89 | 78 | 143 | 256 | 256 | 256 | 256 | 1.39 | 96 |
| GRPO + EMPG | eval | 2384 | 88 | 76 | 119 | 256 | 256 | 256 | 256 | 2.81 | 99 |
| EMPG + curiosity (episodic) | train | 3895 | 90 | 80 | 141 | 256 | 256 | 256 | 256 | 1.62 | 97 |
| EMPG + curiosity (episodic) | train, last 5 it | 3895 | 90 | 80 | 141 | 256 | 256 | 256 | 256 | 1.62 | 97 |
| EMPG + curiosity (episodic) | eval | 2392 | 86 | 74 | 117 | 256 | 256 | 256 | 256 | 3.22 | 100 |
| GRPO + curiosity (group) | train | 3825 | 89 | 79 | 140 | 256 | 256 | 256 | 256 | 1.18 | 100 |
| GRPO + curiosity (group) | train, last 5 it | 3825 | 89 | 79 | 140 | 256 | 256 | 256 | 256 | 1.18 | 100 |
| GRPO + curiosity (group) | eval | 2400 | 84 | 74 | 112 | 256 | 256 | 256 | 256 | 2.62 | 100 |
