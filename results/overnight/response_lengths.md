### Response lengths (tokens) vs the `max_new_tokens` cap

A cap is free only if (almost) no response reaches it. While `% truncated` is near 0, p99 and p99.9 show where a lower cap would start cutting responses off.

| Method | phase | responses | mean | p50 | p90 | p99 | p99.9 | max | cap | % truncated | % of truncated invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct (zero-shot) | eval | 1160 | 87 | 76 | 119 | 256 | 256 | 256 | 256 | 3.02 | 100 |
| GRPO | train | 1556 | 92 | 81 | 147 | 256 | 256 | 256 | 256 | 1.48 | 96 |
| GRPO | train, last 5 it | 956 | 92 | 81 | 150 | 256 | 256 | 256 | 256 | 1.57 | 93 |
| GRPO | eval | 1133 | 88 | 77 | 120 | 256 | 256 | 256 | 256 | 3.18 | 97 |
| GRPO + EMPG | train | 1600 | 94 | 84 | 147 | 256 | 256 | 256 | 256 | 1.81 | 97 |
| GRPO + EMPG | train, last 5 it | 1000 | 96 | 85 | 149 | 256 | 256 | 256 | 256 | 2.10 | 95 |
| GRPO + EMPG | eval | 1200 | 91 | 78 | 131 | 256 | 256 | 256 | 256 | 2.50 | 100 |
| EMPG + curiosity (episodic) | train | 1320 | 91 | 81 | 145 | 256 | 256 | 256 | 256 | 1.36 | 100 |
| EMPG + curiosity (episodic) | train, last 5 it | 920 | 89 | 80 | 139 | 256 | 256 | 256 | 256 | 1.09 | 100 |
| EMPG + curiosity (episodic) | eval | 1173 | 94 | 80 | 143 | 256 | 256 | 256 | 256 | 3.84 | 100 |
| GRPO + curiosity (group) | train | 1562 | 92 | 82 | 142 | 256 | 256 | 256 | 256 | 1.54 | 100 |
| GRPO + curiosity (group) | train, last 5 it | 962 | 90 | 81 | 140 | 256 | 256 | 256 | 256 | 1.56 | 100 |
| GRPO + curiosity (group) | eval | 1160 | 93 | 79 | 146 | 256 | 256 | 256 | 256 | 3.88 | 100 |
