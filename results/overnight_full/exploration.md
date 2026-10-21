### Exploration during training (mean of last 5 iterations)

| Method | train success | revisit rate | no-op rate | unique states / ep | group coverage | P(group ≥1 success) | no-signal groups | steps to success | intrinsic share |
|---|---|---|---|---|---|---|---|---|---|
| GRPO | 0.09 | 0.81 | 0.64 | 10.14 | 27.05 | 0.20 | 0.80 | 34.00 | 0.00 |
| GRPO + EMPG | 0.05 | 0.80 | 0.63 | 10.94 | 30.35 | 0.15 | 0.85 | 41.00 | 0.00 |
| EMPG + curiosity (episodic) | 0.07 | 0.80 | 0.63 | 10.78 | 30.20 | 0.20 | 0.80 | 36.28 | 0.01 |
| GRPO + curiosity (group) | 0.10 | 0.80 | 0.63 | 10.62 | 30.20 | 0.25 | 0.75 | 32.62 | 0.01 |
