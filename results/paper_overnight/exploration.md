### Exploration during training (mean of last 5 iterations)

| Method | train success | revisit rate | no-op rate | unique states / ep | group coverage | P(group ≥1 success) | no-signal groups | steps to success | intrinsic share |
|---|---|---|---|---|---|---|---|---|---|
| GRPO | 0.04 | 0.84 | 0.68 | 8.72 | 36.80 | 0.10 | 0.90 | 33.33 | 0.00 |
| GRPO + EMPG | 0.04 | 0.83 | 0.68 | 9.40 | 44.80 | 0.20 | 0.80 | 12.25 | 0.00 |
