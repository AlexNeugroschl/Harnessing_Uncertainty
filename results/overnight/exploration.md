### Exploration during training (mean of last 5 iterations)

| Method | train success | revisit rate | no-op rate | unique states / ep | group coverage | P(group ≥1 success) | no-signal groups | steps to success | intrinsic share |
|---|---|---|---|---|---|---|---|---|---|
| GRPO | 0.05 | 0.86 | 0.79 | 7.60 | 20.60 | 0.20 | 0.80 | 6.00 | 0.00 |
| GRPO + EMPG | 0.00 | 0.86 | 0.77 | 7.80 | 22.80 | 0.00 | 1.00 | – | 0.00 |
| EMPG + curiosity (episodic) | 0.15 | 0.80 | 0.65 | 9.85 | 31.00 | 0.20 | 0.80 | 23.33 | 0.01 |
| GRPO + curiosity (group) | 0.25 | 0.86 | 0.77 | 7.85 | 20.80 | 0.60 | 0.40 | 41.11 | 0.21 |
