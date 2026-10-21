### Individual runs

Training time includes rollout and updates, excluding model startup, checkpoint saves and evaluation. Missing evaluations are not zero successes. Time-budgeted methods may finish different numbers of updates; compare their recorded costs too.

| Method | Seed | Status | Success (%) | Eval tasks | Iterations | Env steps | Training (h) | Training stop |
|---|---:|---|---:|---:|---:|---:|---:|---|
| base |  | evaluated | 4.2 | 24 |  |  | — | not trained |
| empg_cur_episodic | 0 | evaluated | 4.2 | 24 | 7 | 1320 | 0.71 | max_seconds |
| empg | 0 | evaluated | 0.0 | 24 | 8 | 1600 | 0.70 | max_seconds |
| grpo_cur_group | 0 | evaluated | 4.2 | 24 | 8 | 1562 | 0.68 | max_seconds |
| grpo | 0 | evaluated | 8.3 | 24 | 8 | 1556 | 0.67 | max_seconds |
