### Individual runs

Training time includes rollout and updates, excluding model startup, checkpoint saves and evaluation. Missing evaluations are not zero successes. Time-budgeted methods may finish different numbers of updates; compare their recorded costs too.

| Method | Seed | Status | Success (%) | Eval tasks | Iterations | Env steps | Training (h) | Training stop |
|---|---:|---|---:|---:|---:|---:|---:|---|
| base |  | evaluated | 4.2 | 48 |  |  | — | not trained |
| empg_cur_episodic | 0 | evaluated | 2.1 | 48 | 5 | 3895 | 1.29 | iterations |
| empg | 0 | evaluated | 4.2 | 48 | 5 | 3946 | 1.31 | iterations |
| grpo_cur_group | 0 | evaluated | 0.0 | 48 | 5 | 3825 | 1.24 | iterations |
| grpo | 0 | evaluated | 6.2 | 48 | 5 | 3858 | 1.28 | iterations |
