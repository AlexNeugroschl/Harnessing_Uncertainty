### Individual runs

Training time includes rollout and updates, excluding model startup, checkpoint saves and evaluation. Missing evaluations are not zero successes. Time-budgeted methods may finish different numbers of updates; compare their recorded costs too.

| Method | Seed | Status | Success (%) | Eval tasks | Iterations | Env steps | Training (h) | Training stop |
|---|---:|---|---:|---:|---:|---:|---:|---|
| base |  | evaluated | 6.0 | 134 |  |  | — | not trained |
| empg | 0 | evaluated | 9.0 | 134 | 12 | 9445 | 1.93 | iterations |
| grpo | 0 | evaluated | 6.0 | 134 | 12 | 9475 | 1.95 | iterations |
