# ALFWorld `valid_unseen` benchmark comparison

Success rates are percentages; ± is sample standard deviation across training seeds. Paired intervals below are unadjusted for multiple comparisons. Selecting a winner among many methods needs fresh confirmation.

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | All | seeds |
|---|---|---|---|---|---|---|---|---|
| Zero-shot base | 16.7 | 5.6 | 3.2 | 4.3 | 0.0 | 5.9 | 6.0 | 1 |
| GRPO | 8.3 | 5.6 | 6.5 | 8.7 | 4.8 | 0.0 | 6.0 | 1 |
| GRPO + EMPG | 16.7 | 0.0 | 12.9 | 13.0 | 4.8 | 0.0 | 9.0 | 1 |

### Paired evaluation differences

Intervals resample paired tasks and seeds; one seed measures task variability only. Few seeds give uncertain intervals. Degenerate intervals cannot establish equivalence. Screening and selected variants need fresh confirmation.

| Candidate vs baseline | delta (pp) | 95% interval (pp) | seeds | tasks |
|---|---:|---|---:|---:|
| empg vs grpo | +3.0 | [-2.2, +8.2] | 1 | 134 |

## Curiosity against its matching baseline

These comparisons isolate adding curiosity within each algorithm. One seed is a development screen, not replication across training randomness.

No completed curiosity/baseline pairs available.

### Individual runs

Training time includes rollout and updates, excluding model startup, checkpoint saves and evaluation. Missing evaluations are not zero successes. Time-budgeted methods may finish different numbers of updates; compare their recorded costs too.

| Method | Seed | Status | Success (%) | Eval tasks | Iterations | Env steps | Training (h) | Training stop |
|---|---:|---|---:|---:|---:|---:|---:|---|
| base |  | evaluated | 6.0 | 134 |  |  | — | not trained |
| empg | 0 | evaluated | 9.0 | 134 | 12 | 9445 | 1.93 | iterations |
| grpo | 0 | evaluated | 6.0 | 134 | 12 | 9475 | 1.95 | iterations |
