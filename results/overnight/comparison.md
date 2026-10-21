# ALFWorld `valid_seen` benchmark comparison

Success rates are percentages; ± is sample standard deviation across training seeds. Paired intervals below are unadjusted for multiple comparisons. Selecting a winner among many methods needs fresh confirmation.

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | All | seeds |
|---|---|---|---|---|---|---|---|---|
| Zero-shot base | 0.0 | 0.0 | 20.0 | 0.0 | 0.0 | 0.0 | 4.2 | 1 |
| GRPO | 16.7 | 0.0 | 20.0 | 0.0 | 0.0 | 0.0 | 8.3 | 1 |
| GRPO + EMPG | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| EMPG + curiosity (episodic) | 0.0 | 0.0 | 20.0 | 0.0 | 0.0 | 0.0 | 4.2 | 1 |
| GRPO + curiosity (group) | 0.0 | 0.0 | 20.0 | 0.0 | 0.0 | 0.0 | 4.2 | 1 |

### Paired evaluation differences

Intervals resample paired tasks and seeds; one seed measures task variability only. Few seeds give uncertain intervals. Degenerate intervals cannot establish equivalence. Screening and selected variants need fresh confirmation.

| Candidate vs baseline | delta (pp) | 95% interval (pp) | seeds | tasks |
|---|---:|---|---:|---:|
| empg vs grpo | -8.3 | [-20.8, +0.0] | 1 | 24 |
| empg_cur_episodic vs grpo | -4.2 | [-12.5, +0.0] | 1 | 24 |
| grpo_cur_group vs grpo | -4.2 | [-16.7, +8.3] | 1 | 24 |

## Curiosity against its matching baseline

These comparisons isolate adding curiosity within each algorithm. One seed is a development screen, not replication across training randomness.

### Paired evaluation differences

Intervals resample paired tasks and seeds; one seed measures task variability only. Few seeds give uncertain intervals. Degenerate intervals cannot establish equivalence. Screening and selected variants need fresh confirmation.

| Candidate vs baseline | delta (pp) | 95% interval (pp) | seeds | tasks |
|---|---:|---|---:|---:|
| grpo_cur_group vs grpo | -4.2 | [-16.7, +8.3] | 1 | 24 |

### Paired evaluation differences

Intervals resample paired tasks and seeds; one seed measures task variability only. Few seeds give uncertain intervals. Degenerate intervals cannot establish equivalence. Screening and selected variants need fresh confirmation.

| Candidate vs baseline | delta (pp) | 95% interval (pp) | seeds | tasks |
|---|---:|---|---:|---:|
| empg_cur_episodic vs empg | +4.2 | [+0.0, +12.5] | 1 | 24 |


### Individual runs

Training time includes rollout and updates, excluding model startup, checkpoint saves and evaluation. Missing evaluations are not zero successes. Time-budgeted methods may finish different numbers of updates; compare their recorded costs too.

| Method | Seed | Status | Success (%) | Eval tasks | Iterations | Env steps | Training (h) | Training stop |
|---|---:|---|---:|---:|---:|---:|---:|---|
| base |  | evaluated | 4.2 | 24 |  |  | — | not trained |
| empg_cur_episodic | 0 | evaluated | 4.2 | 24 | 7 | 1320 | 0.71 | max_seconds |
| empg | 0 | evaluated | 0.0 | 24 | 8 | 1600 | 0.70 | max_seconds |
| grpo_cur_group | 0 | evaluated | 4.2 | 24 | 8 | 1562 | 0.68 | max_seconds |
| grpo | 0 | evaluated | 8.3 | 24 | 8 | 1556 | 0.67 | max_seconds |
