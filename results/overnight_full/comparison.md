# ALFWorld `valid_seen` benchmark comparison

Success rates are percentages; ± is sample standard deviation across training seeds. Paired intervals below are unadjusted for multiple comparisons. Selecting a winner among many methods needs fresh confirmation.

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | All | seeds |
|---|---|---|---|---|---|---|---|---|
| Zero-shot base | 0.0 | 0.0 | 11.1 | 0.0 | 11.1 | 0.0 | 4.2 | 1 |
| GRPO | 16.7 | 0.0 | 11.1 | 0.0 | 0.0 | 0.0 | 6.2 | 1 |
| GRPO + EMPG | 0.0 | 0.0 | 11.1 | 0.0 | 11.1 | 0.0 | 4.2 | 1 |
| EMPG + curiosity (episodic) | 0.0 | 0.0 | 11.1 | 0.0 | 0.0 | 0.0 | 2.1 | 1 |
| GRPO + curiosity (group) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |

### Paired evaluation differences

Intervals resample paired tasks and seeds; one seed measures task variability only. Few seeds give uncertain intervals. Degenerate intervals cannot establish equivalence. Screening and selected variants need fresh confirmation.

| Candidate vs baseline | delta (pp) | 95% interval (pp) | seeds | tasks |
|---|---:|---|---:|---:|
| empg vs grpo | -2.1 | [-8.3, +4.2] | 1 | 48 |
| empg_cur_episodic vs grpo | -4.2 | [-10.4, +0.0] | 1 | 48 |
| grpo_cur_group vs grpo | -6.2 | [-14.6, +0.0] | 1 | 48 |

## Curiosity against its matching baseline

These comparisons isolate adding curiosity within each algorithm. One seed is a development screen, not replication across training randomness.

### Paired evaluation differences

Intervals resample paired tasks and seeds; one seed measures task variability only. Few seeds give uncertain intervals. Degenerate intervals cannot establish equivalence. Screening and selected variants need fresh confirmation.

| Candidate vs baseline | delta (pp) | 95% interval (pp) | seeds | tasks |
|---|---:|---|---:|---:|
| grpo_cur_group vs grpo | -6.2 | [-14.6, +0.0] | 1 | 48 |

### Paired evaluation differences

Intervals resample paired tasks and seeds; one seed measures task variability only. Few seeds give uncertain intervals. Degenerate intervals cannot establish equivalence. Screening and selected variants need fresh confirmation.

| Candidate vs baseline | delta (pp) | 95% interval (pp) | seeds | tasks |
|---|---:|---|---:|---:|
| empg_cur_episodic vs empg | -2.1 | [-6.2, +0.0] | 1 | 48 |


### Individual runs

Training time includes rollout and updates, excluding model startup, checkpoint saves and evaluation. Missing evaluations are not zero successes. Time-budgeted methods may finish different numbers of updates; compare their recorded costs too.

| Method | Seed | Status | Success (%) | Eval tasks | Iterations | Env steps | Training (h) | Training stop |
|---|---:|---|---:|---:|---:|---:|---:|---|
| base |  | evaluated | 4.2 | 48 |  |  | — | not trained |
| empg_cur_episodic | 0 | evaluated | 2.1 | 48 | 5 | 3895 | 1.29 | iterations |
| empg | 0 | evaluated | 4.2 | 48 | 5 | 3946 | 1.31 | iterations |
| grpo_cur_group | 0 | evaluated | 0.0 | 48 | 5 | 3825 | 1.24 | iterations |
| grpo | 0 | evaluated | 6.2 | 48 | 5 | 3858 | 1.28 | iterations |
