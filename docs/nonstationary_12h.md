# Calibrated 12-hour non-stationary experiment

The objective is to test whether temporarily increasing curiosity after a rule
change helps a competent agent recover. The runner prepares and checks a shared
baseline policy, measures the actual GPU workload, freezes a matched experiment,
and enforces a 12-hour ceiling across its training and evaluation processes.

## Local prepared run

The checked baseline adapter is
`runs/audit/nonstationary_review_warmup/adapter`. It solved 12/12 held-out
development tasks at both depths 4 and 8; depth 8 requires 32 constructive
baseline actions. The native calibration is
`runs/audit/nonstationary_deduplicated_calibration/calibration.json`.

The plan is prepared with:

```powershell
python run_novelgym_12h.py --prepare-only --initial-adapter runs/audit/nonstationary_review_warmup/adapter --calibration runs/audit/nonstationary_deduplicated_calibration/calibration.json
```

Once `runs/nonstationary/12h/plan.json` exists, use:

```powershell
python run_novelgym_12h.py --estimate-only
python run_novelgym_12h.py
python run_novelgym_12h.py --report-only
```

Preparation does not start the long matrix. `--estimate-only` is read-only and
does not load a model. `--report-only` rebuilds reports from saved artifacts.
The runner adopts unspecified options from its frozen plan; changing the hours,
conditions, seeds, or coefficient requires a fresh `--root`.

The local frozen workload is **12 updates per run**, a switch at update 7, and a
three-update boost window. Its conservative primary forecast is **9.95 hours**
(35,828 seconds including the reporting allowance). The matched coefficient is
0.075. The remaining budget and any savings fund the predeclared reserve blocks.

## Fixed primary protocol

| Setting | Protocol |
|---|---|
| Environment | Custom native suite v3; depth-8 tool-rule change; generic feedback |
| Action horizon | 74 macro actions per episode |
| Training | Two maps per update, eight sibling rollouts per map |
| Seeds | 0 and 1; shared maps, initial weights, and update count across methods |
| Schedules | Baseline then changed rules; stationary changed-rule control |
| Constant | Curiosity coefficient 0.05 throughout |
| High | Coefficient 0.15 throughout |
| Matched | Constant equal to the oracle arm's planned average coefficient |
| Oracle | Coefficient 0.15 for a fixed window beginning at the known change |
| Detected | Coefficient 0.15 after a causal rise in observed operation failures |
| Change | First update of the second half of training |
| Boost window | One quarter of the fitted updates, minimum two |
| Development probes | 12 fixed tasks in both regimes, including initialization and change boundaries |
| Final evaluation | 32 fixed tasks from the untouched 3,000,000+ confirmation seed range |

There are 20 primary conditions and 16 distinct training runs: stationary oracle
and matched conditions reuse their exactly equivalent constant run. Reuse never
adds independent samples. Oracle timing is privileged information supplied only
to the curiosity coefficient; the observed detector never reads regime labels.

The initial checkpoint must solve at least 75% of at least eight held-out baseline
tasks during calibration. Per-run recovery interpretation also requires actual
pre-change competence on its probe panel. Warmup uses training maps, baseline
rules, clean solutions, and recovery from failed operations that leave the native
state unchanged. No changed-rule solution is demonstrated. Development checks
do not consume final confirmation tasks.

## Budget and optional replication

Calibration measures two repetitions of baseline and changed-rule rollout/update
costs, evaluation throughput, startup/shutdown, and a diverse-map kernel stress.
The forecast hedges episode lengths toward the full horizon and budgets at least
35% of response rows as distinct, so it does not assume repetitive rollouts persist.
It includes probe and final-evaluation costs and adds 10% cost headroom.

The runner chooses the largest common multiple-of-four update count fitting 92%
of the available time, subject to 12–200 updates by default. Reporting time is
reserved. The resulting count and complete configurations are written to
`plan.json`; each method keeps that count even if it runs faster than forecast.

If whole matched blocks fit after the primary matrix, predeclared reserve seeds
2–33 compare switched constant, oracle, and detected arms with the same checkpoint,
update count, change point, and evaluation protocol. Admission uses observed
completed-run costs with 25% headroom and the remaining time, never success scores.
Unstarted reserve seeds do not appear as missing primary results. Set
`--no-reserve` before freezing to disable this. `--reserve-count` controls the
predeclared limit. The final block may leave some unused time when another full
comparison would not fit.

When preparation and execution happen together, warmup and calibration consume
the same 12-hour budget. When `--prepare-only` runs first, the frozen workload is
fitted to a fresh 12-hour execution window. This is a measured forecast with a hard
ceiling, not a guarantee that every primary condition will finish if learning
makes future rollouts much more expensive. Cleanup/reporting can add a little time.

## New preparations and failures

For a new study, GPU, or changed execution code:

```powershell
python run_novelgym_12h.py --root runs/nonstationary/new_study --prepare-only
```

Without an explicit adapter/profile, the runner trains a new baseline checkpoint
and performs tuning and calibration. Preparation reuses only checked complete
artifacts; failed attempts stay available for inspection. A fresh preparation's
competence is measured rather than assumed from the validated local checkpoint.

`plan.json` hashes the calibration; calibration records hardware, execution code,
checkpoint contents, policy limits, environment, and training shape. Changed
identities are rejected before the long run. Every treatment uses identical GPU
settings. Full transformer checkpointing remains selected locally because the
faster selective setting failed its threshold-boundary memory check.

At the deadline the active process group is terminated, completed artifacts are
preserved, and partial reports are generated. Interrupted training restarts from
the shared initial checkpoint on rerun; optimizer state is not resumed. Finished
matching runs are reused. A rerun starts a fresh time window. A phase exiting
without complete train/evaluation artifacts is an error, not a successful run.
Missing results are unavailable, never zero success.

Review `results/comparison.md`, `results/summary.json`, and learning curves under
the run root. Primary comparisons are oracle/detected versus constant; high and
matched controls check coefficient exposure, and switch-minus-stationary
interactions check change-specific effects. Reserve seeds are reported in a
separate cohort, with separate comparison estimates and plots per reserve seed;
they do not alter the primary estimate. Their time-based admission can depend
indirectly on earlier episode lengths. Two primary seeds provide limited
uncertainty estimates; report effect sizes and seed variation alongside intervals.
