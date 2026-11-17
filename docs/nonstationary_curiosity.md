# Curiosity when environment rules or rewards change

The research question is whether intrinsic motivation helps LLM agents discover
successful long-horizon behavior after their previous strategy stops working.
GRPO without curiosity is the main outcome-reward control. EMPG is an optional
comparison backbone. This intervention changes the environment, not its task mix.

## Eight-hour overnight run

```powershell
python run_nonstationary.py --tier nonstationary_overnight --hours 8 --fit-budget --estimate-only
python run_nonstationary.py --tier nonstationary_overnight --hours 8 --fit-budget
```

This preset compares GRPO with and without curiosity under **constant inspection
rules** and **original-to-inspection rules**. Both train and final evaluation use
inspection rules in the stationary condition. Generic feedback does not disclose
the prerequisite, and the same inspection actions remain available in both regimes.
It uses a 50-turn horizon, 12 fixed development probe tasks, and a separate fixed,
stratified 24-task `valid_unseen` final panel. One training seed favors deeper
learning within the overnight budget; it is an exploratory comparison, not a
replication across training seeds.

`--fit-budget` selects one even update count for every run before any training.
The switch is halfway through, and probes occur at update zero, immediately
before/after the switch, and the end. The full workload is used for fitting even
on resume, so finishing a run does not expand the remaining runs' training budget.
The default central-estimate target is 90% of the requested time, with an additional
reporting reserve. On the current calibration the eight-hour plan resolves to
**14 updates per run, switching at update 8, with a 6.9-hour central estimate**.
The scenario range is broad (about 3.5–10.4 hours); fitting is not a runtime guarantee.
Use `--budget-fraction=.75` for more headroom, or omit fitting and specify a fixed
workload manually. Fitting controls iteration, switch and probe interval counts.

The runner prints one aggregate estimate with training/probe/evaluation costs,
executes on/off pairs within a seed, assigns each remaining job a proportional
share of the remaining deadline, and reserves time for reports. A slow job cannot
consume the entire night's allocation for later conditions. Timeout cleanup and
unusually slow report generation may extend the eight-hour ceiling slightly.
Incomplete runs remain unavailable in paired comparisons. There are no extra
per-job report subprocesses; each condition is reported once after execution.

`matrix_run.json` records resolved configs, per-job estimates/statuses and elapsed
time. `curiosity_statistics.json` reports paired seed/task bootstrap intervals for
each condition and the interaction. With one seed these reflect **task variability
only**. Missing seeds are not imputed and incomplete pairs do not count as zero.
`--report-only` regenerates reports; rerun with the same fitting arguments/config.

For additional controls or replication, estimate the revised plan first:

```powershell
# Adds constant original rules as a transfer control (six runs per seed).
python run_nonstationary.py --tier nonstationary_overnight --hours 8 --fit-budget --schedules switch stationary stationary_initial --estimate-only

# More seeds require fewer updates, smaller panels, more hours, or several nights.
python run_nonstationary.py --tier nonstationary_overnight --hours 8 --seeds 0 1 --fit-budget --estimate-only -- nonstationary.probe_tasks=6 eval.max_tasks=12

# Prioritizes deeper training in the primary on/off switch comparison (two runs).
# This omits the stationary control and cannot estimate the interaction.
python run_nonstationary.py --tier nonstationary_overnight --hours 8 --fit-budget --schedules switch --estimate-only
```

Competence is checked on the final pre-switch old-regime probe: by default at
least 10% success and at least two successful tasks are required. The report marks
adaptation inconclusive otherwise; it does not move the switch separately for
each method, which would break the matched schedule. The saved original 12-update
GRPO experiment and its zero-shot baseline both scored about 6% on the full unseen
ALFWorld panel. That is a substantive learning limitation: this overnight run may
still find insufficient competence. Longer warm-up, a separately validated shared
initial checkpoint, or additional nights may be needed. No stronger baseline or
learning-rate improvement is assumed by this preset.

## Experiment

The environment offers `inspect <target>` alongside native admissible actions in
**both regimes**. Inspection takes one agent turn and marks the target as prepared.
By default the selected operation is `take`, and its target is the source
receptacle. Repeated inspection of an unchanged target is not a new parsed state.

- `baseline`: native operations and success rewards work as usual.
- `inspection`, `change=rules`: a selected operation fails until its target has
  been inspected. Explicit feedback explains the prerequisite; generic feedback
  only reports failure. Inserting inspections into
  an original successful action sequence preserves the native physical transitions.
- `inspection`, `change=rewards`: native operations still work, but a completed
  task pays only if **every successful selected operation** was prepared, with at
  least one such operation. Completion without payment still terminates the game.
  `base_won` records native task completion separately from rewarded success.

Inspection transitions and observation wording are shared between regimes. The
controller's regime ID and switch iteration never enter prompts or curiosity state
keys. State keys include observable inspection state and route eligibility flags;
they do not include growing action counters. This is a custom ALFWorld extension,
not the unmodified published benchmark. Synthetic inspection/blocked steps leave
native physics untouched and count toward the rollout's total turn budget.

Reward-only changes preserve physical transitions. Terminal reward feedback can
change, as expected. This experiment tests a new rewarded completion route, not
an arbitrary rescaling of the old reward. All compared tasks must admit the
selected operations; `actions=[take]` covers the six standard task families.
Selecting `heat`, for example, should be paired with a suitable task panel rather
than interpreting unaffected tasks as failed adaptation.

## Run the four conditions

```bash
# Cost preview only: no model load, training, or output files.
python run_nonstationary.py --estimate-only

# GRPO / GRPO + episodic curiosity, each with stationary / switched environments.
python run_nonstationary.py --hours 8

# Longer experiment; independently replicated training seeds.
python run_nonstationary.py --tier nonstationary --seeds 0 1 2 --hours 24

# Optional comparison backbone or different existing curiosity arm.
python run_nonstationary.py --backbone empg --estimate-only
python run_nonstationary.py --curiosity-arm grpo_cur_group --estimate-only

# Same design, reward-only change.
python run_nonstationary.py --runs runs/reward_matrix --out results/reward_matrix \
  --estimate-only -- nonstationary.change=rewards
```

Windows entry points forward to the existing WSL vLLM environment. The default
`nonstationary_screen` has six updates, a switch at update four, and a 50-turn
horizon. It checks the mechanism and early behavior; it cannot establish a
reliable learning advantage. `nonstationary` has 40 updates with a switch at 21.
Neither preset guarantees learning before the switch. Inspect the pre-switch
held-out success, and extend the budget if the agent has not learned a strategy.

The matrix fixes final evaluation to the **inspection regime in all conditions**
and preserves the same seeded task schedule across arms and environment schedules.
It writes condition reports under `results/<tier>_matrix/{stationary,switch}` and
`curiosity_matrix.md/.json` with per-seed curiosity effects and their interaction.
Task panels and evaluation protocols must match. Unfinished pairs remain unavailable.
The original screen and long presets retain the constant-original-rules control;
the overnight preset supplies the matched constant-inspection control. The matrix
deadline applies across jobs and probes, with a reserve for reporting. Estimate
before running a larger experiment.

## Independent controls

Every setting below supports dotted YAML/CLI overrides on `run_benchmarks.py` or
the individual `benchmarks.train`/`benchmarks.evaluate` commands.

| Setting | Effect |
|---|---|
| `nonstationary.enabled=false` | Original environment; no wrapper or probes. |
| `nonstationary.change=rules` / `rewards` | Switch physical prerequisites or payment eligibility. |
| `nonstationary.schedule=switch` / `stationary` | One switch or a constant regime. |
| `nonstationary.switch_iteration=21` | First training batch in the final regime. |
| `nonstationary.initial_regime` / `final_regime` | `baseline` / `inspection`; reversing them is supported. |
| `nonstationary.stationary_regime` | Constant `baseline` or `inspection` control. |
| `nonstationary.actions=[take,open]` | Selected verbs: `take`, `open`, `clean`, `heat`, `cool`. |
| `nonstationary.consume_inspection=true` | Preparation is consumed after each successful selected action. |
| `nonstationary.feedback=generic` / `explicit` | Generic feedback leaves the prerequisite to discover; explicit feedback names it. |
| `nonstationary.probe_every=0` | Disable all intermediate evaluation, including boundary probes. |
| `nonstationary.probe_tasks=12` | Fixed held-out panel size; zero means all. |
| `nonstationary.probe_split=valid_seen` | Development panel; reserve `valid_unseen` for confirmation. |
| `nonstationary.probe_regimes=[baseline,inspection]` | Which regimes to probe with each model snapshot. |
| `nonstationary.probe_seed=0` | Panel selection and isolated evaluation sampling seed. |
| `nonstationary.eval_regime=final` | Final evaluation regime; also supports either explicit regime. |
| `nonstationary.recovery_fraction=0.9` | Recovery threshold relative to pre-switch old-regime success. |
| `nonstationary.competence_min_success=0.1` | Minimum pre-switch held-out success fraction for interpreting adaptation. |
| `nonstationary.competence_min_successes=2` | Minimum number of successful pre-switch held-out tasks. |
| `advantage.curiosity.weight=0` | Intrinsic motivation off; positive values set its strength. |
| `advantage.curiosity.scope=episodic` / `group` | Existing within-episode or sibling-rollout novelty. |
| `advantage.curiosity.schedule=constant` | Keep curiosity active after a late shift; linear/exp remain available. |
| `train.reward_source=success` / `environment` | Binary rewarded success or summed environment reward. |
| `train.environment_reward_scale=10` | Scale the extrinsic environment return before advantage normalization. |

The matrix controls schedule, treatment identity and final evaluation to keep
comparisons matched. To vary curiosity weight without changing its zero-weight
control, edit that arm's YAML or run individual arms with separate overrides:

```bash
python run_benchmarks.py --tier nonstationary_screen --arms grpo_cur_episodic \
  --skip-base --runs runs/custom_curiosity --out results/custom_curiosity \
  --estimate-only -- advantage.curiosity.weight=0.1 nonstationary.probe_every=0
```

The existing curiosity bonus is added to advantages by default; it is not silently
converted into environmental reward. `train.reward_mode=step` uses the scaled
episode environment return on each response row, just as the existing success
outcome was broadcast; it is **not** a newly implemented discounted return-to-go
estimator. Formatting penalties remain separate. Disabled intervention settings
preserve historical configuration fingerprints.

## Measurements and efficiency

- Training logs record regime, unshaped environment returns, inspections, blocked
  actions, prepared actions, and successful alternative routes. Raw event counts
  are totals over the batch; `env_return` is its episode mean.
- Probes run at update zero, configured intervals, immediately before/after the
  switch, and the final update. Stationary controls get the same probe points.
  They reuse the loaded model and a separate persistent CPU worker pool. Evaluation
  groups both regimes into one generation session to avoid repeated GPU weight
  transfers (or repeated HF adapter merge/unmerge), and isolates Python, NumPy,
  torch and vLLM sampling RNG state from training.
- `adaptation_probes.jsonl` contains per-task held-out results and training cost
  coordinates. `adaptation_summary.json`, `adaptation_runs.csv`, `adaptation.md`,
  and `adaptation_curves.png` report recovery and post-boundary extrinsic return.
  Recovery is the **first observed threshold crossing**, not sustained recovery.
  A zero pre-switch reference or missing probes gives unavailable recovery.
- Training step/token/time coordinates exclude probes. `probe_seconds_total`
  records their separate overhead. `train.max_seconds` retains its original
  training-only meaning; use the suite/matrix deadline for total wall-clock cost.
- The estimator charges for all configured probe panels. It is a workload
  extrapolation, not measured runtime. Probes can be expensive despite model reuse.
- Probes inherit `eval.selection`. Stratified panels need at least one task per
  subtask (six on the standard splits); use `eval.selection=random` for tiny
  integration checks. Invalid probe panels fail before loading model weights.
- The intervention uses CPU string/set operations, deduplicates inspection commands
  by target, sends configuration only on worker reset, and adds no model forwards,
  backward passes, or worker restarts at the switch. Disabled environments return
  directly through the original construction path.

Validate without loading a model:

```bash
python -m scripts.validate_nonstationary --games 6
python -m scripts.validate_nonstationary --games 1 --feedback generic
pytest tests/test_nonstationary.py tests/test_adaptation_runner.py
```

The real-game validator replays expert solutions under eight combinations per
task family, checks both newly solvable routes and suppressed old rewards, checks
serial/parallel execution including a worker-pool phase switch, and measures CPU
wrapper overhead. It needs installed ALFWorld data, not a model download.
