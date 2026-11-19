# Recovery v4 validation

The five recovery changes were validated on the local RTX 5060 Laptop GPU
(8 GiB) using the existing WSL vLLM environment. Previous v3 artifacts remain
intact. The new long matrix has been prepared without starting its training.

## Baseline initialization

The checked adapter is `runs/audit/recovery_v4_warmup/adapter`, fingerprint
`0e07404502c25faa2a659da850aa80d06c3a3886448adb30b552aed6dcdb93f8`.
It used baseline-only demonstrations and optional tool-selection target mass
0.05, starting from the previous baseline-only skill checkpoint. It stopped
after two epochs: all eight checked development tasks succeeded at each of
depths 2, 4, and 8. The canonical action-catalog diagnostic measured mean
conditional tool-selection probability 0.0519. Supervised training took 156 s;
the warmup record, including validation, took 233 s.

## Actual development learning

The two depth-2 learning pilots completed all 16 rounds, changing rules after
four baseline rounds. Both passed the final gate recomputed from their saved
probes, with the same eight development tasks and sampling protocol:

| Training seed | Initial changed-rule success | Immediately pre-change counterfactual | Final changed-rule success | Final old-rule retention |
|---|---:|---:|---:|---:|
| 101 | 37.5% | 62.5% | 100% | 100% |
| 102 | 37.5% | 62.5% | 100% | 100% |

They produced 176 and 182 successful changed-rule training episodes out of 192
per seed. The selected shared settings are learning rate 3e-6, entropy
coefficient 0.01, and inventory/selection novelty (`key=progress`). Every arm of
the prepared long study starts from the baseline-only adapter above, never a
pilot-adapted checkpoint.

These are development checks with two seeds and a small task panel. They
establish a learning signal and a passing engineering gate, not a benefit from
boosting curiosity or a long-horizon confirmation result. The pilot uses
constant curiosity. Its report and plots are in
`runs/audit/recovery_v4_pilot_03/`.

An independent 16-episode diagnostic, with the same tasks and random streams
before and after adaptation, improved from 75% to 100%. All 16 adapted episodes
selected the axe and harvested successfully. Recorded failed-action repetitions
were zero. These additional diagnostic draws are distinct from the gate's
sampling stream, so their initial success percentages differ.

## Correctness and numerical checks

Tests verify weighted token-mean supervised gradients, full-sequence probability
sums including EOS, finite-catalog normalization, real native action/progress
counters, unchanged extrinsic completion rewards, observable-only novelty, and
disjoint new evaluation panels. Existing objective/gradient tests cover PPO
clipping, dual clipping, entropy/KL weighting, multiple optimizer minibatches,
and weighted response deduplication.

The final full regression completed with **530 passing tests** in 60.9 seconds.
The only warning is the upstream pygame dependency's deprecated `pkg_resources`
import. `git diff --check`, Windows-to-WSL entry points, frozen-plan revalidation,
and inference-worker shutdown checks also passed.

The rearmed detector passed both isolated step-change tests and an actual native
CPU training-loop integration test with a scripted policy: one persistent change
triggered exactly once, affected the next updates, and did not trigger again when
the boost ended. Later changes after a stable baseline were detected. Stationary
constant failure rates did not trigger. Learning evidence checks reject altered
artifacts, mismatched checkpoints/settings, missing seeds, or a manually asserted
pass flag unsupported by the development trajectories.

The scientific-notation CLI parser now preserves numeric overrides such as
`train.lr=3e-06`. Evaluation's independent request streams are reproducible and
do not change temperature, top-p, or the policy's categorical distribution.

## Hardware execution

Native depth-8 calibration used the baseline-only checkpoint, selected learning
settings, and independent request streams. Baseline training success was 16/16;
changed-rule training success was 12/16. The initial checkpoint's held-out
development evaluation was 12/12 under old rules and 3/12 under changed rules
at evaluation temperature 0.4. This leaves a measured adaptation challenge.

The baseline update took 63.65 s for 527 response rows (177 gradient rows), and
the changed-rule update took 113.07 s for 869 rows (387 gradient rows). These
costs exceed the v3 repetitive-policy costs because the new policy produces more
distinct trajectories. Peak reported HF reserved memory was 3.93 GiB, excluding
the separate sleeping inference worker.

A maximum-context stress with eight distinct synthetic prompts at 1,536 prompt
plus 256 response tokens passed with finite updates and 4.79 GiB HF reserved.
It is a memory/kernel check, not a task-performance result. Distinct prompts
prevent response deduplication from silently reducing the stress workload.

The shared-rollout microbatch comparison uses three timing repeats, restores the
same weights and optimizer initialization, and records numerical differences.
Recovery tuning requires at least a 10% median improvement and checks candidate
differences against `max(2e-6, learning_rate, 1.5 * reference_repeatability_noise)`.
This accounts for measured bf16 variation even between reference repeats; it is
an empirical numerical screen, not a bitwise-equivalence guarantee. Faster
settings also have to pass maximum-context and padded-token-budget stresses.
The conservative reference setting remains selected.

Artifacts: `runs/audit/recovery_v4_calibration/`,
`runs/audit/recovery_v4_context_stress.json`, and
`runs/audit/recovery_v4_execution_tuning/selection.json`.

## Prepared long study

`runs/nonstationary/recovery_12h/plan.json` freezes 12 conditions and 10 distinct
runs: constant, oracle, and detected curiosity; switched and stationary rules;
two seeds. Each gets 16 rounds, with a change at round 5 and 12 adaptation rounds.
Oracle boosts last four rounds (5-8); detected boosts start after the observed
trigger and last four rounds. The base coefficient is 0.05 and the boost is 3x.
The conservative primary forecast is 34,571 seconds (9.60 hours) with a 12-hour
whole-process ceiling. Extra-seed reserve replication is disabled.

The same minimum training budget with all five curiosity strategies forecasts
15.31 hours before the additional global scheduling allowance, so it does not
fit the 12-hour protocol. High and matched controls remain configurable in a
fresh study with sufficient budget. Their omission from this primary screen
limits attribution of any gain specifically to timing versus extra coefficient
exposure. All requested treatments remain implemented and supported.

Final confirmation tasks use the fresh 4,000,000+ seed range. None were consumed
by warmup, pilot selection, diagnostics, or hardware calibration. For exact
definitions and commands, see [the protocol](recovery_protocol.md).
