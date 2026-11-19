# Recovery protocol v4

This protocol repairs the exploration and study-design weaknesses found in the
completed v3 run. The v3 configurations, checkpoints, results, task panels, and
legacy detector remain available. New studies use a fresh root through
`run_novelgym_recovery.py`.

## Five changes

1. **Observable behavior diagnostics.** Every training batch records action
   frequencies, failed-action frequencies, consecutive repetitions of the same
   failed action, episodes selecting the axe, episodes harvesting successfully,
   harvested and crafted item counts, and the highest component stage observed
   in inventory. Two uniformly spaced episode indices per batch have full action,
   feedback, and transition traces saved in `behavior_traces.jsonl`. Evaluation
   saves full bounded traces for four tasks. These metrics never shape extrinsic
   rewards and do not read the hidden rule or regime label.
2. **Competence with retained exploration.** Baseline-only demonstrations include
   optional tool selection as an alternative to direct harvesting. Both actions
   are valid under the initial rules. Alternative response weights sum to one at
   each decision; the default optional-tool target mass is 0.05. Recovery examples
   still use failed operations that leave the world unchanged. Validation runs
   every epoch, saves candidate checkpoints, and stops at the earliest checkpoint
   passing baseline competence and the exploration check. No changed-rule
   trajectory is supervised.
3. **A development learnability gate.** A short depth-2 pilot uses at least two
   training seeds and 16 rounds, with the change after four baseline rounds.
   Candidates are tried in a predeclared order: learning rates 3e-6, then 1e-5,
   at entropy coefficient 0.01. A candidate must pass on every seed: final
   changed-rule success at least 50%, improvement of at least 12.5 percentage
   points over the immediately pre-change counterfactual probe, at least eight
   identical development tasks, old-rule retention and pre-change competence at
   least 75%, and at least one successful changed-rule training episode. This is
   an engineering screen, not a significance test. The first passing candidate
   supplies shared hyperparameters. Its adapted weights are never used as the
   long experiment's initialization.
4. **A separate observable novelty abstraction.** `key=progress` hashes displayed
   inventory quantities and selected tool, omitting position, facing, resource
   layout, and hidden rules. Movement alone receives no novelty bonus. This is
   inventory/selection novelty, not a guarantee that every rewarded transition
   helps finish the task. `key=parsed` retains the original full observable-state
   novelty as an explicit alternative. Every timing treatment within one study
   uses the same selected key.
5. **Causal detector rearming.** A trigger still compares the current batch's
   failure rate with the prior exponential moving average and affects the next
   update. The v4 detector disarms after a trigger. Once the boost has ended, it
   rearms only after two consecutive batches lie within 0.05 of the prior EMA.
   This prevents a sustained step increase from repeatedly firing as an earlier
   boost expires. The EMA smoothing factor remains 0.2 and the trigger threshold
   is 0.15. A new change before the detector establishes a stable baseline may be
   missed; this is an explicit limitation of the latch, not an oracle detector.

Fixed-seed vLLM evaluation also now uses a separate reproducible random stream
for each request and generation call. The old path supplied the same fixed seed
to every request, correlating otherwise independent episodes. The sampling
temperature and probabilities are unchanged. The old behavior remains selectable
for legacy artifacts; v4 records independent request streams in the evaluation
protocol.

## Mathematical definitions

For visible-state abstraction `k`, let `N_t(k(s_{t+1}))` count occurrences in the
episode through the new state, including the reset state. The intrinsic term is

```
b_t = eta * multiplier(update) * 1[N_t(k(s_{t+1})) == 1]
A_t = A_GRPO,t + b_t - mean_batch(b)
```

The curiosity multiplier modifies only the intrinsic term. Completion reward,
failure measurements, and reported extrinsic returns are preserved. An unchanged
state receives no repeated first-visit bonus. The `progress` and `parsed` variants
differ only in the explicitly recorded state abstraction. Neither is claimed to
be potential-based reward shaping or an unbiased estimate of the sparse outcome.

Warmup minimizes the weighted response-token negative log likelihood in each
optimizer batch:

```
L = -sum_i w_i * sum_j log p_theta(response_i,j | prompt_i, response_i,<j)
    / sum_i w_i * response_length_i
```

Microbatch accumulation uses that same denominator. The target weights affect
both numerator and denominator. Candidate action diagnostics sum token log
probabilities over the entire exact response, including EOS. They report both
the total probability mass of that finite catalog and the conditional
distribution given emission of a listed response. The conditional values are
not presented as the unrestricted model's action probabilities.

RL continues to sample at temperature 1 with top-p 1. PPO scores that same
conditional token distribution, with the existing clipping, dual clipping,
adapter-off reference KL, full-vocabulary entropy, and token-mean loss. The
entropy coefficient is a selected shared hyperparameter, not a change in the
definition of entropy. No epsilon-greedy actions, forced exploration actions,
temperature mismatch, or hidden-rule reward is added to training.

The inherited `reward_mode=step` protocol repeats each episode's scaled outcome
on its response rows, subtracts the format penalty on invalid response rows, and
normalizes those scores within the task group. It retains the repository's
documented local response-row GRPO protocol; it does not silently substitute
episode-level normalization or claim a full paper replication.

## Workflow

Prepare a fresh study, including warmup, the development pilot, and native GPU
calibration:

```powershell
python run_novelgym_recovery.py --prepare-only
python run_novelgym_recovery.py --estimate-only
python run_novelgym_recovery.py
```

Preparation can accept `--initial-adapter`, `--learning-profile`, and
`--calibration` to reuse content-checked artifacts. A passing profile is mandatory
for the recovery tier. The gate is recomputed from the recorded development
probes; a `passed` flag alone cannot unlock a run. Artifact hashes, checkpoint
identity, shared learning settings, and calibration execution identity are
checked. Frozen plans reject changed evidence.

To inspect an adapter without updating it:

```powershell
python run_novelgym_diagnose.py --adapter PATH --depth 2 --episodes 16 --out runs/audit/decision_diagnostics.json
```

This records decisions after baseline and changed-rule failures. With
`--episodes`, the two novelty keys are audited on exactly the same sampled
trajectories. This shared-data diagnostic measures the reward signal, not the
learning effect of changing the bonus.

For a separate controlled key ablation, run the pilot with the same initialization,
seeds, and hyperparameters in a fresh directory using `--curiosity-key parsed`,
then `--curiosity-key progress`. Learning-rate and entropy grids are exposed via
`--learning-rates` and `--entropy-coefs`. Pilot execution is bounded by `--hours`;
incomplete or nonpassing evidence cannot qualify a long run.

## Budget, panels, and interpretation

The recovery runner gives the baseline four rounds and allocates the remaining
fitted rounds to changed-rule learning. Its minimum is 16 total rounds, providing
at least 12 adaptation rounds, versus six previously. Extra reserve seeds are
disabled by default. The default primary screen compares constant, oracle, and
detected curiosity, with stationary controls, two seeds, and a matched update
count: 12 conditions and 10 distinct runs. All five strategies remain supported
through `--treatments constant high matched oracle detected` in a fresh root with
a budget that fits the measured workload. High and coefficient-matched controls
are needed before attributing a gain specifically to timing rather than greater
coefficient exposure. Both the original and
new detector definitions are explicit configuration choices, with v4 using the
rearmed definition for every observed-failure treatment.

The 12-hour workload is fitted from fresh native measurements with the selected
initial checkpoint and learning settings. It includes probes and final
evaluation, preserves the safe 8-GiB execution settings, and rejects a workload
that cannot fit its minimum. It never silently shortens later arms to make them
finish. Future behavioral diversity can still change runtime; the global process
deadline preserves completed work and marks partial results unavailable.

Suite v4 uses disjoint seed ranges: training 100,000+, development 1,500,000+,
and final confirmation 4,000,000+. Its generic feedback and initial guide are
identical across regimes. Pilot selection consumes development tasks only. The
final confirmation panel remains outside the pilot and preparation checks.

[Local validation](recovery_validation.md) records the checked checkpoint,
two-seed learning results, GPU measurements, and exact prepared workload.

The combined repair package changes initialization, exploration settings, the
novelty abstraction, and evaluation sampling. Improvement over v3 does not
identify which individual change caused the improvement. Within the new long
study, those settings are shared, so the timing comparisons remain controlled.
