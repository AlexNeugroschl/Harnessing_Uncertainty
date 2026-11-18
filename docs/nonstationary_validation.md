# Non-stationary implementation validation

Engineering checks were performed on the local RTX 5060 Laptop GPU (8 GiB),
inside the existing WSL `empg-vllm` environment. The checks establish that the
implementation works; they do not establish a curiosity adaptation benefit.

The optional native gridworld dependency was installed and tested against the
pinned upstream engine. The optional AdaExploreBench checkout and released test
JSONL are available in `.tmp/`; native demonstrations and training artifacts are
under `runs/audit/`. Historical ALFWorld experiments were preserved.

## Native tasks and controls

Actual upstream transitions were tested at dependency depths 2, 4, 8 and 16,
for tool, recipe and resource changes, in both initial and changed regimes.
Constructive solutions finish without failed operations. Hidden changes do not
alter reset observations, action vocabulary or parsed curiosity keys. Repeated
unchanged states earn no first-visit reward. Reward-only changes preserve physical
transitions while changing whether terminal completion pays.

Curiosity controller tests cover finite windows, recurring boundaries, stationary
controls, coefficient scaling and causal failure detection. Observed detection
uses completed batches and cannot reweight the batch that triggered it. Matrix
tests cover matching task/seed protocols, coefficient-matched controls, missing
results, explicit reuse of equivalent stationary conditions, completion manifests,
and adapter content integrity before GPU allocation.

## GPU measurements

| Check | Result | Scope |
|---|---|---|
| Initial rollout | 80 action steps; 10.28 s decoding | Includes a first-use kernel compile; excludes engine initialization |
| Native reset / stepping | 0.112 s / 0.004 s | Same initial 16-trajectory, five-turn rollout |
| Shared-rollout update, score batch 2 | 22.28 s, 4.24 GiB HF reserved | 80 response rows, restored weights, fresh optimizer |
| Shared-rollout update, score batch 4 | 20.90 s, 5.04 GiB HF reserved | Same responses and weights; selected for the preset |
| Disable transformer checkpointing | Allocator OOM | Rejected; checkpointing remains enabled |
| Selective checkpointing candidate | 20.21 s, 5.14 GiB HF reserved | One timing sample; not selected on that evidence alone |
| Maximum-context update | 12.18 s, 4.79 GiB HF reserved, finite gradients | Eight synthetic rows at 1,536 prompt + 256 response tokens, with the real sleeping vLLM worker |

Timing differences include kernel shape/rounding effects and are not precision
equivalence claims. The vocabulary scores remain exact. HF reserved memory
excludes the separate sleeping vLLM process. Raw records are in
`runs/audit/novelgym_gpu_profile.json` and `runs/audit/novelgym_update_tuning.json`.
The maximum-context check is in `runs/audit/novelgym_context_stress.json` and is
a memory/gradient check, not a task-success measurement.

The original 128-token response cap truncated 7.5% of responses, so it was raised
to 256. A revised-format 80-step tuning rollout had no invalid or truncated
responses. A later native-format check reduced mean response length to 5.09 tokens
with no formatting errors or truncation in 104 responses. These are interface
measurements, not evidence of task learning. Initial prompt sizes at depths
2, 4, 8, 16 and 32 were 319, 355, 433, 586 and 891 tokens, below the 1,536 limit.

## Real training and competence

A three-update end-to-end run crossed the tool-rule switch, performed real LoRA
updates, saved its adapter, evaluated both-regime probes, and wrote the report
and PNG curves. Its coefficient schedule was exactly **0.05 → 0.15 → 0.05**.
The measured centered intrinsic contribution increased during the boosted update.
Training used 312 environment steps and about 146 s, excluding probes and startup.
It did not establish pre-change competence and the report correctly marks recovery
inconclusive. See `results/audit/novelgym_mechanism/summary.json`.

A small shared supervised warmup then used 96 baseline-only expert response rows
from training maps (depths 2 and 4), repeated for four epochs. It took about 76 s
and reduced mean target-token NLL from 0.387 to 0.046. It demonstrated no changed
rules. The saved adapter is `runs/audit/novelgym_warmup/adapter`.

On eight held-out development layouts, that checkpoint solved **8/8 at depth 2**.
At depth 4 it solved **1/8 under baseline rules and 0/8 under changed tool rules**.
The latter does not pass the default minimum of two successful tasks. This
checkpoint is useful for a short mechanism pilot; it is not yet a validated
initialization for the longer-horizon confirmation study.

A subsequent recovery-aware warmup used training maps at depths 4 and 8, starting
from that checkpoint. It included baseline failures that leave the world
unchanged, with the correct recovery action demonstrated afterward. It stopped
after two epochs when all 12 held-out development layouts at **both depths 4 and
8** succeeded. Token NLL fell from 0.0674 to 0.00709; supervised training took
210.8 seconds (255.7 seconds including validation). Peak HF
reserved VRAM was 3.78 GiB. No changed-rule solution was demonstrated.

The new checkpoint is `runs/audit/nonstationary_review_warmup/adapter`, with
SHA-256 `3b4306e16848132dff7f2cb555f457c854c68dedcd97a3c5ce2b48bfb2643da3`.
Independent native calibration again solved 12/12 baseline development tasks at
depth 8 and 0/12 under changed tool rules. This passes the initialization gate
and confirms a measurable rule-shift challenge. The 32-action constructive
baseline and 74-action cap are shared by every primary treatment.

These development checks motivated a fresh confirmation seed range in suite
version 3. No final confirmation result is claimed.

## Complementary official runtime

The official AdaExploreBench runtime completed two released episodes in both
dynamic and static conditions using its model-free random baseline. This checks
native rules, history/protocol compatibility, output trajectories and official
metrics. It does not evaluate local model performance. Local client tests confirm
that independent conversations are batched into one GPU call, preserve system /
user / assistant message boundaries, and propagate errors to every waiting task.
See `results/audit/adaexplore_runtime/`.

## Running the prepared pilot

The measured short-task initialization can be used directly:

```powershell
python run_novelgym.py --depths 2 --initial-adapter runs/audit/novelgym_warmup/adapter --treatments constant high matched oracle detected --estimate-only
```

Remove `--estimate-only` and add a whole-matrix `--hours` ceiling to execute it.
For longer tasks, prepare the default warmup and use fixed development probes
to establish competence first. [The protocol](nonstationary_research.md) describes
dose sweeps, recurring changes, reward shifts, and the full paired-seed study.

## Current GPU optimization and 12-hour preparation

The main workflow is now [the calibrated 12-hour experiment](nonstationary_12h.md),
using the competent depth-8 checkpoint above. Core experiment code lives in
`benchmarks/nonstationary/`; the root commands remain compatibility entry points.

Repeated exact prompt/response pairs share scoring work. Duplicate gradient rows
are merged only within the original optimizer minibatch and only for identical
advantages, using their original multiplicities. This preserves the mathematical
token-mean objective, KL and entropy weighting, and optimizer step count. CPU
tests cover mixed advantage signs, clipping ratios, dual clipping, multiple PPO
epochs, GRPO/EMPG, response order, and independent returned tensor storage.

| Native depth-8 measurement | Before sharing | After sharing |
|---|---:|---:|
| Baseline update, 512 response rows | 152.3 s | 23.1 s; 64 gradient rows |
| Changed-rule update, 1,184 response rows | 349.0 s | 19.7 s; 72 gradient rows |
| Shared 96-row GPU equivalence check | 36.63 s | 5.39 s; 12 gradient rows |
| Shared changed-rule 256-row check | 87.34 s | 9.03 s; 24 gradient rows |

The first two measurements come from consecutive native calibrations with the
same checkpoint, seed, execution settings, and task shape. They are indicative
speed measurements, not a learning result. The last comparison restores the same
weights and fresh optimizer on exactly the same cached rollout. Maximum adapter
element difference was 3.46e-7; bf16 batch-shape rounding can change scores and
updates slightly, so numerical equivalence is approximate rather than bitwise.
All treatments use the same selected implementation.

The 256-row comparison uses a real changed-rule rollout with nonzero curiosity
advantages (`adv_std=0.01654`). Intrinsic contribution and advantage statistics
match exactly; the policy-gradient loss differs by 1.28e-9. Maximum adapter
element difference was 1.05e-6 after the real bf16 update. This additional check
is in `runs/audit/nonstationary_dedup_curiosity_equivalence.json`.

Two-repeat tuning found selective checkpointing faster, but the candidate failed
the real 899-token boundary memory check. Production retains the safe full
checkpointing setting. A diverse-map stress used 125 distinct scored rows out of
128 and took 43.3 seconds, preventing the hours forecast from assuming that
exploration will always produce repetitive trajectories. The complete native
profile is `runs/audit/nonstationary_deduplicated_calibration/calibration.json`;
the exact shared-rollout comparison is
`runs/audit/nonstationary_dedup_equivalence.json`.

The frozen primary plan contains 20 conditions, 16 distinct runs, two seeds,
12 matched updates each, and a 9.95-hour conservative primary forecast. Optional
matched reserve blocks can fill the rest of the 12-hour ceiling. Preparation
completed without launching that long matrix. The final full regression passed
**478 tests** in 58.9 seconds, including nested-session deadline cleanup,
coefficient overrides, invalid calibration rejection, reserve admission, and
separate primary/reserve estimates. The only warning comes from the upstream
pygame package's deprecated `pkg_resources` import. The original validation
artifacts above remain available.

## Complete current system check

A genuine GPU matrix completed all six depth-8 conditions: switched and stationary
constant/oracle/detected, with one exact stationary reuse (five trained policies).
Each policy completed four updates, saved its adapter, ran both-regime probes,
and evaluated four development tasks. All six conditions have complete artifacts;
the six paired/interaction comparisons and two PNG figures regenerate successfully
without retraining. There were no truncated responses.

The switched oracle coefficient was `[0.05, 0.05, 0.15, 0.05]`, and its centered
intrinsic contribution at update 3 was three times the constant control. Observed
detection triggered after update 3 and used `[0.05, 0.05, 0.05, 0.15]`. Under
stationary changed rules it never triggered and stayed at 0.05 despite persistent
failures. Pre-change success was established in the switched runs. None recovered
changed-rule success within this short check; it is an engineering test, not
evidence that boosts help or fail over the longer experiment.

Artifacts are in `runs/audit/nonstationary_system/` and
`results/audit/nonstationary_system/`. This check used `eval.split=valid_seen`,
preserving the final confirmation tasks for the frozen study.
