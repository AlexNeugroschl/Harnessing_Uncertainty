# Curiosity-driven adaptation in non-stationary environments

The question is whether temporarily increasing intrinsic motivation after an
environment changes improves recovery on tasks with long action dependencies.
GRPO is the primary policy-gradient backbone; EMPG remains an optional comparison.
The intervention changes transition rules or rewards while keeping the task goal
and task sampling schedule fixed across treatments.

Suite version 3 includes a fixed initial skill tutorial shared by all treatments;
it explains approach, harvest and plank crafting without revealing later changes.
Its default response is a single `action: NAME` line. The native parser also
accepts one action tag or a bare action name and rejects multiple actions.
Reasoning tags are not part of the research objective; the older strict parser
remains an explicit ablation. ALFWorld parsing is unchanged.
The implementation uses the real [NovelGridWorldsV2 engine](https://github.com/tufts-ai-robotics-group/NovelGridWorldsV2),
revision `f5d06d4f5166ca985ca420c809129a79700d904e`, through its native configuration
parser, crafting, resource breaking, item selection, approach, and world updates.
This is an independently specified text crafting suite, **not a reproduction of
published NovelGym scores**. The single-player adapter omits rendering, socket
agents, symbolic planners and multi-agent turn bookkeeping. Native action costs
are set to zero; the task supplies one sparse rewarded-success outcome.

## Tasks and changes

Each task requires crafting a pogo stick through a chain of component recipes.
Each component consumes the previous component and a plank; each plank consumes
one harvested log. Dependency depth is adjustable from 2 to 32. Depths 4, 8 and 16
have constructive baseline solutions of 16, 32 and 64 native macro actions.
These are verified solution lengths, not claims of shortest possible paths.
`approach_*` is one native macro action, and the same action accounting applies
to every condition. The runner gives each depth error-recovery headroom instead
of making the turn limit itself the definition of task difficulty.

All task seeds and interfaces are shared across treatments. Train, development
probes and final evaluation have disjoint seed ranges. They test new layouts with
the same recipe rules, not generalization to unseen recipe templates.
The final confirmation range starts at seed 3,000,000; it excludes the 2,000,000
range used for early interface engineering checks.

| Change | Baseline | Changed regime |
|---|---|---|
| `tool` | Resources can be harvested directly | The existing axe must be selected before harvesting |
| `recipe` | Final craft uses the preceding component and a plank | Final craft also consumes a brace, requiring another resource/craft branch |
| `resource` | Planks consume logs | Planks consume birch logs |
| Reward-only, `tool` | Completion pays | Completion pays only when every successful harvest had the axe selected |

Both resources, the axe, all craft commands, and the initial recipe guide exist in
both regimes. The initial guide stays fixed after changes. Generic feedback says
whether an operation succeeded, without disclosing the changed rule. Explicit
failure explanations are available as a separate diagnostic configuration.
Reward-only shifts preserve native physical transitions and terminal completion;
`base_won` separates completion from rewarded success.

Curiosity hashes only the displayed inventory, selection, position, facing,
resource counts, local view and observable route eligibility. It never hashes
the regime ID, hidden recipe, hidden tool requirement, counters or RNG state.
The default first-visit bonus gives no intrinsic reward for an unchanged state.

## Treatments

The default base coefficient is eta=0.05, with a 3x boost. These are experiment
settings, not established optimal values. Use `--boost-factors` for a dose sweep.

| Treatment | Coefficient |
|---|---|
| `none` | Zero |
| `constant` | eta throughout |
| `high` | eta times the boost factor throughout |
| `matched` | Constant coefficient equal to the oracle arm's planned average coefficient |
| `oracle` | Boosted for K updates beginning at each actual change point |
| `detected` | Boosted for K updates after an observed increase in action failures |
| `clock` (optional) | Boosted at the scheduled boundaries even in stationary environments |

Oracle timing is privileged experimental information, isolated to the training
coefficient. The observed detector never reads change points or regime labels.
It uses the completed batch's failure rate and the previous exponential moving
average. After a warm-up, a rise of at least `failure_threshold` triggers the
*next* update's boost. It does not retrospectively reweight the triggering batch,
or retrigger during an active boost window. A high constant failure rate does
not constitute detected drift. A stationary condition measures false triggers.

The default `mode=step` adds the weighted novelty term to the policy advantage,
then centers it; extrinsic success and environment returns remain unshaped in
all reports. This is the repository's existing curiosity formulation. Increasing
the coefficient scales that contribution. With `mode=trajectory`, intrinsic
reward instead enters the group-normalized episode outcome; normalization can
cancel uniform scaling when there is no extrinsic signal, so treat that mode as
a separate algorithmic ablation.

Matched coefficients do not imply equal realized intrinsic reward: trajectory
lengths and visited states change with the learned policy. Both the raw bonus
and its actual contribution after advantage centering/normalization are logged.

In a stationary-final environment, oracle and matched arms are often exactly
equivalent to constant curiosity. The runner records an explicit `alias_of`
and trains that policy once. Reused results never add independent seeds.

## Running

The recommended bounded study is now `python run_novelgym_12h.py`. Its
[12-hour workflow](nonstationary_12h.md) includes a competent shared initialization,
native GPU calibration, fixed matched update counts, and time-driven additional
seed replications. Use `--estimate-only` to inspect the prepared plan and
`--report-only` to regenerate its reports. The commands below expose the individual
components for experiments with a different scope.

Install the optional dependency inside the existing WSL GPU environment:

```bash
source ~/venvs/empg-vllm/bin/activate
python -m pip install -r requirements-novelgym.txt
```

From Windows or WSL, preview and run the default exploratory screen:

```powershell
python run_novelgym.py --estimate-only
python run_novelgym.py --hours 8
python run_novelgym.py --report-only
```

The small base model did not establish competence in the initial short probes.
For a study of *recovery from previously successful behavior*, prepare a shared
skill checkpoint first and validate it on development maps:

```powershell
python run_novelgym_warmup.py
python run_novelgym.py --initial-adapter runs/novelgym_warmup/adapter --estimate-only
python run_novelgym.py --initial-adapter runs/novelgym_warmup/adapter --hours 8
```

Warmup uses supervised native expert trajectories from **training maps and the
baseline rules only**, including recovery after failed operations that leave the
native world unchanged. It validates every two epochs, saves the best checkpoint,
and can require at least 75% success on held-out baseline layouts at every depth.
The checked local checkpoint solved 12/12 development tasks at depths 4 and 8.
No changed-regime solution is demonstrated. Every arm
then receives the same adapter, fresh optimizer, and paired RL task schedule.
Warmup compute is recorded separately from adaptation compute. Adapter contents
are hashed into each experiment configuration and checked again before training;
changing a checkpoint cannot silently resume a prior experiment. The KL reference
remains the original adapter-off base model, as in the existing training code.
Warmup does not itself establish competence: the held-out pre-change probes
remain the gate for interpreting recovery.

The default is one depth, one change family, two environment schedules, six
treatments, and one seed. Two stationary treatments reuse the identical constant
run, leaving ten distinct runs. `--hours` shares the remaining deadline across
remaining jobs and reserves reporting time. Fixed update counts remain matched;
unfinished evaluations stay unavailable. An interrupted training job restarts
training; it does not restore optimizer state. Completed jobs are reused only
when configuration fingerprints match.

Start smaller if eight hours cannot establish pre-change competence:

```powershell
python run_novelgym.py --treatments constant oracle detected --schedules switch --hours 8
```

Sweep curiosity dose while holding horizon and training budget fixed:

```powershell
python run_novelgym.py --treatments constant high matched oracle detected --boost-factors 2 4 --runs runs/novelgym_dose --out results/novelgym_dose --estimate-only
```

After selecting a configuration on development probes, freeze it before the
larger paired-seed confirmation. This command is a substantial experiment, not
an overnight workload estimate:

```powershell
python run_novelgym.py --tier novelgym --depths 4 8 16 --novelties tool recipe resource --seeds 0 1 2 --treatments constant oracle detected --boost-updates 6 --estimate-only
```

Recurring baseline → changed → baseline transitions test recovery and retention:

```powershell
python run_novelgym.py --tier novelgym --schedules recurring stationary --switches 15 29 --treatments constant oracle detected clock --runs runs/novelgym_recurring --out results/novelgym_recurring --estimate-only
```

Reward-only changes use `--novelties tool` and the override
`nonstationary.change=rewards` after `--`. General dotted overrides also follow
`--`; use a fresh `--runs` and `--out` when changing the protocol. Matrix options
own treatment names, coefficients, output locations, depths and change schedules.

## GPU execution and measurements

The preset targets the local RTX 5060 8 GiB. It collects 16 concurrent training
trajectories (two tasks × eight siblings), uses compiled vLLM with prefix caching,
and sleeps the inference engine while the HF model performs LoRA updates.
Scoring uses the exact response-only analytic vocabulary head, bf16 autocast,
length-sorted token-bounded microbatches, and an HF allocator ceiling of 75%.
The engine context is bounded to the prompt plus response limits. Evaluation
and probes reuse the loaded policy and inference engine.

The native preset also shares score computation for identical tokenized
prompt/response pairs. Within each original optimizer minibatch it collapses only
rows with identical conditional responses and identical advantages, weighting all
loss terms by their original multiplicity. Opposite-sign advantages remain
separate because PPO clipping is nonlinear. Episode rewards, entropy normalization,
curiosity counts, optimizer steps, and reported sample counts retain all rows.
CPU loss/gradient tests and real bf16 GPU checks cover this optimization; floating
point rounding can differ with batch shape. Legacy ALFWorld presets leave it off.

Native calibration compares checkpointing policies on a shared rollout, checks
maximum context and the selective threshold boundary, and records the chosen
settings for every treatment. Locally, full checkpointing remains selected after
the faster selective candidate failed its boundary memory check. See
[validation measurements](nonstationary_validation.md).

Compact state descriptions, short feedback/action history, and an explicit
response-format example reduce unnecessary decoding. Prompt limits raise errors
instead of silently dropping context. The response limit was raised from 128 to
256 after a real 80-step probe found 7.5% truncation. The improved prompt then
produced zero invalid-format or truncated responses in the shared 80-step tuning
rollout; this small result does not guarantee format reliability on longer tasks.

Environment reset and step time in the initial 16-trajectory, five-turn probe
were approximately 0.112 s and 0.004 s. Subprocess IPC would add overhead to this
native workload, so it uses serial stepping alongside batched GPU generation.

Run profiling inside WSL:

```bash
python -m benchmarks.profile --config grpo --tier novelgym_screen --model-only --samples 16 --turns 5 --out runs/audit/novelgym_profile.json
python -m benchmarks.tune_updates --config grpo --tier novelgym_screen --samples 16 --turns 5 --candidates reference score4 bounded_no_checkpoint selective900 --out runs/audit/novelgym_tuning.json --rollout-cache runs/audit/novelgym_rollout.json
```

Without native calibration, the matrix preview gives workload counts. Pass
`--calibration` with a matching profile for measured forecasts and cost-weighted
deadline allocation; the 12-hour workflow does this automatically. The preview
does not apply historical ALFWorld timing to this new environment. Profile timing and successful process completion establish
engineering feasibility, not learning or adaptation effectiveness. Check response
lengths, actual VRAM and success before expanding the matrix.

## Evaluation and interpretation

Every enabled probe panel is fixed across methods. Probes run at initialization,
regular intervals, immediately before each change, after the first update under
the changed rules, and at the end. Both regimes are evaluated at each checkpoint.
The changed-regime probe immediately before the switch measures the immediate
shift effect without a post-change policy update. Probes use isolated evaluation
RNG streams and never update the model or detector.

Primary comparisons are oracle/detected versus constant curiosity. Oracle versus
high and matched controls checks whether timing improves on more exploration
throughout. Stationary controls and optional clock boosts distinguish adaptation
from generic benefits of a temporary exploration pulse.
The report also estimates the switch-minus-stationary interaction using complete
four-cell seed/task pairs, without counting reused controls as additional data.

Interpret recovery only after measured pre-change competence exceeds both the
success-rate threshold and the minimum number of successful held-out tasks.
Otherwise the report marks adaptation inconclusive. Recovery is the first
observed probe crossing 90% of pre-change old-regime success, not a convergence
guarantee. Recurring schedules report each change separately.

Artifacts include `comparison.md`, `summary.json`, `learning_curves.json`, and
per-condition PNG curves of training success, held-out success, effective curiosity
weight and observed failures. Each run records raw extrinsic returns, intrinsic
contribution, detector events, steps, tokens, elapsed training time, truncation,
and peak HF reserved VRAM. The latter excludes the separate vLLM process.
Paired task/seed bootstrap intervals use identical evaluation protocols and task
panels. With one seed, intervals describe task variability only. Constant observed
differences are flagged as degenerate, not evidence of equivalence. Missing and
timed-out runs are never assigned zero success.

## Complementary rule-discovery evaluation

`run_adaexplore.py` uses the official [AdaExploreBench](https://github.com/HKUST-KnowComp/AdaExploreBench)
runtime, history masking, rule transitions, action parser and metric functions.
Its checkout is pinned and verified at revision
`7ac28a364ed19b0461f5a987f8dd81fb57bd0567`. Local base/LoRA policies use the
official conversation projection, greedy text generation, and its JSON parser.
There are no weight updates or intrinsic bonuses during this evaluation. It tests
whether learned exploration transfers to in-context rule adaptation; it does not
replace the sequential sparse-reward crafting experiment.

Inside WSL, prepare the optional runtime and released test set:

```bash
python -m pip install -r requirements-adaexplore.txt
git clone https://github.com/HKUST-KnowComp/AdaExploreBench.git .tmp/AdaExploreBench
git -C .tmp/AdaExploreBench checkout 7ac28a364ed19b0461f5a987f8dd81fb57bd0567
hf download Roryaccout/AdaExploreBench data/test.jsonl --repo-type dataset --local-dir .tmp/adaexplore_data
```

Run the official model-free random baseline to check integration without a GPU:

```powershell
python run_adaexplore.py --source .tmp/AdaExploreBench --dataset .tmp/adaexplore_data/data/test.jsonl --runtime-check --max-episodes 2 --out results/adaexplore_runtime
```

Omit `--runtime-check` for the base model, or pass `--adapter` with a trained LoRA
directory. Multiple independent episodes share each generation batch; a single
vLLM session spans evaluation. The default is four episodes at a time, a 16,384
token context and a 2,048 token response limit. Excess context fails explicitly
instead of truncating benchmark history. Choose a fresh `--out` for each policy.
The manifest records the dataset hash, upstream revision, adapter file hashes
and model configuration. Static and dynamic conditions use the same episodes.
Official JSON trajectories and metrics are retained; the model-free check does
not establish local model performance or cross-benchmark transfer.

The existing ALFWorld inspection experiment remains a secondary continuity check.
