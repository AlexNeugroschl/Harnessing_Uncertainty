# Harnessing Uncertainty

An implementation of **Entropy-Modulated Policy Gradients (EMPG)** from
[*Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents*](https://arxiv.org/pdf/2509.09265)
(Wang et al., arXiv:2509.09265). An LLM agent (Qwen2.5-1.5B-Instruct + LoRA) is trained with GRPO on **real
ALFWorld**, and EMPG reshapes GRPO's advantage using the step-level entropy of the policy being trained.

The [paper fidelity review](docs/paper_fidelity_review.md) distinguishes the implemented equations
from the smaller local training protocol and documents the opt-in corrected, faster presets.

## The algorithm

A long-horizon agent receives one sparse outcome reward per episode. GRPO spreads that single scalar evenly over
every step. The paper's Prop. 1 relates the expected squared **logit-space** score norm to **Rényi-2** entropy.
EMPG uses Shannon token entropy as an empirical credit-assignment heuristic; improvement is a hypothesis,
not a guarantee about neural-network gradients. The equation numbers below are the paper's, and each maps to one function in
[empg/advantages.py](empg/advantages.py):

| Eq. | Quantity | Function |
|---|---|---|
| 7  | `H_t` = mean token entropy of one reason→act response | `LLMPolicy.score` + `trainer.update` |
| 4  | `A_i = (r_i − mean_group r) / (std_group r + ε)` | `grpo_advantages` |
| 12 | `H_norm = (H − min_B H) / (max_B H − min_B H + ε)` | `normalize_entropy` |
| 10 | `g(H_t) = exp(−k·H_norm) / mean_B exp(−k·H_norm)` (batch mean = 1) | `scale_g` |
| 11 | `f(H_t+1) = exp(−k'·H_norm,t+1)` | `clarity_f` |
| 8  | `A_mod = A_i · g(H_t) + ζ · f(H_t+1)` | `compute_advantages` |
| 13 | `A_final = A_mod − mean_B A_mod` | `compute_advantages` |

`g` multiplies the advantage, so it keeps its sign **before the final batch centering**. It **amplifies** confident
correct steps, **penalises confident errors harder**, and **damps** uncertain steps. `f` is additive. It rewards
actions that lead to a low-entropy next state, so it still ranks steps when every rollout in a group has the same
outcome and GRPO's `A_i` is zero. A last sampled step has no `t+1` and gets `f = 0`, following the paper's
appendix pseudocode. This also applies to horizon truncations, where the missing successor is an approximation.

## Layout

```
run_benchmarks.py           one entry point: train, evaluate, estimate runtime and write reports
empg/                      the algorithm (benchmark-agnostic)
  advantages.py            GRPO + EMPG advantage estimators (pure numpy)
  curiosity.py             count-based novelty bonus + exploration metrics (pure numpy; not in the paper)
  policy.py                LLM + LoRA: batched generation, logprob / entropy scoring
  scoring.py               exact response-only output-head chunks + checkpointing
  rollout.py               multi-turn episodes, all envs stepped in lockstep
  loss.py                  token-level PPO-clip + k3 KL to the reference (adapter-off) model
  trainer.py               rollout → entropy → advantages → update
  prompts.py               verl-agent ALFWorld prompt, <think>/<action> parsing
  config.py                dataclass config, YAML `inherit:`, dotted CLI overrides
  envs/base.py             Env / Benchmark protocols (add WebShop, Deep Search here)
  envs/alfworld_env.py     real ALFWorld (TextWorld), one env per game file, + state keys for curiosity
  envs/batch.py worker.py   ordered parallel ALFWorld execution, with a serial reference path
benchmarks/
  configs/base.yaml        shared settings (the `reduced` run size)
  configs/arms/            one small file per method: grpo, empg, ablations, curiosity arms
  configs/tiers/           smoke, small, pilot, overnight, overnight_full, reduced, full
  train.py  evaluate.py    train one arm / evaluate per ALFWorld subtask
  suite.py  report.py      arms × seeds → paper-style tables, exploration table, curves
  explore_stats.py         curiosity plan Phase 0: exploration behaviour without training
  profile.py statistics.py local bottleneck measurements + paired seed/task intervals
docs/curiosity_research_plan.md   research plan for the curiosity extension
tests/                     equation invariants, curiosity, configs, prompts, CPU end-to-end update
```

Extension points:
- **New advantage estimator** (such as DAPO or GiGPO): add a branch in `compute_advantages`. For DAPO's
  clip-higher, set `train.clip_high`.
- **New benchmark**: add a module implementing `Benchmark`/`Env` in `empg/envs/` and register it in
  `make_benchmark`.
- **Bigger model**: set `policy.model_name`.

## Setup (WSL, NVIDIA GPU)

ALFWorld's TextWorld backend is Linux-only, so everything runs in WSL:

```bash
wsl
cd "/mnt/c/.../Harnessing_Uncertainty"
bash scripts/setup_wsl.sh              # venv in ~/venvs/empg, CUDA 12.8 torch, deps, alfworld-download
source ~/venvs/empg/bin/activate
```

## Running

For training, evaluation **and comparison tables in one command**, run the root file:

```bash
python run_benchmarks.py --estimate-only   # preview the remaining workload without loading a model
python run_benchmarks.py                   # expanded overnight: same four methods, seed 0 + base
```

The default is now **`overnight_full`**, targeting approximately **eight hours** on the
profiled RTX 5060 Laptop 8GB. It expands the same four-method development comparison:

| Setting | Expanded overnight (default) | Original overnight | Full confirmation budget |
|---|---|---|---|
| Training seeds | 0 | 0 | 0, 1, 2 |
| Training batch | 4 tasks × 4 siblings (16 trajectories) | 1 task × 4 siblings | 2 tasks × 8 siblings |
| Training iterations | 5 for every method | Up to 12 | 40 |
| Training budget per method | 20 game opportunities, up to 4,000 response steps, 40 optimizer steps | 40 minutes | Iteration budget |
| Train / eval horizon | 50 / 50 turns | 50 / 50 turns | 50 / 50 turns |
| Response cap | 256 tokens | 256 tokens | 256 tokens |
| Evaluation panel | Same stratified 48 `valid_seen` tasks across methods | 24 `valid_seen` tasks | All 134 `valid_unseen` tasks |
| Suite deadline | 8.5 hours, including a completion margin | 8 hours | Unlimited by default |

The larger batch fills more generation slots and samples more games. Eight optimizer
minibatches per iteration retain approximately 100 response steps per optimizer step at
the full horizon. Completed methods receive the same task schedule and number of updates;
actual episode lengths and token costs can differ. The original per-method 40-minute cap
does not apply to the expanded preset. One seed remains development evidence, even with
more training and twice the evaluation panel. It cannot establish a reliable winner.

New artifacts go to **`runs/overnight_full/`** and **`results/overnight_full/`**. Existing
`runs/overnight/` and `results/overnight/` remain available with `--tier overnight`.
The central runtime estimate is approximately 8.2 hours. The default 8.5-hour ceiling
gives the final evaluation extra time; use `--hours 8` for a strict eight-hour cutoff.
See [the expanded plan and timing calibration](docs/overnight_full.md).

`--hours` bounds the entire training/evaluation suite, including phase startup, not just
training updates. At the deadline it terminates the active Linux phase and its environment
workers, preserves completed evaluations/logs, generates partial tables, and exits with code
124. An unfinished evaluation is shown as missing, never as zero success. Cleanup and reports
take additional time. A training time cap is softer: the current rollout/update finishes before
saving the adapter. None of these runtime estimates is a measured timing interval.

```bash
python run_benchmarks.py                       # expanded comparison, 8.5-hour ceiling
python run_benchmarks.py --hours 8              # strict eight-hour cutoff
python run_benchmarks.py --report-only          # regenerate expanded overnight tables
python run_benchmarks.py --tier overnight --report-only  # original run's tables
python run_benchmarks.py --tier full            # opt into the much longer, three-seed benchmark
```

For a shorter night, select one controlled pair, e.g.
`--arms grpo grpo_cur_group --hours 6`. Changing `--hours` alone changes the suite stop time;
it does not change the preset's fixed workload. Uncommenting more methods or adding
seeds increases work and can leave later methods incomplete at the deadline. If a previous
full run is still running, stop its terminal command with Ctrl+C before starting another GPU
job; editing the preset does not change an already-running process.

The `ENABLED_METHODS` list near the top of `run_benchmarks.py` lists all ten methods on
separate lines. Comment out a line to disable its run; uncomment it to enable it.
By default **GRPO**, **EMPG**, **GRPO + group curiosity**, and **EMPG + episodic curiosity**
are enabled. `--arms` overrides the list for one invocation.

The two curiosity priorities test different ideas: group novelty rewards reaching states
that few sibling rollouts visit, while EMPG plus episodic novelty tests whether exploration
complements the clarity bonus. These are hypotheses, not demonstrated winners. Compare each
against its matching baseline; this four-method selection cannot isolate the effect of
episodic versus group scope or establish a curiosity/EMPG interaction. Enable
`grpo_cur_episodic` next for those comparisons, and `grpo_memory` to check whether a gain
comes from compensating for the short prompt history. The oracle, sign-flip and g/f-only
variants are useful diagnostics after a candidate shows a gain in task success.

This also works from Windows Python: the file forwards to `Ubuntu-24.04` in WSL and
`~/venvs/empg/bin/python`. For another setup, set `EMPG_WSL_DISTRO` and/or
`EMPG_WSL_PYTHON` (an absolute Linux Python path). Inside WSL, use the activated environment above.

The `full` tier retains the laptop-sized LoRA setup, with 40 training iterations,
2 tasks × 8 sibling episodes, 50-turn training/evaluation horizons, and all 134
`valid_unseen` evaluation tasks. It is a full **local** comparison, not a replication of the
paper's larger training budget. On this RTX 5060 Laptop 8GB, four methods × three seeds
at `--tier full` require multiple days. Use `python run_benchmarks.py --tier full --estimate-only`
for the current calibrated workload estimate; enabling all ten methods increases work further.
Short profile measurements do not capture long-context costs. These estimates are not
measured timing intervals or a promise that 40 iterations establish an effect.

For a focused development comparison before that commitment:

```bash
python run_benchmarks.py --tier pilot --arms grpo grpo_cur_episodic --seeds 0 --estimate-only
python run_benchmarks.py --tier pilot --arms grpo grpo_cur_episodic --seeds 0
python run_benchmarks.py --tier smoke --arms grpo empg --seeds 0   # plumbing check only
```

The two-method pilot is estimated at **about 1.3–3.8 hours**, using `valid_seen` for development.
It can reveal exploration behavior and obvious failures; a null result cannot rule out a
long-horizon learning effect. For confirmation, fix the candidate before evaluating on
`valid_unseen`, for example `--tier full --arms grpo grpo_cur_episodic --seeds 0 1 2`.
Comparisons across all ten methods are exploratory; their intervals are not adjusted for
selecting the best method.

Results go to `runs/<tier>/` and reports to `results/<tier>/`. The runner generates:

- `comparison.md`: all methods, per-subtask success, seed variability, paired differences and individual runs.
- `runs.md` / `runs.csv`: one row per seed, including success, task count, training steps and measured training time.
- Exploration and response-length tables, paired comparison JSON, and training curves by iteration, steps, tokens and time.
- `curiosity_comparisons.md` / `.json`: each curiosity variant against its matching GRPO or EMPG baseline.
- `benchmark_run.json`: resolved configurations, estimated remaining work, status and observed suite elapsed time.

Re-run the same command to skip completed training/evaluation. An interrupted training run
restarts training; it does not resume optimizer state. `runs.csv` records whether each method
stopped at its iteration/time budget or remained incomplete. The runner validates existing
configurations before starting new work and attempts partial reports when a later run fails.
Use `--report-only` to regenerate tables, or `--runs` and `--out` to keep an experiment separate.
Paths are relative to this repository, or absolute Linux paths when running through WSL.
Reports include only the requested methods and seeds. Shared dotted overrides go after `--`,
for example `-- train.iterations=20`; use a fresh `--runs` directory when changing a protocol.

`--config` takes an arm name (a file in `configs/arms/`) or a YAML path. `--tier` sets the run size, and each
tier writes to its own `runs/<tier>/` folder.

```bash
pytest tests/                                                    # math + curiosity + CPU end-to-end
python -m benchmarks.train    --config empg --tier smoke         # minutes, end to end on ALFWorld
python -m benchmarks.evaluate --config empg --tier smoke
python -m benchmarks.suite    --tier small --arms grpo empg      # base + arms, train then eval
python -m benchmarks.suite    --tier reduced --phase paper --seeds 0 1 2
python -m benchmarks.report   --runs runs/small --split valid_seen --out results/small
```

Any config value can be overridden from the command line, for example `train.iterations=50`,
`advantage.zeta=0.1` or `advantage.curiosity.weight=0.1`. When an override defines a new variant, also give
it a new `name=`, so its runs don't overwrite another arm's.

## Benchmark suite

The suite implements an ALFWorld comparison with these shared features:
- **Metric:** success rate on the 6 subtasks (Pick, Look, Clean, Heat, Cool, Pick2) and overall, measured on
  `valid_unseen` (out of distribution).
- **Arms:** zero-shot base model, GRPO, GRPO + EMPG, and the two ablations (gradient scaling `g` only, clarity
  bonus `f` only).
- **Controlled comparison:** all arms share `base.yaml`, the same seeds and the same task sampling. They differ
  only in the `advantage:` block.

Core settings: `k = k' = 1`, `ζ = 0.05`, KL coefficient 0.01, group-normalised outcome
reward (10 × success), and a 0.1 invalid-format penalty. Inadmissible actions with valid tags are measured as
environment no-ops rather than invalid-format penalties.

### Legacy tier differences from the paper (hardware: one 8GB laptop GPU)

This table describes the older local tiers. The `paper_lora` presets correct several protocol
differences while retaining LoRA and a smaller game batch; see the fidelity review above.

| | Paper | Here |
|---|---|---|
| Model | Qwen2.5-1.5B/7B-Instruct, full fine-tuning | Qwen2.5-1.5B-Instruct (same as the paper's ALFWorld row), LoRA r=16 |
| LR | 1e-6 | 1e-5 (LoRA) |
| Batch | 16 tasks × group 8 | 2 tasks × group 8 |
| Steps | 150, 50-turn episodes | 40 iterations, 20-turn training episodes (50 at eval), `reduced` tier |
| Eval set | all 134 `valid_unseen` games | fixed random 48-game subset (`eval.max_tasks=0` for all) |
| Seeds | 3 (Table 1) | 1 (GRPO vs EMPG); ablations and more seeds pending |
| Entropy regularization | 0.001 (Table 3) | 0 (step entropy shapes advantages only) |
| Framework | verl-agent (vLLM rollouts) | HF `generate` (~15 min per iteration on an RTX 5060 8GB) |

## Curiosity extension (research in progress)

The research question is whether a curiosity bonus, for reaching states the agent hasn't seen, helps on
long-horizon, sparse-reward tasks. [docs/curiosity_research_plan.md](docs/curiosity_research_plan.md) has the
hypotheses, the phases and the metrics. Curiosity is an extra term in the same advantage:

```
A_mod(i,t) = A_i · g(H_t) + ζ · f(H_t+1) + η · c_t        c_t = novelty of s_t+1 (1/√n or first-visit)
```

With `η = 0` the result is bit-identical to GRPO/EMPG. That is tested.

- **State keys** ([empg/envs/alfworld_env.py](empg/envs/alfworld_env.py)):
  - `parsed` (non-privileged, the default): location, inventory, object placements, opened receptacles and
    conditions, all tracked from actions and their feedback.
  - `physical` / `knowledge` (oracle): TextWorld facts with history markers removed (`checked`, `not_*`).
  - `obs`: the observation text.

  On random walks, the parsed key misses a physical change in 1 of 415 states. It is finer than the oracle
  where ALFWorld puts several receptacles at one location.
- **Knobs** (`advantage.curiosity.*` in [base.yaml](benchmarks/configs/base.yaml)):
  - `scope`: `episodic` or `group`, i.e. novel compared with sibling rollouts
  - `bonus`
  - `mode`: step term or trajectory reward
  - `schedule`: constant, linear or exp
  - `gate`: `no_success_signal` applies curiosity only in groups whose rollouts all share one outcome
  - `modulate`: by the policy's confidence
  - `cap`
- **Arms** (`configs/arms/`): `grpo_memory` (the visited-locations prompt control), `grpo_cur_episodic`,
  `grpo_cur_group`, `grpo_cur_oracle`, `empg_cur_episodic`, and `empg_seek_uncertainty` (ζ < 0).
- **Metrics:** every run logs:
  - revisit and no-op rate
  - unique states per episode
  - group coverage
  - P(group has ≥1 success)
  - no-signal groups
  - steps to success
  - intrinsic share of |A|

  `report` writes them to `exploration.md`.

```bash
python -m benchmarks.explore_stats --config grpo --tier small --games 24   # Phase 0 go/no-go (~1 h)
python -m benchmarks.suite --tier small --phase screen                     # Phase 2 screening (8 arms)
python -m benchmarks.suite --tier small --arms grpo_cur_group advantage.curiosity.gate=no_success_signal     name=grpo_cur_group_gated                                              # Phase 3 variant
```

## Speed

The [October 21 fidelity and performance review](docs/paper_fidelity_review.md) found material
protocol differences and measured a substantial decoding improvement. Existing tiers preserve
their original behavior; `paper_lora` opts into corrected framework reward placement, token entropy
regularization, response cap 512, publication-era action parsing, and sampled evaluation at
temperature 0.4. `paper_lora_fast` adds compiled vLLM and a 32-sequence engine limit. Both remain
local LoRA experiments with two games per batch, rather than the original distributed full
fine-tuning experiment. They have no default time limit.

On identical prompts, compiled vLLM decoded about 2.4-4.5x faster than HF across fixed-length probes.
CUDA bf16 autocast reduced a separate cached-trajectory update from about 33 to 28 seconds.
These are component measurements, not a measured full-suite speedup. The runner's current
runtime estimate remains HF-based. See the review for measurements, memory limits, and pilot
commands; use the separate `empg-vllm` environment for the fast preset.

Scoring, entropy (`H_t`) and the update always run on the HF model being trained:

- **Merged LoRA** (`policy.merge_lora: true`, the default for `backend: hf`): LoRA is merged into the
  weights while generating, so decoding skips the adapter's extra matmuls. The frozen base weights are
  restored **bit-exactly** afterwards from a CPU snapshot. That matters because they are also the KL reference.
  `tests/test_trainer_smoke.py` checks the restore and that merged greedy output matches unmerged output.
- **vLLM rollouts** (`policy.backend=vllm`): the vLLM engine keeps batching continuously, and it caches the
  shared prompt prefix. During a rollout the HF model sits in CPU RAM. vLLM samples with the current
  adapter, then sleeps so the update gets the whole GPU. vLLM pins its own torch build, so it needs its own
  venv:
  ```bash
  EMPG_VLLM=1 EMPG_VENV=~/venvs/empg-vllm bash scripts/setup_wsl.sh
  source ~/venvs/empg-vllm/bin/activate
  python -m benchmarks.suite --tier small --arms grpo empg policy.backend=vllm
  ```
  vLLM samples from the same policy, but its numerics differ slightly from HF's. Use the **same backend for
  every arm** in a comparison. Suite overrides apply to every run, including the base eval.
  - **Eager mode for legacy tiers** (`vllm_enforce_eager: true`). The fast preset enables CUDA
    graphs and bounds capture with `vllm_max_num_seqs: 32`. A short graph-backed rollout followed
    by an HF update fits on 8 GiB; long-context memory use still needs a representative pilot.
  - **Environment fixes.** The engine disables the FlashInfer sampler, whose JIT build needs a full CUDA toolkit.
    PyTorch's sampler draws from the same distribution. The engine also puts the pip-installed `nvcc` on PATH
    for `torch.compile`.
  - **One-time startup cost.** About 1–2.5 min per process.
- **Sampling = the policy being scored.** Qwen's `generation_config` sets `repetition_penalty: 1.1`, and HF
  `generate()` used to apply it. Rollouts and greedy eval therefore sampled from a penalised distribution, not
  from π_θ. The measured gap, E[log p] + E[H] over sampled tokens, was −0.65 nats; after the fix it is
  0.00 ± 0.02 for both backends. HF generation now passes `repetition_penalty=1.0`, like verl-agent and
  vLLM. Runs made before this fix are not comparable with runs made after it.
- **No silent VRAM spill.** On Windows/WSL the NVIDIA driver's *CUDA Sysmem Fallback Policy* lets
  allocations beyond the 8GB overflow into shared system RAM. Nothing fails; the run just gets several times
  slower. Turn it off: **NVIDIA Control Panel → Manage 3D settings → CUDA - Sysmem Fallback Policy → Prefer No
  Sysmem Fallback**. After that, running out of memory raises a clear OOM error. If that happens, lower
  `policy.gen_batch_size` / `policy.micro_batch_size`. Training logs `gpu_peak_gb` every iteration and warns when
  the peak reaches physical VRAM.

Historical probe on the RTX 5060 8GB (WSL), 2 games × group 8, with Sysmem Fallback still **on**:

| | generate, 16 prompts | rollout, 5 turns | rollout, 20 turns (it 1 / it 2) | update, 20 turns (it 1 / it 2) |
|---|---|---|---|---|
| HF, unmerged LoRA (before) | – | 124–146 s | – | – |
| HF, merged LoRA | ~10 s | 107–126 s | 366 / 230 s | 255 / 199 s |
| vLLM (eager) | ~3.7 s | 70–72 s | 219 / 183 s | 553 / 195 s |

This older serial-environment probe showed faster vLLM decoding and substantial VRAM spill.
It does not describe the current four-worker implementation. In the completed expanded overnight
run, environment reset/step time totaled about 4.5 minutes across all four methods, compared with
156 minutes generating training responses and 146 minutes updating. Generation and model work
are the main targets for further optimization. See the October 21 review for the newer measurements.

**Response lengths (for choosing a `max_new_tokens` cap later).** Legacy tiers use 256 tokens
(96 in `smoke`); the paper presets use 512. Each
run records every response's length, whether it reached the cap before EOS, and whether it parsed as valid:
- `resp_lengths_train.jsonl`: one line per iteration.
- `resp_lengths_eval_<split>.json`: from eval.
- `metrics.jsonl`: `resp_len_mean/p99/max` and `trunc_rate`.
- the eval JSON: a `resp_len` summary.

`benchmarks.report` pools all of this into `response_lengths.md`, which gives p50/p90/p99/p99.9, max,
`% truncated`, and how often truncation made the step invalid. A lower cap costs nothing only when it sits
above what the model actually produces. Set it from p99.9 in that table, after checking it on a few trained arms, since
lengths can drift during training.

## Results

_Pending: produced by `benchmarks.suite` + `benchmarks.report`._

## Research audit and faster experiments

[docs/research_audit.md](docs/research_audit.md) records the theoretical limits, comparison bugs repaired,
measured performance, and a staged experiment that can produce useful evidence within a smaller budget.
The research question is testable; a short run or a negative pilot cannot establish that a technique works
or fails on long-horizon tasks.

The default ALFWorld configuration now uses four independent CPU workers (`env.workers=0` restores serial
execution). Scoring projects only actual response tokens through the output head in bounded chunks and
recomputes those head activations in backward. It retains exact full-vocabulary entropy and the token-mean PPO
objective. `policy.chunked_head=false` restores the dense scoring path. Scoring and update microbatches are
sorted by sequence length within the same optimizer minibatches to reduce padding. Floating-point rounding
can differ; use the same implementation and backend for all arms.

Training task selection uses a separate RNG from update shuffling, so episode length cannot change subsequent
task assignments. Metrics include cumulative environment steps, response tokens and training seconds.
Evaluation stores per-task outcomes. Reports add paired seed/task bootstrap intervals and curves against those
compute measures. Suite resumption checks configuration fingerprints and rejects old or incompatible results.
Use a fresh output directory for the new implementation.

```bash
# Measure the local bottlenecks and verify serial/parallel environment equivalence.
python -m benchmarks.profile --config empg --tier small --env-only
python -m benchmarks.profile --config empg --tier small --turns 3 --samples 8

# Inspect exploration, successful groups and truncation before training.
python -m benchmarks.explore_stats --config grpo --tier pilot --games 12 --group 4 --games-per-batch 2

# Development screen: 8 iterations, group 4, 30-turn training, 50-turn evaluation.
# valid_seen is used for selection; keep valid_unseen for a fixed confirmation run.
python -m benchmarks.suite --tier pilot --arms grpo grpo_cur_episodic --seeds 0 --skip-base
python -m benchmarks.report --runs runs/pilot --split valid_seen --out results/pilot

# Optional budget: stop after a complete batch, save, then evaluate.
# The elapsed budget covers rollout+update, excludes startup/evaluation, and can overshoot by one batch.
python -m benchmarks.suite --tier pilot --arms grpo grpo_cur_episodic --seeds 0 --skip-base train.max_seconds=3600 out_dir=runs/pilot_1h
```

`pilot` changes group size/horizon and therefore tests a smaller experiment, not an exact paper replication.
Use the same tier and budget for every compared arm. `train.max_env_steps` can also bound training; both
budgets stop at batch boundaries and record the actual completed work in `training_complete.json`.
