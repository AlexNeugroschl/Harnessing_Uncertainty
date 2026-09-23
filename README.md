# Harnessing Uncertainty

An implementation of **Entropy-Modulated Policy Gradients (EMPG)** from
[*Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents*](https://arxiv.org/pdf/2509.09265)
(Wang et al., arXiv:2509.09265). An LLM agent (Qwen2.5-1.5B-Instruct + LoRA) is trained with GRPO on **real
ALFWorld**, and EMPG reshapes GRPO's advantage using the step-level entropy of the policy being trained.

## The algorithm

A long-horizon agent receives one sparse outcome reward per episode. GRPO spreads that single scalar evenly over
every step. The paper also shows (Prop. 1) that the size of a softmax policy gradient is coupled to entropy:
confident steps get small updates and uncertain steps get large, noisy ones. EMPG fixes both problems with the
agent's own uncertainty. The equation numbers below are the paper's, and each maps to one function in
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

`g` multiplies the advantage, so it changes the size of the update but keeps its sign. It **amplifies** confident
correct steps, **penalises confident errors harder**, and **damps** uncertain steps. `f` is additive. It rewards
actions that lead to a low-entropy next state, so it still ranks steps when every rollout in a group has the same
outcome and GRPO's `A_i` is zero. The paper leaves one case undefined: a terminal step has no `t+1`. Here it
gets `f = 0`.

## Layout

```
empg/                      the algorithm (benchmark-agnostic)
  advantages.py            GRPO + EMPG advantage estimators (pure numpy)
  policy.py                LLM + LoRA: batched generation, logprob / entropy scoring
  rollout.py               multi-turn episodes, all envs stepped in lockstep
  loss.py                  token-level PPO-clip + k3 KL to the reference (adapter-off) model
  trainer.py               rollout → entropy → advantages → update
  prompts.py               verl-agent ALFWorld prompt, <think>/<action> parsing
  config.py                dataclass config, YAML `inherit:`, dotted CLI overrides
  envs/base.py             Env / Benchmark protocols (add WebShop, Deep Search here)
  envs/alfworld_env.py     real ALFWorld (TextWorld), one env per game file
benchmarks/
  configs/                 base.yaml + one small file per arm (arms differ only in `advantage:`)
  train.py  evaluate.py    train one arm / evaluate per ALFWorld subtask
  suite.py  report.py      all arms × seeds → paper-style tables + curves in results/
tests/                     equation invariants, prompt parsing, CPU end-to-end update
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

```bash
pytest tests/                                                         # math + CPU end-to-end
python -m benchmarks.train --config benchmarks/configs/smoke.yaml     # few-minute end-to-end on ALFWorld
python -m benchmarks.train --config benchmarks/configs/alfworld_empg.yaml --seed 0
python -m benchmarks.evaluate --config benchmarks/configs/alfworld_empg.yaml --seed 0
python -m benchmarks.suite --seeds 0 1 2                              # base + GRPO + EMPG + ablations
python -m benchmarks.report                                           # → results/
```

Any config value can be overridden from the command line, for example `train.iterations=50`,
`advantage.zeta=0.1` or `eval.max_tasks=40`.

## Benchmark suite

The suite follows the paper's ALFWorld protocol:
- **Metric:** success rate on the 6 subtasks (Pick, Look, Clean, Heat, Cool, Pick2) and overall, measured on
  `valid_unseen` (out of distribution).
- **Arms:** zero-shot base model, GRPO, GRPO + EMPG, and the two ablations (gradient scaling `g` only, clarity
  bonus `f` only).
- **Controlled comparison:** all arms share `base.yaml`, the same seeds and the same task sampling. They differ
  only in the `advantage:` block.

Paper hyperparameters used unchanged: `k = k' = 1`, `ζ = 0.05`, KL coefficient 0.01, group-normalised outcome
reward (10 × success), and a 0.1 invalid-action penalty, as in verl-agent.

### Differences from the paper (hardware: one 8GB laptop GPU)

| | Paper | Here |
|---|---|---|
| Model | Qwen2.5-1.5B/7B-Instruct, full fine-tuning | Qwen2.5-1.5B-Instruct (same as the paper's ALFWorld row), LoRA r=16 |
| LR | 1e-6 | 1e-5 (LoRA) |
| Batch | 16 tasks × group 8 | 2 tasks × group 8 |
| Steps | 150, 50-turn episodes | 40 iterations, 20-turn training episodes (50 at eval) |
| Eval set | all 134 `valid_unseen` games | fixed random 48-game subset (`eval.max_tasks=0` for all) |
| Seeds | not stated | 1 (GRPO vs EMPG); ablations and more seeds pending |
| Framework | verl-agent (vLLM rollouts) | HF `generate` (~15 min per iteration on an RTX 5060 8GB) |

## Results

_Pending: produced by `benchmarks.suite` + `benchmarks.report`._
