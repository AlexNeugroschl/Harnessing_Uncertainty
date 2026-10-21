# EMPG fidelity and performance review

Audit: October 21, 2026. Hardware: RTX 5060 Laptop, 8 GiB, Ubuntu 24.04 in WSL.

## Assessment

The entropy modulation implements the published procedure, but the completed overnight
experiment does not reproduce the paper's training protocol. Both protocol changes and
performance work are needed before interpreting the weak success rates as evidence about EMPG.
The original audit introduced opt-in settings. The follow-up implementation now makes the corrected
LoRA comparison the default; see [implementation and run instructions](paper_implementation.md).
Historical tiers and completed results retain their original settings and fingerprints.

The reference is [Wang et al., EMPG, September 2025](https://arxiv.org/html/2509.09265v1),
especially sections 4, C and E. The public framework is
[Verl-Agent at publication-era commit 29e3e8b](https://github.com/langfengQ/verl-agent/tree/29e3e8b3e3412aff53b1e4a461dac77a8eec19bd),
dated September 3, 2025. Pinning this revision avoids attributing subsequent framework changes
to the original experiment. The available framework is evidence about its defaults, not proof
of every setting in the authors' private experiment. The paper publishes an EMPG function,
but not a complete runnable EMPG experiment in this repository.

## Fidelity findings

The core calculation was already substantially correct: full-vocabulary token entropy averaged
over a response, batch min-max normalization, mean-normalized exponential gradient scaling,
next-response clarity within the same episode, terminal clarity zero, and final step-mean
centering. The follow-up corrected the numerical constants in the paper presets: normalization
and scaling-denominator epsilon are now both 1e-8, while legacy tiers retain their constants. Exact response
boundary parity cannot be established without its unpublished sequence-processing helper.

| Setting | Published experiment | Completed `overnight_full` |
|---|---|---|
| Training | 150 iterations, 16 games, 8 siblings | 5 iterations, 4 games, 4 siblings |
| Episode opportunities per seed | 19,200 | 80 |
| Learning rate | 1e-6 | 1e-5 |
| Token entropy regularization | 0.001 | 0 |
| Hardware / parameters | 4 A100s for 1.5B, full parameters | One 8 GiB GPU, LoRA rank 16 |
| Evaluation | 134 `valid_unseen` tasks, three seeds | 48 `valid_seen` tasks, one seed |

The training opportunity gap is **240 times per seed**. More accurate settings do not remove
this compute gap. Even the new local preset uses two games instead of sixteen.

Additional differences come from the
[original framework's ALFWorld script](https://github.com/langfengQ/verl-agent/blob/29e3e8b3e3412aff53b1e4a461dac77a8eec19bd/examples/grpo_trainer/run_alfworld.sh):
response limit 512 versus 256 locally, sampled evaluation at temperature 0.4 versus greedy,
and CUDA graphs enabled versus the local vLLM eager default. The previous overnight report's
roughly 3% evaluation truncation rate includes truncations in more than half of evaluation
episodes; a small per-response fraction can affect many whole tasks.

### Reward placement changes learning

The framework's
[episode reward manager](https://github.com/langfengQ/verl-agent/blob/29e3e8b3e3412aff53b1e4a461dac77a8eec19bd/agent_system/reward_manager/episode.py)
assigns the episode outcome to each response row and penalizes that row if its format is invalid.
Its [GRPO estimator](https://github.com/langfengQ/verl-agent/blob/29e3e8b3e3412aff53b1e4a461dac77a8eec19bd/verl/trainer/ppo/core_algos.py)
normalizes across response rows for a task group when `compute_mean_std_cross_steps=True`.
The local legacy implementation instead sums format penalties into one episode reward,
normalizes across episodes, and broadcasts the result. That penalizes valid steps in an
episode containing one malformed response and weights unequal-length episodes differently.

For two failed two-step episodes, with only the first response malformed:

| Calculation | Four response advantages, approximately |
|---|---|
| Legacy trajectory normalization | -0.707, -0.707, +0.707, +0.707 |
| Framework response-row normalization | -1.5, +0.5, +0.5, +0.5 |

`train.reward_mode=step` now implements the framework behavior. `trajectory` retains the old
behavior. This distinction is separate from the paper's simplified trajectory-return equations;
the exact authors' experiment overrides are unavailable.

The optional `env.action_parser=verl` also follows the original
[action projection](https://github.com/langfengQ/verl-agent/blob/29e3e8b3e3412aff53b1e4a461dac77a8eec19bd/agent_system/environments/env_package/alfworld/projection.py):
first action block, lowercase action, literal thought tags, CJK rejection, and last-30-character
fallback. The old strict parser remains available. Neither format validity nor the parsed
no-op metric establishes whether a generated plan is useful.

`train.entropy_coef` now adds differentiable, masked token entropy to the actor objective.
This is separate from the detached entropy used to compute EMPG advantages. Merely computing
entropy for EMPG did not supply the missing entropy regularizer.

The paper presets also select the pinned actor optimizer's weight decay of 0.01; legacy local
tiers use zero. Corrected tiers now enforce the framework's 2,048-token prompt limit without
silent truncation and use sequential 256-response optimizer minibatches. A remaining framework
difference is unpadding: the local trainer uses length-sorted, padded microbatches rather than
the reference's distributed unpadded actor.
The final actor-loss comparison also identified the framework's negative-advantage dual-clip
bound of 3, which the corrected presets now enable alongside the symmetric 0.2 PPO clip.
These are additional reasons to call the presets scaled protocol checks rather than exact
reproductions. Prompt templates and two-step observation/action history are substantially
the same; the first ALFWorld observation already contains the task goal.

## Measured performance

Summing the four completed runs gives approximately 156 minutes generating training responses,
146 minutes updating (including 48 minutes scoring), and 4.5 minutes resetting/stepping
environments. Evaluation and process startup account for most of the remaining 144 minutes.
The earlier README claim that environments dominate the remaining runtime does not describe
this completed run. HF downloads are not the principal explanation; the new probes used the
cached model with network access disabled.

| Bounded local probe | Before | After | Interpretation |
|---|---|---|---|
| Update on 48 cached response steps, two repeats | 32.83-33.19 s | 28.16-28.17 s | bf16 autocast: about 1.17x |
| 16 prompts, exactly 64 greedy tokens each, warm repeats | HF 4.13-4.16 s | Compiled vLLM 0.92-0.93 s | About 4.5x decode speed |
| Same probe with the 32-sequence limit | HF 4.15-4.16 s | Compiled vLLM 0.92-1.20 s | About 3.5-4.5x; one late JIT compilation |
| Additional 32-sequence repeat | HF 4.14 s | Compiled vLLM 1.35-1.76 s | About 2.4-3.1x; machine load and warmup vary |
| vLLM sleep then corrected HF update | Not previously checked with graphs | 30.42 s | Short-context update fits; not a long-context guarantee |

The update probe retains gradient checkpointing, the same cached responses, and exact entropy.
PEFT's 392 trainable LoRA tensors were float32; the old forward path lacked CUDA autocast.
`policy.amp=true` uses bf16 compute while retaining float32 trainable parameters. Autocast changes
rounding; this is not an exact arithmetic equivalence claim.

Profiling confirmed fused memory-efficient scaled dot-product attention is already active.
There was no accidental dense attention fallback to fix. Disabling checkpointing saturated
physical VRAM and stalled; that probe was stopped. It is not a safe default optimization on 8 GiB.
CUDA kernel timing was unavailable because CUPTI rejected this WSL device; attention-path
identification uses recorded operator names rather than reported GPU kernel durations.

The vLLM decode comparison used identical prompt IDs in one environment, Torch 2.13.0+cu130
and vLLM 0.30.0. The separate autocast comparison used Torch 2.11.0+cu128. They are component
measurements, not a measured end-to-end suite speedup. Response length, EOS timing, batch
occupancy, model transfers, scoring and optimizer work all affect the eventual gain. Enabling
the missing entropy objective and raising the response cap also adds work.

The fast preset enables CUDA graphs and limits `vllm_max_num_seqs` to 32, matching the local
generation batch ceiling. The engine's previous limit of 256 caused unnecessary graph capture
up to 512-token scheduling shapes. Existing eager mode remains the default for old tiers.
The initial graph probe took 74 seconds to initialize and left about 2.3 GiB in the sleeping
engine. Bounding the engine to 32 sequences reduced startup to 56 seconds, captured graph
memory to 0.08 GiB, and sleeping-engine memory to 1.87 GiB. A corrected 48-response update
then completed in 30.13 seconds, followed by generation with adapter ID 2 after reloading.
Both probes completed the update, but vLLM emitted a native allocator warning during process
cleanup. This backend requires a representative pilot before a long unattended experiment.

An additional repeat completed the corrected update in 46.43 seconds and checked generated
token log probabilities directly after adapter refresh: across 1,446 sampled tokens, mean
HF-minus-vLLM log probability was -0.0024 nats, mean absolute difference 0.0197, and maximum
absolute difference 0.240. Mean `logp + entropy` was -0.0011. These results support sampler/scorer
consistency within the observed numerical differences; they do not establish exact equivalence.
That repeat used cached compilation (26-second engine startup), incurred an 8.8-second cold
generation, and ended with forced worker cleanup and a semaphore warning. Its timings underline
why a long-run speedup cannot be inferred from the best warm decode alone.

Local evidence is saved under ignored `runs/audit/`: `paper_perf_probe.json`,
`vllm_graphs_probe.json`, `vllm_graphs_32_baseline_probe.json`, `vllm_graphs_32_probe.json`,
the graph probe logs, and their helper scripts. The cached probe
trajectories are for performance checks; injected outcomes in one probe are not learning results.

## Running the corrected local comparison

`paper_lora` enables the protocol corrections above with HF decoding. `paper_lora_fast`
inherits them and selects compiled vLLM. Both use 150 iterations, two games and eight siblings,
50-turn episodes, full `valid_unseen` evaluation, and LoRA. They remain scaled experiments.
Full-parameter training with a separate frozen KL reference is not implemented here; setting
LoRA rank to zero does not reproduce the original distributed trainer.

Use the separate vLLM environment for the fast tier. On Windows, set
`EMPG_WSL_PYTHON=/home/alex_neugroschl/venvs/empg-vllm/bin/python` in the launching environment;
inside WSL activate it directly. Use new directories for every changed protocol.

```bash
source ~/venvs/empg-vllm/bin/activate
python run_benchmarks.py --tier paper_lora_fast --arms grpo empg --seeds 0 --estimate-only

# Short integration check; two-turn episodes do not test learning.
python run_benchmarks.py --tier paper_lora_fast --arms grpo empg --seeds 0 \
  --runs runs/paper_fast_smoke --out results/paper_fast_smoke -- \
  train.iterations=1 train.games_per_iter=1 train.group_size=2 \
  train.max_steps=2 eval.max_tasks=6 eval.max_steps=2

# Larger development comparison with a suite deadline and fixed work per arm.
python -m benchmarks.suite --tier paper_lora_fast --arms grpo empg --seeds 0 \
  --max-seconds 28800 -- train.iterations=10 eval.split=valid_seen \
  eval.max_tasks=48 out_dir=runs/paper_fast_pilot
python -m benchmarks.report --runs runs/paper_fast_pilot --split valid_seen \
  --out results/paper_fast_pilot --baseline grpo --arms grpo empg --seeds 0
```

The follow-up runner estimate includes conservative component speedups, the entropy objective,
and transfer/startup costs. It remains an extrapolation, not a measured overnight runtime.
The 150-iteration paper tiers have no default time limit; `paper_overnight` has an 8.5-hour
default deadline. A deadline can leave later methods incomplete. Increase
the fixed iteration budget only after a full-horizon pilot measures rollout, update and
evaluation costs. Start with GRPO and EMPG; adding curiosity arms divides the same compute
budget before the basic reproduction is established. Use three seeds for confirmation.

New non-default settings participate in configuration fingerprints and evaluation protocol
checks. Disabled new options preserve legacy fingerprints. An estimate-only check confirmed
that all five existing `overnight_full` evaluations are still recognized as complete.

## Validation

All 198 tests passed after the changes. Coverage includes local format penalties, unequal
episode lengths, the analytic entropy-loss gradient and padding mask, legacy fingerprint
compatibility, parser behavior, and restoring the training sampling temperature after evaluation.
A real ALFWorld integration check completed base evaluation and one update plus evaluation for
both GRPO and EMPG, with two-turn episodes and six evaluation tasks, in 108 seconds. It used
the new reward mode, entropy loss, autocast, parser, response limit and evaluation temperature.
That is a pipeline check, not a learning experiment. No new long benchmark was launched.
