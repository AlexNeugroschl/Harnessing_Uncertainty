# Implementation and runtime review

Reviewed October 19, 2026 against the current working tree, the saved October 14
overnight run, and short probes in the installed WSL environment. Existing
experiment artifacts and training code were preserved. Reproduction script:
[`runs/audit/implementation_review.py`](../runs/audit/implementation_review.py).
Raw measurements: [`implementation_review.json`](../runs/audit/implementation_review.json).

## Assessment

The implementation performs online, multi-turn reinforcement learning: it generates
fresh episodes, scores them under the policy and reference, and then updates LoRA.
It is substantially different from supervised fine-tuning over an existing dataset.
There are concrete throughput and experimental-design problems. There is no evidence
in the examined run of an inactive optimizer, CPU-only model, disabled decoding cache,
or a training loop stuck indefinitely.

The central problem is spending most of the budget collecting poorly batched,
low-success episodes while making very few optimizer steps over very few tasks.
Successful process completion does not establish efficient training or useful learning.

## Measured cost

The suite took 14,041.57 seconds (234.03 minutes):

| Component | Minutes | Evidence |
|---|---:|---|
| Training response generation | 105.49 | Sum of `t_generate` |
| Policy/reference scoring before updates | 19.19 | Sum of `t_score` |
| Remaining update work, including backward and optimizer | 38.58 | Sum of `t_update - t_score` |
| Other training work | 2.49 | Remaining rollout time, including environment work |
| Evaluation, startup, saves and reports | 68.27 | Suite elapsed minus recorded training time |

The last row is a residual, not a separately instrumented evaluation measurement.
Environment reset and stepping together consumed approximately two minutes across
all training runs; optimizing the environment cannot recover most of this runtime.
Training generated 556,413 response tokens; evaluation generated another 527,621.
These counts exclude input tokens and padded decoding work.

## 1. Training batches are much smaller than the configured generation ceiling

[`overnight.yaml`](../benchmarks/configs/tiers/overnight.yaml) sets
`games_per_iter=1` and `group_size=4`.
[`trainer.py`](../empg/trainer.py) constructs exactly four trajectories per batch.
[`rollout.py`](../empg/rollout.py) advances them in lockstep, completing the entire
batch before the update. Each turn waits for `policy.generate()` to return.
`gen_batch_size=32` in [`policy.py`](../empg/policy.py) only divides an existing
prompt list; it does not create additional work or fill the batch. Actual training
decoding therefore has at most four prompts, decreasing if episodes end early.

The fresh GPU probe used Qwen2.5-1.5B with the configured merged LoRA path, the
actual first training task's 377-token reset prompt, and fixed 64-token greedy
decoding. Two repeats reversed the order of batch sizes. Throughput pooled over
the repeats was:

| Batch size | Mean call time | Generated tokens/second |
|---|---:|---:|
| 4 | 1.382 s | 185.3 |
| 8 | 1.661 s | 308.3 |
| 16 | 2.024 s | 505.9 |

Batch sixteen provided 2.73 times the token throughput of batch four. Peak reserved
memory in this short probe remained below 3.84 GiB. This does not establish the
same gain or memory requirement for varied, long-context rollouts and updates.

Priority: benchmark collecting multiple independent task groups at once while
retaining four siblings per task. Adjust task, token and optimizer budgets explicitly:
raising `games_per_iter` alone increases the amount of work per iteration and changes
the update schedule. Increasing `gen_batch_size` alone will have no effect here.

## 2. Decoding retains finished responses and waits for stragglers

The HF path calls `model.generate()` once for all active prompts at a turn. In the
installed Transformers `_sample` implementation, finished rows receive padding while
the full batch continues until the last response stops. The rollout then waits for
every environment response before beginning another turn. There is no continuous
refill of finished response slots from other episodes.

Saved response lengths allow reconstruction of this wasted decoding work. For each
turn, compare `active_rows * longest_response` with the sum of actual response lengths:

| Phase | Padded decoding slots / actual response tokens |
|---|---:|
| Full-horizon training batches | 1.48–1.51 |
| Evaluation batches | 2.33–2.48 |

Across evaluation, 527,621 actual response tokens required approximately 1,267,534
decoding token slots. Only approximately 3% of individual evaluation responses hit
the 256-token cap, yet an outlier made 21–32 of each evaluation's 50 turn calls
decode all the way to that cap. This is a real consequence of static batching.
These slot ratios are work proxies, not measured end-to-end speedup factors.

Priority: benchmark the existing vLLM path or an implementation that removes/refills
finished response rows and schedules independent episodes efficiently. Keep the same
backend across compared arms and validate sampler/scorer agreement. Neither backend
eliminates the need to finish each task's current action before its next environment step.

## 3. Very little task-success signal reaches the optimizer

| Method | Distinct training games | Rollout batches | Optimizer steps | Batches with no binary outcome contrast |
|---|---:|---:|---:|---:|
| GRPO | 8 | 8 | 16 | 7 |
| EMPG | 8 | 8 | 16 | 8 |
| GRPO + group curiosity | 8 | 8 | 16 | 4 |
| EMPG + episodic curiosity | 7 | 7 | 14 | 6 |

Thus 25 of 31 batches had no success/failure contrast among siblings. GRPO's
outcome advantage is group-relative. In the all-failed GRPO batches, differences
in formatting penalties can still provide gradients; an absence of binary outcome
contrast does not imply an absence of all gradients. In GRPO iteration seven,
`adv_std=0` and `pg_loss=0`; that update still incurs the KL term and its computation.
EMPG and curiosity also supply their configured shaping signals.

The question is whether those signals teach task completion. Saved results do not
establish that: evaluation success was 0–8.3%, and 115 of 120 evaluation episodes
reached the complete 50-turn horizon. High repeated-state rates do not by themselves
prove a code defect, but they explain why rollouts rarely become cheaper by finishing.

The per-method 2,400-second budget also stopped methods after different numbers of
batches. They therefore saw different numbers of games and optimizer steps. This is
a valid compute-budget screen if labeled accordingly, but it confounds an interpretation
as an otherwise equal-data, equal-update algorithm comparison.

Priority: establish a usable task-success signal on a controlled task panel and
report the number of informative groups, distinct games and optimizer steps. Use
explicitly matched training budgets when the intended comparison requires them.

## 4. Scoring involves multiple full passes, but entropy is not the main culprit

[`trainer.update`](../empg/trainer.py) always runs a policy scoring pass with full
vocabulary entropy, a reference scoring pass when KL is enabled, and gradient
forwards/backward. With microbatch size two, a 200-response rollout needs roughly
100 policy scoring microbatches, 100 reference microbatches and 100 gradient
microbatches per epoch, plus checkpoint recomputation.

These passes are not all interchangeable: the PPO denominator, fixed reference and
gradient computation have different purposes. Blindly dropping them changes the
objective. The current response-only, chunked output head already avoids projecting
prompt positions and is covered by dense-path gradient comparisons.

Full entropy is unnecessary for plain GRPO's advantage when no entropy-dependent
curiosity modulation is enabled; it is currently computed for every method partly
for metrics. However, the isolated eight-response probe took approximately 0.643 s
with entropy versus 0.633 s without. This short probe supplies no evidence that
entropy alone explains the multi-hour runtime. It is a lower-priority optimization,
and EMPG's entropy must be retained.

## 5. Interrupted training is restarted, not resumed

[`suite.py`](../benchmarks/suite.py) skips completed runs, but an unfinished training
phase calls `train()` again. The trainer creates a fresh policy and optimizer and
opens metrics with `"w"`. It does not restore adapter/optimizer/RNG/task-schedule
state. With overnight `save_every=0`, no intermediate adapter is saved either.

This does not explain the successfully completed overnight run. It does mean that
interrupting and rerunning a long phase can lose that phase's work, even if earlier
completed methods are retained. True checkpoint resume is needed for that use case.

## Checks and limits

- WSL environment: PyTorch 2.11.0+cu128, Transformers 5.17.0, PEFT 0.21.0; CUDA available.
- Actual generation forward hooks showed KV caching enabled, one-token decoder inputs
  after prefill, and `logits_to_keep=1`. Attention implementation was SDPA. There is no
  evidence of repeated full-context decoding or full-prompt vocabulary projection.
- Saved training peak reserved GPU memory was approximately 4.7–5.25 GiB; these
  measurements do not suggest exceeding the physical 8 GiB during recorded training.
- The first real reset observation contains the task text, so the first prompt's
  lack of a separate `{task}` field does not remove the goal in this environment.
- All **151 existing tests passed**, offline, in 13.76 seconds. They check numerical
  and control-flow correctness; they do not prove useful learning or good throughput.
- No end-to-end run was repeated, no backend speedup was assumed, and no research
  hyperparameters or production source files were changed for this review.

The first performance work should address actual rollout batch occupancy and static
decoding waste, followed by validation that the resulting training batches carry a
useful task-success signal. A shorter smoke test only checks that the pipeline runs.
