# Update throughput and experiment budgets

The completed `paper_overnight` suite took 5.1 hours. Its 12 iterations per method
covered 192 episodes on 24 distinct training tasks. Scoring and gradient updates
accounted for approximately 65% of training time. Changes here preserve the paper
reward, EMPG advantages, exact vocabulary entropy, entropy regularization, KL,
PPO/dual clipping and optimizer minibatch membership/token weighting.

## Execution controls

- `policy.score_batch_size`: independent no-gradient policy/reference scoring batch;
  zero retains the historical `micro_batch_size` behavior. Transfers happen once per
  batch rather than once per response vector. Length sorting and response order are preserved.
- `policy.micro_batch_size`: training rows per gradient accumulation microbatch.
  Increasing it does not change optimizer minibatch size or optimizer step count.
- `policy.micro_batch_tokens`: bounds padded prompt-plus-response tokens in both
  scoring and training microbatches. The optimized preset uses 2,560, allowing
  larger batches for short contexts and fewer rows for long contexts without
  dropping responses or changing their loss weights.
- `policy.cuda_memory_fraction`: optional process-wide HF allocator ceiling.
  The optimized preset uses 0.75 to reserve room for the sleeping inference worker
  and desktop. Zero leaves the existing process limit unchanged. Excessive allocations
  raise OOM rather than allowing the HF allocator to grow without a bound.
- `policy.head_backend=analytic`: exact linear-head log probabilities and entropy with
  a chunked analytic backward. Only hidden states, head parameters and targets are
  retained; each vocabulary projection is recomputed one chunk at a time. The derivative
  is `dH/dz = -p * (log(p) + H)`. Arbitrary upstream derivatives from PPO, KL and the
  entropy regularizer are supported. Nonlinear/transformed/adapter heads fall back
  to the checkpoint reference instead of silently bypassing their behavior.
- `policy.checkpoint_min_tokens`: an optional selective transformer-checkpoint
  threshold. Zero keeps checkpointing always on. Short-context checkpoint-free
  candidates exceeded the measured laptop memory budget, so the optimized preset
  retains zero. The option is available for profiling hardware with more memory.
- `policy.head_backend=checkpoint`: the original reference implementation.
  `policy.head_checkpointing=false` lets profiling compare keeping head activations
  with recomputation. `policy.gradient_checkpointing` controls the transformer separately.

No extra training framework or kernel package is required. This is not a substitution
of a stock GRPO loss for the EMPG objective. Historical default values are omitted from
configuration fingerprints, keeping saved experiments readable.

## Profiling

```bash
python -m benchmarks.tune_updates --config empg --tier paper_overnight \
  --turns 12 --samples 16 --rollout-cache runs/audit/update_tuning_rollout.json \
  --out runs/audit/update_tuning.json
```

Each candidate receives the same real rollout, restored initial adapter and a fresh
optimizer. The profiler records scoring time, total update time, optimizer steps,
adapter differences and the HF process's peak reserved VRAM. A 75% HF allocator ceiling
leaves room for the sleeping vLLM worker and desktop, converting excessive allocations
into recorded OOM candidates instead of permitting shared-RAM spill. The worker is
initialized even for cached rollouts so its memory footprint is represented.
Results are published after each candidate, and the optional JSON rollout cache avoids
repeated decoding. A different config/horizon/sample count needs a new cache path.
The maximum prompt/response lengths are recorded: a short probe alone does not establish
memory safety for all possible long responses. Use `--turns 50` with a fresh cache for
a full-horizon validation. `--repeats 2` reverses candidate order on the second pass.

Timing improvements are hardware/workload dependent. No learning improvement follows
from numerical equivalence or a faster timing probe alone.

## Measurements on the RTX 5060 Laptop 8 GB

All timing candidates below used the same cached real 192-response rollout
(maximum prompt 1,035 tokens, maximum response 493). Times include scoring and
one optimizer update. These are short update probes, not full-suite speedups.

| Configuration | Update time | HF peak reserved VRAM |
|---|---:|---:|
| Reference: two rows, checkpoint head, chunks of 128 | 147.2 s | 4.34 GiB |
| Four scoring rows, two training rows | 142.5 s | 5.93 GiB |
| Four scoring and training rows | 142.9 s | 5.93 GiB |
| Eight training rows, analytic head | 148.7 s | 5.97 GiB |
| Selected: two scoring/training rows, analytic head, chunks of 256 | 137.4 s | 5.90 GiB |

The selected path reduced measured update time by 6.6% (about 1.07x throughput).
The inference worker retained approximately 1.35 GiB in addition to these process
figures. Larger batches did not provide a useful speed gain. Unconditional
checkpoint-free updates failed, including with a 2,560-token padded budget.
Selective one-row variants at 1,280/900-token thresholds also exceeded the allocator
ceiling. The selected preset therefore retains transformer checkpointing throughout.

Evidence: `runs/audit/update_tuning_safe.json`, `update_tuning_bounded.json` and
`update_tuning_selective.json` (and their logs). The profiler's CPU checks compare
full update weights, entropy/log-probability gradients and optimizer step counts;
BF16 batch-shape rounding can still change individual gradients on GPU.

The final preset also completed a synthetic maximum-context GPU check with eight
2,048-token prompts and 512-token responses: 20.2 seconds, finite gradients, and
4.81 GiB peak HF reserved memory with the sleeping worker present. See
`runs/audit/update_max_context.json`. This exercises the context and activation
limits; it is not a task-success measurement. All 285 CPU tests pass, and the
separate real CUDA BF16 score/gradient equivalence check passes.

The optimized runner forecast uses the completed 5.1-hour suite's component costs,
scaled by the measured scoring/update ratios. Its evaluation estimate is also
recalibrated from that completed suite. A lower forecast than the historical
conservative estimate is not an additional measured speedup. Context/task variation,
cold startup and device power affect actual runtime; the 8.5-hour deadline remains.
The default forecast is approximately 7.5 hours with a fresh zero-shot evaluation,
or 7.1 hours when reusing the existing baseline. Adding both missing replication
seeds is forecast at about 9.2 hours; it is not promised to fit one eight-hour night.

## Experiment presets

- `paper_optimized`: 20 iterations, 320 episodes per method/seed, one seed by default,
  all 134 unseen tasks, 8.5-hour suite deadline. Fresh output directories preserve seed 0.
- `paper_replication`: the same execution path with the original 12-iteration budget.
  `--seeds 1 2 --skip-base --include-runs runs/paper_overnight` trains only the missing
  independent seeds and pools them with the completed seed 0 and its zero-shot result.
  Three seeds are the default when `--seeds` is omitted. No default deadline.
- `paper_lr_pilot`: eight iterations and a fixed stratified 48-task `valid_seen` panel.
  This keeps the final `valid_unseen` panel out of learning-rate selection. No automatic
  learning-rate change or pilot launch occurs. The learning rate remains 1e-6 by default.

For a learning-rate pilot, run the following in separate directories, changing the
rate to `3e-6` or `1e-5` and the output names for subsequent pilots. All methods receive
the same learning rate. Use `--skip-base` after the first pilot to avoid repeating
the zero-shot evaluation; its existing result remains in the first report.

```bash
python run_benchmarks.py --tier paper_lr_pilot --runs runs/lr_1e6 \
  --out results/lr_1e6 --estimate-only -- train.lr=1e-6
python run_benchmarks.py --tier paper_lr_pilot --runs runs/lr_1e6 \
  --out results/lr_1e6 -- train.lr=1e-6
```

Pilots are sparse-reward development screens, not reliable winner selection. Confirm
a chosen shared setting on additional seeds before drawing a conclusion about EMPG.

## Combining compatible seeds

`run_benchmarks.py --include-runs ...` adds completed external seeds to the report,
without scheduling them for training. `benchmarks.report` accepts the same option.
Pooling validates per-method configurations, evaluation protocols and exact task panels.
Only execution details (microbatches, head implementation/checkpointing, environment
worker count and checkpoint frequency) may differ. Precision, model, learning rate,
rewards, optimizer minibatches, training budgets, and sampling settings must match.
Existing source artifacts stay untouched; `summary_sources.json` records their paths.
Duplicate trained seeds are errors. A zero-shot result is included once.
Pass `--skip-base --include-runs runs/paper_overnight/base` to reuse only the
compatible zero-shot evaluation for a deeper run with a different training budget.
Its recorded evaluation protocol must match; trained seeds from different budgets
are still rejected. No source files are copied or overwritten.
