# Research and performance audit

Audit date: October 11, 2026. Local platform: RTX 5060 Laptop GPU, 8 GiB, WSL Ubuntu 24.04.

## Assessment

The useful research question is whether count-based state novelty improves sparse-reward LLM agent learning,
and whether it complements or conflicts with EMPG's confidence-based step weighting. This is a sound **empirical
question**. The existing implementation's EMPG equations match the paper's appendix procedure, including zero
clarity on the last sampled step. It is a scaled LoRA experiment, not an exact reproduction of the paper.

Count-based exploration is established prior work. The contribution here would be the controlled comparison
with clarity, group-relative novelty, gating, memory, and actual task success in this agent setting, rather than
the existence of a novelty bonus. [Bellemare et al., 2016](https://arxiv.org/abs/1606.01868).

Related work checked as of this audit also limits the novelty claim:

- [ExTra, June 2026](https://arxiv.org/abs/2606.24994) combines a post-GRPO-normalization novelty bonus for
  diverse correct solutions with entropy-guided prefix regeneration on mathematical reasoning. This overlaps
  novelty-plus-entropy conceptually, while this repository measures environment-state novelty during interaction.
- [Look Before You Leap, May 2026](https://arxiv.org/abs/2605.16143) studies autonomous environmental
  exploration using checkpoint coverage and interleaved exploration/task training. Here novelty is applied
  inside task episodes, without a separate preparatory exploration phase.
- [The Dark Room in the Reward Channel, August 2026 revision](https://arxiv.org/abs/2607.21273) reports failures
  of prediction-based reward shaping under GRPO standardization on ALFWorld and other settings. It is a preprint
  about a different signal, not evidence that this count bonus fails; its normalization mechanism is a useful
  reason to compare where a bonus enters the advantage pipeline.

A defensible project framing is the **controlled interaction of state novelty, clarity and memory under a small
compute budget**. A claim of a new general curiosity/GRPO algorithm needs a more extensive prior-art comparison.

The paper's identity is valid: for a categorical softmax, the expected squared logit-space score norm is
`1 - sum(p_j**2)`, equivalently `1 - exp(-H_2)`. It does not prove monotonicity in Shannon entropy or in a neural
network's parameter-gradient norm. For example, `p=(.499,.499,.002)` has lower Shannon entropy but higher
`1-sum(p²)` than `q=(.7,.15,.15)`. The network Jacobian and advantages also affect updates. The appendix's
immediate clarity reward is a shaped, detached learning signal; it omits reward-to-go and the derivative of its
policy-dependent reward. Treat the proposed cumulative-objective interpretation cautiously.
[Wang et al., 2025, Proposition 1 and Appendix B](https://arxiv.org/html/2509.09265v1).

The same caution applies to step curiosity: rewarding the next state's novelty is local advantage shaping,
not an unbiased gradient of the episode's cumulative novelty return. The existing `mode: trajectory` is a
different estimator: it adds **mean** episode novelty before group reward normalization. These should be
labeled distinct experiments. Neither automatically preserves the optimal extrinsic-task policy.

In a group with tied extrinsic rewards, trajectory mode produces
`A_i = (c_i - mean(c)) / (std(c) + eps/eta)` for positive `eta` when novelty varies. Once `eta*std(c)` dominates
epsilon, decreasing `eta` barely decreases the shaped advantage. This is algebraically different from step
mode's post-normalization `eta` term. Keep step mode for the initial weight sweep; do not interpret a small
trajectory-mode coefficient as a small update, or its failure as a clean test of count novelty alone.

## Problems repaired

1. **Training tasks were not actually controlled across arms.** Task sampling and minibatch shuffling shared
   one NumPy RNG. Different episode lengths change permutation consumption, then subsequent sampled games.
   Independent task/update RNGs now preserve the task schedule. A regression exercises this with different
   episode lengths through the training loop.
2. **The sampler could leave the scored policy.** Training rejects temperature other than 1 or top-p other than
   1, because the scorer does not apply those transforms. Existing repetition-penalty disabling is preserved.
   Module dropout is disabled during updates while transformer checkpointing remains active. Mixed-precision
   merged-LoRA/vLLM numerics still differ slightly from unmerged HF scoring; compare arms with one backend.
3. **No individual evaluation outcomes or inferential comparison.** Evaluation now saves task identity, outcome,
   episode length, seed, and protocol. Reports compare identical seed/task panels using a crossed paired
   bootstrap: the task draw is shared across seed draws. Mismatched panels, seeds, or protocols are rejected.
4. **Stale results could be silently reused.** The suite now checks evaluation configuration fingerprints and
   training configurations. Existing artifacts remain intact; use a fresh output directory after this audit.
5. **Intrinsic share overstated influence.** It previously used the raw positive bonus, including its constant
   part removed by centering. It now measures `mean|A_with - A_without| / mean|A_with|`, including trajectory
   normalization effects. The ratio can exceed one through cancellation; it is not a probability.
6. **Environment resources survived exceptions.** Rollouts close all episodes in `finally`; training/evaluation
   close persistent workers. Independent workers communicate through ordered JSON messages, keeping CUDA and
   model weights out of worker processes. Direct TextWorld construction avoids accumulating gym registrations.
7. **KL silently disappeared without LoRA.** A requested KL penalty now errors when no supported frozen
   reference exists. Full fine-tuning needs `kl_coef=0` or an implemented separate reference.

## Interpretation limits that remain

- The old `small` tier trains for 10 turns. That can systematically truncate the very chains curiosity is
  intended to improve. The new `pilot` uses 30 turns; confirmation should check 50-turn training as well.
  Last-step clarity is still zero at an artificial horizon; horizon sensitivity is part of the experiment.
- All-failed groups need not have zero GRPO advantage: invalid-format penalties can differ across rollouts.
  `no_signal_frac` measures equal binary outcomes; `flat_group_frac` measures equal actual rewards. Report both.
- Reasoning-token entropy measures token uncertainty, not calibrated semantic correctness or action uncertainty.
  Low entropy can mean a confident mistake. Negative clarity weight is not a pure controlled exploration
  measure: terminal zero bonuses and batch centering introduce horizon/length effects.
- Parsed state is a hand-engineered, partial-observation abstraction. Initial location is `start`; only
  feedback-tracked changes are represented. Prior random-walk checks do not establish exact equivalence to all
  physical states. Oracle and parsed keys also differ in granularity, so oracle performance is a diagnostic,
  not a guaranteed upper bound. Group novelty is hindsight sibling coverage, not an online lifelong count.
- Inv-sqrt bonuses remain positive on revisits. Group counts ignore repeat visits within a sibling, so a state
  reached by only one sibling can keep earning a large bonus. Watch no-op rate, loops, episode length and task
  success; consider the existing first-visit/cap controls before adding new mechanisms.
- Centering removes a constant curiosity bonus. An all-failed group gains a useful shaping contrast only when
  the relevant bonuses vary; adding a positive constant does not create information. Invalid-rate metrics here
  measure response formatting; legal-looking but inadmissible actions appear in no-op metrics instead.
- A correlation between novelty/revisit rate and success in base rollouts is diagnostic, not causal. Episode
  length and task difficulty can explain it. Increasing state coverage alone does not establish useful learning.
- LoRA, learning rate, batch size, training horizon, missing token-entropy regularization, and response caps
  differ from the paper. The paper reports three seeds, not an unspecified seed count (Table 1).
- Equal iterations are equal episode opportunities but not equal realized environment steps, tokens or time.
  New logs/plots expose all three budgets. Budget stops occur after complete batches and may overshoot.

## Smaller experiment that answers a useful question

1. **Check the mechanism without training.** Run base-policy exploration on training games, group 4, at least
   30 turns. Inspect valid-format rate, truncation, no-ops, real revisits (`revisit - noop`), group coverage,
   successful groups, and per-subtask results. If there is no usable extrinsic signal at that horizon, a null
   training result will be hard to interpret. Do not lower the response cap while truncation is material.
2. **Screen one prespecified pair.** Run GRPO vs episodic curiosity using `pilot` and seed 0, with the same
   configuration/backend. This uses 2 games x 4 siblings x 8 iterations: at most 1,920 training turns per arm.
   Evaluate a fixed proportional 24-game `valid_seen` panel at 50 turns. This is development evidence.
3. **Test complementarity and the memory confound.** If novelty improves task success, test EMPG vs EMPG plus
   that same novelty configuration, and run GRPO plus memory. Prefer this staged selection to eight arms before
   knowing whether any agent is learning. Prespecify success as the primary endpoint; exploration metrics
   explain possible mechanisms and do not replace it.
4. **Freeze the candidate before confirmation.** Run at least 3 paired seeds on a fresh output path. Evaluate
   all `valid_unseen` tasks at 50 turns (`eval.max_tasks=0`, `eval.split=valid_unseen`). Report paired gains,
   seed-level differences, intervals, subtasks, actual compute, and horizon sensitivity. If budget is limited,
   choose fewer prespecified arms instead of interpreting tiny differences on a tiny task panel.

For illustration, a single unpaired 16-game success rate has a worst-case normal-approximation 95% half-width
near 25 percentage points; 24 games near 20 points; 48 near 14 points. Pairing can improve precision, depending
on discordant outcomes. Repeating the same games across seeds does not create independent new tasks. Three
seeds are a practical minimum, not a guarantee of adequate power. A pilot that detects a large improvement is
useful; a null pilot remains inconclusive. Selecting the best of many arms on `valid_unseen` would contaminate
the intended confirmation set. Constant observed paired differences give degenerate empirical bootstrap
intervals; a zero-width interval from a tiny all-failed panel is not evidence of equivalence.

## Performance changes and validation

- Four ALFWorld worker processes parallelize reset and stepping; the serial path remains available.
- Exact full-vocabulary scoring projects **only valid response hidden states**, in chunks of 128 tokens.
  Output-head checkpointing recomputes chunks during backward, avoiding vocabulary-sized activations for the
  whole response batch. Prompt/pad positions do not need vocabulary projection. No entropy approximation.
- Scoring restores original response order after sorting by length. Updates retain random optimizer-minibatch
  membership and token-count weighting, while sorting accumulation microbatches to reduce padding.
- Loss statistics stay on device until aggregate reporting, reducing GPU synchronization.
- Optional environment-step/time budgets save a final adapter and a completion record before evaluation.

The default chunked path supports ordinary output heads in Qwen2/Qwen3/Llama/Mistral; other model architectures
fall back to their causal-LM forward to preserve model-specific output transforms. Dense scoring is available
with `policy.chunked_head=false`. Floating-point kernels and accumulation order can change rounding, so
bitwise-identical training is not promised. The underlying formulas, full vocabulary and loss weighting remain.

Raw profiles are under `runs/audit/`; these use real ALFWorld games and, for the update profile,
Qwen2.5-1.5B-Instruct plus LoRA. Update timing uses injected mixed rewards to guarantee nonzero gradients:
those scores are performance data, not evidence that the method improves success.

The real training/evaluation smoke suite completed for GRPO and EMPG plus episodic curiosity, on a fresh path
`runs/audit/smoke`. Both had zero success on a 4-game, 4-turn evaluation. Training/evaluation response truncation
was substantial at the smoke-only 96-token cap; this demonstrates why that tier cannot adjudicate the research
hypothesis. Both arms used exactly the same training tasks in both iterations and the same evaluation panel.
Curiosity's measured advantage contribution was nonzero, while GRPO's was zero. Reports include the degenerate
paired interval warning, actual steps/tokens/time curves, lengths and per-task outcomes.

Final isolated profiles:

| Measurement | Serial/dense | Parallel/chunked | Scope |
|---|---:|---:|---|
| Environment reset, warm | 35.10 s | 12.72 s | 16 identical training games, four workers |
| Environment stepping, warm | 7.79 s | 2.51 s | Same games and scripted actions, five turns |
| Environment total, warm | 42.89 s | 15.23 s | 2.82x faster; observations, info and state keys match exactly |
| Policy update, warm | 43.61 s | 15.90 s | Same 24 response steps; alternating path order |
| Peak allocated GPU memory | 4.19 GiB | 3.55 GiB | Update profile; excludes allocator-reserved unused blocks |

The final update profile shows 2.74x speedup; an earlier repeat measured 22.00 vs 15.92 seconds (1.38x), so
performance varies and these component gains should not be multiplied into an end-to-end claim. The response
cap, prompts, real environment and full vocabulary were unchanged in each dense/chunked comparison. Step
entropy means differed by about 2.5e-9 nats in this replay, but mixed-precision updates are not bit-identical.
Raw records: [environment_final.json](../runs/audit/environment_final.json),
[model_final.json](../runs/audit/model_final.json), and
[the earlier repeated profile](../runs/audit/model_profile_repeated.json).

Validation: **115 tests passed**; CPU tests compare output-head log probabilities, full entropy and gradients
against the dense path, including variable prompt/response lengths and trainable/frozen heads. Training schedule
tests use different episode lengths and exercise iteration, step and time budgets. Real ALFWorld checks cover
serial/parallel trace equivalence, GRPO and EMPG+curiosity training/evaluation, the Phase 0 diagnostic, adapter
saving/loading, and report generation. No long-horizon effectiveness claim has been established by these
engineering checks; the pilot/confirmation protocol above remains the actual research experiment.
