# Expanded overnight benchmark

Run `python run_benchmarks.py` for the expanded development comparison, or add
`--estimate-only` to preview it without training. The default tier is `overnight_full`.
This update prepares the benchmark; it does not automatically launch an eight-hour job.

## Workload

The same four methods remain enabled: GRPO, EMPG, GRPO with group curiosity, and EMPG
with episodic curiosity. Each starts from the same base model using seed 0. The
untrained base is also evaluated. Model, response format, reward/loss definitions,
curiosity settings, 256-token response cap, and 50-turn horizons are retained.

Each method runs five training iterations. Each iteration samples four games, with
four sibling rollouts per game: 16 simultaneous trajectories instead of four.
The complete workload provides 20 game opportunities, up to 80 episodes and 4,000
response steps per method. Games are distinct within an iteration; a game can recur
in a subsequent iteration. Independent task/update RNGs preserve the task schedule
across methods, even when episode lengths differ.

Eight optimizer minibatches per iteration yield 40 optimizer steps per method. At
the full horizon, each optimizer step covers approximately 100 response steps, as
in the original smaller configuration. Microbatch size remains two, preserving
the existing memory constraint. Larger rollouts also mean more optimizer steps
between fresh policy rollouts; this is a changed training batch size, not a claim
of numerically identical training to the original run.

All policies are evaluated on the same deterministic stratified panel of 48
`valid_seen` tasks, twice the previous panel size. The task types and selection seed
are retained. Increasing the panel size resamples the proportional panel; the old
24-task panel is not guaranteed to be a subset. `valid_unseen` remains available
for later confirmation after choosing a candidate.

## Timing and deadline

The workload's central estimate is approximately 8.2 hours on the profiled RTX 5060
Laptop 8GB. The default ceiling is 8.5 hours, allowing a small completion margin for
the final evaluation. Use `--hours 8` for a strict eight-hour cutoff. The deadline
includes model startup, training, and evaluation. Cleanup and reports occur afterward.
Hardware variation, longer responses, and longer contexts can exceed the estimate; the deadline can leave a
later method incomplete. Missing evaluations are reported as missing.

Training uses a fixed iteration budget, with no separate per-method time stop.
This gives completed methods equal game opportunities and optimizer steps. It
also prevents the old 40-minute cap from silently reducing the expanded workload.
`--hours` changes the deadline, not the amount of work. Selecting additional seeds
or methods requires proportionally more work.

Calibration on October 21, 2026 used the real configured model and 16 trajectories
on four distinct ALFWorld training games. A five-turn rollout took 58.99 seconds,
including 50.62 seconds generating responses. The subsequent update took 51.08
seconds. Peak reserved GPU memory was 6.04 GiB. Mixed rewards were injected for
the update timing to exercise nonzero gradients; this is a performance check,
not evidence of task learning. Raw results and the reproduction script are at
`runs/audit/overnight_full_profile.json` and `runs/audit/overnight_full_profile.py`.

Separate probes of microbatch size four and output-head chunk size 256 did not
improve update time (52.53 s and 59.45 s, respectively). They were not enabled.
The expanded preset retains microbatch size two and head chunk size 128.

The estimator combines that larger-batch probe with the completed original
overnight run's four-trajectory generation rate and evaluation costs. It counts
the actual supplied training trajectories, rather than assuming `gen_batch_size`
fills the batch, and accounts for partially filled evaluation batches. Its range
is a heuristic workload range, not a measured confidence interval or guarantee.

## Outputs and monitoring

New runs and reports use `runs/overnight_full/` and `results/overnight_full/`.
The previous overnight configuration and artifacts are preserved. Regenerate its
reports with `python run_benchmarks.py --tier overnight --report-only`.

Training logs each batch's start and announces the scoring/update phase. Rollouts
and evaluation print progress every five turns and at completion. Child Python
processes run unbuffered so progress is visible when redirecting output to a log.

The suite retains completed phases on rerun. Interrupted training itself still
starts fresh; this update does not implement optimizer/checkpoint resumption.
