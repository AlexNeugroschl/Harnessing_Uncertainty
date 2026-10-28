"""Run training, held-out evaluation, and comparison reports with one command.

    python run_benchmarks.py --estimate-only
    python run_benchmarks.py
    python run_benchmarks.py --tier pilot --arms grpo grpo_cur_episodic --seeds 0

On Windows, forwards to Ubuntu-24.04, selecting ~/venvs/empg-vllm for fast tiers
and ~/venvs/empg for HF tiers. Set
EMPG_WSL_DISTRO / EMPG_WSL_PYTHON to use another distro / Linux Python path.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DEFAULT_TIER = "paper_optimized"
FAST_TIERS = ("paper_lora_fast", "paper_overnight", "paper_optimized", "paper_replication", "paper_lr_pilot")

# Comment/uncomment one line per method to choose the default experiment.
# --arms overrides this list. The untrained base model is evaluated separately
# unless --skip-base is passed. "grpo" is the regular trained baseline.
# These are research priorities, not an empirically established performance ranking.
ENABLED_METHODS = [
    "grpo",                   # Regular GRPO: outcome-reward baseline.
    "empg",                   # EMPG: entropy scaling + future clarity.
    # "grpo_cur_group",       # Optional extension after the GRPO/EMPG comparison.
    # "empg_cur_episodic",    # Optional extension after the GRPO/EMPG comparison.
    # "grpo_cur_episodic",     # Per-episode novelty; next control for novelty scope.
    # "grpo_memory",           # Visited-location prompt; control for forgetting.
    # "grpo_cur_oracle",       # Privileged physical-state novelty diagnostic.
    # "empg_g_only",           # Ablation: entropy gradient scaling only.
    # "empg_f_only",           # Ablation: future-clarity bonus only.
    # "empg_seek_uncertainty", # Sign-flipped clarity; exploratory diagnostic.
]


def launch_in_wsl(argv):
    root = PureWindowsPath(str(ROOT))
    if not root.drive or root.drive.startswith("\\\\"):
        raise ValueError("Put the repository on a local Windows drive or run inside WSL.")
    linux_root = "/mnt/" + root.drive[0].lower() + "/" + "/".join(root.parts[1:])
    # Paths and user arguments remain separate argv items, never shell source.
    python = os.environ.get("EMPG_WSL_PYTHON")
    tier = DEFAULT_TIER
    for i, arg in enumerate(argv):
        if arg == "--tier" and i + 1 < len(argv):
            tier = argv[i + 1]
        elif arg.startswith("--tier="):
            tier = arg.split("=", 1)[1]
    backend = "vllm" if tier in FAST_TIERS else "hf"
    for arg in argv:
        if arg.startswith("policy.backend="):
            backend = arg.split("=", 1)[1]
    venv = "empg-vllm" if backend == "vllm" else "empg"
    invocation = ([python, "run_benchmarks.py", *argv] if python else
                  ["bash", "-lc", f'exec "$HOME/venvs/{venv}/bin/python" "$@"',
                   "run_benchmarks", "run_benchmarks.py", *argv])
    cmd = ["wsl.exe", "--distribution", os.environ.get("EMPG_WSL_DISTRO", "Ubuntu-24.04"),
           "--cd", linux_root, "--exec", *invocation]
    return subprocess.call(cmd)


def duration(seconds):
    if seconds >= 86400:
        return f"{seconds / 86400:.1f} days"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 60:.1f} minutes"


def optimized_calibration(cfg):
    p = cfg.policy
    return (p.model_name == "Qwen/Qwen2.5-1.5B-Instruct" and p.dtype == "bfloat16"
            and p.backend == "vllm" and not p.vllm_enforce_eager and p.amp and p.lora_r == 16
            and p.chunked_head and p.head_backend == "analytic" and p.head_chunk_size == 256
            and p.gradient_checkpointing and p.checkpoint_min_tokens == 0
            and p.micro_batch_size == p.score_batch_size == 2 and p.micro_batch_tokens == 2560
            and p.max_new_tokens == 512 and p.max_prompt_tokens == 2048
            and cfg.train.entropy_coef == .001 and cfg.train.kl_coef == .01)


def estimate(cfg, *, train=True, evaluate=True):
    """Workload extrapolation, not a statistical interval or effectiveness claim.

    RTX 5060 Laptop 8GB, Qwen2.5-1.5B LoRA/HF, chunked head, four CPU workers:
    Expanded-batch probe: 80 response steps, generation 50.62s, update 51.08s,
    mean 92.45 tokens; 16 trajectories on four distinct games. Generation scales
    conservatively between this probe and the completed four-trajectory run
    (1.05s/step). Evaluation uses the completed overnight suite's ~0.65s/step
    generation cost at 24 environments. See docs/overnight_full.md and
    runs/audit/overnight_full_profile.json. These are workload extrapolations.
    """
    tokens = max(0.1, min(1., cfg.policy.max_new_tokens / 92.45))
    if cfg.policy.max_new_tokens > 256:
        # Roughly 3% of old eval responses reached 256; a larger ceiling does
        # not double every response. Charge those tails for the whole extension.
        tokens *= 1 + .03 * (cfg.policy.max_new_tokens - 256) / 92.45
    compiled = cfg.policy.backend == "vllm" and not cfg.policy.vllm_enforce_eager
    decode_speedup = 3. if compiled else 1.  # short probes: 2.4-4.5x, not suite timings
    full_horizon_calibration = (compiled and cfg.policy.amp and cfg.policy.chunked_head
                               and cfg.policy.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
                               and cfg.policy.micro_batch_size == 2
                               and cfg.train.entropy_coef == .001)
    if full_horizon_calibration:
        # Real 16-trajectory, 50-turn batch: 800 responses, mean length 92.86,
        # decode 232.51s, update 551.64s. Includes exact entropy regularization.
        # The 512-token cap is already represented, so do not charge it twice.
        tokens = max(.1, min(1., cfg.policy.max_new_tokens / 92.85875))
        if cfg.policy.max_new_tokens > 512:
            tokens *= 1 + .03 * (cfg.policy.max_new_tokens - 512) / 92.85875
        decode_speedup = (50.616117672 / 80) / (232.511240086 / 800)
    workers = int(cfg.env.get("workers", 0))
    env_scale = 1. if workers >= 4 else (2.82 if workers <= 1 else 4. / workers)
    episodes = cfg.train.games_per_iter * cfg.train.group_size
    steps = episodes * cfg.train.max_steps
    # gen_batch_size is a ceiling: it cannot fill missing trajectory slots.
    active_batch = min(episodes, cfg.policy.gen_batch_size, 16)
    generation_rate = (50.616117672 / 80) * (16 / max(1, active_batch)) ** .365
    generation_rate /= decode_speedup
    update_rate = 51.082308316 / 80 * cfg.train.ppo_epochs
    if cfg.policy.amp:
        update_rate /= 1.17
    if cfg.train.entropy_coef:
        update_rate *= 1.1
    if full_horizon_calibration:
        update_rate = 551.638224246 / 800 * cfg.train.ppo_epochs
    if not cfg.policy.chunked_head:
        update_rate = 43.61 / 24 * cfg.train.ppo_epochs
    observed = optimized_calibration(cfg)
    if observed:
        # Completed 5.1h suite: 18,920 response rows, 9,122.25s scoring/update,
        # 4,291.36s decoding. Apply only the measured shared-rollout head ratios:
        # 54.28 -> 51.05s scoring; 92.92 -> 86.39s backward/optimizer (192 rows).
        # This is a workload extrapolation, not a measured optimized full suite.
        tokens = 1.
        generation_rate = (4291.36 / 18920) * (16 / max(1, active_batch)) ** .365
        score_share = 3172.31 / 9122.25
        score_ratio = 51.049164766 / 54.276215761
        grad_ratio = (137.437746908 - 51.049164766) / (147.194682376 - 54.276215761)
        update_rate = (9122.25 / 18920) * (
            score_share * score_ratio + (1 - score_share) * grad_ratio * cfg.train.ppo_epochs)
        batch_seconds = (steps * (generation_rate + update_rate + .011 * env_scale)
                         + episodes * .235 * env_scale + 10.075)
    else:
        batch_seconds = (steps * ((generation_rate + update_rate) * tokens + 2.51 / 80 * env_scale)
                         + episodes * 12.72 / 16 * env_scale)
    if cfg.policy.backend == "vllm" and not observed:
        batch_seconds += 20.  # weight transfers, sleep/wake, adapter publication
    batches = cfg.train.iterations
    if cfg.train.max_env_steps:
        batches = min(batches, max(1, (cfg.train.max_env_steps + steps - 1) // steps))
    training = batches * batch_seconds if train else 0.
    if train and cfg.train.max_seconds:
        training = min(training, cfg.train.max_seconds + batch_seconds)
    # Actual task count if the data is installed; no GPU/model load.
    task_count = cfg.eval.max_tasks
    if evaluate:
        from empg.envs.base import make_benchmark
        bench = make_benchmark(cfg.benchmark, **cfg.env)
        try:
            available = len(bench.tasks(cfg.eval.split))
            task_count = min(task_count, available) if task_count > 0 else available
        finally:
            if hasattr(bench, "close"):
                bench.close()
    evaluation = 0.
    if evaluate:
        # Account for the partially filled final environment/generation batch.
        for start in range(0, task_count, cfg.eval.batch_envs):
            chunk = min(cfg.eval.batch_envs, task_count - start)
            for offset in range(0, chunk, cfg.policy.gen_batch_size):
                n = min(cfg.policy.gen_batch_size, chunk - offset)
                rate = .65 * (24 / min(n, 32)) ** .5 / decode_speedup
                evaluation += n * cfg.eval.max_steps * (
                    rate * tokens + 2.51 / 80 * env_scale)
        evaluation += task_count * 12.72 / 16 * env_scale
        if observed:
            # Completed suite's non-training time, allowing for process startup,
            # was about 65% of the earlier evaluation extrapolation.
            evaluation *= .65
    startup = 75 if cfg.policy.backend == "vllm" else 30
    phases = 1 if train and evaluate and cfg.eval.sampling_seed is not None else int(train) + int(evaluate)
    central = training + evaluation + startup * phases
    return {"seconds": central, "low_seconds": central * .5, "high_seconds": central * 1.5,
            "training_seconds": training, "evaluation_seconds": evaluation,
            "eval_tasks": task_count, "training_steps_at_caps": batches * steps if train else 0}


def execute(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main(argv=None):
    if os.name == "nt":
        return launch_in_wsl(sys.argv[1:] if argv is None else argv)

    from benchmarks.common import BASE, arms, load, tiers
    from benchmarks.evaluate import eval_path
    from benchmarks.suite import _finished, _matching_eval

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=tiers(), default=DEFAULT_TIER)
    ap.add_argument("--arms", nargs="+", choices=list(arms()),
                    help="override the uncommented ENABLED_METHODS lines at the top of this file")
    ap.add_argument("--seeds", nargs="+", type=int,
                    help="default: 0 for overnight/pilot tiers; 0 1 2 for replication/full/reduced/paper_lora[_fast]")
    ap.add_argument("--hours", type=float,
                    help="training/evaluation deadline; default 8.5 for paper_optimized/paper_overnight/overnight_full, 8 for overnight, 0 otherwise")
    ap.add_argument("--runs", help="run directory (default runs/<tier>)")
    ap.add_argument("--out", help="report directory (default results/<tier>)")
    ap.add_argument("--baseline", default="grpo", choices=list(arms()))
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--include-runs", nargs="+",
                    help="include compatible completed seeds from these roots in the report; training stays in --runs")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--estimate-only", action="store_true", help="print remaining workload; do not train")
    modes.add_argument("--report-only", action="store_true", help="regenerate tables from existing runs")
    ap.add_argument("overrides", nargs="*", help="shared dotted config overrides; precede with -- after --seeds/--arms")
    args = ap.parse_args(argv)
    if args.seeds is None:
        args.seeds = [0, 1, 2] if args.tier in ("full", "reduced", "paper_lora", "paper_lora_fast", "paper_replication") else [0]
    if args.hours is None:
        args.hours = {"paper_optimized": 8.5, "paper_overnight": 8.5, "overnight_full": 8.5, "overnight": 8.}.get(args.tier, 0.)
    if not math.isfinite(args.hours) or args.hours < 0:
        ap.error("--hours must be finite and nonnegative (0 disables the deadline)")
    if args.arms is None:
        args.arms = list(ENABLED_METHODS)
    if not args.arms:
        ap.error("no methods enabled; uncomment an ENABLED_METHODS line or pass --arms")
    unknown = set(args.arms) - set(arms())
    if unknown:
        ap.error(f"unknown enabled methods: {', '.join(sorted(unknown))}")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        ap.error("arms and seeds must be unique")
    if any(o.split("=", 1)[0] in ("out_dir", "seed", "name", "eval.split") for o in args.overrides):
        ap.error("use --runs / --seeds / --tier to set output, seeds, and split; shared name= is ambiguous")
    # All commands use the repository as cwd, even when launched elsewhere.
    os.chdir(ROOT)
    runs = Path(args.runs or f"runs/{args.tier}")
    out = Path(args.out or f"results/{args.tier}")
    overrides = [*args.overrides, f"out_dir={runs}"]
    configs = [load(arm, args.tier, overrides, seed) for seed in args.seeds for arm in args.arms]
    base = load(BASE, args.tier, overrides)
    split = configs[0].eval.split
    report_seeds = list(args.seeds)
    included_pairs = set()
    if args.include_runs:
        from benchmarks.report import run_dirs, validate_combined_runs
        included = [Path(p) for p in args.include_runs]
        if any(not p.is_dir() for p in included):
            ap.error("every --include-runs directory must exist")
        sources = validate_combined_runs([runs, *included], split, arms=args.arms, planned=configs)
        if "base" in sources:
            args.skip_base = True
            print(f"Reuse zero-shot evaluation: {sources['base']}", flush=True)
        report_seeds = sorted(set(report_seeds) | {
            int(d.name.rsplit("_s", 1)[1]) for root in included for d, arm in run_dirs(root)
            if arm in args.arms and (d / f"eval_{split}.json").exists()})
        included_pairs = {(arm, int(d.name.rsplit("_s", 1)[1]))
                          for d, arm in run_dirs(included, arms=args.arms, split=split)
                          if arm != "base" and (d / f"eval_{split}.json").exists()}
    report_cmd = [sys.executable, "-m", "benchmarks.report", "--runs", str(runs),
                  "--split", split, "--out", str(out), "--baseline", args.baseline,
                  "--arms", *args.arms, "--seeds", *map(str, report_seeds)]
    if args.include_runs:
        report_cmd.extend(["--include-runs", *args.include_runs])
    if args.report_only:
        execute(report_cmd)
        print(f"Reports: {(out / 'comparison.md').resolve()}")
        return 0

    # Validate every destination before spending time on the first new run.
    remaining = []
    if not args.skip_base and not _matching_eval(eval_path(base, base=True), base):
        remaining.append({"arm": "base", "seed": None, **estimate(base, train=False)})
    for cfg in configs:
        if (cfg.name, cfg.seed) in included_pairs:
            continue
        if _matching_eval(eval_path(cfg), cfg):
            continue
        finished = _finished(cfg)
        needs_train = not finished or not (cfg.run_dir / "adapter").exists()
        remaining.append({"arm": cfg.name, "seed": cfg.seed, **estimate(cfg, train=needs_train)})
    totals = {k: sum(r[k] for r in remaining) for k in ("seconds", "low_seconds", "high_seconds")}
    # Subprocess imports and report generation still take time when all runs are done.
    totals["seconds"] += 45
    totals["low_seconds"] += 22.5
    totals["high_seconds"] += 67.5
    print(f"\n{args.tier}: {len(args.arms)} methods x {len(args.seeds)} seeds = {len(configs)} trained runs"
          f"; {len(remaining)} evaluations remaining (including base if requested).", flush=True)
    print(f"Estimated remaining runtime: {duration(totals['low_seconds'])} to "
          f"{duration(totals['high_seconds'])}; full-horizon extrapolation {duration(totals['seconds'])}.")
    if args.hours:
        print(f"Training/evaluation deadline: {args.hours:g} hours, then cleanup and reports. "
              "If reached, unfinished methods remain explicitly incomplete.", flush=True)
        if totals["seconds"] > args.hours * 3600:
            print("The central estimate exceeds the deadline; allow more hours or reduce the workload to finish all runs.")
        elif totals["high_seconds"] > args.hours * 3600:
            print("Slower runs can reach the deadline; completed results are preserved and unfinished runs are marked incomplete.")
    print("Calibration: RTX 5060 Laptop 8GB, Qwen2.5-1.5B LoRA, HF, four workers; "
          "real 16-trajectory probe and completed overnight run. "
          "Range models early finishes / timing variation; it is not a measured confidence interval. "
          "Longer prompts, response lengths, other models/backends or hardware can exceed it.")
    if any(c.policy.backend == "vllm" for c in configs):
        if any(optimized_calibration(c) for c in configs):
            print("Optimized estimate: completed 5.1-hour suite plus the shared-rollout analytic-head probe "
                  "(147.2s to 137.4s per 192-response update). Revised forecasts are not measured "
                  "whole-suite speedups; task/context variation can still reach the deadline.", flush=True)
        else:
            print("vLLM estimate: the corrected 1.5B/AMP/chunked path uses the real 800-response, "
                  "50-turn probe (232.5s decoding, 551.6s updating), plus transfer/startup costs. "
                  "Other settings use the earlier component extrapolation. "
                  "Evaluation and whole-suite runtime remain extrapolations.", flush=True)
    print(f"Runs: {runs.resolve()}\nReports: {out.resolve()}", flush=True)
    if args.tier in ("overnight_full", "paper_overnight", "paper_optimized", "paper_replication"):
        cfg = configs[0]
        steps_per_iteration = (math.ceil(cfg.train.games_per_iter * cfg.train.group_size *
                                       cfg.train.max_steps / cfg.train.mini_batch_size)
                               if cfg.train.mini_batch_size else cfg.train.mini_batches)
        print(f"Expanded plan per method/seed: {cfg.train.iterations} batches x "
              f"{cfg.train.games_per_iter} games x {cfg.train.group_size} siblings; "
              f"up to {cfg.train.iterations * steps_per_iteration * cfg.train.ppo_epochs} optimizer steps; "
              f"{cfg.eval.max_tasks or 'all'} "
              "evaluation tasks. Fixed workload across methods; runtime can vary.", flush=True)
    if args.estimate_only:
        return 0

    runs.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "tier": args.tier,
                "hours": args.hours,
                "arms": args.arms, "seeds": args.seeds, "baseline": args.baseline,
                "include_runs": args.include_runs or [], "report_seeds": report_seeds,
                "included_seed_runs": sorted(included_pairs),
                "runs": str(runs.resolve()), "split": split, "remaining_estimates": remaining,
                "estimate": totals, "configs": [dataclasses.asdict(c) for c in configs],
                "status": "running"}
    manifest_path = out / "benchmark_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    suite_cmd = [sys.executable, "-u", "-m", "benchmarks.suite", "--tier", args.tier,
                 "--arms", *args.arms, "--seeds", *map(str, args.seeds)]
    if args.skip_base:
        suite_cmd.append("--skip-base")
    if args.include_runs:
        suite_cmd.extend(["--include-runs", *args.include_runs])
    if args.hours:
        suite_cmd.extend(["--max-seconds", str(args.hours * 3600)])
    suite_cmd.extend(["--", *overrides])
    start = time.perf_counter()
    try:
        execute(suite_cmd)
        manifest["status"] = "complete"
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 124:
            manifest["status"] = "deadline_reached"
        else:
            manifest["status"] = "interrupted_or_failed"
            raise
    except KeyboardInterrupt:
        manifest["status"] = "interrupted_or_failed"
        raise
    finally:
        manifest["suite_seconds"] = time.perf_counter() - start
        # Preserve completed results if a later method fails or Ctrl+C stops the suite.
        try:
            if runs.exists():
                execute(report_cmd)
                manifest["report_status"] = "complete"
            else:
                manifest["report_status"] = "no_runs"
        except (subprocess.CalledProcessError, KeyboardInterrupt) as exc:
            manifest["report_status"] = "failed"
            if manifest["status"] == "complete":
                manifest["status"] = "report_failed"
                raise
            print(f"Partial report failed: {exc}", file=sys.stderr)
        finally:
            manifest["elapsed_seconds"] = time.perf_counter() - start
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outcome = "Deadline reached; partial results" if manifest["status"] == "deadline_reached" else "Finished"
    print(f"{outcome} in {duration(manifest['elapsed_seconds'])}. Tables: {(out / 'comparison.md').resolve()}")
    return 124 if manifest["status"] == "deadline_reached" else 0


if __name__ == "__main__":
    raise SystemExit(main())
