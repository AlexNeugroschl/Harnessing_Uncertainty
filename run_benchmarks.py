"""Run training, held-out evaluation, and comparison reports with one command.

    python run_benchmarks.py --estimate-only
    python run_benchmarks.py
    python run_benchmarks.py --tier pilot --arms grpo grpo_cur_episodic --seeds 0

On Windows, forwards to Ubuntu-24.04 and ~/venvs/empg in WSL. Set
EMPG_WSL_DISTRO / EMPG_WSL_PYTHON to use another distro / Linux Python path.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def launch_in_wsl(argv):
    root = PureWindowsPath(str(ROOT))
    if not root.drive or root.drive.startswith("\\\\"):
        raise ValueError("Put the repository on a local Windows drive or run inside WSL.")
    linux_root = "/mnt/" + root.drive[0].lower() + "/" + "/".join(root.parts[1:])
    # Paths and user arguments remain separate argv items, never shell source.
    python = os.environ.get("EMPG_WSL_PYTHON")
    invocation = ([python, "run_benchmarks.py", *argv] if python else
                  ["bash", "-lc", 'exec "$HOME/venvs/empg/bin/python" "$@"',
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


def estimate(cfg, *, train=True, evaluate=True):
    """Workload extrapolation, not a statistical interval or effectiveness claim.

    RTX 5060 Laptop 8GB, Qwen2.5-1.5B LoRA/HF, chunked head, four CPU workers:
    24 real response steps: generation 19.65s, warm update 15.90s, mean 112.6 tokens.
    16 distinct environments: warm reset 12.72s, 80 scripted steps 2.51s.
    See docs/research_audit.md. Short profiles cannot capture long-context costs.
    """
    tokens = max(0.1, min(1., cfg.policy.max_new_tokens / 112.6))
    if cfg.policy.max_new_tokens > 256:
        tokens *= cfg.policy.max_new_tokens / 256
    workers = int(cfg.env.get("workers", 0))
    env_scale = 1. if workers >= 4 else (2.82 if workers <= 1 else 4. / workers)
    episodes = cfg.train.games_per_iter * cfg.train.group_size
    steps = episodes * cfg.train.max_steps
    update_rate = 15.90 / 24 * cfg.train.ppo_epochs
    if not cfg.policy.chunked_head:
        update_rate = 43.61 / 24 * cfg.train.ppo_epochs
    batch_seconds = (steps * ((19.65 / 24 + update_rate) * tokens + 2.51 / 80 * env_scale)
                     + episodes * 12.72 / 16 * env_scale)
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
    eval_steps = task_count * cfg.eval.max_steps if evaluate else 0
    evaluation = (eval_steps * (19.65 / 24 * tokens + 2.51 / 80 * env_scale)
                  + task_count * 12.72 / 16 * env_scale) if evaluate else 0.
    central = training + evaluation + 30 * (int(train) + int(evaluate))
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
    ap.add_argument("--tier", choices=tiers(), default="full")
    ap.add_argument("--arms", nargs="+", choices=list(arms()), default=list(arms()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--runs", help="run directory (default runs/<tier>)")
    ap.add_argument("--out", help="report directory (default results/<tier>)")
    ap.add_argument("--baseline", default="grpo", choices=list(arms()))
    ap.add_argument("--skip-base", action="store_true")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--estimate-only", action="store_true", help="print remaining workload; do not train")
    modes.add_argument("--report-only", action="store_true", help="regenerate tables from existing runs")
    ap.add_argument("overrides", nargs="*", help="shared dotted config overrides; precede with -- after --seeds/--arms")
    args = ap.parse_args(argv)
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
    report_cmd = [sys.executable, "-m", "benchmarks.report", "--runs", str(runs),
                  "--split", split, "--out", str(out), "--baseline", args.baseline,
                  "--arms", *args.arms, "--seeds", *map(str, args.seeds)]
    if args.report_only:
        execute(report_cmd)
        print(f"Reports: {(out / 'comparison.md').resolve()}")
        return 0

    # Validate every destination before spending time on the first new run.
    remaining = []
    if not args.skip_base and not _matching_eval(eval_path(base, base=True), base):
        remaining.append({"arm": "base", "seed": None, **estimate(base, train=False)})
    for cfg in configs:
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
    print("Calibration: RTX 5060 Laptop 8GB, Qwen2.5-1.5B LoRA, HF, four workers. "
          "Range models early finishes / timing variation; it is not a measured confidence interval. "
          "Longer prompts, response lengths, other models/backends or hardware can exceed it.")
    print(f"Runs: {runs.resolve()}\nReports: {out.resolve()}", flush=True)
    if args.estimate_only:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "tier": args.tier,
                "arms": args.arms, "seeds": args.seeds, "baseline": args.baseline,
                "runs": str(runs.resolve()), "split": split, "remaining_estimates": remaining,
                "estimate": totals, "configs": [dataclasses.asdict(c) for c in configs],
                "status": "running"}
    manifest_path = out / "benchmark_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    suite_cmd = [sys.executable, "-m", "benchmarks.suite", "--tier", args.tier,
                 "--arms", *args.arms, "--seeds", *map(str, args.seeds)]
    if args.skip_base:
        suite_cmd.append("--skip-base")
    suite_cmd.extend(["--", *overrides])
    start = time.perf_counter()
    try:
        execute(suite_cmd)
        manifest["status"] = "complete"
    except (subprocess.CalledProcessError, KeyboardInterrupt):
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
    print(f"Finished in {duration(manifest['elapsed_seconds'])}. Tables: {(out / 'comparison.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
