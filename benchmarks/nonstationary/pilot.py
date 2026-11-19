"""Bounded development learning screen, with fresh initialization for every seed.

Candidate order is predeclared. The first candidate passing the paired-task gate
on every seed supplies shared hyperparameters, never its trained adapter.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time
import subprocess

from run_benchmarks import ROOT, launch_in_wsl
from .artifacts import write_json, read_jsonl
from .evidence import file_sha256, inspect_run


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initial-adapter")
    ap.add_argument("--out", default="runs/nonstationary/recovery_pilot")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--novelty", choices=["tool", "recipe", "resource"], default="tool")
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    ap.add_argument("--updates", type=int, default=16)
    ap.add_argument("--pre-change-updates", type=int, default=4)
    ap.add_argument("--learning-rates", type=float, nargs="+", default=[3e-6, 1e-5])
    ap.add_argument("--entropy-coefs", type=float, nargs="+", default=[.01])
    ap.add_argument("--curiosity-key", choices=["progress", "parsed"], default="progress")
    ap.add_argument("--hours", type=float, default=1.)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--tier", default="novelgym_recovery", help=argparse.SUPPRESS)
    return ap


def write_pilot_report(profile, root):
    from empg.config import Config, _build
    from .reporting import plot_curves, plot_behavior_curves
    root = Path(root)
    jobs, panels = [], {}
    text = ("# Development recovery pilot\n\n"
            "Same baseline-only initialization; constant curiosity; independent reproducible evaluation requests. "
            "These are development learnability checks on eight fixed tasks, not confirmation of a boost benefit.\n\n"
            "| Candidate | Seed | Pre-change counterfactual | Final changed-rule | Gain | Old-rule retention | Gate |\n"
            "|---|---:|---:|---:|---:|---:|---|\n")
    for row in profile["runs"]:
        cfg = _build(Config, row["config"])
        jobs.append({"cfg": cfg, "depth": cfg.env["recipe_depth"], "novelty": cfg.env["novelty"],
                     "schedule": cfg.nonstationary.schedule, "seed": cfg.seed, "treatment": row["candidate"]})
        panels[str(cfg.run_dir)] = {"metrics": read_jsonl(cfg.run_dir / "metrics.jsonl"),
                                    "probes": read_jsonl(cfg.run_dir / "adaptation_probes.jsonl")}
        checked = inspect_run(cfg)
        text += (f"| {row['candidate']} | {row['seed']} | {100 * checked['initial_success']:.1f}% | "
                 f"{100 * checked['final_success']:.1f}% | {100 * checked['gain']:+.1f} pp | "
                 f"{100 * checked['old_rule_retention']:.1f}% | {'pass' if checked['passed'] else 'fail'} |\n")
    (root / "comparison.md").write_text(text, encoding="utf-8")
    write_json(root / "learning_curves.json", panels)
    plot_curves(jobs, panels, root)
    plot_behavior_curves(jobs, panels, root)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.name == "nt":
        return launch_in_wsl(["--tier", "novelgym_recovery", *argv], entrypoint="run_novelgym_pilot.py")
    import math
    args = parser().parse_args(argv)
    if args.report_only:
        profile = json.loads((Path(args.out) / "learning_profile.json").read_text())
        write_pilot_report(profile, args.out)
        return 0
    if not args.initial_adapter:
        raise ValueError("--initial-adapter is required to run a learning pilot")
    if (not 2 <= args.depth <= 8 or args.updates < 4 or not 1 <= args.pre_change_updates < args.updates
            or len(set(args.seeds)) < 2 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0
            or not math.isfinite(args.hours) or args.hours <= 0
            or any(not math.isfinite(v) or v <= 0 for v in args.learning_rates)
            or any(not math.isfinite(v) or v < 0 for v in args.entropy_coefs)):
        raise ValueError("invalid learning pilot configuration")
    os.chdir(ROOT)
    from benchmarks.common import load
    from benchmarks.process import run_process
    from benchmarks.suite import _finished
    from empg.config import adapter_fingerprint
    from empg.envs.novelgym_env import nominal_solution_length
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.hours * 3600
    result = {"schema": 1, "complete": False, "passed": False, "initial_adapter_sha256": adapter_fingerprint(args.initial_adapter),
              "initial_adapter": str(Path(args.initial_adapter).resolve()), "options": vars(args), "runs": [],
              "selection_rule": "first predeclared candidate passing on every development seed; pilot adapters are discarded"}
    write_json(root / "learning_profile.json", result)
    for lr in args.learning_rates:
        for entropy in args.entropy_coefs:
            candidate = f"lr{lr:g}_entropy{entropy:g}"
            candidate_rows = []
            for seed in args.seeds:
                horizon = max(24, 2 * nominal_solution_length(args.depth, args.novelty, "novelty") + 8)
                overrides = [f"name={candidate}", f"out_dir={root / 'runs'}", f"env.recipe_depth={args.depth}",
                    f"env.novelty={args.novelty}", f"train.initial_adapter={Path(args.initial_adapter).resolve()}",
                    f"train.iterations={args.updates}", f"train.lr={lr}", f"train.entropy_coef={entropy}",
                    f"train.max_steps={horizon}", f"eval.max_steps={horizon}",
                    f"nonstationary.switch_iteration={args.pre_change_updates + 1}",
                    f"nonstationary.probe_every={max(1, args.updates // 4)}",
                    f"advantage.curiosity.key={args.curiosity_key}"]
                cfg = load("grpo", args.tier, overrides, seed)
                if not _finished(cfg):
                    try:
                        run_process([sys.executable, "-u", "-m", "benchmarks.train", "--config", "grpo", "--tier", args.tier,
                            "--seed", str(seed), *overrides], timeout=max(0., deadline - time.monotonic()))
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, KeyboardInterrupt) as error:
                        status = 124 if isinstance(error, subprocess.TimeoutExpired) else 130 if isinstance(error, KeyboardInterrupt) else error.returncode
                        result.update(status="deadline_reached" if status == 124 else "interrupted" if status == 130 else "phase_failed")
                        write_json(root / "learning_profile.json", result)
                        return status
                checked = inspect_run(cfg)
                artifacts = {str(cfg.run_dir / filename): file_sha256(cfg.run_dir / filename)
                    for filename in ("config.json", "metrics.jsonl", "adaptation_probes.jsonl", "training_complete.json")}
                row = {"candidate": candidate, "seed": seed, "gate": checked, "config": dataclasses.asdict(cfg), "artifacts": artifacts}
                candidate_rows.append(row)
                result["runs"].append(row)
                write_json(root / "learning_profile.json", result)
                print(json.dumps({"candidate": candidate, "seed": seed, "gate": checked}), flush=True)
            if all(row["gate"]["passed"] for row in candidate_rows):
                result.update(passed=True, selected=candidate, settings={"train.lr": lr, "train.entropy_coef": entropy,
                              "advantage.curiosity.key": args.curiosity_key})
                break
        if result["passed"]:
            break
    result["complete"] = True
    write_json(root / "learning_profile.json", result)
    write_pilot_report(result, root)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
