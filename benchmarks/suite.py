"""Run the benchmark suite: base model + arms x seeds, train then evaluate.

    python -m benchmarks.suite --tier small --phase screen            # plan Phase 2
    python -m benchmarks.suite --tier reduced --arms grpo empg --seeds 0 1 2
    python -m benchmarks.suite --tier small --arms grpo_cur_group advantage.curiosity.gate=no_success_signal name=gated

Finished runs (eval json present) are skipped, so the suite can be resumed.
Each run is a subprocess so GPU memory is fully released between runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.common import BASE, arms, load, tiers
from benchmarks.evaluate import eval_path
from benchmarks.process import run_process
from empg.config import config_fingerprint, config_matches

# Named arm sets from docs/curiosity_research_plan.md.
PHASES = {
    "paper": ["grpo", "empg", "empg_g_only", "empg_f_only"],
    "screen": ["grpo", "grpo_memory", "grpo_cur_episodic", "grpo_cur_group", "grpo_cur_oracle",
               "empg", "empg_cur_episodic", "empg_seek_uncertainty"],
}


def run(module: str, config: str, tier, seed: int, overrides: list[str], *extra: str, deadline=None):
    cmd = [sys.executable, "-u", "-m", module, "--config", str(config), "--seed", str(seed),
           *(["--tier", tier] if tier else []), *extra, *overrides]
    print("$", " ".join(cmd), flush=True)
    timeout = max(0., deadline - time.monotonic()) if deadline is not None else None
    run_process(cmd, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=list(arms()))
    ap.add_argument("--phase", choices=list(PHASES), help="named set of arms")
    ap.add_argument("--tier", choices=tiers())
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--include-runs", nargs="+", help="reuse compatible completed seeds/base from these roots")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="wall-clock limit across training/evaluation; 0 disables it")
    ap.add_argument("overrides", nargs="*", help="dotted overrides applied to every run")
    args = ap.parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds < 0:
        ap.error("--max-seconds must be finite and nonnegative")
    selected = args.arms or PHASES.get(args.phase) or PHASES["paper"]
    names = [load(arm, args.tier, args.overrides).name for arm in selected]
    if len(set(names)) != len(names):
        ap.error("selected arms resolve to the same run name; remove the shared name= override")
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else None
    try:
        run_suite(args, selected, deadline)
    except subprocess.TimeoutExpired:
        print("Suite deadline reached. Completed evaluations and training logs are preserved.", flush=True)
        return 124
    return 0


def run_suite(args, selected, deadline=None):
    provided, reuse_base = set(), False
    if getattr(args, "include_runs", None):
        from benchmarks.report import run_dirs, validate_combined_runs
        roots = [Path(p) for p in args.include_runs]
        if any(not p.is_dir() for p in roots):
            raise ValueError("every --include-runs directory must exist")
        planned = [load(arm, args.tier, args.overrides, seed) for seed in args.seeds for arm in selected]
        sources = validate_combined_runs([Path(planned[0].out_dir), *roots], planned[0].eval.split,
                                        arms=selected, planned=planned)
        reuse_base = "base" in sources
        provided = {(arm, int(d.name.rsplit("_s", 1)[1]))
                    for d, arm in run_dirs(roots, arms=selected, split=planned[0].eval.split)
                    if arm != "base" and (d / f"eval_{planned[0].eval.split}.json").exists()}
    if not args.skip_base and not reuse_base:
        cfg = load(BASE, args.tier, args.overrides)
        if not _matching_eval(eval_path(cfg, base=True), cfg):
            run("benchmarks.evaluate", BASE, args.tier, 0, args.overrides, "--base", deadline=deadline)

    for seed in args.seeds:
        for arm in selected:
            cfg = load(arm, args.tier, args.overrides, seed)
            if (arm, seed) in provided:
                print(f"skip {arm} s{seed} (compatible included run)")
                continue
            if _matching_eval(eval_path(cfg), cfg):
                print(f"skip {cfg.name} s{seed} (done)")
                continue
            finished = _finished(cfg)  # validate config even if a partial run has no adapter
            if not (cfg.run_dir / "adapter").exists() or not finished:
                # Explicit evaluation RNG makes engine reuse independent of how
                # many samples training consumed. Legacy runs keep separate phases.
                extra = ("--evaluate",) if cfg.eval.sampling_seed is not None else ()
                run("benchmarks.train", arm, args.tier, seed, args.overrides, *extra, deadline=deadline)
                if extra:
                    continue
            run("benchmarks.evaluate", arm, args.tier, seed, args.overrides, deadline=deadline)


def _finished(cfg) -> bool:
    saved = cfg.run_dir / "config.json"
    if saved.exists() and not config_matches(json.loads(saved.read_text()), cfg):
        raise ValueError(f"{cfg.run_dir} contains a different configuration. "
                         "Choose a new name= or out_dir= to preserve that experiment.")
    completed = cfg.run_dir / "training_complete.json"
    if completed.exists():
        return json.loads(completed.read_text()).get("config_fingerprint") == config_fingerprint(cfg)
    if cfg.train.max_seconds or cfg.train.reward_mode == "step":
        # A killed budgeted run can have all metric rows but an unfinished adapter save.
        # Corrected protocols always require the marker written after the final save.
        return False
    m = cfg.run_dir / "metrics.jsonl"
    return m.exists() and sum(1 for _ in m.open()) >= cfg.train.iterations


def _matching_eval(path, cfg):
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    if saved.get("config_fingerprint") != config_fingerprint(cfg):
        raise ValueError(f"{path} has an old or different experiment configuration. "
                         "Choose a new out_dir= (or name= for an arm) before resuming.")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
