"""Run the benchmark suite: base model + arms x seeds, train then evaluate.

    python -m benchmarks.suite --tier small --phase screen            # plan Phase 2
    python -m benchmarks.suite --tier reduced --arms grpo empg --seeds 0 1 2
    python -m benchmarks.suite --tier small --arms grpo_cur_group advantage.curiosity.gate=no_success_signal name=gated

Finished runs (eval json present) are skipped, so the suite can be resumed.
Each run is a subprocess so GPU memory is fully released between runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from benchmarks.common import BASE, arms, load, tiers
from benchmarks.evaluate import eval_path

# Named arm sets from docs/curiosity_research_plan.md.
PHASES = {
    "paper": ["grpo", "empg", "empg_g_only", "empg_f_only"],
    "screen": ["grpo", "grpo_memory", "grpo_cur_episodic", "grpo_cur_group", "grpo_cur_oracle",
               "empg", "empg_cur_episodic", "empg_seek_uncertainty"],
}


def run(module: str, config: str, tier, seed: int, overrides: list[str], *extra: str):
    cmd = [sys.executable, "-m", module, "--config", str(config), "--seed", str(seed),
           *(["--tier", tier] if tier else []), *extra, *overrides]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=list(arms()))
    ap.add_argument("--phase", choices=list(PHASES), help="named set of arms")
    ap.add_argument("--tier", choices=tiers())
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("overrides", nargs="*", help="dotted overrides applied to every run")
    args = ap.parse_args()
    selected = args.arms or PHASES.get(args.phase) or PHASES["paper"]

    if not args.skip_base:
        cfg = load(BASE, args.tier, args.overrides)
        if not eval_path(cfg, base=True).exists():
            run("benchmarks.evaluate", BASE, args.tier, 0, args.overrides, "--base")

    for seed in args.seeds:
        for arm in selected:
            cfg = load(arm, args.tier, args.overrides, seed)
            if eval_path(cfg).exists():
                print(f"skip {cfg.name} s{seed} (done)")
                continue
            if not (cfg.run_dir / "adapter").exists() or not _finished(cfg):
                run("benchmarks.train", arm, args.tier, seed, args.overrides)
            run("benchmarks.evaluate", arm, args.tier, seed, args.overrides)


def _finished(cfg) -> bool:
    m = cfg.run_dir / "metrics.jsonl"
    return m.exists() and sum(1 for _ in m.open()) >= cfg.train.iterations


if __name__ == "__main__":
    main()
