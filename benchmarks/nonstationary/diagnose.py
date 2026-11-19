"""Read-only policy diagnostics; no gradients, optimizer, or checkpoint selection."""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

from run_benchmarks import launch_in_wsl
from .artifacts import write_json
from .diagnostics import action_diagnostics


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.name == "nt":
        return launch_in_wsl(["--tier", "novelgym_recovery", *argv], entrypoint="run_novelgym_diagnose.py")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=0, help="optional changed-rule rollout for a shared-data key audit")
    ap.add_argument("--steps", type=int, default=26)
    ap.add_argument("--tier", default="novelgym_recovery", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not 2 <= args.depth <= 32 or not 1 <= args.tasks <= 64 or not 0 <= args.episodes <= 64 or args.steps < 1:
        ap.error("invalid diagnostics workload")
    import numpy as np
    from benchmarks.common import load
    from empg.config import config_dict
    from empg.curiosity import intrinsic_rewards
    from empg.diagnostics import behavior_metrics
    from empg.envs.base import make_benchmark
    from empg.policy import LLMPolicy
    from empg.rollout import run_episodes
    from empg.trainer import seed_everything
    cfg = load("grpo", args.tier, [f"env.recipe_depth={args.depth}", f"train.initial_adapter={args.adapter}"])
    seed_everything(17)
    policy = LLMPolicy(cfg.policy, seed=17)
    bench = make_benchmark(cfg.benchmark, **cfg.env)
    try:
        policy.load_adapter(args.adapter)
        tasks = bench.tasks("valid_seen")[:args.tasks]
        result = {"schema": 1, "config": config_dict(cfg), "updates": 0,
                  "decisions": [action_diagnostics(policy, bench, tasks, regime=r) for r in ("baseline", "novelty")]}
        if args.episodes:
            bench.intervention = cfg.nonstationary.intervention("novelty")
            sampled = [bench.tasks("valid_seen")[i] for i in range(args.episodes)]
            with policy.sampling(17):
                trajectories = run_episodes(policy, bench, sampled, list(range(len(sampled))), args.steps)
            audit = {}
            for key in ("parsed", "progress"):
                cur = dataclasses.replace(cfg.advantage.curiosity, key=key)
                values = intrinsic_rewards([t.key_seq(key) for t in trajectories],
                    [t.group for t in trajectories], [t.won for t in trajectories], cur)
                audit[key] = {"weighted_bonus_sum": float(values.sum()), "positive_steps": int((values > 0).sum()),
                              "centered_mean_abs": float(np.abs(values - values.mean()).mean())}
            result.update(shared_rollout_key_audit=audit, behavior=behavior_metrics(trajectories),
                          episodes=len(trajectories), success=float(np.mean([t.won for t in trajectories])),
                          note="Same on-policy trajectories, alternative observable novelty keys; no learning-effect claim")
        write_json(args.out, result)
        print(f"wrote {args.out}", flush=True)
    finally:
        bench.close()
        policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
