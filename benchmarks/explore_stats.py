"""Plan Phase 0: measure exploration behaviour without training.

Rolls out a policy (the base model by default) on training games, G samples
each, and reports revisit rate, group coverage, P(group has >= 1 success), and
whether revisiting predicts failure. This is the go/no-go gate: if the policy
rarely revisits and groups are already diverse, curiosity has little room.

    python -m benchmarks.explore_stats --config grpo --tier small --games 24
    python -m benchmarks.explore_stats --config grpo --tier small --adapter runs/small/grpo_s0/adapter
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from benchmarks.common import add_args, load
from empg.curiosity import episodic_counts, exploration_stats
from empg.envs.base import make_benchmark
from empg.policy import LLMPolicy
from empg.rollout import length_stats, response_lengths, run_episodes
from empg.trainer import seed_everything

KEYS = ("parsed", "physical", "knowledge", "obs")


def trajectory_record(tr, bench) -> dict:
    rec = {"task": tr.task, "group": tr.group, "task_type": bench.task_type(tr.task),
           "won": tr.won, "length": len(tr.steps), "invalid": tr.n_invalid,
           "truncated": sum(s.truncated for s in tr.steps),
           "response_tokens": sum(len(s.response_ids) for s in tr.steps)}
    for key in KEYS:
        seq = tr.key_seq(key)
        counts = episodic_counts([seq])[0]
        rec[f"unique_{key}"] = len(set(seq))
        rec[f"revisit_{key}"] = float((counts > 1).mean()) if len(counts) else 0.0
    return rec


def summarize(records: list[dict], trajs) -> dict:
    won = np.array([r["won"] for r in records])
    out = {"n_trajectories": len(records), "success": float(won.mean())}
    groups = [tr.group for tr in trajs]
    for key in KEYS:
        out[key] = exploration_stats([tr.key_seq(key) for tr in trajs], groups, won)
    # Does revisiting predict failure? Success rate by revisit-rate tercile (parsed key).
    rev = np.array([r["revisit_parsed"] for r in records])
    edges = np.quantile(rev, [1 / 3, 2 / 3])
    bins = np.digitize(rev, edges)
    out["success_by_revisit_tercile"] = {
        name: (float(won[bins == b].mean()) if (bins == b).any() else None)
        for b, name in enumerate(["low", "mid", "high"])}
    if won.any() and (~won).any():
        out["revisit_won_vs_lost"] = [float(rev[won].mean()), float(rev[~won].mean())]
    by_type = {}
    for r in records:
        by_type.setdefault(r["task_type"], []).append(r)
    out["by_task_type"] = {
        t: {"n": len(rs), "success": float(np.mean([r["won"] for r in rs])),
            "revisit_parsed": float(np.mean([r["revisit_parsed"] for r in rs])),
            "length": float(np.mean([r["length"] for r in rs]))}
        for t, rs in sorted(by_type.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--games", type=int, default=24, help="training games to sample")
    ap.add_argument("--group", type=int, default=8, help="rollouts per game")
    ap.add_argument("--games-per-batch", type=int, default=4)
    ap.add_argument("--adapter", help="LoRA adapter dir (default: untrained base model)")
    ap.add_argument("--out", help="output json (default: <out_dir>/explore_stats.json)")
    args = ap.parse_args()
    cfg = load(args.config, args.tier, args.overrides, args.seed)
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    bench = make_benchmark(cfg.benchmark, **cfg.env)
    train_tasks = bench.tasks("train")
    chosen = [train_tasks[i] for i in rng.choice(len(train_tasks), args.games, replace=False)]
    policy = LLMPolicy(cfg.policy, seed=cfg.seed)
    if args.adapter:
        policy.load_adapter(args.adapter)

    trajs = []
    with ExitStack() as cleanup, policy.generation():
        if hasattr(bench, "close"):
            cleanup.callback(bench.close)
        for s in range(0, len(chosen), args.games_per_batch):
            batch = chosen[s:s + args.games_per_batch]
            tasks = [t for t in batch for _ in range(args.group)]
            groups = [s + g for g in range(len(batch)) for _ in range(args.group)]
            trajs += run_episodes(policy, bench, tasks, groups, cfg.train.max_steps,
                                  cfg.train.success_reward, cfg.train.invalid_penalty)
            print(f"  {min(s + args.games_per_batch, len(chosen))}/{len(chosen)} games", flush=True)

    records = [trajectory_record(tr, bench) for tr in trajs]
    summary = summarize(records, trajs)
    summary["resp_len"] = length_stats(response_lengths(trajs), cfg.policy.max_new_tokens)
    out = Path(args.out) if args.out else Path(cfg.out_dir) / "explore_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "trajectories": records}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
