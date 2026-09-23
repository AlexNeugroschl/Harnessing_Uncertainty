"""Evaluate a policy on a benchmark split; reports success per task type (paper Table 1).

    python -m benchmarks.evaluate --config benchmarks/configs/alfworld_empg.yaml --seed 0
    python -m benchmarks.evaluate --config benchmarks/configs/base.yaml --base   # zero-shot model
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from empg.config import Config, load_config
from empg.envs.base import make_benchmark
from empg.policy import LLMPolicy
from empg.rollout import run_episodes
from empg.trainer import seed_everything


def evaluate(policy: LLMPolicy, cfg: Config, log=print) -> dict:
    ec = cfg.eval
    bench = make_benchmark(cfg.benchmark, **cfg.env)
    tasks = bench.tasks(ec.split)
    if ec.max_tasks:
        tasks = tasks[:ec.max_tasks]
    by_type = defaultdict(list)
    for s in range(0, len(tasks), ec.batch_envs):
        chunk = tasks[s:s + ec.batch_envs]
        trajs = run_episodes(policy, bench, chunk, list(range(len(chunk))), ec.max_steps,
                             cfg.train.success_reward, cfg.train.invalid_penalty,
                             greedy=ec.greedy)
        for tr in trajs:
            by_type[bench.task_type(tr.task)].append(tr.won)
        log(f"  eval {min(s + ec.batch_envs, len(tasks))}/{len(tasks)}")
    res = {t: float(np.mean(v)) for t, v in sorted(by_type.items())}
    res["all"] = float(np.mean([w for v in by_type.values() for w in v]))
    res["n"] = {t: len(v) for t, v in by_type.items()}
    return res


def eval_path(cfg: Config, base: bool = False) -> Path:
    root = Path(cfg.out_dir) / "base" if base else cfg.run_dir
    return root / f"eval_{cfg.eval.split}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", action="store_true", help="evaluate the untrained model")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. eval.max_tasks=20")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg.seed = args.seed
    seed_everything(cfg.seed)
    policy = LLMPolicy(cfg.policy)
    if not args.base:
        policy.load_adapter(str(cfg.run_dir / "adapter"))
    res = evaluate(policy, cfg)
    out = eval_path(cfg, args.base)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
