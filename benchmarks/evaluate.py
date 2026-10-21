"""Evaluate a policy on a benchmark split; reports success per task type (paper Table 1).

    python -m benchmarks.evaluate --config empg --tier small --seed 0
    python -m benchmarks.evaluate --config benchmarks/configs/base.yaml --tier small --base  # zero-shot
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from contextlib import ExitStack
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.common import add_args, load
from empg.config import Config, config_fingerprint
from empg.envs.base import make_benchmark
from empg.policy import LLMPolicy
from empg.rollout import length_stats, response_lengths, run_episodes
from empg.trainer import seed_everything


def select_tasks(tasks, bench, ec):
    """A deterministic panel independent of the arm and training seed."""
    if ec.selection not in ("random", "stratified") or ec.max_tasks < 0:
        raise ValueError("invalid evaluation selection/max_tasks")
    if not ec.max_tasks or ec.max_tasks >= len(tasks):
        return tasks
    rng = np.random.default_rng(ec.selection_seed)
    if ec.selection == "random":
        pick = rng.choice(len(tasks), ec.max_tasks, replace=False)
    else:
        types = sorted({bench.task_type(t) for t in tasks})
        buckets = [np.array([i for i, t in enumerate(tasks) if bench.task_type(t) == typ])
                   for typ in types]
        if ec.max_tasks < len(types):
            raise ValueError("stratified panel needs at least one task per type")
        sizes = np.array([len(b) for b in buckets])
        ideal = ec.max_tasks * sizes / sizes.sum()
        counts = np.minimum(sizes, np.maximum(1, np.floor(ideal).astype(int)))
        while counts.sum() > ec.max_tasks:
            i = np.argmax(np.where(counts > 1, counts - ideal, -np.inf))
            counts[i] -= 1
        while counts.sum() < ec.max_tasks:
            i = np.argmax(np.where(counts < sizes, ideal - counts, -np.inf))
            counts[i] += 1
        pick = np.concatenate([rng.choice(b, n, replace=False) for b, n in zip(buckets, counts)])
    return [tasks[i] for i in sorted(pick)]


def evaluate(policy: LLMPolicy, cfg: Config, log=print) -> tuple[dict, dict]:
    ec = cfg.eval
    if not ec.greedy and ec.temperature is not None and (
            not np.isfinite(ec.temperature) or ec.temperature <= 0):
        raise ValueError("stochastic eval.temperature must be finite and positive")
    bench = make_benchmark(cfg.benchmark, **cfg.env)
    tasks = select_tasks(bench.tasks(ec.split), bench, ec)
    if not tasks:
        raise ValueError("empty evaluation split")
    by_type = defaultdict(list)
    per_task = []
    lengths = {"len": [], "truncated": [], "valid": []}
    with ExitStack() as cleanup:
        if hasattr(bench, "close"):
            cleanup.callback(bench.close)
        if not ec.greedy and ec.temperature is not None:
            original = policy.cfg
            policy.cfg = dataclasses.replace(original, temperature=ec.temperature)
            cleanup.callback(setattr, policy, "cfg", original)
        cleanup.enter_context(policy.generation())  # one session for all chunks
        for s in range(0, len(tasks), ec.batch_envs):
            chunk = tasks[s:s + ec.batch_envs]
            log(f"  eval tasks {s + 1}-{s + len(chunk)}/{len(tasks)} starting")
            trajs = run_episodes(policy, bench, chunk, list(range(len(chunk))), ec.max_steps,
                                 cfg.train.success_reward, cfg.train.invalid_penalty,
                                 greedy=ec.greedy, progress=lambda turn, active: log(
                                     f"    turn {turn}/{ec.max_steps}, {active} active"))
            for tr in trajs:
                by_type[bench.task_type(tr.task)].append(tr.won)
                per_task.append({"task": tr.task, "type": bench.task_type(tr.task),
                                 "won": tr.won, "steps": len(tr.steps), "invalid": tr.n_invalid})
            for k, v in response_lengths(trajs).items():
                lengths[k] += v
            log(f"  eval {min(s + ec.batch_envs, len(tasks))}/{len(tasks)}")
    res = {t: float(np.mean(v)) for t, v in sorted(by_type.items())}
    res["all"] = float(np.mean([w for v in by_type.values() for w in v]))
    res["n"] = {t: len(v) for t, v in by_type.items()}
    res["resp_len"] = length_stats(lengths, policy.cfg.max_new_tokens)
    res.update(seed=cfg.seed, per_task=per_task, config_fingerprint=config_fingerprint(cfg),
               protocol={"split": ec.split, "max_steps": ec.max_steps, "greedy": ec.greedy,
                         "backend": cfg.policy.backend, "model": cfg.policy.model_name,
                         "max_new_tokens": cfg.policy.max_new_tokens,
                         "selection": ec.selection, "selection_seed": ec.selection_seed,
                         "history_length": cfg.env.get("history_length", 2)})
    if not ec.greedy and ec.temperature is not None:
        res["protocol"]["temperature"] = ec.temperature
    if cfg.env.get("action_parser", "strict") != "strict":
        res["protocol"]["action_parser"] = cfg.env["action_parser"]
    return res, {"cap": policy.cfg.max_new_tokens, **lengths}


def eval_path(cfg: Config, base: bool = False) -> Path:
    root = Path(cfg.out_dir) / "base" if base else cfg.run_dir
    return root / f"eval_{cfg.eval.split}.json"


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--base", action="store_true", help="evaluate the untrained model")
    args = ap.parse_args()
    cfg = load(args.config, args.tier, args.overrides, args.seed)
    seed_everything(cfg.seed)
    policy = LLMPolicy(cfg.policy, seed=cfg.seed)
    if not args.base:
        policy.load_adapter(str(cfg.run_dir / "adapter"))
    res, lengths = evaluate(policy, cfg)
    out = eval_path(cfg, args.base)
    out.parent.mkdir(parents=True, exist_ok=True)
    length_out = out.parent / f"resp_lengths_eval_{cfg.eval.split}.json"
    temporary = length_out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(lengths))
    temporary.replace(length_out)
    # A suite deadline must not leave a half-written evaluation looking complete.
    # Publish the score last, when its companion artifact is also complete.
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(res, indent=2))
    temporary.replace(out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
