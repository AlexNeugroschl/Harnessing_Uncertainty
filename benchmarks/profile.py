"""Measure bottlenecks on this machine, using identical games/data for each path.

    python -m benchmarks.profile --config empg --tier small --env-only
    python -m benchmarks.profile --config empg --tier small --turns 3 --samples 8

The second command compares dense/chunked updates on the very same real rollout,
restoring the adapter and optimizer initialization between measurements.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.common import add_args, load


def environment_profile(cfg, tasks, turns, workers):
    from empg.envs.alfworld_env import ALFWorldBenchmark
    records, expected = [], None
    for n_workers in workers:
        bench = ALFWorldBenchmark(**{**cfg.env, "workers": n_workers})
        try:
            # Include first-use imports; reuse the worker pool for the warm repeat.
            for repeat in range(2):
                rng = np.random.default_rng(123)
                batch = bench.episode_batch(tasks)
                start = time.perf_counter()
                try:
                    states = batch.reset()
                    reset_s = time.perf_counter() - start
                    trace = [states]
                    active = list(range(len(tasks)))
                    infos = [s[1] for s in states]
                    start = time.perf_counter()
                    for _ in range(turns):
                        actions = [infos[i]["admissible"][rng.integers(len(infos[i]["admissible"]))]
                                   for i in active]
                        rows = batch.step(active, actions)
                        trace.append(rows)
                        for i, row in zip(active, rows):
                            infos[i] = row[3]
                        active = [i for i, row in zip(active, rows) if not row[2]]
                        if not active:
                            break
                    step_s = time.perf_counter() - start
                    # JSON transport represents environment tuples as lists.
                    trace = json.loads(json.dumps(trace))
                    if expected is None:
                        expected = trace
                    if trace != expected:
                        raise AssertionError("parallel environment changed the scripted trace")
                    row = dict(workers=n_workers, repeat=repeat, reset_s=reset_s,
                               step_s=step_s, trace_equal=True)
                    records.append(row)
                    print(json.dumps(row), flush=True)
                finally:
                    batch.close()
        finally:
            bench.close()
    return records


def update_profile(cfg, tasks, turns, repeats=2):
    import torch
    from empg.envs.base import make_benchmark
    from empg.policy import LLMPolicy
    from empg.rollout import run_episodes
    from empg.trainer import seed_everything, update

    seed_everything(cfg.seed)
    bench = make_benchmark(cfg.benchmark, **cfg.env)
    policy = LLMPolicy(cfg.policy, seed=cfg.seed)
    try:
        times = {}
        # All samples are siblings of one game; force a mixed reward below to
        # benchmark nonzero gradients independently of early policy success.
        trajs = run_episodes(policy, bench, [tasks[0]] * len(tasks), [0] * len(tasks),
                             turns, timings=times)
        print(json.dumps({"rollout": times}), flush=True)
    finally:
        if hasattr(bench, "close"):
            bench.close()
    trajs[0].reward = 10.0
    trajs[1].reward = 0.0
    saved = [p.detach().cpu().clone() for p in policy.parameters()]
    records, result_weights = [], []
    for repeat, chunked in [(repeat, chunked) for repeat in range(repeats)
                            for chunked in ((False, True) if repeat % 2 == 0 else (True, False))]:
        print(f"profiling update, chunked_head={chunked}", flush=True)
        with torch.no_grad():
            for p, initial in zip(policy.parameters(), saved):
                p.copy_(initial)
        policy.cfg.chunked_head = chunked
        optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.train.lr,
                                      weight_decay=cfg.train.weight_decay)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        stats = update(policy, optimizer, trajs, cfg, np.random.default_rng(321))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        row = dict(chunked=chunked, repeat=repeat, seconds=time.perf_counter() - start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30
                   if torch.cuda.is_available() else 0.0, stats=stats)
        if repeat == 0:
            result_weights.append([p.detach().cpu().clone() for p in policy.parameters()])
        records.append(row)
        print(json.dumps(row), flush=True)
        del optimizer
    max_delta = max((a - b).abs().max().item()
                    for a, b in zip(*result_weights))
    return {"rollout": times, "updates": records, "adapter_max_abs_delta": max_delta,
            "note": "Mixed rewards injected for performance/gradient checks; not research evidence."}


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--env-only", action="store_true")
    mode.add_argument("--model-only", action="store_true")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=2, help="alternating-order GPU measurements")
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 4])
    ap.add_argument("--out", default="runs/audit/profile.json")
    args = ap.parse_args()
    if args.samples < 2 or args.turns < 1 or args.repeats < 1:
        ap.error("samples must be >=2; turns/repeats must be >=1")
    cfg = load(args.config, args.tier, args.overrides, args.seed)
    if cfg.benchmark != "alfworld":
        ap.error("this profiler currently supports ALFWorld only")
    from empg.envs.base import make_benchmark
    bench = make_benchmark(cfg.benchmark, **cfg.env)
    available = bench.tasks("train")
    tasks = [available[i] for i in np.random.default_rng(0).choice(
        len(available), args.samples, replace=False)]
    results = {}
    if not args.model_only:
        results["environment"] = environment_profile(cfg, tasks, args.turns, args.workers)
    if not args.env_only:
        results["model"] = update_profile(cfg, tasks, args.turns, args.repeats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
