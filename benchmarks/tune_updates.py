"""Compare exact update implementations on one shared ALFWorld rollout.

No learning-rate or reward changes. Every candidate starts with identical LoRA
weights and a fresh optimizer. OOM candidates are recorded and skipped. This
does not choose a default automatically or make a learning-performance claim.

    python -m benchmarks.tune_updates --tier paper_overnight --turns 12
"""

import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmarks.common import add_args, load
from empg.config import config_dict, config_fingerprint
from empg.envs.base import make_benchmark
from empg.policy import LLMPolicy
from empg.rollout import Step, Trajectory, run_episodes
from empg.trainer import seed_everything, update


def candidates(cfg):
    p = cfg.policy
    return [
        ("reference", replace(p)),
        ("score4", replace(p, score_batch_size=4)),
        ("micro4", replace(p, score_batch_size=4, micro_batch_size=4)),
        ("chunk256", replace(p, score_batch_size=4, micro_batch_size=4, head_chunk_size=256)),
        ("analytic", replace(p, score_batch_size=4, micro_batch_size=4,
                             head_chunk_size=256, head_backend="analytic")),
        ("micro8", replace(p, score_batch_size=4, micro_batch_size=8,
                           head_chunk_size=256, head_backend="analytic")),
        ("no_transformer_checkpoint", replace(p, score_batch_size=4, micro_batch_size=4,
                                             head_chunk_size=256, head_backend="analytic",
                                             gradient_checkpointing=False)),
        ("bounded_no_checkpoint", replace(p, score_batch_size=4, micro_batch_size=4,
                                         micro_batch_tokens=2560, cuda_memory_fraction=.75,
                                         head_chunk_size=256, head_backend="analytic",
                                         gradient_checkpointing=False)),
        ("bounded_analytic2", replace(p, score_batch_size=2, micro_batch_size=2,
                                     micro_batch_tokens=4096, cuda_memory_fraction=.75,
                                     head_chunk_size=256, head_backend="analytic")),
        ("selective", replace(p, score_batch_size=4, micro_batch_size=1,
                              micro_batch_tokens=2560, cuda_memory_fraction=.75,
                              gradient_checkpointing=True, checkpoint_min_tokens=1280,
                              head_chunk_size=256, head_backend="analytic")),
        ("selective900", replace(p, score_batch_size=4, micro_batch_size=1,
                                 micro_batch_tokens=2560, cuda_memory_fraction=.75,
                                 gradient_checkpointing=True, checkpoint_min_tokens=900,
                                 head_chunk_size=256, head_backend="analytic")),
    ]


def profile(cfg, turns, samples, repeats, selected=None, memory_fraction=.75, rollout_cache=None, out=None,
            stress_max_length=False):
    seed_everything(cfg.seed)
    if torch.cuda.is_available():
        # Leave room for the sleeping inference worker and desktop. A failed
        # candidate should raise OOM rather than spill into slow shared RAM.
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    bench = make_benchmark(cfg.benchmark, **cfg.env)
    policy = None
    try:
        available = bench.tasks("train")
        n_groups = samples // cfg.train.group_size
        chosen = np.random.default_rng(cfg.seed).choice(len(available), n_groups, replace=False)
        tasks = [available[i] for i in chosen for _ in range(cfg.train.group_size)]
        groups = [i for i in range(n_groups) for _ in range(cfg.train.group_size)]
        policy = LLMPolicy(cfg.policy, seed=cfg.seed)
        rollout = {}
        request = {"config": config_fingerprint(cfg), "turns": turns, "samples": samples,
                   "stress_max_length": stress_max_length}
        cache = Path(rollout_cache) if rollout_cache else None
        if stress_max_length:
            with policy.generation():
                pass
            # Exercise the configured context limits with bounded cost. This is
            # a memory/finite-gradient check, not a learning or rollout benchmark.
            prompt_len = cfg.policy.max_prompt_tokens or 2048
            trajs = []
            for i, (task, group) in enumerate(zip(tasks, groups)):
                step = Step([3] * prompt_len, [4 + i % 2] * cfg.policy.max_new_tokens,
                            "synthetic context stress", "look", True, state_keys={"obs": "next"})
                trajs.append(Trajectory(task, group, [step], won=bool(i % 2),
                                        init_keys={"obs": "initial"}))
            rollout["synthetic_context_stress"] = True
        elif cache and cache.exists():
            saved = json.loads(cache.read_text())
            cached_request = {"stress_max_length": False, **saved["request"]}
            if cached_request != request:
                raise ValueError("Rollout cache has a different configuration; choose a new cache path")
            trajs = [Trajectory(**{**t, "steps": [Step(**s) for s in t["steps"]]}) for t in saved["trajs"]]
            with policy.generation():
                pass  # retain the real sleeping-worker memory footprint
            rollout["reused_cache"] = str(cache)
        else:
            trajs = run_episodes(policy, bench, tasks, groups, turns,
                                 cfg.train.success_reward, cfg.train.invalid_penalty,
                                 timings=rollout, progress=lambda turn, active: print(
                                     f"rollout {turn}/{turns}, {active} active", flush=True))
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"request": request, "trajs": [asdict(t) for t in trajs]}))
        initial = [p.detach().cpu().clone() for p in policy.parameters()]
        reference = None
        records = []
        result = {"config": config_dict(cfg), "rollout": rollout, "updates": records,
                  "response_rows": sum(len(t.steps) for t in trajs),
                  "max_prompt_tokens": max(len(s.prompt_ids) for t in trajs for s in t.steps),
                  "max_response_tokens": max(len(s.response_ids) for t in trajs for s in t.steps),
                  "memory_fraction": memory_fraction,
                  "note": "Shared real rollout, restored weights/fresh optimizer; timing, not learning evidence."}
        for repeat in range(repeats):
            variants = candidates(cfg)
            if selected:
                variants = [(name, p) for name, p in variants if name in selected]
            if repeat % 2:
                variants.reverse()
            for name, pc in variants:
                with torch.no_grad():
                    for p, original in zip(policy.parameters(), initial):
                        p.copy_(original)
                policy.cfg = pc
                if pc.gradient_checkpointing:
                    policy.model.gradient_checkpointing_enable()
                else:
                    policy.model.gradient_checkpointing_disable()
                candidate = replace(cfg, policy=pc)
                optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.train.lr,
                                              weight_decay=cfg.train.weight_decay)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                row = {"name": name, "repeat": repeat, "policy": config_dict(candidate)["policy"]}
                print(f"profiling {name}, repeat {repeat}", flush=True)
                start = time.perf_counter()
                try:
                    stats = update(policy, optimizer, trajs, candidate, np.random.default_rng(321))
                    if not all(np.isfinite(stats[k]) for k in ("pg_loss", "kl", "grad_norm", "entropy_mean", "adv_std")):
                        raise RuntimeError(f"{name} produced nonfinite update statistics")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    current = [p.detach().cpu().clone() for p in policy.parameters()]
                    if name == "reference" and reference is None:
                        reference = current
                    delta = (max((a - b).abs().max().item() for a, b in zip(reference, current))
                             if reference is not None else None)
                    row.update(status="ok", seconds=time.perf_counter() - start, stats=stats,
                               adapter_max_abs_delta=delta)
                except torch.cuda.OutOfMemoryError:
                    row.update(status="oom", seconds=time.perf_counter() - start)
                finally:
                    row.update(peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30
                               if torch.cuda.is_available() else 0.)
                    del optimizer
                    policy.model.zero_grad(set_to_none=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                records.append(row)
                if out:
                    target = Path(out)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(result, indent=2))
                    temporary.replace(target)
                print(json.dumps(row), flush=True)
        return result
    finally:
        bench.close()
        if policy is not None:
            policy.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_args(ap)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--memory-fraction", type=float, default=.75,
                    help="HF allocator ceiling as fraction of physical VRAM; prevents shared-RAM spill")
    ap.add_argument("--candidates", nargs="+", choices=[name for name, _ in candidates(load("empg", "paper_overnight"))])
    ap.add_argument("--out", default="runs/audit/update_tuning.json")
    ap.add_argument("--rollout-cache", help="save/replay a fingerprinted local JSON rollout to avoid repeated decoding")
    ap.add_argument("--stress-max-length", action="store_true",
                    help="synthetic maximum-context memory/finite-gradient check; not a timing or learning benchmark")
    args = ap.parse_args()
    cfg = load(args.config, args.tier, args.overrides, args.seed)
    if args.turns < 1 or args.repeats < 1 or args.samples < cfg.train.group_size or args.samples % cfg.train.group_size:
        ap.error("positive turns/repeats and samples divisible by group_size are required")
    if not 0 < args.memory_fraction <= 1:
        ap.error("memory-fraction must be in (0, 1]")
    result = profile(cfg, args.turns, args.samples, args.repeats, args.candidates, args.memory_fraction,
                     args.rollout_cache, args.out, args.stress_max_length)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    if not any(row["status"] == "ok" for row in result["updates"]):
        raise SystemExit("No candidate completed within the memory limit")


if __name__ == "__main__":
    main()
