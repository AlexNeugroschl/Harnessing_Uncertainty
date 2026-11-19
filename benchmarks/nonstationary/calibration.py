"""Measure the actual native workload; no ALFWorld runtime extrapolation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from .artifacts import write_json

SCHEMA = 1


def runtime_identity(cfg):
    from empg.config import config_dict
    import torch
    root = Path(__file__).resolve().parents[2]
    files = [root / "empg" / name for name in ("policy.py", "scoring.py", "trainer.py", "loss.py", "rollout.py",
             "curiosity.py", "advantages.py", "diagnostics.py", "envs/novelgym_env.py")]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    data = config_dict(cfg)
    return {"schema": SCHEMA, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "vllm": importlib.metadata.version("vllm") if cfg.policy.backend == "vllm" else None,
            "policy": data["policy"], "environment": data["env"], "benchmark": cfg.benchmark,
            "evaluation": {k: getattr(cfg.eval, k) for k in ("batch_envs", "max_steps", "greedy", "temperature")},
            "intervention": {"change": cfg.nonstationary.change, "feedback": cfg.nonstationary.feedback},
            "intrinsic": {k: getattr(cfg.advantage.curiosity, k) for k in ("key", "bonus", "scope", "mode")},
            "initial_adapter_sha256": cfg.train.initial_adapter_sha256, "execution_sha256": digest.hexdigest(),
            "train": {k: getattr(cfg.train, k) for k in ("games_per_iter", "group_size", "max_steps", "ppo_epochs",
                "mini_batch_size", "mini_batches", "shuffle", "kl_coef", "entropy_coef", "reward_mode", "reward_source",
                "lr", "weight_decay", "clip_low", "clip_high", "dual_clip")}}


def measure(cfg, out, repeats=2, validation_tasks=12):
    import numpy as np
    import torch
    from benchmarks.evaluate import evaluate
    from empg.envs.base import make_benchmark
    from empg.policy import LLMPolicy
    from empg.rollout import run_episodes
    from empg.trainer import seed_everything, update

    if not torch.cuda.is_available():
        raise RuntimeError("GPU calibration requires CUDA")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    started = time.perf_counter()
    policy = LLMPolicy(cfg.policy, seed=cfg.seed)
    bench = make_benchmark(cfg.benchmark, **cfg.env)
    result = {"schema": SCHEMA, "identity": runtime_identity(cfg), "training": [], "evaluation": []}
    try:
        if cfg.train.initial_adapter:
            policy.load_adapter(cfg.train.initial_adapter)
        initial = [p.detach().cpu().clone() for p in policy.parameters()]
        with policy.generation():
            pass  # measure engine startup once, outside warm batch rates
        result["startup_seconds"] = time.perf_counter() - started
        available = bench.tasks("train")
        chosen = np.random.default_rng(cfg.seed).choice(len(available), cfg.train.games_per_iter, replace=False)
        tasks = [available[i] for i in chosen for _ in range(cfg.train.group_size)]
        groups = [i for i in range(cfg.train.games_per_iter) for _ in range(cfg.train.group_size)]
        for repeat in range(repeats):
            for regime in ("baseline", "novelty") if repeat % 2 == 0 else ("novelty", "baseline"):
                with torch.no_grad():
                    for p, original in zip(policy.parameters(), initial):
                        p.copy_(original)
                bench.intervention = cfg.nonstationary.intervention(regime)
                times = {}
                start = time.perf_counter()
                with policy.sampling(200 + repeat):
                    trajectories = run_episodes(policy, bench, tasks, groups, cfg.train.max_steps,
                        cfg.train.success_reward, cfg.train.invalid_penalty, timings=times,
                        reward_source=cfg.train.reward_source, environment_reward_scale=cfg.train.environment_reward_scale)
                rollout_seconds = time.perf_counter() - start
                optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                stats = update(policy, optimizer, trajectories, cfg, np.random.default_rng(321))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if not all(math.isfinite(stats[k]) for k in ("pg_loss", "kl", "grad_norm", "entropy_mean")):
                    raise RuntimeError("nonfinite calibration update")
                row = {"repeat": repeat, "regime": regime, "rollout_seconds": rollout_seconds,
                       "update_seconds": elapsed, "steps": stats["n_steps"], "success": stats["success"],
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "timings": times,
                       "unique_scored_steps": stats["unique_scored_steps"], "gradient_rows": stats["gradient_rows"],
                       "response_tokens": sum(len(s.response_ids) for tr in trajectories for s in tr.steps)}
                result["training"].append(row)
                write_json(out / "calibration.json", result)
                print(json.dumps(row), flush=True)
                del optimizer
                policy.model.zero_grad(set_to_none=True)
        # Kernel stress with distinct training maps, not an adaptation result.
        # This prevents fitting twelve hours assuming sibling rollouts remain
        # identical after curiosity starts producing diverse strategies.
        with torch.no_grad():
            for p, original in zip(policy.parameters(), initial):
                p.copy_(original)
        bench.intervention = cfg.nonstationary.intervention("baseline")
        count = cfg.train.games_per_iter * cfg.train.group_size
        diverse_tasks = [available[i] for i in np.random.default_rng(999).choice(len(available), count, replace=False)]
        with policy.sampling(999):
            diverse = run_episodes(policy, bench, diverse_tasks, groups, min(8, 4 * cfg.env["recipe_depth"] - 1),
                                  reward_source=cfg.train.reward_source)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        start = time.perf_counter()
        stats = update(policy, optimizer, diverse, cfg, np.random.default_rng(321))
        torch.cuda.synchronize()
        result["diversity"] = {"update_seconds": time.perf_counter() - start, "steps": stats["n_steps"],
                               "gradient_rows": stats["gradient_rows"], "unique_scored_steps": stats["unique_scored_steps"],
                               "note": "distinct training maps; kernel stress only, no research outcome"}
        del optimizer
        policy.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, original in zip(policy.parameters(), initial):
                p.copy_(original)
        ec = dataclasses.replace(cfg.eval, split="valid_seen", max_tasks=validation_tasks, sampling_seed=17, selection_seed=17)
        evaluation_cfg = dataclasses.replace(cfg, eval=ec)
        session_start = time.perf_counter()
        with policy.generation():
            for regime in ("baseline", "novelty"):
                start = time.perf_counter()
                score, _ = evaluate(policy, evaluation_cfg, regime=regime)
                row = {"regime": regime, "seconds": time.perf_counter() - start,
                       "steps": sum(task["steps"] for task in score["per_task"]),
                       "tasks": len(score["per_task"]), "success": score["all"]}
                result["evaluation"].append(row)
                write_json(out / f"competence_{regime}.json", score)
        result["evaluation_session_overhead_seconds"] = max(0., time.perf_counter() - session_start - sum(r["seconds"] for r in result["evaluation"]))
    finally:
        bench.close()
        closing = time.perf_counter()
        policy.close()
    result["shutdown_seconds"] = time.perf_counter() - closing
    result["seconds"] = time.perf_counter() - started
    result["complete"] = True
    write_json(out / "calibration.json", result)
    return result


def checked_candidates(updates, repeats, tolerance):
    """Require both complete timing repeats and bounded real update differences."""
    candidates, valid, rejected = {}, {}, {}
    for row in updates:
        candidates.setdefault(row["name"], []).append(row)
    for name, rows in candidates.items():
        if len(rows) != repeats or any(r["status"] != "ok" for r in rows):
            rejected[name] = "incomplete or failed update measurements"
        elif any(r.get("adapter_max_abs_delta") is None or not math.isfinite(r["adapter_max_abs_delta"])
                 or r["adapter_max_abs_delta"] > tolerance for r in rows):
            rejected[name] = f"adapter update differs from reference by more than {tolerance:g}"
        else:
            valid[name] = rows
    return valid, rejected


def update_tolerance(updates, learning_rate):
    noise = max((r["adapter_max_abs_delta"] for r in updates if r["name"] == "reference"
                 and r["status"] == "ok" and r.get("adapter_max_abs_delta") is not None
                 and math.isfinite(r["adapter_max_abs_delta"])), default=0.)
    return max(2e-6, learning_rate, 1.5 * noise), noise


def tune(cfg, out, repeats=3):
    """Compare checkpoint policies on shared data; retain a measured safe choice."""
    from benchmarks.tune_updates import profile
    out = Path(out)
    variants = (["reference", "micro4"] if cfg.env.get("suite_version") == 4 else
                ["reference", "native_selective2", "selective900"])
    result = profile(cfg, 6, cfg.train.games_per_iter * cfg.train.group_size, repeats,
        variants, rollout_cache=out / "tuning_rollout.json",
        out=out / "tuning.json")
    tolerance, reference_noise = update_tolerance(result["updates"], cfg.train.lr)
    valid, rejected = checked_candidates(result["updates"], repeats, tolerance)
    if "reference" not in valid:
        raise RuntimeError("reference GPU update failed")
    median = {name: statistics.median(r["seconds"] for r in rows) for name, rows in valid.items()}
    best = min(median, key=median.get)
    # A small timing gain does not justify changing the production kernel shape.
    fraction = .90 if cfg.env.get("suite_version") == 4 else .95
    if median[best] >= fraction * median["reference"]:
        best = "reference"
    selected = valid[best][0]["policy"]
    cfg = dataclasses.replace(cfg, policy=dataclasses.replace(cfg.policy, **{
        key: selected.get(key, getattr(cfg.policy, key)) for key in
        ("micro_batch_size", "score_batch_size", "micro_batch_tokens", "checkpoint_min_tokens")}))
    if best != "reference":
        # Force the selected adaptive checkpoint policy through the configured
        # maximum context with a real sleeping inference worker before use.
        stress = profile(cfg, 1, cfg.train.group_size, 1, ["reference"],
                         out=out / "selected_context_stress.json", stress_max_length=True)
        if any(row["status"] != "ok" for row in stress["updates"]):
            raise RuntimeError("selected GPU settings failed the maximum-context stress check")
        if cfg.policy.micro_batch_tokens and cfg.policy.micro_batch_size > 1:
            # Also fill the padded token budget with distinct rows, so response
            # deduplication cannot accidentally turn this into a smaller test.
            prompt = cfg.policy.micro_batch_tokens // cfg.policy.micro_batch_size - cfg.policy.max_new_tokens
            if prompt > 0:
                packed_cfg = dataclasses.replace(cfg, policy=dataclasses.replace(cfg.policy,
                    max_prompt_tokens=min(cfg.policy.max_prompt_tokens, prompt)))
                packed = profile(packed_cfg, 1, cfg.train.group_size, 1, ["reference"],
                    out=out / "selected_padded_budget_stress.json", stress_max_length=True)
                if any(row["status"] != "ok" for row in packed["updates"]):
                    raise RuntimeError("selected GPU settings failed the full padded-token stress check")
        boundary = cfg.policy.checkpoint_min_tokens - 1
        if boundary > cfg.policy.max_new_tokens:
            boundary_cfg = dataclasses.replace(cfg, policy=dataclasses.replace(cfg.policy,
                max_prompt_tokens=min(cfg.policy.max_prompt_tokens, boundary - cfg.policy.max_new_tokens)))
            check = profile(boundary_cfg, 1, cfg.train.group_size, 1, ["reference"],
                out=out / "selected_checkpoint_boundary.json", stress_max_length=True)
            if any(row["status"] != "ok" for row in check["updates"]):
                # Unsafe thresholds never become a production choice.
                best = "reference"
                cfg = dataclasses.replace(cfg, policy=dataclasses.replace(cfg.policy,
                    micro_batch_size=result["config"]["policy"]["micro_batch_size"],
                    score_batch_size=result["config"]["policy"].get("score_batch_size", 0),
                    micro_batch_tokens=result["config"]["policy"].get("micro_batch_tokens", 0),
                    checkpoint_min_tokens=result["config"]["policy"].get("checkpoint_min_tokens", 0)))
    write_json(out / "selection.json", {"selected": best, "median_seconds": median, "policy": dataclasses.asdict(cfg.policy),
                                        "adapter_tolerance": tolerance, "reference_repeatability_noise": reference_noise,
                                        "minimum_relative_speedup": 1 - fraction, "rejected": rejected})
    return cfg


def main(argv=None):
    from run_benchmarks import launch_in_wsl
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.name == "nt":
        return launch_in_wsl(["--tier", "novelgym_screen", *argv], entrypoint="run_novelgym_calibrate.py")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initial-adapter", required=True)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--novelty", choices=["tool", "recipe", "resource"], default="tool")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="runs/nonstationary/calibration")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--learning-profile")
    ap.add_argument("--tier", default="novelgym_screen", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not 2 <= args.depth <= 32 or args.repeats < 1:
        ap.error("invalid depth/repeats")
    from benchmarks.common import load
    from empg.envs.novelgym_env import nominal_solution_length
    horizon = max(48, 2 * nominal_solution_length(args.depth, args.novelty, "novelty") + 8)
    overrides = [f"train.initial_adapter={args.initial_adapter}",
        f"env.recipe_depth={args.depth}", f"env.novelty={args.novelty}", f"train.max_steps={horizon}", f"eval.max_steps={horizon}"]
    if args.learning_profile:
        from .evidence import load_learning_profile
        learning = load_learning_profile(args.learning_profile, args.initial_adapter)
        overrides += [f"{k}={json.dumps(v)}" for k, v in learning["settings"].items()]
    cfg = load("grpo", args.tier, overrides)
    if args.tune:
        cfg = tune(cfg, args.out)
    measure(cfg, args.out, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
