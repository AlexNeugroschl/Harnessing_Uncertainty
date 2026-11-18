"""Shared baseline skills with recovery examples and held-out competence checks."""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
from pathlib import Path
import sys
import time

from run_benchmarks import launch_in_wsl
from .artifacts import write_json


def expert_actions(depth):
    for i in range(1, depth + 1):
        yield from ("approach_log", "break_block", "craft_planks",
                    "craft_pogo_stick" if i == depth else f"craft_component_{i:02d}")


def baseline_examples(bench, task, recovery_probability, rng):
    """Initial-rule clean/recovery prompts; no changed-rule solution is shown."""
    env = bench.make_env(task)
    history = []
    try:
        obs, info = env.reset()
        for action in expert_actions(bench.depth):
            yield bench.prompt(info["task"], obs, info["admissible"], history), action
            if rng.random() < recovery_probability:
                perturbation = "break_block" if env._visible()["ahead"] == "air" else "craft_pogo_stick"
                before = info["state_keys"]
                result, _, done, new_info = env.step(perturbation)
                if done or new_info["state_keys"] != before or not new_info["intervention_metrics"]["action_failures"]:
                    raise RuntimeError("recovery demonstration changed the native world")
                history.append((obs, perturbation))
                obs, info = result, new_info
                yield bench.prompt(info["task"], obs, info["admissible"], history), action
            previous = obs
            obs, _, done, info = env.step(action)
            history.append((previous, action))
            if info["intervention_metrics"]["action_failures"]:
                raise RuntimeError("expert trace failed on a native training map")
        if not done or not info["won"]:
            raise RuntimeError("expert trace did not finish")
    finally:
        env.close()


def supervised_epoch(policy, optimizer, data, order, batch_size):
    import torch
    from empg.policy import token_microbatches

    lengths = [len(p) + len(r) for p, r in data]
    total_loss = torch.zeros((), device=policy.device)
    total_tokens = 0
    parameters = policy.parameters()
    policy.model.train()
    for module in policy.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    for at in range(0, len(order), batch_size):
        rows = sorted(order[at:at + batch_size], key=lambda i: lengths[i])
        tokens = sum(len(data[i][1]) for i in rows)
        optimizer.zero_grad(set_to_none=True)
        for micro in token_microbatches(rows, lengths, policy.cfg.micro_batch_size, policy.cfg.micro_batch_tokens):
            lp, _, mask = policy.forward_logprobs([data[i][0] for i in micro], [data[i][1] for i in micro])
            loss = -lp[mask].sum()
            (loss / tokens).backward()
            total_loss.add_(loss.detach())
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        total_tokens += tokens
    policy.model.eval()
    return total_loss.item() / total_tokens


def validate(policy, cfg, depths, tasks, seed, out, epoch):
    from benchmarks.evaluate import evaluate
    results = {}
    with policy.generation():
        for depth in depths:
            candidate = dataclasses.replace(cfg, env={**cfg.env, "recipe_depth": depth},
                eval=dataclasses.replace(cfg.eval, split="valid_seen", max_tasks=tasks,
                    max_steps=max(48, depth * 6), selection_seed=seed, sampling_seed=seed))
            result, _ = evaluate(policy, candidate, regime="baseline")
            results[str(depth)] = result
    write_json(out / f"validation_epoch_{epoch:03d}.json", results)
    return results


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/novelgym_warmup")
    ap.add_argument("--depths", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8, help="maximum; stop once all depths pass validation")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--initial-adapter")
    ap.add_argument("--recovery-probability", type=float, default=.3)
    ap.add_argument("--validate-every", type=int, default=2)
    ap.add_argument("--validation-tasks", type=int, default=16)
    ap.add_argument("--validation-seed", type=int, default=17)
    ap.add_argument("--min-success", type=float, default=.75)
    ap.add_argument("--require-competence", action="store_true")
    ap.add_argument("--tier", default="novelgym_screen", help=argparse.SUPPRESS)
    return ap


def main(argv=None):
    import json
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.name == "nt":
        return launch_in_wsl(["--tier", "novelgym_screen", *argv], entrypoint="run_novelgym_warmup.py")
    ap = parser()
    args = ap.parse_args(argv)
    if (min(args.games, args.epochs, args.batch_size, args.validation_tasks, args.validate_every) < 1
            or args.games > 64 or args.validation_tasks > 64
            or any(not 2 <= d <= 32 for d in args.depths) or len(set(args.depths)) != len(args.depths)
            or min(args.seed, args.validation_seed) < 0 or not math.isfinite(args.lr) or args.lr <= 0
            or not 0 <= args.recovery_probability <= 1 or not 0 < args.min_success <= 1):
        ap.error("invalid warmup counts, depths, seeds or learning parameters")
    import numpy as np
    import torch
    from benchmarks.common import load
    from empg.config import adapter_fingerprint
    from empg.envs.novelgym_env import NovelGymBenchmark, SUITE_VERSION
    from empg.policy import LLMPolicy
    from empg.trainer import seed_everything

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        ap.error("choose a new or empty warmup directory")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    cfg = load("grpo", args.tier)
    policy = LLMPolicy(cfg.policy, seed=args.seed)
    records, best_score, ready = [], -1., False
    start = time.perf_counter()
    try:
        if args.initial_adapter:
            policy.load_adapter(args.initial_adapter)
        data = []
        demo_rng = np.random.default_rng(args.seed)
        for depth in args.depths:
            bench = NovelGymBenchmark(recipe_depth=depth, action_parser="native")
            for task in bench.tasks("train")[:args.games]:
                for prompt, action in baseline_examples(bench, task, args.recovery_probability, demo_rng):
                    target = policy.tokenizer("action: " + action, add_special_tokens=False)["input_ids"] + [policy.tokenizer.eos_token_id]
                    data.append((policy.encode_prompt(prompt), target))
        optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=.01)
        rng = np.random.default_rng(args.seed)
        training_seconds = 0.
        manifest = {"method": "baseline-only supervised skill initialization with failure recovery",
                    "suite_version": SUITE_VERSION, "options": vars(args), "policy": dataclasses.asdict(cfg.policy),
                    "response_rows": len(data), "training_task_split": "train", "changed_regime_demonstrations": 0}
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            nll = supervised_epoch(policy, optimizer, data, rng.permutation(len(data)), args.batch_size)
            if not math.isfinite(nll):
                raise RuntimeError("nonfinite supervised loss")
            training_seconds += time.perf_counter() - epoch_start
            row = {"epoch": epoch, "nll": nll, "training_seconds": training_seconds}
            if epoch % args.validate_every == 0 or epoch == args.epochs:
                results = validate(policy, cfg, args.depths, args.validation_tasks, args.validation_seed, out, epoch)
                row["validation"] = {d: result["all"] for d, result in results.items()}
                score = min(row["validation"].values())
                if score > best_score:
                    best_score = score
                    policy.save(str(out / "adapter"))
                    manifest["best_epoch"] = epoch
                ready = score >= args.min_success
            row["seconds"] = time.perf_counter() - start
            records.append(row)
            manifest.update(epochs=records, ready=best_score >= args.min_success, best_min_success=best_score,
                            seconds=time.perf_counter() - start, training_seconds=training_seconds)
            write_json(out / "warmup.json", manifest)
            print(json.dumps(row), flush=True)
            if ready:
                break
        manifest.update(adapter_sha256=adapter_fingerprint(out / "adapter"),
            peak_reserved_vram_gib=torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.)
        write_json(out / "warmup.json", manifest)
        write_json(out / "complete.json", {"ready": manifest["ready"], "adapter_sha256": manifest["adapter_sha256"]})
    finally:
        policy.close()
    return 2 if args.require_competence and not manifest["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
