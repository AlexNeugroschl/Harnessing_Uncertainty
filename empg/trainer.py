"""Training loop: rollout -> score (entropy) -> advantages -> policy update."""

from __future__ import annotations

import dataclasses
import json
import random
import time

import numpy as np
import torch

from empg.advantages import compute_advantages
from empg.config import Config
from empg.envs.base import make_benchmark
from empg.loss import policy_loss
from empg.policy import LLMPolicy
from empg.rollout import Trajectory, run_episodes


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _right_aligned(rows: list[torch.Tensor], T: int, device) -> torch.Tensor:
    out = torch.zeros(len(rows), T)
    for i, r in enumerate(rows):
        out[i, T - len(r):] = r
    return out.to(device)


def update(policy: LLMPolicy, opt, trajs: list[Trajectory], cfg: Config,
           rng: np.random.Generator) -> dict:
    """One policy-gradient update on a batch of trajectories. Returns stats."""
    tc = cfg.train
    steps = [s for tr in trajs for s in tr.steps]
    step_traj = np.array([i for i, tr in enumerate(trajs) for _ in tr.steps])
    P = [s.prompt_ids for s in steps]
    R = [s.response_ids for s in steps]

    # Step entropy H_t (Eq. 7): mean full-vocab token entropy over the response.
    old_logp, ents = policy.score(P, R, with_entropy=True)
    for s, e in zip(steps, ents):
        s.entropy = e.mean().item()
    ref_logp = None
    if tc.kl_coef > 0 and policy.use_lora:  # reference = base model (adapter off)
        ref_logp, _ = policy.score(P, R, with_entropy=False, reference=True)

    adv = compute_advantages(
        [tr.reward for tr in trajs], [tr.group for tr in trajs], step_traj,
        [s.entropy for s in steps], cfg.advantage)

    policy.model.train()
    dev, mb_size = policy.device, policy.cfg.micro_batch_size
    sums = {"pg": 0.0, "kl": 0.0, "clipfrac": 0.0, "tokens": 0}
    grad_norms = []
    for _ in range(tc.ppo_epochs):
        for mb in np.array_split(rng.permutation(len(steps)), tc.mini_batches):
            if len(mb) == 0:
                continue
            n_tok = sum(len(R[i]) for i in mb)
            opt.zero_grad(set_to_none=True)
            for s in range(0, len(mb), mb_size):
                idx = mb[s:s + mb_size]
                logp, _, mask = policy.forward_logprobs([P[i] for i in idx], [R[i] for i in idx])
                T = logp.shape[1]
                old = _right_aligned([old_logp[i] for i in idx], T, dev)
                ref = _right_aligned([ref_logp[i] for i in idx], T, dev) if ref_logp else None
                a = torch.tensor(adv[idx], dtype=torch.float32, device=dev)[:, None].expand(-1, T)
                loss, st = policy_loss(logp, old, ref, a, mask, tc.clip_low, tc.clip_high,
                                       tc.kl_coef)
                (loss / n_tok).backward()
                for k in ("pg", "kl", "clipfrac"):
                    sums[k] += st[k]
            sums["tokens"] += n_tok
            grad_norms.append(torch.nn.utils.clip_grad_norm_(
                policy.parameters(), tc.max_grad_norm).item())
            opt.step()
    policy.model.eval()

    H = np.array([s.entropy for s in steps])
    rewards = np.array([tr.reward for tr in trajs])
    groups = np.array([tr.group for tr in trajs])
    flat = np.mean([np.ptp(rewards[groups == g]) == 0 for g in np.unique(groups)])
    n = max(sums["tokens"], 1)
    return {
        "success": float(np.mean([tr.won for tr in trajs])),
        "reward": float(rewards.mean()),
        "ep_len": float(np.mean([len(tr.steps) for tr in trajs])),
        "invalid_rate": float(np.mean([not s.valid for s in steps])),
        "entropy_mean": float(H.mean()), "entropy_std": float(H.std()),
        "adv_std": float(adv.std()),
        "flat_group_frac": float(flat),  # groups where GRPO's Eq. 4 is all zero
        "pg_loss": sums["pg"] / n, "kl": sums["kl"] / n, "clipfrac": sums["clipfrac"] / n,
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "n_steps": len(steps),
    }


def train(cfg: Config, log=print) -> LLMPolicy:
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    bench = make_benchmark(cfg.benchmark, **cfg.env)
    train_tasks = bench.tasks("train")
    policy = LLMPolicy(cfg.policy)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    tc = cfg.train
    log(f"[{cfg.name} s{cfg.seed}] {len(train_tasks)} train tasks, "
        f"advantage={cfg.advantage.method} (g={cfg.advantage.use_g}, f={cfg.advantage.use_f})")

    with open(run_dir / "metrics.jsonl", "w") as mf:
        for it in range(1, tc.iterations + 1):
            t0 = time.time()
            chosen = rng.choice(len(train_tasks), tc.games_per_iter, replace=False)
            tasks = [train_tasks[c] for c in chosen for _ in range(tc.group_size)]
            groups = [g for g in range(tc.games_per_iter) for _ in range(tc.group_size)]
            trajs = run_episodes(policy, bench, tasks, groups, tc.max_steps,
                                 tc.success_reward, tc.invalid_penalty)
            t1 = time.time()
            stats = update(policy, opt, trajs, cfg, rng)
            stats.update(iteration=it, t_rollout=t1 - t0, t_update=time.time() - t1)
            mf.write(json.dumps(stats) + "\n")
            mf.flush()
            log(f"it {it:4d} | succ {stats['success']:.2f} | R {stats['reward']:+.2f} "
                f"| len {stats['ep_len']:.1f} | inval {stats['invalid_rate']:.2f} "
                f"| H {stats['entropy_mean']:.3f} | kl {stats['kl']:.4f} "
                f"| {stats['t_rollout']:.0f}s+{stats['t_update']:.0f}s")
            if tc.save_every and it % tc.save_every == 0:
                policy.save(str(run_dir / "adapter"))
    policy.save(str(run_dir / "adapter"))
    return policy
