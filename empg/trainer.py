"""Training loop: rollout -> score (entropy) -> advantages -> policy update."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
import random
import time

import numpy as np
import torch

from empg.advantages import compute_advantages, normalize_entropy
from empg.config import Config, config_dict, config_fingerprint, config_matches
from empg.curiosity import exploration_stats, intrinsic_rewards
from empg.envs.base import make_benchmark
from empg.loss import policy_loss
from empg.policy import LLMPolicy, token_microbatches
from empg.rollout import Trajectory, length_stats, response_lengths, run_episodes


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _right_aligned(rows: list[torch.Tensor], T: int, device) -> torch.Tensor:
    out = torch.zeros(len(rows), T)
    for i, r in enumerate(rows):
        out[i, T - len(r):] = r
    return out.to(device)


def optimizer_minibatches(n_steps, tc, rng):
    order = rng.permutation(n_steps) if tc.shuffle else np.arange(n_steps)
    if tc.mini_batch_size:
        return [order[s:s + tc.mini_batch_size] for s in range(0, n_steps, tc.mini_batch_size)]
    return [mb for mb in np.array_split(order, tc.mini_batches) if len(mb)]


# Exploration metrics always use the non-privileged key, so every arm is measured
# the same way whatever key its curiosity bonus uses. Envs without state keys
# fall back to the observation text.
METRICS_KEY = "parsed"


@contextmanager
def _close_policy_on_error(policy):
    try:
        yield
    except BaseException:
        if hasattr(policy, "close"):
            policy.close()
        raise


def update(policy: LLMPolicy, opt, trajs: list[Trajectory], cfg: Config,
           rng: np.random.Generator, iteration: int = 1) -> dict:
    """One policy-gradient update on a batch of trajectories. Returns stats."""
    tc = cfg.train
    if policy.cfg.temperature != 1.0 or policy.cfg.top_p != 1.0:
        raise ValueError("on-policy training requires temperature=1 and top_p=1; "
                         "scoring does not include sampling transforms")
    steps = [s for tr in trajs for s in tr.steps]
    if not steps:
        raise ValueError("cannot update from an empty rollout")
    if tc.kl_coef > 0 and not policy.use_lora:
        raise ValueError("KL reference currently requires LoRA; use kl_coef=0 for full fine-tuning")
    step_traj = np.array([i for i, tr in enumerate(trajs) for _ in tr.steps])
    P = [s.prompt_ids for s in steps]
    R = [s.response_ids for s in steps]
    lengths = [len(p) + len(r) for p, r in zip(P, R)]

    # Step entropy H_t (Eq. 7): mean full-vocab token entropy over the response.
    t_score = time.perf_counter()
    old_logp, ents = policy.score(P, R, with_entropy=True)
    for s, e in zip(steps, ents):
        s.entropy = e.mean().item()
    ref_logp = None
    if tc.kl_coef > 0 and policy.use_lora:  # reference = base model (adapter off)
        ref_logp, _ = policy.score(P, R, with_entropy=False, reference=True)
    score_seconds = time.perf_counter() - t_score

    groups = [tr.group for tr in trajs]
    won = [tr.won for tr in trajs]
    step_h = [s.entropy for s in steps]
    step_rewards = None
    if tc.reward_mode == "step":
        step_rewards = np.array([tc.success_reward * trajs[t].won - tc.invalid_penalty * (not s.valid)
                                 for t, s in zip(step_traj, steps)])
    elif tc.reward_mode != "trajectory":
        raise ValueError("train.reward_mode must be 'trajectory' or 'step'")
    cur = cfg.advantage.curiosity
    intrinsic = None
    if cur.weight != 0:
        intrinsic = intrinsic_rewards(
            [tr.key_seq(cur.key) for tr in trajs], groups, won, cur, iteration,
            tc.iterations, h_norm=normalize_entropy(step_h, cfg.advantage.entropy_eps))
    adv = compute_advantages([tr.reward for tr in trajs], groups, step_traj, step_h,
                             cfg.advantage, intrinsic=intrinsic, step_rewards=step_rewards)

    # Keep transformer gradient checkpointing active; disable LoRA/module
    # dropout so PPO uses the conditional distribution used in rollout.
    policy.model.train()
    for module in policy.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    dev, mb_size = policy.device, policy.cfg.micro_batch_size
    sums = {"pg": 0.0, "kl": 0.0, "clipfrac": 0.0, "tokens": 0}
    grad_norms = []
    for _ in range(tc.ppo_epochs):
        for mb in optimizer_minibatches(len(steps), tc, rng):
            n_tok = sum(len(R[i]) for i in mb)
            # Keep random minibatch membership and token-mean weighting, while
            # reducing padding inside its gradient-accumulation microbatches.
            mb = np.array(sorted(mb, key=lambda i: lengths[i]))
            opt.zero_grad(set_to_none=True)
            for idx in token_microbatches(mb, lengths, mb_size, policy.cfg.micro_batch_tokens):
                logp, entropy, mask = policy.forward_logprobs(
                    [P[i] for i in idx], [R[i] for i in idx], with_entropy=tc.entropy_coef > 0)
                T = logp.shape[1]
                old = _right_aligned([old_logp[i] for i in idx], T, dev)
                ref = _right_aligned([ref_logp[i] for i in idx], T, dev) if ref_logp else None
                a = torch.tensor(adv[idx], dtype=torch.float32, device=dev)[:, None].expand(-1, T)
                loss, st = policy_loss(logp, old, ref, a, mask, tc.clip_low, tc.clip_high,
                                       tc.kl_coef, tensor_stats=True, entropy=entropy,
                                       entropy_coef=tc.entropy_coef, dual_clip=tc.dual_clip)
                (loss / n_tok).backward()
                for k in ("pg", "kl", "clipfrac"):
                    sums[k] += st[k]
            sums["tokens"] += n_tok
            grad_norms.append(torch.nn.utils.clip_grad_norm_(
                policy.parameters(), tc.max_grad_norm).item())
            opt.step()
    policy.model.eval()
    if torch.cuda.is_available():
        # Hand the update's cached blocks back before the next rollout: on an 8GB
        # card a bloated cache pushes generation into (slow) shared system memory.
        torch.cuda.empty_cache()

    H = np.array(step_h)
    rewards = np.array([tr.reward for tr in trajs])
    lens = length_stats(response_lengths(trajs))
    groups = np.array(groups)
    score_groups = groups[step_traj] if step_rewards is not None else groups
    scores = step_rewards if step_rewards is not None else rewards
    flat = np.mean([np.ptp(scores[score_groups == g]) == 0 for g in np.unique(groups)])
    n = max(sums["tokens"], 1)
    explore = exploration_stats([tr.key_seq(METRICS_KEY) for tr in trajs], groups, won)
    # Measure the actual contribution after centering/trajectory normalization.
    # Raw positive bonuses can exaggerate influence when most are constant.
    base_adv = compute_advantages([tr.reward for tr in trajs], groups, step_traj, step_h,
                                 cfg.advantage, step_rewards=step_rewards)
    effective_intrinsic = adv - base_adv
    share = float(np.abs(effective_intrinsic).mean() / max(np.abs(adv).mean(), 1e-8))
    return {
        "t_score": score_seconds,
        **explore,
        "intrinsic_share": share,  # mean |A_with_curiosity - A_without| / mean |A_with|
        "success": float(np.mean([tr.won for tr in trajs])),
        "reward": float(scores.mean()),
        "ep_len": float(np.mean([len(tr.steps) for tr in trajs])),
        "invalid_rate": float(np.mean([not s.valid for s in steps])),
        "resp_len_mean": lens["mean"], "resp_len_p99": lens["p99"], "resp_len_max": lens["max"],
        "trunc_rate": lens["trunc_rate"],  # share of responses cut off at max_new_tokens
        "entropy_mean": float(H.mean()), "entropy_std": float(H.std()),
        "adv_std": float(adv.std()),
        "flat_group_frac": float(flat),  # groups where GRPO's Eq. 4 is all zero
        "pg_loss": float(sums["pg"] / n), "kl": float(sums["kl"] / n),
        "clipfrac": float(sums["clipfrac"] / n),
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "n_steps": len(steps),
        "optimizer_steps": len(grad_norms),
    }


def gpu_memory_check(log=print, _warned=[False]) -> dict:
    """Peak VRAM this process reserved since the last call. With the Windows/WSL
    "CUDA Sysmem Fallback Policy" on, going past physical VRAM does not raise OOM:
    it silently spills into shared system RAM and runs several times slower."""
    if not torch.cuda.is_available():
        return {}
    peak = torch.cuda.max_memory_reserved()
    total = torch.cuda.mem_get_info()[1]
    torch.cuda.reset_peak_memory_stats()
    if peak > 0.98 * total and not _warned[0]:
        _warned[0] = True
        log(f"WARNING: peak reserved VRAM {peak / 2**30:.2f} GiB >= physical {total / 2**30:.2f} "
            "GiB, so the run is probably spilling into shared system memory (very slow). Set "
            "NVIDIA Control Panel > Manage 3D settings > CUDA - Sysmem Fallback Policy to "
            "'Prefer No Sysmem Fallback' (see README).")
    return {"gpu_peak_gb": peak / 2**30}


def train(cfg: Config, log=print) -> LLMPolicy:
    tc = cfg.train
    if min(tc.iterations, tc.games_per_iter, tc.max_steps, tc.mini_batches, tc.ppo_epochs) < 1:
        raise ValueError("training counts/horizon must be positive")
    if tc.group_size < 2:
        raise ValueError("GRPO comparisons require group_size >= 2")
    if tc.mini_batch_size < 0:
        raise ValueError("train.mini_batch_size must be nonnegative")
    if not np.isfinite(tc.dual_clip) or (tc.dual_clip != 0 and tc.dual_clip <= 1):
        raise ValueError("train.dual_clip must exceed 1, or be 0 to disable")
    if tc.max_env_steps < 0 or tc.max_seconds < 0:
        raise ValueError("training budgets must be nonnegative")
    if not np.isfinite(tc.entropy_coef) or tc.entropy_coef < 0:
        raise ValueError("train.entropy_coef must be finite and nonnegative")
    if tc.reward_mode not in ("trajectory", "step"):
        raise ValueError("train.reward_mode must be 'trajectory' or 'step'")
    if (not np.isfinite(cfg.advantage.entropy_eps) or cfg.advantage.entropy_eps <= 0 or
            not np.isfinite(cfg.advantage.scale_eps) or cfg.advantage.scale_eps < 0):
        raise ValueError("advantage entropy_eps must be positive and scale_eps nonnegative")
    if cfg.policy.temperature != 1.0 or cfg.policy.top_p != 1.0:
        raise ValueError("on-policy training requires temperature=1 and top_p=1")
    seed_everything(cfg.seed)
    task_rng = np.random.default_rng(cfg.seed)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 1]))
    run_dir = cfg.run_dir
    saved_config = run_dir / "config.json"
    if saved_config.exists() and not config_matches(json.loads(saved_config.read_text()), cfg):
        raise ValueError(f"{run_dir} contains a different configuration; choose a new name= or out_dir=")
    run_dir.mkdir(parents=True, exist_ok=True)
    # An explicitly restarted run must not retain a previous completion marker
    # while its metrics and adapter are being replaced.
    (run_dir / "training_complete.json").unlink(missing_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config_dict(cfg), indent=2))

    bench = make_benchmark(cfg.benchmark, **cfg.env)
    train_tasks = bench.tasks("train")
    if tc.games_per_iter > len(train_tasks):
        raise ValueError("games_per_iter exceeds the available training tasks")
    policy = LLMPolicy(cfg.policy, seed=cfg.seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    tc = cfg.train
    ac, cur = cfg.advantage, cfg.advantage.curiosity
    log(f"[{cfg.name} s{cfg.seed}] {len(train_tasks)} train tasks, advantage={ac.method}"
        + (f" (g={ac.use_g}, f={ac.use_f}, zeta={ac.zeta})" if ac.method == "empg" else "")
        + f", curiosity eta={cur.weight}"
        + (f" ({cur.key}/{cur.scope}/{cur.bonus}/{cur.mode}, schedule={cur.schedule}, "
           f"gate={cur.gate})" if cur.weight else ""))

    gpu_memory_check(log)  # reset the peak counter
    cumulative_steps, cumulative_tokens, cumulative_seconds = 0, 0, 0.0
    stop_reason = "iterations"
    with ExitStack() as cleanup, open(run_dir / "metrics.jsonl", "w") as mf, \
            open(run_dir / "resp_lengths_train.jsonl", "w") as lf:
        if hasattr(bench, "close"):
            cleanup.callback(bench.close)
        cleanup.enter_context(_close_policy_on_error(policy))
        for it in range(1, tc.iterations + 1):
            t0 = time.time()
            chosen = task_rng.choice(len(train_tasks), tc.games_per_iter, replace=False)
            tasks = [train_tasks[c] for c in chosen for _ in range(tc.group_size)]
            groups = [g for g in range(tc.games_per_iter) for _ in range(tc.group_size)]
            rollout_times = {}
            log(f"[{cfg.name} s{cfg.seed}] batch {it}/{tc.iterations}: "
                f"{tc.games_per_iter} games, {len(tasks)} trajectories")
            trajs = run_episodes(policy, bench, tasks, groups, tc.max_steps,
                                 tc.success_reward, tc.invalid_penalty, timings=rollout_times,
                                 progress=lambda turn, active: log(
                                     f"  rollout turn {turn}/{tc.max_steps}, {active} active"))
            t1 = time.time()
            log(f"  scoring and updating batch {it}/{tc.iterations}")
            stats = update(policy, opt, trajs, cfg, rng, iteration=it)
            stats.update(iteration=it, t_rollout=t1 - t0, t_update=time.time() - t1,
                         **rollout_times, **gpu_memory_check(log))
            cumulative_steps += sum(len(tr.steps) for tr in trajs)
            cumulative_tokens += sum(len(s.response_ids) for tr in trajs for s in tr.steps)
            cumulative_seconds += stats["t_rollout"] + stats["t_update"]
            stats.update(env_steps_total=cumulative_steps, response_tokens_total=cumulative_tokens,
                         seconds_total=cumulative_seconds, tasks=[train_tasks[c] for c in chosen])
            mf.write(json.dumps(stats) + "\n")
            mf.flush()
            lf.write(json.dumps({"iteration": it, "cap": cfg.policy.max_new_tokens,
                                 **response_lengths(trajs)}) + "\n")
            lf.flush()
            log(f"it {it:4d} | succ {stats['success']:.2f} | R {stats['reward']:+.2f} "
                f"| len {stats['ep_len']:.1f} | inval {stats['invalid_rate']:.2f} "
                f"| H {stats['entropy_mean']:.3f} | kl {stats['kl']:.4f} "
                f"| revisit {stats['revisit_rate']:.2f} | anysucc {stats['any_success_frac']:.2f} "
                f"| resp p99 {stats['resp_len_p99']:.0f} trunc {stats['trunc_rate']:.2f} "
                f"| {stats['t_rollout']:.0f}s+{stats['t_update']:.0f}s")
            if tc.save_every and it % tc.save_every == 0:
                policy.save(str(run_dir / "adapter"))
            if tc.max_env_steps and cumulative_steps >= tc.max_env_steps:
                stop_reason = "max_env_steps"
                break
            if tc.max_seconds and cumulative_seconds >= tc.max_seconds:
                stop_reason = "max_seconds"
                break
        policy.save(str(run_dir / "adapter"))
        completed = run_dir / "training_complete.json"
        temporary = completed.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "config_fingerprint": config_fingerprint(cfg), "reason": stop_reason,
            "iterations": it, "env_steps": cumulative_steps, "response_tokens": cumulative_tokens,
            "seconds": cumulative_seconds}, indent=2))
        temporary.replace(completed)
    return policy
