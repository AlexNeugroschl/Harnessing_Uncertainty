"""Training loop: the same policy gradient, two ways of weighting it.

The only difference between the GRPO arm and the EMPG arm is how the per-step
advantage is computed:

    GRPO  A(i,t) = A_i                       (Eq. 4, broadcast to every step)
    EMPG  A(i,t) = A_i*g(H_t) + z*f(H_t+1)   (Eq. 8), then re-centred (Eq. 13)

Everything else - rollouts, seeds, features, optimiser, loss - is identical, so
any difference in the learning curve is attributable to the advantage
weighting rather than to sampling or tuning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from .agent import PolicyHead, Trajectory, rollout
from .empg import (
    DEFAULT_K,
    DEFAULT_K_PRIME,
    DEFAULT_ZETA,
    broadcast_advantages,
    compute_empg_advantages,
    finalize_advantages,
    group_advantages,
)
from .env import TASKS, make_env
from .llm import OllamaClient


@dataclass
class TrainConfig:
    iterations: int = 12
    group_size: int = 8          # G in the paper (8 for ALFWorld)
    lr: float = 3e-3
    temperature: float = 1.0
    k: float = DEFAULT_K
    k_prime: float = DEFAULT_K_PRIME
    zeta: float = DEFAULT_ZETA
    seed: int = 0
    env_seed: int = 3


@dataclass
class TrainLog:
    success_rate: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    mean_entropy: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)


def collect_group(
    client: OllamaClient,
    head: PolicyHead,
    task_name: str,
    cfg: TrainConfig,
) -> list[Trajectory]:
    """Sample G rollouts on one task - the group GRPO computes advantages over."""
    return [
        rollout(
            make_env(task_name, seed=cfg.env_seed),
            client,
            head=head,
            temperature=cfg.temperature,
        )
        for _ in range(cfg.group_size)
    ]


def flatten(trajectories: list[Trajectory]):
    """Flatten a group into the step-ordered arrays EMPG's equations expect."""
    entropies, done, lengths = [], [], []
    for traj in trajectories:
        lengths.append(len(traj.steps))
        for i, step in enumerate(traj.steps):
            entropies.append(step.entropy)
            done.append(i == len(traj.steps) - 1)
    return np.array(entropies), np.array(done, dtype=bool), np.array(lengths)


def compute_advantages(
    trajectories: list[Trajectory], method: str, cfg: TrainConfig
) -> np.ndarray:
    """Per-step advantages for one group, under GRPO or EMPG.

    ``method`` is one of: ``grpo``, ``empg``, ``empg_g`` (scaling only),
    ``empg_f`` (clarity bonus only). The last two reproduce the paper's
    Table 2 ablation.
    """
    rewards = np.array([t.reward for t in trajectories])
    entropies, done, lengths = flatten(trajectories)

    # Eq. 4, then broadcast the one sparse outcome scalar over every step.
    traj_adv = group_advantages(rewards)
    step_adv = broadcast_advantages(traj_adv, lengths)

    if method == "grpo":
        # The baseline: uniform credit, re-centred so both arms are compared
        # on equal footing (Eq. 13 applies to EMPG, so apply it here too).
        return finalize_advantages(step_adv)

    use_scaling = method in ("empg", "empg_g")
    use_clarity = method in ("empg", "empg_f")
    return compute_empg_advantages(
        step_adv, entropies, done,
        k=cfg.k, k_prime=cfg.k_prime, zeta=cfg.zeta,
        use_scaling=use_scaling, use_clarity=use_clarity,
    )


def policy_loss(
    head: PolicyHead, trajectories: list[Trajectory], advantages: np.ndarray
) -> torch.Tensor:
    """REINFORCE-style loss: -(A * log pi(a_t|s_t)), averaged over steps."""
    log_probs = []
    for traj in trajectories:
        for step in traj.steps:
            feats = torch.tensor(step.features, dtype=torch.float32)
            log_probs.append(head.action_log_probs(feats)[step.action_index])

    lp = torch.stack(log_probs)
    adv = torch.tensor(advantages, dtype=torch.float32)
    return -(adv * lp).mean()


def train(
    method: str,
    client: OllamaClient,
    cfg: TrainConfig,
    tasks: list[str] | None = None,
    verbose: bool = True,
) -> tuple[PolicyHead, TrainLog]:
    """Train one arm. Seeds are fixed so arms differ only in the advantage."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    head = PolicyHead()
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr)
    log = TrainLog()
    task_names = tasks or [t.name for t in TASKS]

    for it in range(cfg.iterations):
        successes, all_ent, losses, grads = [], [], [], []

        for task_name in task_names:
            group = collect_group(client, head, task_name, cfg)
            successes.extend(t.success for t in group)
            all_ent.extend(s.entropy for t in group for s in t.steps)

            adv = compute_advantages(group, method, cfg)
            loss = policy_loss(head, group, adv)

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            opt.step()

            losses.append(float(loss.item()))
            grads.append(float(grad_norm))

        log.success_rate.append(float(np.mean(successes)))
        log.mean_entropy.append(float(np.mean(all_ent)))
        log.loss.append(float(np.mean(losses)))
        log.grad_norm.append(float(np.mean(grads)))

        if verbose:
            print(
                f"  [{method:7s}] iter {it + 1:2d}/{cfg.iterations}  "
                f"success={log.success_rate[-1]:.1%}  "
                f"loss={log.loss[-1]:+.4f}  |grad|={log.grad_norm[-1]:.3f}"
            )

    return head, log


def evaluate(
    head: PolicyHead,
    client: OllamaClient,
    cfg: TrainConfig,
    tasks: list[str] | None = None,
) -> float:
    """Greedy success rate after training - the headline metric.

    Greedy action selection over cached LLM scores is deterministic, so one
    episode per task is sufficient; repeating it would return the same result.
    """
    task_names = tasks or [t.name for t in TASKS]
    wins = sum(
        rollout(make_env(t, seed=cfg.env_seed), client, head=head, greedy=True).success
        for t in task_names
    )
    return wins / len(task_names)
