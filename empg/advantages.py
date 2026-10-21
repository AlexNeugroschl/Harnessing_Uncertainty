"""Advantage estimators: GRPO baseline and EMPG (arXiv 2509.09265, Sec. 3).

Everything here is pure numpy so the math is testable without a model.

Data layout (shared by every estimator):
    traj_rewards  [n_traj]   one outcome reward per trajectory
    group_ids     [n_traj]   rollouts of the same task share a group id
    step_traj     [n_steps]  trajectory index of each step; steps of a trajectory
                             are contiguous and in time order
    step_entropy  [n_steps]  raw step entropy H_t in nats (Eq. 7, from the policy)

Equation map
    Eq. 4   grpo_advantages      group-relative outcome advantage
    Eq. 12  normalize_entropy    batch min-max scaling
    Eq. 10  scale_g              self-calibrating gradient scale (batch mean 1)
    Eq. 11  clarity_f            future-clarity bonus on H_{t+1}
    Eq. 8   A_mod = A_i * g(H_t) + zeta * f(H_{t+1})
    Eq. 13  A_final = A_mod - batch mean

Optional, not in the paper: a curiosity term ``eta * c_t`` (``empg/curiosity.py``)
added to A_mod before Eq. 13, or to the outcome reward before Eq. 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from empg.curiosity import CuriosityConfig

EPS = 1e-6


@dataclass
class AdvantageConfig:
    method: str = "empg"      # "grpo" | "empg"
    use_g: bool = True        # Eq. 10 gradient scaling
    use_f: bool = True        # Eq. 11 future-clarity bonus
    k: float = 1.0            # Eq. 10 exponent
    k_prime: float = 1.0      # Eq. 11 exponent
    zeta: float = 0.05        # Eq. 8 bonus weight (0.05 ALFWorld/WebShop, 0.1 Deep Search)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)  # not in the paper


def grpo_advantages(traj_rewards, group_ids, eps: float = EPS) -> np.ndarray:
    """Eq. 4: A_i = (r_i - mean_group r) / (std_group r + eps), per trajectory.

    Uses the unbiased std like verl; a singleton group gets mean 0 / std 1.
    """
    r = np.asarray(traj_rewards, dtype=np.float64)
    g = np.asarray(group_ids)
    adv = np.zeros_like(r)
    for gid in np.unique(g):
        m = g == gid
        if m.sum() == 1:
            mean, std = 0.0, 1.0
        else:
            mean, std = r[m].mean(), r[m].std(ddof=1)
        adv[m] = (r[m] - mean) / (std + eps)
    return adv


def normalize_entropy(h, eps: float = EPS) -> np.ndarray:
    """Eq. 12: min-max normalise step entropies over the whole batch."""
    h = np.asarray(h, dtype=np.float64)
    return (h - h.min()) / (h.max() - h.min() + eps)


def scale_g(h_norm, k: float = 1.0) -> np.ndarray:
    """Eq. 10: g = exp(-k H_norm) / mean_batch exp(-k H_norm).

    Mean over the batch is exactly 1, so g redistributes update magnitude from
    uncertain (g < 1) to confident (g > 1) steps. It multiplies A, so the sign is
    kept: confident errors are penalised harder.
    """
    raw = np.exp(-k * np.asarray(h_norm, dtype=np.float64))
    return raw / raw.mean()


def clarity_f(h_norm_next, k_prime: float = 1.0) -> np.ndarray:
    """Eq. 11: f = exp(-k' H_norm,t+1). NaN (no successor) -> 0 bonus."""
    h = np.asarray(h_norm_next, dtype=np.float64)
    return np.where(np.isnan(h), 0.0, np.exp(-k_prime * np.nan_to_num(h)))


def next_step_entropy(h_norm, step_traj) -> np.ndarray:
    """H_norm of step t+1 within the same trajectory; NaN for terminal steps.

    The paper does not define f for the last step (there is no t+1). We give it
    no bonus, which also keeps entropy from leaking across trajectories.
    """
    h = np.asarray(h_norm, dtype=np.float64)
    tr = np.asarray(step_traj)
    nxt = np.full_like(h, np.nan)
    same = tr[1:] == tr[:-1]
    nxt[:-1][same] = h[1:][same]
    return nxt


def compute_advantages(traj_rewards, group_ids, step_traj, step_entropy,
                       cfg: AdvantageConfig, intrinsic=None, step_rewards=None) -> np.ndarray:
    """Per-step advantages for the batch, according to ``cfg.method``.

    ``intrinsic`` is an optional, already-weighted per-step curiosity term
    (``empg.curiosity.intrinsic_rewards``). With ``cfg.curiosity.mode ==
    "trajectory"`` its per-episode mean is added to the outcome reward before
    Eq. 4; otherwise it is added per step (then centred, like Eq. 13).
    """
    step_traj = np.asarray(step_traj)
    rewards = np.asarray(traj_rewards, dtype=np.float64)
    scores = np.asarray(step_rewards, dtype=np.float64) if step_rewards is not None else None
    if scores is not None and scores.shape != step_traj.shape:
        raise ValueError("step_rewards must contain one score per response step")
    if intrinsic is not None and cfg.curiosity.mode == "trajectory":
        per_traj = np.bincount(step_traj, weights=intrinsic, minlength=len(rewards))
        bonus = per_traj / np.maximum(np.bincount(step_traj, minlength=len(rewards)), 1)
        if scores is None:
            rewards = rewards + bonus
        else:
            scores = scores + bonus[step_traj]
        intrinsic = None
    elif intrinsic is not None and cfg.curiosity.mode != "step":
        raise ValueError(f"unknown curiosity mode: {cfg.curiosity.mode!r}")

    if scores is None:
        a_traj = grpo_advantages(rewards, group_ids)
        a_step = a_traj[step_traj]  # legacy: episode advantage broadcast to every step
    else:
        # Verl-Agent represents each response as a row with its episode outcome
        # minus that response's invalid-format penalty, then normalizes over rows.
        a_step = grpo_advantages(scores, np.asarray(group_ids)[step_traj])

    if cfg.method == "grpo":
        if intrinsic is None:
            return a_step
        return a_step + (intrinsic - np.mean(intrinsic))
    if cfg.method != "empg":
        raise ValueError(f"unknown advantage method: {cfg.method!r}")

    h_norm = normalize_entropy(step_entropy)
    a_mod = a_step * scale_g(h_norm, cfg.k) if cfg.use_g else a_step.copy()
    if cfg.use_f:
        a_mod += cfg.zeta * clarity_f(next_step_entropy(h_norm, step_traj), cfg.k_prime)
    if intrinsic is not None:
        a_mod += intrinsic
    return a_mod - a_mod.mean()  # Eq. 13
