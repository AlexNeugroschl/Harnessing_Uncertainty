"""Curiosity: count-based novelty bonuses over visited states (pure numpy).

See docs/curiosity_research_plan.md. The bonus for step t is computed on the
state the action led to, s_{t+1} (like EMPG's clarity bonus f), and is added to
the advantage in ``compute_advantages``:

    A_mod(i,t) = A_i * g(H_t) + zeta * f(H_{t+1}) + eta * c_t

Data layout: one key sequence per trajectory, ``seq = [k_0, k_1, ..., k_T]``,
where k_0 is the reset state and k_t (t >= 1) the state after step t. Per-step
outputs are flat arrays in trajectory-then-time order, matching
``compute_advantages``.

Scopes
    episodic  n = visits to s_{t+1} so far in this episode (k_0 counts)
    group     n = number of sibling rollouts (same task) that ever reach s_{t+1}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CuriosityConfig:
    weight: float = 0.0          # eta; 0 disables curiosity (bit-identical to the baseline)
    key: str = "parsed"          # state key: parsed | physical | knowledge | obs
    scope: str = "episodic"      # episodic | group
    bonus: str = "inv_sqrt"      # inv_sqrt: 1/sqrt(n) | first_visit: 1[n == 1]
    mode: str = "step"           # step: per-step advantage term | trajectory: added to outcome reward
    schedule: str = "constant"   # constant | linear | exp
    decay_iters: int = 0         # linear: iterations to reach 0 (0 = whole run); exp: half-life (0 = whole run)
    gate: str = "none"           # none | no_success_signal (only groups whose rollouts all share one outcome)
    modulate: str = "none"       # none | confident (x (1 - H_norm_t)) | uncertain (x H_norm_t)
    cap: float = 0.0             # max summed bonus per episode (0 = no cap)


def episodic_counts(seqs: list[list[str]]) -> list[np.ndarray]:
    """For each step t >= 1: how many times s_t has occurred in seq[0..t]."""
    out = []
    for seq in seqs:
        seen: dict[str, int] = {}
        counts = []
        for i, k in enumerate(seq):
            seen[k] = seen.get(k, 0) + 1
            if i > 0:
                counts.append(seen[k])
        out.append(np.array(counts, dtype=np.float64))
    return out


def group_counts(seqs: list[list[str]], groups) -> list[np.ndarray]:
    """For each step t >= 1: how many trajectories in the same group ever reach s_t."""
    groups = np.asarray(groups)
    visitors: dict[tuple, int] = {}
    for seq, g in zip(seqs, groups):
        for k in set(seq):
            visitors[(g, k)] = visitors.get((g, k), 0) + 1
    return [np.array([visitors[(g, k)] for k in seq[1:]], dtype=np.float64)
            for seq, g in zip(seqs, groups)]


def novelty_bonus(counts: np.ndarray, kind: str = "inv_sqrt") -> np.ndarray:
    if kind == "inv_sqrt":
        return 1.0 / np.sqrt(counts)
    if kind == "first_visit":
        return (counts == 1).astype(np.float64)
    raise ValueError(f"unknown bonus: {kind!r}")


def apply_cap(bonus: np.ndarray, cap: float) -> np.ndarray:
    """Truncate so one episode's summed bonus never exceeds ``cap``."""
    if cap <= 0:
        return bonus
    before = np.concatenate([[0.0], np.cumsum(bonus)[:-1]])
    return np.clip(np.minimum(bonus, cap - before), 0.0, None)


def schedule_multiplier(cfg: CuriosityConfig, iteration: int, total_iters: int) -> float:
    """Weight multiplier at a 1-based training iteration."""
    horizon = cfg.decay_iters or max(total_iters, 1)
    k = max(iteration - 1, 0)
    if cfg.schedule == "constant":
        return 1.0
    if cfg.schedule == "linear":
        return max(0.0, 1.0 - k / horizon)
    if cfg.schedule == "exp":
        return 0.5 ** (k / horizon)
    raise ValueError(f"unknown schedule: {cfg.schedule!r}")


def no_success_signal(groups, won) -> np.ndarray:
    """Per trajectory: True if every rollout in its group had the same outcome
    (all failed or all succeeded), i.e. GRPO gets no task signal from the group."""
    groups, won = np.asarray(groups), np.asarray(won, dtype=bool)
    flat = {g: won[groups == g].min() == won[groups == g].max() for g in np.unique(groups)}
    return np.array([flat[g] for g in groups])


def intrinsic_rewards(seqs, groups, won, cfg: CuriosityConfig, iteration: int = 1,
                      total_iters: int = 1, h_norm=None) -> np.ndarray:
    """Weighted per-step curiosity term eta_k * c_t, flat over the batch."""
    if cfg.scope == "episodic":
        counts = episodic_counts(seqs)
    elif cfg.scope == "group":
        counts = group_counts(seqs, groups)
    else:
        raise ValueError(f"unknown scope: {cfg.scope!r}")
    bonus = [apply_cap(novelty_bonus(c, cfg.bonus), cfg.cap) for c in counts]

    if cfg.gate == "no_success_signal":
        keep = no_success_signal(groups, won)
        bonus = [b * keep[i] for i, b in enumerate(bonus)]
    elif cfg.gate != "none":
        raise ValueError(f"unknown gate: {cfg.gate!r}")

    c = np.concatenate(bonus) if bonus else np.zeros(0)
    if cfg.modulate != "none":
        if h_norm is None:
            raise ValueError("modulate needs h_norm")
        h = np.asarray(h_norm, dtype=np.float64)
        c = c * ((1.0 - h) if cfg.modulate == "confident" else h)
    return cfg.weight * schedule_multiplier(cfg, iteration, total_iters) * c


def exploration_stats(seqs, groups, won) -> dict:
    """Exploration metrics for a batch (plan Sec. 6), from key sequences."""
    groups, won = np.asarray(groups), np.asarray(won, dtype=bool)
    ep = episodic_counts(seqs)
    steps = np.concatenate(ep) if ep else np.zeros(0)
    lengths = np.array([len(s) - 1 for s in seqs])
    per_group = [(groups == g) for g in np.unique(groups)]
    stays = [a == b for s in seqs for a, b in zip(s[:-1], s[1:])]
    return {
        "unique_states": float(np.mean([len(set(s)) for s in seqs])),
        # A step revisits if s_{t+1} was seen earlier this episode. That includes
        # no-ops (failed/idle actions: s_{t+1} == s_t); noop_rate separates them
        # from genuine loops (revisit_rate - noop_rate).
        "revisit_rate": float((steps > 1).mean()) if len(steps) else 0.0,
        "noop_rate": float(np.mean(stays)) if stays else 0.0,
        "group_coverage": float(np.mean([len({k for i in np.flatnonzero(m) for k in seqs[i]})
                                         for m in per_group])),
        "any_success_frac": float(np.mean([won[m].any() for m in per_group])),
        "no_signal_frac": float(np.mean([won[m].min() == won[m].max() for m in per_group])),
        "success_len": float(lengths[won].mean()) if won.any() else float("nan"),
    }
