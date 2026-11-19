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
    boost: str = "none"         # none | oracle_window | clock_window | observed_failure
    boost_factor: float = 3.0
    boost_updates: int = 4
    failure_threshold: float = 0.15  # increase over prior EMA, not absolute failure rate
    failure_ema: float = 0.2
    detector_warmup: int = 2
    detector_rearm: str = "legacy"  # legacy | stable: one event until a new stable failure baseline
    rearm_updates: int = 2
    rearm_tolerance: float = 0.05

    def validate(self):
        if self.boost not in ("none", "oracle_window", "clock_window", "observed_failure"):
            raise ValueError("unknown curiosity boost")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("curiosity weight must be finite and nonnegative")
        if not np.isfinite(self.boost_factor) or self.boost_factor < 1:
            raise ValueError("curiosity boost_factor must be finite and >= 1")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (self.boost_updates, self.detector_warmup)):
            raise ValueError("curiosity boost_updates/detector_warmup must be positive integers")
        if not 0 < self.failure_ema <= 1 or not 0 < self.failure_threshold <= 1:
            raise ValueError("invalid curiosity failure detector parameters")
        if (self.detector_rearm not in ("legacy", "stable")
                or not isinstance(self.rearm_updates, int) or isinstance(self.rearm_updates, bool)
                or self.rearm_updates < 1 or not 0 < self.rearm_tolerance <= 1
                or (self.detector_rearm == "stable" and self.rearm_tolerance >= self.failure_threshold)):
            raise ValueError("invalid curiosity detector rearming parameters")


class CuriosityController:
    """One scalar per update; observed detection uses completed batches only.

    Oracle timing is experimental privileged information. It never enters an
    observation, action set, state key, or the observable detector.
    """

    def __init__(self, cfg):
        cfg.validate()
        self.cfg = cfg
        self.ema = None
        self.seen = 0
        self.until = 0
        self.start = 0
        self.triggered = False
        self.armed = True
        self.stable_updates = 0

    def multiplier(self, iteration, nonstationary):
        c = self.cfg
        if c.boost == "none":
            return 1.0
        if c.boost in ("oracle_window", "clock_window"):
            if not nonstationary.enabled:
                return 1.0
            if c.boost == "oracle_window" and nonstationary.schedule == "stationary":
                return 1.0
            points = nonstationary.change_points()
            active = any(p <= iteration < p + c.boost_updates for p in points)
        else:
            active = self.start <= iteration <= self.until
        return c.boost_factor if active else 1.0

    def observe(self, iteration, failure_rate):
        self.triggered = False
        if self.cfg.boost != "observed_failure":
            return
        if failure_rate is None or not np.isfinite(failure_rate):
            raise ValueError("observed_failure boost requires observable action failure feedback")
        if (self.ema is not None and self.seen >= self.cfg.detector_warmup
                and self.armed
                and iteration > self.until
                and failure_rate - self.ema >= self.cfg.failure_threshold):
            # The next update is the first affected one; no hindsight weighting.
            self.until = iteration + self.cfg.boost_updates
            self.start = iteration + 1
            self.triggered = True
            if self.cfg.detector_rearm == "stable":
                self.armed, self.stable_updates = False, 0
        elif self.cfg.detector_rearm == "stable" and not self.armed:
            # Compare to the PRIOR EMA. Persistent drift cannot rearm merely
            # because the boost window ended. A stable new baseline can.
            stable = iteration > self.until and abs(failure_rate - self.ema) <= self.cfg.rearm_tolerance
            self.stable_updates = self.stable_updates + 1 if stable else 0
            if self.stable_updates >= self.cfg.rearm_updates:
                self.armed = True
        alpha = self.cfg.failure_ema
        self.ema = failure_rate if self.ema is None else (1 - alpha) * self.ema + alpha * failure_rate
        self.seen += 1


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
                      total_iters: int = 1, h_norm=None, multiplier: float = 1.0) -> np.ndarray:
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
    if not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("curiosity multiplier must be finite and nonnegative")
    return cfg.weight * schedule_multiplier(cfg, iteration, total_iters) * multiplier * c


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
