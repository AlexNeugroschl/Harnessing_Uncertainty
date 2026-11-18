"""Optional, deterministic ALFWorld interventions. No model/numpy imports.

The action interface is identical in both regimes. Inspection is a real extra
step with observable state; no phase label enters observations or state keys.
"""

from dataclasses import dataclass, field, replace
import hashlib
import re


REGIMES = ("baseline", "inspection", "novelty")
_TARGETS = {
    "take": re.compile(r"^take .+ from (.+)$"),
    "open": re.compile(r"^open (.+)$"),
    "clean": re.compile(r"^clean .+ with (.+)$"),
    "heat": re.compile(r"^heat .+ with (.+)$"),
    "cool": re.compile(r"^cool .+ with (.+)$"),
}


@dataclass
class NonstationaryConfig:
    enabled: bool = False
    change: str = "rules"                 # rules | rewards
    schedule: str = "switch"              # switch | stationary
    switch_iteration: int = 11            # first batch in final regime
    switch_iterations: list[int] = field(default_factory=list)  # recurring toggles; empty = one switch
    initial_regime: str = "baseline"
    final_regime: str = "inspection"
    stationary_regime: str = "baseline"
    actions: list[str] = field(default_factory=lambda: ["take"])
    consume_inspection: bool = False       # otherwise reusable until reset
    feedback: str = "explicit"            # explicit hints | generic failure feedback
    probe_every: int = 0                  # 0 disables ALL intermediate evaluation
    probe_tasks: int = 12                 # fixed panel; 0 = entire split
    probe_split: str = "valid_seen"
    probe_regimes: list[str] = field(default_factory=lambda: ["baseline", "inspection"])
    probe_seed: int = 0
    eval_regime: str = "final"            # final | baseline | inspection
    recovery_fraction: float = 0.9
    competence_min_success: float = 0.1
    competence_min_successes: int = 2

    def validate(self, iterations=None):
        if not self.enabled:
            return
        counts = (self.switch_iteration, self.probe_every, self.probe_tasks, self.probe_seed)
        if any(not isinstance(v, int) or isinstance(v, bool) for v in counts):
            raise ValueError("nonstationary iteration, probe counts and seed must be integers")
        if self.change not in ("rules", "rewards") or self.schedule not in ("switch", "stationary", "recurring"):
            raise ValueError("nonstationary change must be rules/rewards; schedule must be switch/stationary")
        if self.feedback not in ("explicit", "generic"):
            raise ValueError("nonstationary.feedback must be explicit or generic")
        if any(r not in REGIMES for r in (self.initial_regime, self.final_regime, self.stationary_regime)):
            raise ValueError("nonstationary regimes must be baseline or inspection")
        if self.eval_regime not in ("final", *REGIMES):
            raise ValueError("invalid nonstationary.eval_regime")
        if not self.actions or len(set(self.actions)) != len(self.actions) or any(a not in _TARGETS for a in self.actions):
            raise ValueError("nonstationary.actions must be unique supported action verbs")
        if self.schedule in ("switch", "recurring") and (self.switch_iteration < 2 or
                (iterations is not None and self.switch_iteration > iterations)):
            raise ValueError("nonstationary.switch_iteration must lie in [2, train.iterations]")
        if self.schedule in ("switch", "recurring") and self.initial_regime == self.final_regime:
            raise ValueError("a switch requires different initial/final regimes; use stationary instead")
        if self.switch_iterations:
            if (any(not isinstance(i, int) or isinstance(i, bool) or i < 2 or
                    (iterations is not None and i > iterations) for i in self.switch_iterations)
                    or self.switch_iterations != sorted(set(self.switch_iterations))
                    or self.switch_iterations[0] != self.switch_iteration):
                raise ValueError("switch_iterations must be sorted unique boundaries starting at switch_iteration")
            if self.schedule == "switch" and len(self.switch_iterations) > 1:
                raise ValueError("multiple boundaries require schedule=recurring or a stationary clock control")
        if self.schedule == "recurring" and len(self.switch_iterations) < 2:
            raise ValueError("recurring requires at least two switch_iterations")
        if self.probe_every < 0 or self.probe_tasks < 0 or self.probe_seed < 0:
            raise ValueError("nonstationary probe counts/seed must be nonnegative")
        if self.probe_split not in ("valid_seen", "valid_unseen"):
            raise ValueError("probes must use a held-out split")
        if not self.probe_regimes or len(set(self.probe_regimes)) != len(self.probe_regimes) or any(
                r not in REGIMES for r in self.probe_regimes):
            raise ValueError("invalid nonstationary.probe_regimes")
        if not 0 < self.recovery_fraction <= 1:
            raise ValueError("nonstationary.recovery_fraction must be in (0, 1]")
        if not 0 < self.competence_min_success <= 1 or not isinstance(self.competence_min_successes, int) or self.competence_min_successes < 1:
            raise ValueError("invalid nonstationary pre-switch competence thresholds")

    def regime_at(self, iteration):
        if self.schedule == "stationary":
            return self.stationary_regime
        changes = sum(iteration >= p for p in self.change_points())
        return self.final_regime if changes % 2 else self.initial_regime

    def change_points(self):
        return self.switch_iterations or [self.switch_iteration]

    def evaluation_regime(self):
        if self.eval_regime != "final":
            return self.eval_regime
        if self.schedule == "recurring":
            return self.final_regime if len(self.change_points()) % 2 else self.initial_regime
        return self.stationary_regime if self.schedule == "stationary" else self.final_regime

    def intervention(self, regime):
        if regime not in REGIMES:
            raise ValueError(f"invalid regime: {regime}")
        spec = {"change": self.change, "regime": regime, "actions": self.actions,
                "consume_inspection": self.consume_inspection}
        if self.feedback != "explicit":
            spec["feedback"] = self.feedback
        return spec

    def probe_iterations(self, iterations):
        if not self.enabled or not self.probe_every:
            return []
        points = {0, iterations, *range(self.probe_every, iterations + 1, self.probe_every)}
        # Include identical boundary probes for stationary controls, too.
        points.update(i for p in self.change_points() for i in (p - 1, p) if 0 <= i <= iterations)
        return sorted(points)

    def probe_eval_config(self, ec):
        return replace(ec, split=self.probe_split, max_tasks=self.probe_tasks,
                       selection_seed=self.probe_seed, sampling_seed=self.probe_seed)


class InspectionEnv:
    """Add an inspection prerequisite or a sparse route-dependent reward.

    Rules: an otherwise admissible selected action fails until its target is
    inspected. Rewards: physical transitions are unchanged, but success pays
    only if every successful selected action was prepared (at least one).
    Both regimes offer the same inspection actions and observation wording.
    """

    def __init__(self, env, *, change, regime, actions, consume_inspection=False, feedback="explicit"):
        if regime not in ("baseline", "inspection"):
            raise ValueError("ALFWorld inspection wrapper requires baseline/inspection regimes")
        self.env, self.change, self.regime = env, change, regime
        self.patterns = {a: _TARGETS[a] for a in actions}
        self.consume = consume_inspection
        self.feedback = feedback

    def _target(self, action):
        pattern = self.patterns.get(action.partition(" ")[0])
        match = pattern.fullmatch(action) if pattern else None
        return match[1] if match else None

    def _decorate(self, obs, info, **metrics):
        info = dict(info)
        self.native_actions = frozenset(info["admissible"])
        self.targets = {t for a in self.native_actions if (t := self._target(a)) is not None}
        info["admissible"] = [*info["admissible"], *(f"inspect {t}" for t in sorted(self.targets))]
        # Preserve all native key variants, including privileged diagnostic keys.
        # Inspection history is observable in both regimes; the regime is not.
        if self.inspected or self.prepared or self.unprepared:
            # Route eligibility is history-dependent but observable; use flags,
            # not growing counters that would reward endless repetition.
            state = (sorted(self.inspected), self.prepared > 0, self.unprepared > 0)
            suffix = hashlib.sha256(repr(state).encode()).hexdigest()[:16]
            info["state_keys"] = {k: f"{v}|inspection:{suffix}" for k, v in info.get("state_keys", {}).items()}
        info["intervention_metrics"] = metrics
        self.obs, self.info = obs, info
        return obs, info

    def reset(self):
        self.inspected, self.prepared, self.unprepared = set(), 0, 0
        obs, info = self.env.reset()
        self._base_info = info
        return self._decorate(obs, info)

    def step(self, action):
        if action.startswith("inspect ") and action[8:] in self.targets:
            target = action[8:]
            self.inspected.add(target)
            obs = f"You inspect {target}."
            if self.feedback == "explicit":
                obs += " It is ready for use."
            obs, info = self._decorate(obs, self._base_info, inspections=1)
            return obs, 0.0, False, info
        target = self._target(action) if action in self.native_actions else None
        ready = target in self.inspected
        if self.change == "rules" and self.regime == "inspection" and target is not None and not ready:
            obs = (f"Nothing happens. You need to inspect {target} before that action."
                   if self.feedback == "explicit" else "Nothing happens. The operation failed.")
            obs, info = self._decorate(obs, self._base_info, rule_blocks=1)
            return obs, 0.0, False, info
        obs, reward, done, info = self.env.step(action)
        # Native ALFWorld reports failed actions with this exact prefix.
        successful = target is not None and not obs.strip().startswith("Nothing happens")
        if successful:
            self.prepared += int(ready)
            self.unprepared += int(not ready)
            if self.consume:
                self.inspected.discard(target)
        base_won = bool(info.get("won", False))
        qualified = self.prepared > 0 and self.unprepared == 0
        if self.change == "rewards" and self.regime == "inspection":
            # Native intermediate scores cannot accidentally pay the old route.
            reward = float(base_won and qualified)
            info = dict(info, won=base_won and qualified)
            if base_won:
                obs += ("\nReward received." if qualified else
                        "\nNo reward: the selected operations were not all inspected first."
                        if self.feedback == "explicit" else "\nNo reward received.")
        self._base_info = info
        obs, info = self._decorate(obs, info, base_won=base_won,
                                  prepared_actions=int(successful and ready),
                                  alternative_success=int(base_won and qualified))
        return obs, reward, done, info

    def close(self):
        self.env.close()


def wrap_env(env, intervention):
    return InspectionEnv(env, **intervention) if intervention else env
