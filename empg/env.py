"""A small ALFWorld-style text household environment.

Deliberately shaped to reproduce the setting EMPG targets (the paper's
ALFWorld benchmark), with three properties that matter:

* **Sparse, outcome-only reward.** +1 on task completion, 0 everywhere else.
  No shaped intermediate rewards - that absence *is* the credit-assignment
  problem the paper attacks.
* **Long horizon.** 5-10 steps, so a single terminal scalar has to explain a
  whole sequence of decisions.
* **A mix of forced and genuinely ambiguous steps.** Some steps have one
  sensible action (low entropy); others ask the agent to pick among several
  indistinguishable containers (high entropy). Without both kinds there is no
  spread for Eq. 12 to normalise and EMPG would have nothing to modulate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Task:
    name: str
    goal: str
    setting: str
    # Ambiguous steps: the target is hidden among look-alike containers, so no
    # amount of reasoning can single it out. These are the high-entropy steps.
    containers: list[str] = field(default_factory=list)
    target_container: str | None = None


TASKS: list[Task] = [
    Task(
        name="clean_mug",
        goal="put a clean mug on the desk",
        setting="You are in a kitchen. You see a counter, a sink, a desk, and a fridge.",
        containers=["cabinet 1", "cabinet 2", "cabinet 3"],
        target_container="cabinet 2",
    ),
    Task(
        name="find_key",
        goal="find the key and unlock the box",
        setting="You are in a cluttered attic. You see a locked box, a shelf, and a rug.",
        containers=["drawer 1", "drawer 2", "drawer 3", "trunk"],
        target_container="drawer 3",
    ),
    Task(
        name="heat_food",
        goal="heat the food and place it on the table",
        setting="You are in a kitchen. You see a microwave, a table, and a pantry.",
        containers=["shelf 1", "shelf 2", "shelf 3"],
        target_container="shelf 1",
    ),
]


class TextHouseholdEnv:
    """Scripted multi-step household task with sparse terminal reward.

    The plan is a fixed sequence of *phases*. Most phases accept exactly one
    correct action; the ``search`` phase accepts only the one container that
    actually holds the object, and the agent cannot tell which from the
    observation. That is the deliberate source of irreducible uncertainty.
    """

    MAX_STEPS = 10

    def __init__(self, task: Task, rng: random.Random | None = None) -> None:
        self.task = task
        self.rng = rng or random.Random(0)
        self.reset()

    # -- plan construction ---------------------------------------------------
    def _phases(self) -> list[dict]:
        t = self.task
        if t.name == "clean_mug":
            return [
                {"kind": "goto",
                 "obs": t.setting + " The mug is stored in one of the cabinets.",
                 "correct": ["go to counter"],
                 "options": ["go to counter", "go to sink", "go to desk", "go to fridge"]},
                {"kind": "search", "obs": "The counter is bare. The mug must be in a cabinet.",
                 "correct": [f"open {t.target_container}"],
                 "options": [f"open {c}" for c in t.containers]},
                {"kind": "take", "obs": "You see a dirty mug inside.",
                 "correct": ["take mug"],
                 "options": ["take mug", "close cabinet", "go to desk"]},
                {"kind": "act", "obs": "You are holding a dirty mug.",
                 "correct": ["clean mug with sink"],
                 "options": ["clean mug with sink", "put mug on desk", "heat mug"]},
                {"kind": "goto", "obs": "The mug is now clean.",
                 "correct": ["go to desk"],
                 "options": ["go to desk", "go to fridge", "go to counter"]},
                {"kind": "final", "obs": "You are at the desk holding a clean mug.",
                 "correct": ["put mug on desk"],
                 "options": ["put mug on desk", "put mug on counter", "drop mug"]},
            ]
        if t.name == "find_key":
            return [
                {"kind": "goto", "obs": t.setting,
                 "correct": ["go to shelf"],
                 "options": ["go to shelf", "go to box", "look under rug"]},
                {"kind": "search", "obs": "The shelf holds several closed drawers and a trunk.",
                 "correct": [f"open {t.target_container}"],
                 "options": [f"open {c}" for c in t.containers]},
                {"kind": "take", "obs": "A small brass key lies inside.",
                 "correct": ["take key"],
                 "options": ["take key", "close drawer", "go to box"]},
                {"kind": "goto", "obs": "You are holding the key.",
                 "correct": ["go to box"],
                 "options": ["go to box", "go to shelf", "look under rug"]},
                {"kind": "final", "obs": "You are at the locked box, holding the key.",
                 "correct": ["unlock box with key"],
                 "options": ["unlock box with key", "open box", "shake box"]},
            ]
        return [
            {"kind": "goto", "obs": t.setting,
             "correct": ["go to pantry"],
             "options": ["go to pantry", "go to microwave", "go to table"]},
            {"kind": "search", "obs": "The pantry has several shelves.",
             "correct": [f"open {t.target_container}"],
             "options": [f"open {c}" for c in t.containers]},
            {"kind": "take", "obs": "You find a bowl of cold food.",
             "correct": ["take bowl"],
             "options": ["take bowl", "close shelf", "go to table"]},
            {"kind": "act", "obs": "You are holding cold food.",
             "correct": ["heat bowl with microwave"],
             "options": ["heat bowl with microwave", "put bowl on table", "clean bowl"]},
            {"kind": "final", "obs": "The food is hot.",
             "correct": ["put bowl on table"],
             "options": ["put bowl on table", "put bowl in pantry", "drop bowl"]},
        ]

    # -- gym-ish API ---------------------------------------------------------
    def reset(self) -> tuple[str, list[str]]:
        self.plan = self._phases()
        self.idx = 0
        self.steps = 0
        self.done = False
        self.success = False
        self.failed_here = 0
        # Success requires clearing the *final* phase correctly, not merely
        # running out of plan, so a skipped-ahead episode can never be scored
        # as a win.
        self.cleared_final = False
        return self.observation(), self.valid_actions()

    def observation(self) -> str:
        if self.done:
            return "Task finished."
        return self.plan[self.idx]["obs"]

    def valid_actions(self) -> list[str]:
        if self.done:
            return []
        opts = list(self.plan[self.idx]["options"])
        self.rng.shuffle(opts)
        return opts

    def correct_actions(self) -> list[str]:
        return [] if self.done else list(self.plan[self.idx]["correct"])

    @property
    def phase_kind(self) -> str:
        return "done" if self.done else self.plan[self.idx]["kind"]

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Advance one step. Reward is 0 until the task completes (+1)."""
        if self.done:
            return self.observation(), 0.0, True, {"success": self.success}

        phase = self.plan[self.idx]
        correct = action.strip().lower() in [c.lower() for c in phase["correct"]]
        self.steps += 1
        info = {"correct": correct, "kind": phase["kind"], "phase_index": self.idx}

        if correct:
            was_final = phase["kind"] == "final"
            self.idx += 1
            self.failed_here = 0
            if self.idx >= len(self.plan):
                self.done = True
                # Only a correctly-executed final phase counts as success.
                self.success = was_final
                if self.success:
                    # Sparse reward: the only non-zero signal in the episode.
                    return "You completed the task.", 1.0, True, {**info, "success": True}
                return "The episode ended unfinished.", 0.0, True, {**info, "success": False}
        else:
            # A wrong action costs a step. After a couple of misses on the same
            # phase we advance anyway with a nudge, so a single bad guess does
            # not deadlock the whole episode. Without this the agent can burn
            # its entire budget re-picking one wrong option, every rollout in
            # the group fails identically, and GRPO's advantage collapses to
            # zero - leaving nothing to compare EMPG against.
            self.failed_here += 1
            if self.failed_here >= 2:
                self.failed_here = 0
                self.idx += 1
                if self.idx >= len(self.plan):
                    self.done = True
                    return "You ran out of moves.", 0.0, True, {**info, "success": False}
                return (
                    f"That did not work. {self.observation()}",
                    0.0, False, info,
                )

        if self.steps >= self.MAX_STEPS:
            self.done = True
            return "You ran out of time.", 0.0, True, {**info, "success": False}

        return self.observation(), 0.0, False, info


def make_env(task_name: str | None = None, seed: int = 0) -> TextHouseholdEnv:
    rng = random.Random(seed)
    if task_name:
        task = next(t for t in TASKS if t.name == task_name)
    else:
        task = rng.choice(TASKS)
    return TextHouseholdEnv(task, rng)
