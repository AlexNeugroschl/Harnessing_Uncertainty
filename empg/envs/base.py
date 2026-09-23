"""Minimal interfaces an environment/benchmark must provide.

Adding a benchmark (WebShop, Deep Search, ...) means implementing these two
protocols in a new module and registering it in ``make_benchmark``.
"""

from __future__ import annotations

from typing import Protocol


class Env(Protocol):
    def reset(self) -> tuple[str, dict]:
        """Returns (observation, info). info must contain 'admissible' (list[str])
        and 'task' (the goal description)."""

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Returns (observation, reward, done, info). info must contain
        'admissible' and 'won' (bool, task success).

        Optional (reset and step): info['state_keys'], a dict of named hashable
        state identifiers used by curiosity (e.g. {'parsed': ..., 'physical': ...}).
        The rollout always adds 'obs' (the observation text) as a fallback key."""

    def close(self) -> None: ...


class Benchmark(Protocol):
    name: str

    def tasks(self, split: str) -> list[str]:
        """Task identifiers (e.g. game files) for a split."""

    def make_env(self, task: str) -> Env:
        """A fresh env that always plays ``task``."""

    def task_type(self, task: str) -> str:
        """Category used for per-subtask reporting."""

    def prompt(self, task_desc: str, observation: str, admissible: list[str],
               history: list[tuple[str, str]]) -> str:
        """User prompt for the current turn."""


def make_benchmark(name: str, **kwargs) -> Benchmark:
    if name == "alfworld":
        from empg.envs.alfworld_env import ALFWorldBenchmark
        return ALFWorldBenchmark(**kwargs)
    raise ValueError(f"unknown benchmark: {name!r}")
