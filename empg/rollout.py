"""Multi-turn rollouts: many envs stepped in lockstep with batched generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from empg.envs.base import Benchmark
from empg.envs.batch import SerialBatch
from empg.prompts import parse_action


@dataclass
class Step:
    prompt_ids: list[int]
    response_ids: list[int]
    text: str
    action: str
    valid: bool
    entropy: float = 0.0  # H_t (Eq. 7), filled in after scoring
    obs: str = ""         # observation the action led to (s_{t+1})
    state_keys: dict = field(default_factory=dict)  # keys of s_{t+1} (curiosity)
    truncated: bool = False  # response hit max_new_tokens before EOS


@dataclass
class Trajectory:
    task: str
    group: int
    steps: list[Step] = field(default_factory=list)
    won: bool = False
    reward: float = 0.0  # outcome reward used by the advantage estimator
    init_keys: dict = field(default_factory=dict)  # state keys of the reset state s_0

    @property
    def n_invalid(self) -> int:
        return sum(not s.valid for s in self.steps)

    def key_seq(self, key: str) -> list[str]:
        """[k_0, k_1, ..., k_T] for one key; falls back to the observation text."""
        return [d.get(key, d["obs"]) for d in [self.init_keys, *(s.state_keys for s in self.steps)]]


def _keys(obs: str, info: dict) -> dict:
    keys = dict(info.get("state_keys", {}))
    keys["obs"] = obs
    return keys


def run_episodes(policy, bench: Benchmark, tasks: list[str], groups: list[int],
                 max_steps: int, success_reward: float = 10.0,
                 invalid_penalty: float = 0.1, greedy: bool = False,
                 timings: dict | None = None) -> list[Trajectory]:
    """Play one episode per (task, group) pair; all envs advance one turn per
    batched generate() call."""
    if len(tasks) != len(groups) or not tasks or max_steps < 1:
        raise ValueError("rollout needs matching nonempty tasks/groups and positive max_steps")
    batch = bench.episode_batch(tasks) if hasattr(bench, "episode_batch") else SerialBatch(bench, tasks)
    trajs = [Trajectory(task=t, group=g) for t, g in zip(tasks, groups)]
    times = {"t_env_reset": 0.0, "t_generate": 0.0, "t_env_step": 0.0}
    try:
        start = time.perf_counter()
        initial = batch.reset()
        times["t_env_reset"] = time.perf_counter() - start
        obs, infos = map(list, zip(*initial))
        history = [[] for _ in tasks]
        for tr, o, info in zip(trajs, obs, infos):
            tr.init_keys = _keys(o, info)
        active = list(range(len(tasks)))
        with policy.generation():
            for _ in range(max_steps):
                if not active:
                    break
                prompts = [bench.prompt(infos[i]["task"], obs[i], infos[i]["admissible"], history[i])
                           for i in active]
                start = time.perf_counter()
                p_ids, r_ids, texts = policy.generate(prompts, greedy=greedy)
                times["t_generate"] += time.perf_counter() - start
                if any(len(rows) != len(active) for rows in (p_ids, r_ids, texts)):
                    raise ValueError("generation must return one response per active episode")
                parsed = [parse_action(text) for text in texts]
                start = time.perf_counter()
                results = batch.step(active, [a for a, _ in parsed])
                times["t_env_step"] += time.perf_counter() - start
                still = []
                for i, p, r, text, (action, valid), (o, _, done, info) in zip(
                        active, p_ids, r_ids, texts, parsed, results):
                    trajs[i].steps.append(Step(p, r, text, action, valid, obs=o,
                                               state_keys=_keys(o, info),
                                               truncated=policy.truncated(r)))
                    info.setdefault("task", infos[i]["task"])
                    history[i].append((obs[i], action))
                    obs[i], infos[i] = o, info
                    if info.get("won"):
                        trajs[i].won = True
                    if not done:
                        still.append(i)
                active = still
    finally:
        batch.close()
        if timings is not None:
            timings.update(times)

    for tr in trajs:
        tr.reward = success_reward * tr.won - invalid_penalty * tr.n_invalid
    return trajs


# ------------------------------------------------------------ response lengths
# Logged so a max_new_tokens cap can be chosen from data: a cap is free only if
# (almost) no response reaches it.

def response_lengths(trajs: list[Trajectory]) -> dict:
    """Raw per-response lengths, truncation and validity flags (for pooled percentiles)."""
    steps = [s for tr in trajs for s in tr.steps]
    return {"len": [len(s.response_ids) for s in steps],
            "truncated": [int(s.truncated) for s in steps],
            "valid": [int(s.valid) for s in steps]}


def length_stats(raw: dict, cap: int | None = None) -> dict:
    """Summary of ``response_lengths`` output (possibly pooled over many calls)."""
    n = np.asarray(raw["len"], dtype=float)
    if n.size == 0:
        return {}
    tr = np.asarray(raw["truncated"], dtype=bool)
    valid = np.asarray(raw["valid"], dtype=bool)
    out = {"n": int(n.size), "mean": float(n.mean()), "max": int(n.max()),
           **{f"p{q}": float(np.percentile(n, q)) for q in (50, 90, 99, 99.9)},
           "trunc_rate": float(tr.mean()),
           # P(invalid | truncated): how often hitting the cap cost the step its action
           "trunc_invalid": float((~valid[tr]).mean()) if tr.any() else 0.0}
    if cap is not None:
        out["cap"] = int(cap)
    return out
