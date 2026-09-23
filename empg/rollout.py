"""Multi-turn rollouts: many envs stepped in lockstep with batched generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from empg.envs.base import Benchmark
from empg.prompts import parse_action


@dataclass
class Step:
    prompt_ids: list[int]
    response_ids: list[int]
    text: str
    action: str
    valid: bool
    entropy: float = 0.0  # H_t (Eq. 7), filled in after scoring


@dataclass
class Trajectory:
    task: str
    group: int
    steps: list[Step] = field(default_factory=list)
    won: bool = False
    reward: float = 0.0  # outcome reward used by the advantage estimator

    @property
    def n_invalid(self) -> int:
        return sum(not s.valid for s in self.steps)


def run_episodes(policy, bench: Benchmark, tasks: list[str], groups: list[int],
                 max_steps: int, success_reward: float = 10.0,
                 invalid_penalty: float = 0.1, greedy: bool = False) -> list[Trajectory]:
    """Play one episode per (task, group) pair; all envs advance one turn per
    batched generate() call."""
    envs = [bench.make_env(t) for t in tasks]
    trajs = [Trajectory(task=t, group=g) for t, g in zip(tasks, groups)]
    obs, infos, history = [], [], [[] for _ in envs]
    for env in envs:
        o, info = env.reset()
        obs.append(o)
        infos.append(info)
    active = list(range(len(envs)))

    for _ in range(max_steps):
        if not active:
            break
        prompts = [bench.prompt(infos[i]["task"], obs[i], infos[i]["admissible"], history[i])
                   for i in active]
        p_ids, r_ids, texts = policy.generate(prompts, greedy=greedy)
        still = []
        for i, p, r, text in zip(active, p_ids, r_ids, texts):
            action, valid = parse_action(text)
            trajs[i].steps.append(Step(p, r, text, action, valid))
            o, _, done, info = envs[i].step(action)
            info.setdefault("task", infos[i]["task"])
            history[i].append((obs[i], action))
            obs[i], infos[i] = o, info
            if info.get("won"):
                trajs[i].won = True
            if done:
                continue
            still.append(i)
        active = still

    for env, tr in zip(envs, trajs):
        env.close()
        tr.reward = success_reward * tr.won - invalid_penalty * tr.n_invalid
    return trajs
