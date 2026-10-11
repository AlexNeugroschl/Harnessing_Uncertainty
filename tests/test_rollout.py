from contextlib import nullcontext

import pytest

from empg.rollout import run_episodes


class Env:
    def __init__(self, task):
        self.task, self.closed = task, False

    def reset(self):
        return "start", {"task": self.task, "admissible": ["look"]}

    def step(self, action):
        return "done", 1.0, True, {"won": True, "admissible": []}

    def close(self):
        self.closed = True


class Bench:
    def __init__(self):
        self.envs = []

    def make_env(self, task):
        env = Env(task)
        self.envs.append(env)
        return env

    def prompt(self, *args):
        return "prompt"


class Policy:
    fail = False

    def generation(self):
        return nullcontext()

    def generate(self, prompts, greedy=False):
        if self.fail:
            raise RuntimeError("generation failed")
        return [[1]] * len(prompts), [[2]] * len(prompts), [
            "<think>ok</think><action>look</action>"] * len(prompts)

    def truncated(self, ids):
        return False


def test_rollout_cleanup_success_and_failure():
    bench, policy = Bench(), Policy()
    timings = {}
    trajs = run_episodes(policy, bench, ["a", "b"], [0, 1], 5, timings=timings)
    assert [t.task for t in trajs] == ["a", "b"]
    assert all(t.won and t.reward == 10 and len(t.steps) == 1 for t in trajs)
    assert all(e.closed for e in bench.envs)
    assert all(t >= 0 for t in timings.values())
    policy.fail = True
    with pytest.raises(RuntimeError, match="generation failed"):
        run_episodes(policy, bench, ["a"], [0], 2)
    assert all(e.closed for e in bench.envs)


def test_rollout_rejects_mismatched_groups():
    with pytest.raises(ValueError, match="matching"):
        run_episodes(Policy(), Bench(), ["a"], [], 2)
