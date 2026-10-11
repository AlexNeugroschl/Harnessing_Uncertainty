import json

import torch
import pytest

from benchmarks.common import load
from empg.rollout import Step, Trajectory
from empg import trainer


@pytest.mark.parametrize("step_budget,time_budget", [(0, 0), (1, 0), (0, 1e-12)])
def test_episode_length_and_update_shuffle_cannot_change_task_schedule(
        tmp_path, monkeypatch, step_budget, time_budget):
    class Bench:
        def tasks(self, split):
            return [f"game_{i}" for i in range(30)]

    class Policy:
        def __init__(self, *args, **kwargs):
            self.p = torch.nn.Parameter(torch.tensor(0.0))

        def parameters(self):
            return [self.p]

        def save(self, path):
            pass

    monkeypatch.setattr(trainer, "make_benchmark", lambda *a, **k: Bench())
    monkeypatch.setattr(trainer, "LLMPolicy", Policy)
    monkeypatch.setattr(trainer, "gpu_memory_check", lambda *a, **k: {})
    schedules = []
    for length in (1, 5):
        cfg = load("grpo", "smoke", ["train.iterations=4", "train.games_per_iter=2"])
        cfg.out_dir = str(tmp_path / str(length))
        cfg.train.max_env_steps, cfg.train.max_seconds = step_budget, time_budget

        def episodes(policy, bench, tasks, groups, *a, **k):
            return [Trajectory(t, g, [Step([1], [2], "", "", True)] * length,
                               init_keys={"obs": "start"}) for t, g in zip(tasks, groups)]

        def update(policy, opt, trajs, cfg, rng, **k):
            rng.permutation(sum(len(t.steps) for t in trajs))
            return {key: 0.0 for key in ("success", "reward", "ep_len", "invalid_rate",
                    "entropy_mean", "kl", "revisit_rate", "any_success_frac", "resp_len_p99", "trunc_rate")}

        monkeypatch.setattr(trainer, "run_episodes", episodes)
        monkeypatch.setattr(trainer, "update", update)
        trainer.train(cfg, log=lambda *a: None)
        schedules.append([json.loads(line)["tasks"] for line in
                          (cfg.run_dir / "metrics.jsonl").read_text().splitlines()])
        completion = json.loads((cfg.run_dir / "training_complete.json").read_text())
        assert completion["iterations"] == (1 if step_budget or time_budget else 4)
        assert completion["reason"] == ("max_env_steps" if step_budget else
                                        "max_seconds" if time_budget else "iterations")
    assert schedules[0] == schedules[1]
