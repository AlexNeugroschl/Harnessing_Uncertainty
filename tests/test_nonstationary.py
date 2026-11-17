"""Intervention semantics, native rewards, paired controls and probe isolation."""

from contextlib import nullcontext
from types import SimpleNamespace
import dataclasses
import json
import random

import numpy as np
import pytest

from empg.nonstationary import InspectionEnv, NonstationaryConfig, wrap_env
from empg.config import Config, config_dict, config_fingerprint, config_matches, load_config
from empg.rollout import run_episodes


TAKE = "take apple 1 from table 1"
INSPECT = "inspect table 1"
PUT = "put apple 1 in/on box 1"


class World:
    def __init__(self):
        self.calls = []
        self.closed = False

    def info(self, won=False):
        return {"task": "put apple in box", "admissible": [PUT] if self.holding else [TAKE, "look"],
                "won": won, "state_keys": {"parsed": str(self.holding), "physical": str(self.holding)}}

    def reset(self):
        self.holding = False
        return "An apple on a table.", self.info()

    def step(self, action):
        self.calls.append(action)
        won = action == PUT and self.holding
        if action == TAKE and not self.holding:
            self.holding = True
            return "You take the apple.", 0., False, self.info()
        return "Done." if won else "Nothing happens.", float(won), won, self.info(won)

    def close(self):
        self.closed = True


def environment(change="rules", regime="inspection", consume=False):
    return InspectionEnv(World(), **NonstationaryConfig(
        change=change, consume_inspection=consume).intervention(regime))


def test_rules_block_native_transition_but_inspection_restores_solution():
    env = environment()
    _, initial = env.reset()
    assert initial["admissible"] == [TAKE, "look", INSPECT]
    obs, reward, done, info = env.step(TAKE)
    assert not env.env.calls and not done and reward == 0
    assert "inspect table 1" in obs and info["intervention_metrics"]["rule_blocks"] == 1
    assert info["state_keys"] == initial["state_keys"]
    _, _, _, ready = env.step(INSPECT)
    assert ready["state_keys"] != initial["state_keys"]
    _, _, _, repeated = env.step(INSPECT)
    assert ready["state_keys"] == repeated["state_keys"]
    assert len(repeated["admissible"]) == len(set(repeated["admissible"]))
    env.step(TAKE)
    _, reward, done, info = env.step(PUT)
    assert reward == 1 and done and info["won"]
    assert info["intervention_metrics"]["alternative_success"] == 1
    _, reset = env.reset()
    assert reset["state_keys"] == initial["state_keys"] and not env.inspected
    env.close()
    assert env.env.closed


def test_same_interface_and_inspection_observation_in_both_regimes():
    a, b = environment(regime="baseline"), environment()
    assert a.reset() == b.reset()
    assert a.step(INSPECT) == b.step(INSPECT)
    a.step(TAKE)
    b.step(TAKE)
    assert a.env.calls == b.env.calls == [TAKE]


def test_generic_feedback_does_not_explain_the_hidden_prerequisite():
    spec = NonstationaryConfig(feedback="generic").intervention("inspection")
    env = InspectionEnv(World(), **spec)
    env.reset()
    obs, _, _, _ = env.step(TAKE)
    assert "inspect" not in obs and "table" not in obs
    obs, _, _, _ = env.step(INSPECT)
    assert obs == "You inspect table 1."
    env.step(TAKE)
    assert env.step(PUT)[1:3] == (1., True)
    env = InspectionEnv(World(), **dict(spec, change="rewards"))
    env.reset()
    env.step(TAKE)
    obs, reward, done, _ = env.step(PUT)
    assert done and reward == 0 and obs.endswith("No reward received.") and "inspect" not in obs


def test_reward_change_preserves_native_physics_and_success_but_changes_payment():
    a, b = environment("rewards", "baseline"), environment("rewards")
    a.reset()
    b.reset()
    assert a.step(TAKE) == b.step(TAKE)
    _, ra, da, ia = a.step(PUT)
    _, rb, db, ib = b.step(PUT)
    assert a.env.calls == b.env.calls == [TAKE, PUT]
    assert da == db == True and ia["intervention_metrics"]["base_won"] and ib["intervention_metrics"]["base_won"]
    assert ra == 1 and rb == 0 and ia["won"] and not ib["won"]
    b.reset()
    b.step(INSPECT)
    b.step(TAKE)
    assert b.step(PUT)[1:3] == (1., True)


def test_disabled_wrapper_is_identity_and_consumption_is_configurable():
    native = World()
    assert wrap_env(native, None) is native
    reusable, consumed = environment(consume=False), environment(consume=True)
    for env in (reusable, consumed):
        env.reset()
        env.step(INSPECT)
        env.step(TAKE)
    assert reusable.inspected == {"table 1"} and not consumed.inspected


def test_schedule_boundaries_and_stationary_control_probe_points_match():
    nc = NonstationaryConfig(enabled=True, switch_iteration=4, probe_every=2)
    assert [nc.regime_at(i) for i in range(1, 7)] == ["baseline"] * 3 + ["inspection"] * 3
    assert nc.probe_iterations(6) == [0, 2, 3, 4, 6]
    stationary = dataclasses.replace(nc, schedule="stationary")
    assert stationary.probe_iterations(6) == nc.probe_iterations(6)
    assert all(stationary.regime_at(i) == "baseline" for i in range(1, 7))
    assert dataclasses.replace(nc, probe_every=0).probe_iterations(6) == []


@pytest.mark.parametrize("overrides", [
    ["nonstationary.switch_iteration=1"], ["nonstationary.switch_iteration=101"],
    ["nonstationary.actions=[dance]"], ["nonstationary.probe_split=train"],
    ["nonstationary.recovery_fraction=.nan"], ["nonstationary.change=typo"],
    ["nonstationary.probe_every=1.5"], ["nonstationary.switch_iteration=.nan"],
])
def test_invalid_experiments_fail_before_model_load(overrides):
    with pytest.raises(ValueError):
        load_config(overrides=["nonstationary.enabled=true", *overrides])


def test_disabled_feature_preserves_legacy_configuration_fingerprints():
    cfg = Config()
    old = dataclasses.asdict(cfg)
    old.pop("nonstationary")
    old["train"].pop("reward_source")
    old["train"].pop("environment_reward_scale")
    assert config_matches(old, cfg)
    before = config_fingerprint(cfg)
    cfg.nonstationary.switch_iteration = 2
    assert config_fingerprint(cfg) == before
    cfg.nonstationary.enabled = True
    assert config_fingerprint(cfg) != before and "nonstationary" in config_dict(cfg)


class ScriptedPolicy:
    def __init__(self, actions):
        self.actions = iter(actions)

    def generation(self):
        return nullcontext()

    def generate(self, prompts, greedy=False):
        action = next(self.actions)
        return [[1]], [[2]], [f"<think>act</think><action>{action}</action>"]

    def truncated(self, ids):
        return False


def test_rollout_trains_on_environment_rewards_even_when_won_is_false():
    class RewardEnv(World):
        def step(self, action):
            return "reward", -2.5, True, {"won": False, "admissible": []}

    class Bench:
        def make_env(self, task):
            return RewardEnv()

        def prompt(self, *args):
            return "prompt"

    tr, = run_episodes(ScriptedPolicy(["look"]), Bench(), ["t"], [0], 3,
                       reward_source="environment", environment_reward_scale=2.)
    assert not tr.won and tr.env_return == -2.5 and tr.reward == -5.


def test_recovery_is_heldout_and_missing_recovery_is_not_zero():
    from benchmarks.adaptation import summarize
    nc = NonstationaryConfig(enabled=True, switch_iteration=3)
    rows = [{"iteration": i, "env_steps_total": 10 * i, "n_steps": 10, "episodes": 2,
             "env_return_sum": 1.} for i in range(1, 6)]
    probes = [{"iteration": 2, "regime": "baseline", "all": .8},
              {"iteration": 3, "regime": "inspection", "all": .1, "env_steps_total": 30},
              {"iteration": 5, "regime": "inspection", "all": .75, "env_steps_total": 50}]
    result = summarize(rows, probes, nc)
    assert result["recovery_updates"] == 3 and result["recovery_env_steps"] == 30
    assert result["post_boundary_env_return"] == 3.
    assert summarize(rows, probes[:-1], nc)["recovery_updates"] is None
    assert summarize(rows, [], nc)["recovery_target"] is None
    assert not summarize(rows[:2], probes[:1], nc)["shift_observed"]


def test_adaptation_conclusions_require_measured_pre_switch_competence():
    from benchmarks.adaptation import summarize
    nc = NonstationaryConfig(enabled=True, switch_iteration=3)
    rows = [{"iteration": 2}, {"iteration": 3}]
    probes = [{"iteration": 2, "regime": "baseline", "all": .5,
               "per_task": [{"won": True}, {"won": False}]},
              {"iteration": 3, "regime": "inspection", "all": .5}]
    result = summarize(rows, probes, nc)
    assert not result["adaptation_interpretable"]
    assert result["competence_status"] == "insufficient pre-switch success"
    probes[0]["per_task"] = [{"won": True}, {"won": True}]
    assert summarize(rows, probes, nc)["adaptation_interpretable"]


def test_probes_reuse_environment_and_restore_rng_on_failure(tmp_path, monkeypatch):
    from benchmarks import adaptation
    closed = []
    class Bench:
        def close(self):
            closed.append(True)
    bench = Bench()
    monkeypatch.setattr(adaptation, "make_benchmark", lambda *a, **kw: bench)
    seen = []
    def evaluate(policy, cfg, log, *, regime, benchmark):
        assert benchmark is bench and cfg.eval.sampling_seed == 0
        random.random()
        np.random.rand()
        seen.append(regime)
        if regime == "inspection":
            raise RuntimeError("probe failed")
        return {"all": .5}, {}
    monkeypatch.setattr(adaptation, "evaluate", evaluate)
    cfg = Config(out_dir=str(tmp_path), nonstationary=NonstationaryConfig(enabled=True, switch_iteration=2))
    cfg.run_dir.mkdir()
    before_py, before_np = random.getstate(), np.random.get_state()
    sessions = []
    policy = SimpleNamespace(sampling=lambda seed: nullcontext(),
                             generation=lambda: sessions.append("generation") or nullcontext())
    probes = adaptation.AdaptationProbes(policy, cfg, log=lambda *a: None)
    try:
        with pytest.raises(RuntimeError, match="probe failed"):
            probes.run(0)
        assert random.getstate() == before_py
        np.testing.assert_equal(np.random.get_state(), before_np)
        assert seen == ["baseline", "inspection"]
    finally:
        probes.close()
    assert closed == [True]
    assert sessions == ["generation"]
    assert len((cfg.run_dir / "adaptation_probes.jsonl").read_text().splitlines()) == 1


def test_training_switch_keeps_optimizer_and_probes_do_not_change_training(tmp_path, monkeypatch):
    import torch
    from empg import trainer
    from benchmarks import adaptation

    class Bench:
        intervention = None

        def tasks(self, split):
            return ["a", "b", "c", "d"]

        def task_type(self, task):
            return "Pick"

        def make_env(self, task):
            return wrap_env(World(), self.intervention)

        def prompt(self, task, obs, admissible, history):
            return json.dumps([obs, admissible])

    class Policy:
        def __init__(self, cfg, **kw):
            self.cfg, self.p = cfg, torch.nn.Parameter(torch.tensor(0.))

        def parameters(self):
            return [self.p]

        def generation(self):
            return nullcontext()

        def sampling(self, seed):
            return torch.random.fork_rng(devices=[])

        def generate(self, prompts, greedy=False):
            torch.rand(1)  # probes must not consume the training RNG stream
            actions = []
            for prompt in prompts:
                obs, admissible = json.loads(prompt)
                action = PUT if PUT in admissible else (INSPECT if "need to inspect" in obs else TAKE)
                actions.append(f"<think>act</think><action>{action}</action>")
            return [[1]] * len(prompts), [[2]] * len(prompts), actions

        def truncated(self, ids):
            return False

        def save(self, path):
            pass

    monkeypatch.setattr(trainer, "LLMPolicy", Policy)
    monkeypatch.setattr(trainer, "make_benchmark", lambda *a, **kw: Bench())
    monkeypatch.setattr(adaptation, "make_benchmark", lambda *a, **kw: Bench())
    monkeypatch.setattr(trainer, "gpu_memory_check", lambda *a: {})
    observed = []
    current = []
    optimizer_ids = []
    def update(policy, optimizer, trajectories, cfg, rng, **kw):
        current.append((torch.rand(1).item(), [tr.task for tr in trajectories],
                        sum(tr.intervention_metrics.get("rule_blocks", 0) for tr in trajectories)))
        optimizer_ids.append(id(optimizer))
        stats = {key: 0. for key in ("reward", "invalid_rate", "entropy_mean", "kl", "revisit_rate",
                                    "any_success_frac", "resp_len_p99", "trunc_rate")}
        stats.update(success=1., ep_len=float(np.mean([len(t.steps) for t in trajectories])))
        return stats
    monkeypatch.setattr(trainer, "update", update)
    for every in (0, 2):
        cfg = Config(out_dir=str(tmp_path / str(every)), nonstationary=NonstationaryConfig(
            enabled=True, switch_iteration=3, probe_every=every, probe_tasks=2))
        cfg.train.iterations, cfg.train.games_per_iter, cfg.train.group_size = 4, 2, 2
        cfg.train.max_steps, cfg.train.save_every = 8, 0
        current, optimizer_ids = [], []
        trainer.train(cfg, log=lambda *a: None)
        assert len(set(optimizer_ids)) == 1
        assert [entry[2] for entry in current] == [0, 0, 4, 4]
        observed.append(current)
        if every:
            records = [json.loads(s) for s in (cfg.run_dir / "adaptation_probes.jsonl").read_text().splitlines()]
            assert {r["iteration"] for r in records} == {0, 2, 3, 4}
            assert all(r["all"] == 1. for r in records)
    assert observed[0] == observed[1]


def test_matrix_executes_training_evaluation_and_reports_with_scripted_policy(tmp_path, monkeypatch):
    """Exercise actual phase CLIs/artifacts; no downloads, GPU, or learning claim."""
    import sys
    import torch
    import run_nonstationary as matrix
    from benchmarks import adaptation, evaluate, report, train as train_cli
    from empg import trainer

    class Bench:
        intervention = None
        def tasks(self, split):
            return ["a", "b", "c", "d"]
        def task_type(self, task):
            return "Pick"
        def make_env(self, task):
            return wrap_env(World(), self.intervention)
        def prompt(self, task, obs, admissible, history):
            return json.dumps([obs, admissible])

    class Policy:
        def __init__(self, cfg, **kwargs):
            self.cfg, self.p = cfg, torch.nn.Parameter(torch.tensor(0.))
        def parameters(self):
            return [self.p]
        def generation(self):
            return nullcontext()
        def sampling(self, seed):
            return torch.random.fork_rng(devices=[])
        def generate(self, prompts, greedy=False):
            actions = []
            for prompt in prompts:
                obs, admissible = json.loads(prompt)
                action = PUT if PUT in admissible else INSPECT if "operation failed" in obs else TAKE
                actions.append(f"<think>act</think><action>{action}</action>")
            return [[1]] * len(prompts), [[2]] * len(prompts), actions
        def truncated(self, ids):
            return False
        def save(self, path):
            from pathlib import Path
            Path(path).mkdir(parents=True, exist_ok=True)
        def close(self):
            pass

    def update(policy, optimizer, trajectories, cfg, rng, **kwargs):
        stats = {k: 0. for k in ("invalid_rate", "entropy_mean", "kl", "revisit_rate", "any_success_frac", "resp_len_p99", "trunc_rate")}
        stats.update(success=1., reward=10., ep_len=float(np.mean([len(t.steps) for t in trajectories])),
                     n_steps=sum(len(t.steps) for t in trajectories), episodes=len(trajectories),
                     env_return_sum=sum(t.env_return for t in trajectories))
        return stats
    monkeypatch.setattr(trainer, "LLMPolicy", Policy)
    monkeypatch.setattr(trainer, "gpu_memory_check", lambda *a: {})
    monkeypatch.setattr(trainer, "update", update)
    for module in (trainer, adaptation, evaluate):
        monkeypatch.setattr(module, "make_benchmark", lambda *a, **kw: Bench())
    calls = []
    def execute(command, timeout=None):
        module = command[command.index("-m") + 1]
        calls.append(module)
        monkeypatch.setattr(sys, "argv", [module, *command[command.index("-m") + 2:]])
        if module == "benchmarks.train":
            train_cli.main()
        elif module == "benchmarks.report":
            report.main()
        else:
            raise AssertionError(module)
    monkeypatch.setattr(matrix, "execute_phase", execute)
    monkeypatch.setattr(matrix, "estimate", lambda *a, **kw: {"seconds": 100.})
    runs, out = tmp_path / "runs", tmp_path / "out"
    assert matrix.main(["--tier", "nonstationary_overnight", "--hours", "0", "--runs", str(runs), "--out", str(out),
                        "--", "train.iterations=2", "train.games_per_iter=1", "train.group_size=2", "train.max_steps=8",
                        "nonstationary.switch_iteration=2", "nonstationary.probe_every=1", "nonstationary.probe_tasks=2",
                        "eval.max_tasks=2", "eval.selection=random"]) == 0
    assert calls.count("benchmarks.train") == 4 and calls.count("benchmarks.report") == 2
    manifest = json.loads((out / "matrix_run.json").read_text())
    assert manifest["status"] == "complete" and all(p["complete"] for p in manifest["jobs"])
    stats = json.loads((out / "curiosity_statistics.json").read_text())
    assert stats["switch"]["delta"] == 0 and stats["switch_minus_stationary_effect"]["delta"] == 0
    summaries = json.loads((out / "switch/adaptation_summary.json").read_text())
    assert all(s["adaptation_interpretable"] for s in summaries)
    assert (out / "stationary/adaptation_curves.png").exists()
