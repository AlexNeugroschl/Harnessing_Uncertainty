import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from empg.curiosity import CuriosityConfig, CuriosityController, intrinsic_rewards
from empg.diagnostics import behavior_metrics
from empg.rollout import Step, Trajectory
from benchmarks.nonstationary.diagnostics import catalog_distribution, decision_prompts
from benchmarks.nonstationary.evidence import gate
from benchmarks.nonstationary.warmup import supervised_epoch, weighted_baseline_examples


def test_stable_detector_fires_once_for_a_step_and_can_detect_a_later_step():
    controller = CuriosityController(CuriosityConfig(boost="observed_failure", detector_rearm="stable", boost_updates=3))
    events = []
    rates = [0.] * 6 + [.5] * 30 + [.95] * 30
    for iteration, rate in enumerate(rates, 1):
        controller.observe(iteration, rate)
        if controller.triggered:
            events.append(iteration)
    assert events == [7, 37]


@pytest.mark.parametrize("rate", [0., .75, 1.])
def test_stable_detector_never_fires_on_stationary_failure_rates(rate):
    c = CuriosityController(CuriosityConfig(boost="observed_failure", detector_rearm="stable"))
    for i in range(1, 40):
        c.observe(i, rate)
        assert not c.triggered


def test_native_progress_key_ignores_movement_and_uses_only_visible_facts():
    pytest.importorskip("gym_novel_gridworlds2")
    from empg.envs.novelgym_env import NovelGymBenchmark
    bench = NovelGymBenchmark(recipe_depth=2, suite_version=4)
    env = bench.make_env(bench.tasks("train")[0])
    try:
        _, reset = env.reset()
        _, _, _, moved = env.step("approach_log")
        assert reset["state_keys"]["progress"] == moved["state_keys"]["progress"]
        assert reset["state_keys"]["parsed"] != moved["state_keys"]["parsed"]
        _, _, _, equipped = env.step("select_axe")
        assert equipped["state_keys"]["progress"] != moved["state_keys"]["progress"]
        assert equipped["diagnostics"]["tool_selected"] == 1
        _, _, _, harvested = env.step("break_block")
        assert harvested["diagnostics"]["successful_harvest"] == 1
        assert harvested["diagnostics"]["harvested_items"] == 1
        _, _, _, crafted = env.step("craft_planks")
        assert crafted["diagnostics"]["crafted_items"] == 1
    finally:
        env.close()


def test_new_suite_has_fresh_panels_and_unchanged_reset_between_regimes():
    pytest.importorskip("gym_novel_gridworlds2")
    from empg.envs.novelgym_env import NovelGymBenchmark
    old, new = NovelGymBenchmark(), NovelGymBenchmark(suite_version=4)
    assert not set(old.tasks("valid_unseen")) & set(new.tasks("valid_unseen"))
    task = new.tasks("train")[0]
    observations, keys = [], []
    for regime in ("baseline", "novelty"):
        new.intervention = {"regime": regime, "change": "rules", "feedback": "generic"}
        env = new.make_env(task)
        try:
            obs, info = env.reset()
            observations.append(obs)
            keys.append(info["state_keys"])
        finally:
            env.close()
    assert observations[0] == observations[1] and keys[0] == keys[1]


def test_finite_catalog_uses_sequence_sums_and_reports_conditional_mass():
    d = catalog_distribution(np.log([.2, .3]))
    assert d["catalog_mass"] == pytest.approx(.5)
    np.testing.assert_allclose(d["conditional"], [.4, .6])
    assert d["conditional_entropy"] == pytest.approx(-.4 * math.log(.4) - .6 * math.log(.6))
    with pytest.raises(ValueError):
        catalog_distribution([float("nan")])


def test_action_diagnostics_sums_all_response_tokens_including_eos(monkeypatch):
    from benchmarks.nonstationary import diagnostics
    monkeypatch.setattr(diagnostics, "decision_prompts", lambda *a, **kw: [("t", "prompt", ["select_axe", "break_block"])])
    class Tokenizer:
        eos_token_id = 0
        def __call__(self, text, **kwargs):
            return {"input_ids": [10, 11] if text.endswith("select_axe") else [12]}
    class Policy:
        tokenizer = Tokenizer()
        def encode_prompt(self, prompt):
            return [1]
        def score(self, prompts, responses, **kwargs):
            assert responses == [[10, 11, 0], [12, 0]]
            return [torch.tensor([-1., -2., -3.]), torch.tensor([-1., -1.])], []
    result = diagnostics.action_diagnostics(Policy(), None, ["t"])
    assert result["mean_tool_probability"] == pytest.approx(math.exp(-6) / (math.exp(-6) + math.exp(-2)))
    assert result["tasks"][0]["sequence_log_probabilities"] == {"select_axe": -6., "break_block": -2.}


def test_weighted_supervised_objective_matches_full_token_gradient():
    class Policy:
        device = "cpu"
        cfg = SimpleNamespace(micro_batch_size=1, micro_batch_tokens=32)
        def __init__(self):
            self.model = torch.nn.Linear(1, 3, bias=False)
            self.model.weight.data.zero_()
        def parameters(self):
            return list(self.model.parameters())
        def forward_logprobs(self, prompts, responses):
            lp = self.model(torch.ones(1)).log_softmax(0)
            values = torch.stack([lp[torch.tensor(r)] for r in responses])
            return values, None, torch.ones_like(values, dtype=torch.bool)
    p = Policy()
    data = [([0], [0], .75), ([0], [1, 1], .25), ([0], [2], 1.)]
    nll = supervised_epoch(p, torch.optim.SGD(p.parameters(), lr=.1), data, np.arange(3), 3)
    expected = -.1 * (torch.full((3,), 1 / 3) - torch.tensor([.75, .5, 1.]) / 2.25)
    torch.testing.assert_close(p.model.weight[:, 0], expected)
    assert nll == pytest.approx(math.log(3))


def test_baseline_optional_targets_are_both_valid_and_keep_exact_target_mass():
    pytest.importorskip("gym_novel_gridworlds2")
    from empg.envs.novelgym_env import NovelGymBenchmark
    bench = NovelGymBenchmark(recipe_depth=2, suite_version=4, action_parser="native")
    examples = list(weighted_baseline_examples(bench, bench.tasks("train")[0], 1., np.random.default_rng(5), .05))
    assert any(action == "select_axe" for _, action, _ in examples)
    for i, (prompt, action, weight) in enumerate(examples):
        if action == "select_axe":
            previous = examples[i - 1]
            assert previous[0] == prompt and previous[1] == "break_block"
            assert previous[2] + weight == pytest.approx(1.)


def test_behavior_statistics_count_outcomes_without_using_hidden_regime():
    steps = [Step([1], [2], "", "select_axe", True, diagnostics={"tool_selected": 1}),
             Step([1], [2], "", "break_block", True, diagnostics={"failed": True}),
             Step([1], [2], "", "break_block", True, diagnostics={"failed": True}),
             Step([1], [2], "", "craft_component_01", True, diagnostics={"crafted_items": 1, "component_depth": 1})]
    stats = behavior_metrics([Trajectory("t", 0, steps)])
    assert stats["tool_selection_episodes"] == 1
    assert stats["failed_action_counts"]["break_block"] == 2 and stats["repeated_failed_actions"] == 1
    assert stats["max_component_depth"] == 1


def test_development_gate_requires_learning_on_identical_panels():
    def panel(wins):
        return {"protocol": {"split": "valid_seen"}, "per_task": [{"task": str(i), "won": w} for i, w in enumerate(wins)]}
    initial, final = panel([False] * 8), panel([True] * 4 + [False] * 4)
    assert gate(initial, final)["passed"]
    assert not gate(final, final)["passed"]
    with pytest.raises(ValueError, match="protocol"):
        gate(initial, {**final, "protocol": {"split": "valid_unseen"}})


@pytest.mark.parametrize("value", ["3e-06", "3.0e-6", "1E-5", "0.00001"])
def test_scientific_learning_rate_overrides_remain_numeric(value):
    from benchmarks.common import load
    cfg = load("grpo", "novelgym_recovery", [f"train.lr={value}"])
    assert isinstance(cfg.train.lr, float) and cfg.train.lr == float(value)


def test_gpu_tuner_rejects_faster_updates_that_exceed_numerical_tolerance():
    from benchmarks.nonstationary.calibration import checked_candidates
    rows = [{"name": name, "status": "ok", "adapter_max_abs_delta": difference}
            for name, difference in (("reference", 0.), ("micro4", 6e-6)) for _ in range(2)]
    valid, rejected = checked_candidates(rows, 2, 3e-6)
    assert set(valid) == {"reference"} and "micro4" in rejected


def test_gpu_tolerance_accounts_for_measured_reference_repeatability():
    from benchmarks.nonstationary.calibration import update_tolerance
    rows = [{"name": "reference", "status": "ok", "adapter_max_abs_delta": v} for v in (0., 6e-6)]
    tolerance, noise = update_tolerance(rows, 3e-6)
    assert noise == 6e-6 and tolerance == pytest.approx(9e-6)


def test_real_changed_rule_rollout_instruments_tool_discovery_without_shaping_reward():
    pytest.importorskip("gym_novel_gridworlds2")
    from contextlib import nullcontext
    from empg.envs.novelgym_env import NovelGymBenchmark
    from empg.rollout import run_episodes
    class Teacher:
        def __init__(self):
            self.actions = iter(["approach_log", "break_block", "select_axe", "break_block", "craft_planks",
                                 "craft_component_01", "approach_log", "break_block", "craft_planks", "craft_pogo_stick"])
        def generation(self):
            return nullcontext()
        def generate(self, prompts, greedy=False):
            action = next(self.actions)
            return [[1]] * len(prompts), [[2]] * len(prompts), ["action: " + action] * len(prompts)
        def truncated(self, tokens):
            return False
    bench = NovelGymBenchmark(recipe_depth=2, suite_version=4, action_parser="native",
                              intervention={"regime": "novelty", "change": "rules", "feedback": "generic"})
    trajectories = run_episodes(Teacher(), bench, bench.tasks("train")[:2], [0, 0], 12)
    stats = behavior_metrics(trajectories)
    assert all(t.won and t.env_return == 1. and t.reward == 10. for t in trajectories)
    assert stats["tool_selection_episodes"] == stats["harvesting_episodes"] == 2
    assert stats["harvested_items"] == 4 and stats["max_component_depth"] == 2
    assert stats["failed_action_counts"] == {"break_block": 2}


def test_native_training_loop_applies_one_causal_boost_for_a_persistent_change(tmp_path, monkeypatch):
    pytest.importorskip("gym_novel_gridworlds2")
    import json
    from contextlib import contextmanager
    from benchmarks.common import load
    from empg import trainer
    class ScriptedPolicy:
        def __init__(self, cfg, seed):
            self.cfg, self.device, self.use_lora = cfg, "cpu", False
            self.model = torch.nn.Linear(1, 1)
        def parameters(self):
            return list(self.model.parameters())
        @contextmanager
        def generation(self):
            self.turn = 0
            yield
        def generate(self, prompts, greedy=False):
            actions = ["approach_log", "break_block", "craft_planks", "craft_component_01",
                       "approach_log", "break_block", "craft_planks", "craft_pogo_stick"]
            action = actions[self.turn % len(actions)]
            self.turn += 1
            return [[1, 2]] * len(prompts), [[3]] * len(prompts), ["action: " + action] * len(prompts)
        def truncated(self, response):
            return False
        def score(self, prompts, responses, with_entropy=True, **kwargs):
            return [torch.tensor([-1.]) for _ in prompts], [torch.tensor([1.]) for _ in prompts] if with_entropy else []
        def forward_logprobs(self, prompts, responses, with_entropy=False):
            lp = self.model(torch.ones(len(prompts), 1))
            return lp, None, torch.ones_like(lp, dtype=torch.bool)
        def save(self, path):
            from pathlib import Path
            Path(path).mkdir(parents=True, exist_ok=True)
        def close(self):
            pass
    monkeypatch.setattr(trainer, "LLMPolicy", ScriptedPolicy)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = load("grpo", "novelgym_recovery", [f"out_dir={tmp_path}", "train.iterations=8", "train.games_per_iter=1",
        "train.group_size=2", "train.max_steps=26", "train.kl_coef=0", "train.entropy_coef=0", "train.trace_episodes=0",
        "nonstationary.probe_every=0", "nonstationary.switch_iteration=3", "advantage.curiosity.boost=observed_failure",
        "advantage.curiosity.boost_updates=2"])
    trainer.train(cfg, log=lambda *a: None).close()
    rows = [json.loads(line) for line in (cfg.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in rows if r["curiosity_detector_triggered"]] == [3]
    np.testing.assert_allclose([r["curiosity_weight"] for r in rows], [.05, .05, .05, .15, .15, .05, .05, .05])
    assert all(r["action_failure_rate"] > .5 for r in rows[2:])
