"""Actual upstream transitions, not a fake simulator. Optional dependency."""
import pytest

pytest.importorskip("gym_novel_gridworlds2")

from empg.envs.novelgym_env import NovelGymBenchmark, NovelGymEnv, nominal_solution_length


def solution(depth, novelty, regime):
    actions = ["select_axe"] if novelty == "tool" and regime == "novelty" else []
    wood = "birch_log" if novelty == "resource" and regime == "novelty" else "log"
    harvest = [f"approach_{wood}", "break_block", "craft_planks"]
    for i in range(1, depth + 1):
        if i == depth and novelty == "recipe" and regime == "novelty":
            actions += harvest + ["craft_brace"]
        actions += harvest + ["craft_pogo_stick" if i == depth else f"craft_component_{i:02d}"]
    return actions


@pytest.mark.parametrize("depth", [2, 4, 8, 16])
@pytest.mark.parametrize("novelty", ["tool", "recipe", "resource"])
@pytest.mark.parametrize("regime", ["baseline", "novelty"])
def test_native_tasks_remain_solvable_at_all_horizons(depth, novelty, regime):
    env = NovelGymEnv(f"craft:{depth}:7", novelty=novelty, regime=regime)
    try:
        env.reset()
        actions = solution(depth, novelty, regime)
        assert len(actions) == nominal_solution_length(depth, novelty, regime)
        for i, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            assert not info["intervention_metrics"]["action_failures"], (action, obs)
            assert done == (i == len(actions) - 1)
        assert reward == 1 and info["won"]
    finally:
        env.close()


def test_hidden_changes_do_not_leak_into_reset_or_state_key():
    for novelty in ("tool", "recipe", "resource"):
        a = NovelGymEnv("craft:4:13", novelty=novelty)
        b = NovelGymEnv("craft:4:13", novelty=novelty, regime="novelty")
        assert a.reset() == b.reset()
        if novelty == "tool":
            assert a.step("approach_log") == b.step("approach_log")
            before = b._info(False, 0)["state_keys"]
            obs, _, _, info = b.step("break_block")
            assert obs.startswith("Operation failed.")
            assert before == info["state_keys"]  # failure feedback cannot create fake novelty
        a.close()
        b.close()


def test_reward_only_change_keeps_physics_but_changes_payment():
    a = NovelGymEnv("craft:2:7", novelty="tool", change="rewards")
    b = NovelGymEnv("craft:2:7", novelty="tool", change="rewards", regime="novelty")
    assert a.reset() == b.reset()
    for action in solution(2, "tool", "baseline"):
        left, right = a.step(action), b.step(action)
        assert a._visible() == b._visible()
    assert left[1] == 1 and right[1] == 0 and left[2] and right[2]
    a.close()
    b.close()


def test_panels_are_disjoint_deterministic_and_prompts_fit_compact_history():
    bench = NovelGymBenchmark(recipe_depth=16)
    panels = [set(bench.tasks(split)) for split in ("train", "valid_seen", "valid_unseen")]
    assert not (panels[0] & panels[1] or panels[1] & panels[2] or panels[0] & panels[2])
    assert all(int(task.split(":")[-1]) >= 3000000 for task in panels[2])
    env = bench.make_env(bench.tasks("train")[0])
    obs, info = env.reset()
    assert env.reset() == (obs, info)
    prompt = bench.prompt(info["task"], obs, info["admissible"], [(obs, "nop")] * 100)
    assert len(prompt) < 5000 and prompt.count('"inventory"') == 1
    env.close()


@pytest.mark.parametrize("response,action,valid", [
    ("action: break_block", "break_block", True),
    ("Reasoning\naction: craft_planks", "craft_planks", True),
    ("<action>select_axe</action>", "select_axe", True),
    ("approach_log", "approach_log", True),
    ("action: nop\naction: break_block", "nop", False),
    ("<action>nop</action>\naction: break_block", "nop", False),
    ("I will approach a log", "nop", False),
])
def test_native_action_format_is_explicit_and_rejects_multiple_actions(response, action, valid):
    assert NovelGymBenchmark(action_parser="native").parse_action(response) == (action, valid)
