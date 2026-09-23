import pytest

from benchmarks.common import arms, load, tiers


@pytest.mark.parametrize("arm", list(arms()))
@pytest.mark.parametrize("tier", [None, *tiers()])
def test_every_arm_loads_in_every_tier(arm, tier):
    cfg = load(arm, tier)
    assert cfg.name == arm
    assert cfg.advantage.method in ("grpo", "empg")


def test_arms_differ_only_in_advantage_and_names_are_unique():
    base = load("grpo")
    for arm in arms():
        cfg = load(arm)
        assert cfg.train == base.train and cfg.policy == base.policy
    assert len({load(a).name for a in arms()}) == len(arms())


def test_tier_then_override_precedence():
    cfg = load("grpo_cur_group", "small", ["train.iterations=3", "advantage.curiosity.weight=0.2"])
    assert cfg.train.iterations == 3 and cfg.train.max_steps == 10  # override > tier > base
    assert cfg.advantage.curiosity.weight == 0.2 and cfg.advantage.curiosity.scope == "group"
    assert str(cfg.run_dir).replace("\\", "/").startswith("runs/small/")
