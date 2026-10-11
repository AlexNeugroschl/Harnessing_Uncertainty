import numpy as np
import pytest

from empg.advantages import AdvantageConfig, compute_advantages, grpo_advantages
from empg.curiosity import (
    CuriosityConfig, apply_cap, episodic_counts, exploration_stats, group_counts,
    intrinsic_rewards, no_success_signal, novelty_bonus, schedule_multiplier,
)
from empg.envs.alfworld_env import ParsedState
from empg.prompts import alfworld_prompt, visited_locations

# Two groups of two trajectories. Key sequences are [s_0, s_1, ..., s_T].
SEQS = [["a", "b", "a", "c"],   # group 0: revisits a
        ["a", "b", "d"],        # group 0: shares a, b with its sibling
        ["x", "y"],             # group 1
        ["x", "z"]]             # group 1
GROUPS = [0, 0, 1, 1]
WON = [True, False, False, False]  # group 0 has signal, group 1 all failed
N_STEPS = sum(len(s) - 1 for s in SEQS)


def test_episodic_counts_revisit_and_first_visit():
    c = episodic_counts(SEQS)
    np.testing.assert_array_equal(c[0], [1, 2, 1])  # b new, a revisited, c new
    np.testing.assert_array_equal(c[1], [1, 1])


def test_revisit_gets_no_first_visit_bonus_and_smaller_inv_sqrt():
    c = episodic_counts(SEQS)[0]
    np.testing.assert_array_equal(novelty_bonus(c, "first_visit"), [1, 0, 1])
    b = novelty_bonus(c, "inv_sqrt")
    assert b[0] == b[2] == 1.0 and b[1] == pytest.approx(1 / np.sqrt(2))


def test_group_counts_split_shared_states():
    c = group_counts(SEQS, GROUPS)
    np.testing.assert_array_equal(c[0], [2, 2, 1])  # b, a shared with sibling; c unique
    np.testing.assert_array_equal(c[1], [2, 1])     # b shared; d unique
    np.testing.assert_array_equal(c[2], [1])        # y unique within group 1


def test_cap_limits_episode_total():
    capped = apply_cap(np.ones(5), cap=2.5)
    np.testing.assert_allclose(capped, [1, 1, 0.5, 0, 0])
    assert apply_cap(np.ones(3), 0).sum() == 3


def test_schedules():
    cfg = CuriosityConfig(schedule="linear")
    assert schedule_multiplier(cfg, 1, 10) == 1.0
    assert schedule_multiplier(cfg, 6, 10) == pytest.approx(0.5)
    assert schedule_multiplier(cfg, 20, 10) == 0.0
    cfg = CuriosityConfig(schedule="exp", decay_iters=4)
    assert schedule_multiplier(cfg, 5, 100) == pytest.approx(0.5)
    assert schedule_multiplier(CuriosityConfig(), 99, 10) == 1.0


def test_no_success_signal_gate():
    np.testing.assert_array_equal(no_success_signal(GROUPS, WON), [False, False, True, True])
    cfg = CuriosityConfig(weight=1.0, gate="no_success_signal", bonus="first_visit")
    c = intrinsic_rewards(SEQS, GROUPS, WON, cfg)
    assert len(c) == N_STEPS
    assert np.all(c[:5] == 0)       # group 0 has a success -> gated off
    np.testing.assert_array_equal(c[5:], [1, 1])


def test_weight_and_modulation():
    cfg = CuriosityConfig(weight=0.5, bonus="first_visit")
    c = intrinsic_rewards(SEQS, GROUPS, WON, cfg)
    assert c.max() == 0.5
    h = np.linspace(0, 1, N_STEPS)
    conf = intrinsic_rewards(SEQS, GROUPS, WON, CuriosityConfig(weight=1, modulate="confident"), h_norm=h)
    unc = intrinsic_rewards(SEQS, GROUPS, WON, CuriosityConfig(weight=1, modulate="uncertain"), h_norm=h)
    base = intrinsic_rewards(SEQS, GROUPS, WON, CuriosityConfig(weight=1))
    np.testing.assert_allclose(conf + unc, base)


# ------------------------------------------------ integration with advantages
REWARDS = [10.0, 0.0, 0.0, 0.0]
STEP_TRAJ = np.repeat(np.arange(4), [len(s) - 1 for s in SEQS])
ENT = np.linspace(0.1, 1.0, N_STEPS)


@pytest.mark.parametrize("method", ["grpo", "empg"])
def test_zero_curiosity_is_identical(method):
    cfg = AdvantageConfig(method=method)
    a = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENT, cfg)
    b = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENT, cfg, intrinsic=None)
    np.testing.assert_array_equal(a, b)
    weighted_zero = intrinsic_rewards(SEQS, GROUPS, WON, cfg.curiosity)
    c = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENT, cfg, intrinsic=weighted_zero)
    np.testing.assert_array_equal(a, c)


def test_step_mode_adds_centred_bonus():
    cfg = AdvantageConfig(method="grpo")
    intr = intrinsic_rewards(SEQS, GROUPS, WON, CuriosityConfig(weight=0.1))
    a = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENT, cfg, intrinsic=intr)
    base = grpo_advantages(REWARDS, GROUPS)[STEP_TRAJ]
    np.testing.assert_allclose(a - base, intr - intr.mean())
    # All-failed group 1: GRPO gives 0, curiosity ranks the steps.
    assert np.all(base[5:] == 0) and a[5] != a[4]


def test_empg_with_curiosity_still_zero_mean():
    cfg = AdvantageConfig(method="empg")
    intr = intrinsic_rewards(SEQS, GROUPS, WON, CuriosityConfig(weight=0.1))
    a = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENT, cfg, intrinsic=intr)
    assert a.mean() == pytest.approx(0.0, abs=1e-12)


def test_trajectory_mode_shapes_outcome_reward():
    cfg = AdvantageConfig(method="grpo")
    cfg.curiosity.mode = "trajectory"
    intr = intrinsic_rewards(SEQS, GROUPS, WON, CuriosityConfig(weight=1.0, bonus="first_visit"))
    a = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENT, cfg, intrinsic=intr)
    # Group 1: traj 2 visits new y (mean bonus 1), traj 3 visits new z (1) -> tie, still 0.
    assert np.allclose(a[STEP_TRAJ == 2], 0) and np.allclose(a[STEP_TRAJ == 3], 0)
    # Per-step credit within a trajectory stays uniform.
    assert np.ptp(a[STEP_TRAJ == 0]) == 0


def test_exploration_stats():
    s = exploration_stats(SEQS, GROUPS, WON)
    assert s["revisit_rate"] == pytest.approx(1 / N_STEPS)
    assert s["noop_rate"] == 0.0
    assert exploration_stats([["a", "a", "b"]], [0], [False])["noop_rate"] == 0.5
    assert s["unique_states"] == pytest.approx((3 + 3 + 2 + 2) / 4)
    assert s["group_coverage"] == pytest.approx((4 + 3) / 2)  # {a,b,c,d}, {x,y,z}
    assert s["any_success_frac"] == 0.5 and s["no_signal_frac"] == 0.5
    assert s["success_len"] == 3


# ------------------------------------------------------- parsed state + memory
def test_parsed_state_tracks_world_changes():
    p = ParsedState()
    k0 = p.key()
    p.update("go to desk 1", "You arrive at desk 1. On the desk 1, you see a mug 1.")
    k1 = p.key()
    p.update("examine desk 1", "On the desk 1, you see a mug 1.")
    assert p.key() == k1                      # examine changes nothing
    p.update("take mug 1 from desk 1", "Nothing happens.")
    assert p.key() == k1                      # failed action changes nothing
    p.update("take mug 1 from desk 1", "You pick up the mug 1 from the desk 1.")
    assert "mug 1" in p.inventory
    p.update("go to sinkbasin 1", "You arrive at sinkbasin 1.")
    p.update("clean mug 1 with sinkbasin 1", "You clean the mug 1 using the sinkbasin 1.")
    p.update("go to desk 1", "You arrive at desk 1.")
    p.update("move mug 1 to desk 1", "You move the mug 1 to the desk 1.")
    assert not p.inventory and "clean:mug 1" in p.conditions
    assert len({k0, k1, p.key()}) == 3        # back at desk 1, but the mug is now clean there


def test_parsed_state_tracks_object_placement():
    p = ParsedState()
    p.update("go to desk 1", "You arrive at desk 1.")
    at_start = p.key()
    p.update("take mug 1 from desk 1", "You pick up the mug 1 from the desk 1.")
    p.update("go to shelf 1", "You arrive at shelf 1.")
    p.update("move mug 1 to shelf 1", "You move the mug 1 to the shelf 1.")
    p.update("go to desk 1", "You arrive at desk 1.")
    assert p.key() != at_start               # same place, empty hands, but the mug moved
    p.update("go to shelf 1", "You arrive at shelf 1.")
    p.update("take mug 1 from shelf 1", "You pick up the mug 1 from the shelf 1.")
    p.update("go to desk 1", "You arrive at desk 1.")
    p.update("move mug 1 to desk 1", "You move the mug 1 to the desk 1.")
    assert p.key() == at_start               # mug back where it began: world unchanged


def test_parsed_state_open_close_roundtrip():
    p = ParsedState()
    p.update("go to drawer 1", "You arrive at drawer 1. The drawer 1 is closed.")
    k = p.key()
    p.update("open drawer 1", "You open the drawer 1. The drawer 1 is open.")
    assert p.key() != k
    p.update("close drawer 1", "You close the drawer 1.")
    assert p.key() == k


def test_visited_locations_memory_prompt():
    hist = [("o0", "go to desk 1"), ("You arrive at desk 1.", "go to shelf 9"),
            ("Nothing happens.", "go to bed 1"), ("You arrive at bed 1.", "go to desk 1")]
    visited = visited_locations(hist, "You arrive at desk 1.")
    assert visited == ["desk 1", "bed 1"]     # shelf 9 failed; desk 1 deduplicated
    p = alfworld_prompt("task", "obs", ["look"], hist, visited=visited)
    assert "already visited: desk 1, bed 1." in p
    assert "already visited" not in alfworld_prompt("task", "obs", ["look"], hist)
