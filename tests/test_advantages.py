import numpy as np
import pytest

from empg.advantages import (
    AdvantageConfig, clarity_f, compute_advantages, grpo_advantages,
    next_step_entropy, normalize_entropy, scale_g,
)

# Two groups, 2 trajectories each. Group 0: one success, one failure.
# Group 1: both fail (GRPO has no signal there).
REWARDS = np.array([10.0, 0.0, 0.0, 0.0])
GROUPS = np.array([0, 0, 1, 1])
STEP_TRAJ = np.array([0, 0, 0, 1, 1, 2, 2, 3])
ENTROPY = np.array([0.1, 0.5, 0.2, 0.05, 1.5, 0.8, 0.3, 1.0])


def test_grpo_group_relative():
    a = grpo_advantages(REWARDS, GROUPS)
    assert a[0] > 0 > a[1]
    assert a[0] == pytest.approx(-a[1])
    assert a[2] == a[3] == 0.0


def test_grpo_singleton_group():
    assert grpo_advantages([3.0], [0])[0] == pytest.approx(3.0, rel=1e-4)


def test_normalize_range():
    h = normalize_entropy(ENTROPY)
    assert h.min() == 0.0 and h.max() == pytest.approx(1.0, abs=1e-5)


def test_g_mean_one_and_monotone():
    h = normalize_entropy(ENTROPY)
    g = scale_g(h)
    assert g.mean() == pytest.approx(1.0)
    assert g[h.argmin()] > 1 > g[h.argmax()]
    assert np.all(np.diff(g[np.argsort(h)]) < 0)


def test_next_step_entropy_respects_boundaries():
    h = np.arange(8, dtype=float)
    nxt = next_step_entropy(h, STEP_TRAJ)
    np.testing.assert_array_equal(nxt[[0, 1, 3, 5]], [1, 2, 4, 6])
    assert np.isnan(nxt[[2, 4, 6, 7]]).all()  # terminal steps


def test_terminal_clarity_is_zero():
    f = clarity_f(np.array([0.0, 1.0, np.nan]))
    np.testing.assert_allclose(f, [1.0, np.exp(-1.0), 0.0])


def test_grpo_method_is_broadcast():
    a = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENTROPY, AdvantageConfig(method="grpo"))
    np.testing.assert_allclose(a, grpo_advantages(REWARDS, GROUPS)[STEP_TRAJ])


def test_empg_zero_mean():
    a = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENTROPY, AdvantageConfig())
    assert a.mean() == pytest.approx(0.0, abs=1e-12)


def test_empg_without_mechanisms_is_centered_grpo():
    cfg = AdvantageConfig(use_g=False, use_f=False)
    a = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENTROPY, cfg)
    base = grpo_advantages(REWARDS, GROUPS)[STEP_TRAJ]
    np.testing.assert_allclose(a, base - base.mean())


def test_confident_error_penalised_harder():
    # Trajectory 1 failed: step 3 is confident (H=0.05), step 4 uncertain (H=1.5).
    a = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENTROPY, AdvantageConfig(use_f=False))
    assert a[3] < a[4] < 0


def test_confident_success_amplified():
    # Trajectory 0 succeeded: step 0 (H=0.1) gets more credit than step 1 (H=0.5).
    a = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENTROPY, AdvantageConfig(use_f=False))
    grpo = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENTROPY, AdvantageConfig(method="grpo"))
    assert a[0] > a[1] > 0
    assert a[0] > grpo[0]


def test_clarity_gives_signal_when_grpo_has_none():
    # Group 1 all failed -> GRPO advantage identically 0 on its steps.
    grpo = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENTROPY, AdvantageConfig(method="grpo"))
    assert np.all(grpo[5:] == 0)
    a = compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENTROPY, AdvantageConfig(use_g=False))
    # Step 5 leads to a low-entropy successor (0.3), step 6 is terminal (no bonus).
    assert a[5] > a[6]


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        compute_advantages(REWARDS, GROUPS, STEP_TRAJ, ENTROPY, AdvantageConfig(method="ppo"))
