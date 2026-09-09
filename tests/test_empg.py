"""Unit tests for the EMPG math (Eq. 4-13).

These run without Ollama or torch: they pin the properties the paper's
abstract actually claims, so a regression in the math is caught here rather
than being mistaken for a bad training run.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from empg.empg import (  # noqa: E402
    broadcast_advantages,
    clarity_bonus_f,
    compute_empg_advantages,
    finalize_advantages,
    gradient_scale_g,
    group_advantages,
    modulated_advantage,
    normalize_entropy,
    successor_entropy,
)


# --- Eq. 4 ------------------------------------------------------------------

def test_group_advantages_matches_hand_computation():
    rewards = np.array([1.0, 0.0, 0.0, 1.0])
    # mean 0.5, population std 0.5 -> (r - 0.5) / 0.5 = +1/-1
    np.testing.assert_allclose(
        group_advantages(rewards), [1.0, -1.0, -1.0, 1.0], atol=1e-6
    )


def test_group_advantages_all_equal_rewards_gives_zero():
    """The degenerate case that motivates the clarity bonus: if every rollout
    in a group succeeds (or all fail), GRPO's advantage is identically zero
    and the group contributes no learning signal at all."""
    np.testing.assert_allclose(group_advantages(np.ones(8)), np.zeros(8), atol=1e-6)


# --- Eq. 12 -----------------------------------------------------------------

def test_normalize_entropy_maps_to_unit_range():
    h_norm = normalize_entropy(np.array([0.2, 0.7, 1.2]))
    assert h_norm.min() == pytest.approx(0.0)
    assert h_norm.max() == pytest.approx(1.0, abs=1e-6)


def test_normalize_entropy_constant_input_does_not_divide_by_zero():
    h_norm = normalize_entropy(np.full(5, 0.42))
    assert np.all(np.isfinite(h_norm))


# --- Eq. 10: the self-calibration property ----------------------------------

def test_gradient_scale_has_unit_batch_mean():
    """g is normalised so it redistributes gradient mass rather than
    rescaling the effective learning rate."""
    rng = np.random.default_rng(0)
    g = gradient_scale_g(normalize_entropy(rng.random(64)))
    assert g.mean() == pytest.approx(1.0, abs=1e-9)


def test_confident_steps_amplified_uncertain_damped():
    g = gradient_scale_g(normalize_entropy(np.array([0.1, 0.5, 1.5])))
    assert g[0] > 1.0, "low-entropy (confident) step must be amplified"
    assert g[-1] < 1.0, "high-entropy (uncertain) step must be damped"
    assert g[0] > g[1] > g[2], "g must decrease monotonically in entropy"


# --- The abstract's three headline behaviours -------------------------------

def test_confident_correct_action_gets_larger_update():
    """Abstract: 'strengthens updates for confident correct actions'."""
    h_norm = np.array([0.0, 1.0])          # confident, uncertain
    a = np.array([1.0, 1.0])               # both correct
    a_mod = modulated_advantage(a, h_norm, np.zeros(2), use_clarity=False)
    assert a_mod[0] > a[0], "confident+correct should exceed the raw advantage"
    assert a_mod[1] < a[1], "uncertain step should be damped"


def test_confident_error_gets_stronger_penalty():
    """Abstract: 'penalizes confident errors'.

    The key subtlety: g multiplies, so it scales *magnitude* and preserves
    sign. A confident mistake is pushed further negative, not toward zero.
    """
    h_norm = np.array([0.0, 1.0])
    a = np.array([-1.0, -1.0])             # both wrong
    a_mod = modulated_advantage(a, h_norm, np.zeros(2), use_clarity=False)
    assert a_mod[0] < a[0], "confident+wrong must be penalised harder"
    assert abs(a_mod[0]) > abs(a_mod[1]), "confident error > uncertain error"
    assert a_mod[0] < 0, "sign must be preserved"


def test_uncertain_steps_damped_regardless_of_sign():
    """Abstract: 'reduces updates from uncertain steps' - for both outcomes."""
    h_norm = np.array([0.0, 1.0, 0.0, 1.0])
    a = np.array([1.0, 1.0, -1.0, -1.0])
    a_mod = modulated_advantage(a, h_norm, np.zeros(4), use_clarity=False)
    assert abs(a_mod[1]) < abs(a_mod[0])   # positive case
    assert abs(a_mod[3]) < abs(a_mod[2])   # negative case


# --- Eq. 11 -----------------------------------------------------------------

def test_clarity_bonus_prefers_predictable_successors():
    f = clarity_bonus_f(np.array([0.0, 1.0]))
    assert f[0] > f[1], "leading to a low-entropy state must score higher"
    assert np.all(f > 0), "bonus is strictly positive"


def test_clarity_bonus_gives_signal_when_grpo_is_silent():
    """With all-equal rewards A_i == 0, so the scaling term vanishes and only
    the clarity bonus can still differentiate steps."""
    a = np.zeros(3)
    h_norm = np.array([0.0, 0.5, 1.0])
    a_final = finalize_advantages(
        modulated_advantage(a, h_norm, h_norm, zeta=0.05)
    )
    assert a_final.std() > 0, "clarity bonus must break the tie"
    assert a_final[0] > a_final[2]


def test_successor_entropy_does_not_leak_across_trajectories():
    h_norm = np.array([0.1, 0.2, 0.9, 0.3])
    done = np.array([False, True, False, True])   # two 2-step trajectories
    nxt = successor_entropy(h_norm, done)
    assert nxt[0] == pytest.approx(0.2)   # within trajectory 1
    assert nxt[1] == pytest.approx(0.0)   # terminal -> max clarity, no leak
    assert nxt[2] == pytest.approx(0.3)   # within trajectory 2
    assert nxt[3] == pytest.approx(0.0)   # terminal


# --- Eq. 13 -----------------------------------------------------------------

def test_final_advantages_are_zero_mean():
    rng = np.random.default_rng(1)
    a_final = compute_empg_advantages(
        rng.normal(size=32), rng.random(32), np.zeros(32, dtype=bool)
    )
    assert a_final.mean() == pytest.approx(0.0, abs=1e-9)


# --- Plumbing ---------------------------------------------------------------

def test_broadcast_spreads_outcome_reward_over_steps():
    """The GRPO baseline: one sparse outcome number per trajectory, smeared
    uniformly across its steps."""
    out = broadcast_advantages(np.array([1.0, -1.0]), np.array([3, 2]))
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0, -1.0, -1.0])


def test_ablation_toggles_change_the_result():
    """Paper Table 2 runs g-only, f-only and both; all three must differ."""
    a = np.array([1.0, -1.0, 1.0])
    h = np.array([0.2, 0.8, 0.5])
    done = np.array([False, False, True])
    both = compute_empg_advantages(a, h, done)
    g_only = compute_empg_advantages(a, h, done, use_clarity=False)
    f_only = compute_empg_advantages(a, h, done, use_scaling=False)
    assert not np.allclose(both, g_only)
    assert not np.allclose(both, f_only)
    assert not np.allclose(g_only, f_only)


def test_empg_reduces_to_grpo_when_both_mechanisms_off():
    """Sanity anchor: with no modulation EMPG must be plain (re-centred) GRPO."""
    a = np.array([1.0, -1.0, 1.0, -1.0])
    h = np.array([0.2, 0.8, 0.5, 0.1])
    done = np.array([False, True, False, True])
    out = compute_empg_advantages(a, h, done, use_scaling=False, use_clarity=False)
    np.testing.assert_allclose(out, a - a.mean(), atol=1e-9)
