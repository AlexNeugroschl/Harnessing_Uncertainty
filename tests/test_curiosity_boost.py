import dataclasses

import numpy as np
import pytest

from empg.curiosity import CuriosityConfig, CuriosityController, intrinsic_rewards
from empg.nonstationary import NonstationaryConfig


def test_oracle_windows_end_and_do_not_boost_stationary_controls():
    c = CuriosityConfig(boost="oracle_window", boost_factor=4, boost_updates=2)
    controller = CuriosityController(c)
    nc = NonstationaryConfig(enabled=True, switch_iteration=4)
    assert [controller.multiplier(i, nc) for i in range(1, 8)] == [1, 1, 1, 4, 4, 1, 1]
    nc.schedule = "stationary"
    assert all(controller.multiplier(i, nc) == 1 for i in range(1, 8))
    c.boost = "clock_window"
    assert controller.multiplier(4, nc) == 4


@pytest.mark.parametrize("rearm", ["legacy", "stable"])
def test_detector_is_causal_and_never_uses_regime_or_success_labels(rearm):
    c = CuriosityConfig(boost="observed_failure", boost_updates=2, detector_warmup=2, detector_rearm=rearm)
    controller = CuriosityController(c)
    nc = NonstationaryConfig(enabled=False)
    for i in (1, 2):
        controller.observe(i, .1)
    assert controller.multiplier(3, nc) == 1
    controller.observe(3, .8)
    assert controller.triggered
    assert controller.multiplier(3, nc) == 1  # even a later query cannot reweight the triggering update
    assert controller.multiplier(4, nc) == 3
    assert controller.multiplier(5, nc) == 3
    assert controller.multiplier(6, nc) == 1
    with pytest.raises(ValueError, match="feedback"):
        controller.observe(6, None)


def test_boost_scales_curiosity_without_changing_extrinsic_outcomes():
    c = CuriosityConfig(weight=.05, bonus="first_visit")
    seqs = [["a", "b", "b", "c"], ["a", "b", "a"]]
    base = intrinsic_rewards(seqs, [0, 0], [False, True], c)
    boosted = intrinsic_rewards(seqs, [0, 0], [False, True], c, multiplier=3)
    np.testing.assert_allclose(boosted, 3 * base)
    assert boosted[1] == 0  # repeating an unchanged state earns nothing
    zero = dataclasses.replace(c, weight=0)
    assert not intrinsic_rewards(seqs, [0, 0], [False, True], zero, multiplier=3).any()


def test_recurring_boundaries_and_probes():
    nc = NonstationaryConfig(enabled=True, schedule="recurring", initial_regime="baseline",
                            final_regime="novelty", switch_iteration=3, switch_iterations=[3, 6], probe_every=4)
    nc.validate(8)
    assert [nc.regime_at(i) for i in range(1, 9)] == ["baseline"] * 2 + ["novelty"] * 3 + ["baseline"] * 3
    assert nc.probe_iterations(8) == [0, 2, 3, 4, 5, 6, 8]
    assert nc.evaluation_regime() == "baseline"
    controller = CuriosityController(CuriosityConfig(boost="oracle_window", boost_updates=1))
    assert [i for i in range(1, 9) if controller.multiplier(i, nc) > 1] == [3, 6]


@pytest.mark.parametrize("kwargs", [{"boost_factor": float("nan")}, {"boost_updates": 0},
                                  {"failure_ema": 0}, {"weight": -1}, {"boost": "typo"}])
def test_invalid_boosts_fail_early(kwargs):
    with pytest.raises(ValueError):
        CuriosityConfig(**kwargs).validate()
