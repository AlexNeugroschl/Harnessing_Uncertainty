import dataclasses

import pytest

from benchmarks.common import load
from benchmarks.nonstationary.budget import common_workload, estimate, validate_measurements, verify_identity


def measurements():
    return {"schema": 1, "complete": True, "identity": {"test": True}, "startup_seconds": 30.,
        "shutdown_seconds": 5., "evaluation_session_overhead_seconds": 4.,
        "training": [{"regime": "baseline", "steps": 512, "rollout_seconds": 10., "update_seconds": 50.},
                     {"regime": "novelty", "steps": 1184, "rollout_seconds": 20., "update_seconds": 100.}],
        "evaluation": [{"regime": "baseline", "tasks": 12, "steps": 384, "seconds": 6., "success": 1.},
                       {"regime": "novelty", "tasks": 12, "steps": 888, "seconds": 10., "success": 0.}]}


def configuration(iterations=16):
    return load("grpo", "novelgym_12h", [f"train.iterations={iterations}",
        f"nonstationary.switch_iteration={iterations // 2 + 1}", f"nonstationary.probe_every={iterations // 4}",
        "train.max_steps=74", "eval.max_steps=74"])


def test_budget_accounts_for_costlier_stationary_final_rules_and_probes():
    cfg = configuration()
    stationary = dataclasses.replace(cfg, nonstationary=dataclasses.replace(cfg.nonstationary, schedule="stationary"))
    left, right = estimate(cfg, measurements()), estimate(stationary, measurements())
    assert right["seconds"] > left["seconds"]
    assert left["probe_seconds"] > 0 and left["evaluation_seconds"] > 0
    assert estimate(cfg, measurements(), train=False)["training_seconds"] == 0


def test_fitting_freezes_common_updates_and_excludes_reused_conditions():
    def jobs(n):
        return [{"cfg": configuration(n)}, {"cfg": configuration(n)}, {"cfg": configuration(n), "alias_of": 0}]
    fit = common_workload(jobs, measurements(), 12000, minimum=12)
    n = fit["iterations"]
    assert n >= 12 and n % 4 == 0 and fit["unique_runs"] == 2
    assert fit["estimated_seconds"] <= fit["target_seconds"]
    next_cost = 90 + 2 * estimate(configuration(n + 4), measurements())["seconds"]
    assert next_cost > fit["target_seconds"]
    with pytest.raises(ValueError, match="minimum matched"):
        common_workload(jobs, measurements(), 1, minimum=12)


@pytest.mark.parametrize("mutation", ["incomplete", "missing_regime", "zero_steps", "missing_eval", "nan_success", "zero_diversity"])
def test_incomplete_or_invalid_profiles_cannot_fit_an_experiment(mutation):
    data = measurements()
    if mutation == "incomplete":
        data["complete"] = False
    elif mutation == "missing_regime":
        data["training"].pop()
    elif mutation == "zero_steps":
        data["training"][0]["steps"] = 0
    elif mutation == "nan_success":
        data["evaluation"][0]["success"] = float("nan")
    elif mutation == "zero_diversity":
        data["diversity"] = {"steps": 128, "gradient_rows": 0, "update_seconds": 10.}
    else:
        data["evaluation"] = []
    with pytest.raises(ValueError):
        validate_measurements(data)


def test_changed_execution_identity_rejects_calibration(monkeypatch):
    from benchmarks.nonstationary import calibration
    monkeypatch.setattr(calibration, "runtime_identity", lambda cfg: {"test": True})
    verify_identity(configuration(), measurements())
    monkeypatch.setattr(calibration, "runtime_identity", lambda cfg: {"test": False})
    with pytest.raises(ValueError, match="calibration differs"):
        verify_identity(configuration(), measurements())
