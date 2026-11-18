import dataclasses
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks.common import load
from empg.config import config_dict
from benchmarks.nonstationary import overnight


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    adapter = tmp_path / "initial"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    cfg = load("grpo", "novelgym_12h")
    profile = {"schema": 1, "complete": True, "identity": {"policy": config_dict(cfg)["policy"]},
        "startup_seconds": 30., "shutdown_seconds": 5., "evaluation_session_overhead_seconds": 4.,
        "training": [{"regime": "baseline", "steps": 512, "rollout_seconds": 10., "update_seconds": 50.},
                     {"regime": "novelty", "steps": 1184, "rollout_seconds": 20., "update_seconds": 100.}],
        "evaluation": [{"regime": "baseline", "tasks": 12, "steps": 384, "seconds": 6., "success": 1.},
                       {"regime": "novelty", "tasks": 12, "steps": 888, "seconds": 10., "success": 0.}]}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(profile))
    args = overnight.parser().parse_args(["--root", str(root), "--initial-adapter", str(adapter), "--calibration", str(path)])
    monkeypatch.setattr(overnight, "verify_identity", lambda *a: None)
    return args, adapter, path, profile


def test_default_12_hour_plan_is_matched_and_contains_predeclared_reserve(prepared):
    args, adapter, path, _ = prepared
    frozen = overnight.freeze(args, adapter, path, 12 * 3600)
    jobs = overnight.deserialize_jobs(frozen["jobs"])
    assert len(jobs) == 20 and sum("alias_of" not in j for j in jobs) == 16
    assert {j["cfg"].train.iterations for j in jobs} == {frozen["workload"]["iterations"]}
    assert frozen["workload"]["estimated_seconds"] <= 12 * 3600 * .92
    assert len(frozen["reserve_jobs"]) == 3 * args.reserve_count
    assert {j["seed"] for j in frozen["reserve_jobs"]} == set(range(2, 2 + args.reserve_count))
    assert all(j["schedule"] == "switch" for j in frozen["reserve_jobs"])


def test_insufficient_baseline_competence_stops_before_fitting(prepared):
    args, adapter, path, profile = prepared
    profile["evaluation"][0]["success"] = .5
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="competence gate"):
        overnight.freeze(args, adapter, path, 12 * 3600)


def test_custom_curiosity_coefficient_is_preserved_in_both_blocks(prepared):
    args, adapter, path, _ = prepared
    args.eta = .123
    args.reserve_count = 1
    frozen = overnight.freeze(args, adapter, path, 43200)
    for job in overnight.combined_jobs(frozen):
        if job["treatment"] == "constant":
            assert job["cfg"].advantage.curiosity.weight == args.eta


def test_estimate_without_measurements_is_read_only(tmp_path):
    root = tmp_path / "not_created"
    assert overnight.main(["--root", str(root), "--estimate-only"]) == 0
    assert not root.exists()


def test_prepared_plan_can_be_inspected_and_reported_without_starting_a_phase(prepared, monkeypatch):
    args, adapter, path, _ = prepared
    root = Path(args.root)
    root.mkdir()
    frozen = overnight.freeze(args, adapter, path, 43200)
    (root / "plan.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(overnight, "_phase", lambda *a, **kw: pytest.fail("must not start a process"))
    assert overnight.main(["--root", str(root), "--estimate-only"]) == 0
    assert overnight.main(["--root", str(root), "--prepare-only"]) == 0
    assert overnight.main(["--root", str(root), "--report-only"]) == 0
    assert (root / "results" / "comparison.md").exists()
    with pytest.raises(SystemExit):
        overnight.main(["--root", str(root), "--hours", "11", "--estimate-only"])


def test_reserve_aliases_point_to_their_own_seed_after_combining(prepared):
    args, adapter, path, _ = prepared
    args.boost_factor = 1.
    frozen = overnight.freeze(args, adapter, path, 43200)
    jobs = overnight.combined_jobs(frozen)
    reserve = [j for j in jobs if j.get("optional_reserve")]
    assert reserve
    for job in reserve:
        if "alias_of" in job:
            assert jobs[job["alias_of"]]["seed"] == job["seed"]


def test_expired_phase_budget_never_starts_a_subprocess(monkeypatch):
    from benchmarks import process
    monkeypatch.setattr(overnight.time, "monotonic", lambda: 20.)
    monkeypatch.setattr(process.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not spawn"))
    with pytest.raises(subprocess.TimeoutExpired):
        overnight._phase(["python", "phase.py"], 19.)


def test_changed_calibration_cannot_resume_frozen_workload(prepared):
    args, adapter, path, profile = prepared
    root = Path(args.root)
    root.mkdir()
    frozen = overnight.freeze(args, adapter, path, 43200)
    (root / "plan.json").write_text(json.dumps(frozen))
    profile["startup_seconds"] = 100
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="calibration changed"):
        overnight.main(["--root", str(root), "--prepare-only"])


def test_reserve_blocks_use_remaining_time_and_omit_unstarted_seeds(prepared, monkeypatch):
    args, adapter, path, _ = prepared
    args.reserve_count = 4
    root = Path(args.root)
    root.mkdir()
    frozen = overnight.freeze(args, adapter, path, 43200)
    (root / "plan.json").write_text(json.dumps(frozen))
    clock = [0.]
    commands = []
    reports = []
    monkeypatch.setattr(overnight.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(overnight, "reserve_cost", lambda *a: 50.)
    monkeypatch.setattr(overnight.matrix, "report", lambda jobs, out: reports.append(jobs))
    def phase(command, deadline, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            clock[0] += 42800.
        else:
            clock[0] += 100.
            seed = int(command[command.index("--seeds") + 1])
            for job in overnight.deserialize_jobs(frozen["reserve_jobs"]):
                if job["seed"] == seed:
                    job["cfg"].run_dir.mkdir(parents=True, exist_ok=True)
                    (job["cfg"].run_dir / "config.json").write_text("{}")
    monkeypatch.setattr(overnight, "_phase", phase)
    assert overnight.main(["--root", str(root)]) == 0
    state = json.loads((root / "system_run.json").read_text())
    assert state["status"] == "complete" and state["reserve_completed_seeds"] == [2, 3, 4]
    assert len(commands) == 4
    assert {j["seed"] for j in reports[-1] if j.get("optional_reserve")} == {2, 3, 4}
