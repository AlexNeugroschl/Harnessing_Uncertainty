import json

import pytest

import run_benchmarks as runner
import run_nonstationary as matrix
from benchmarks.common import load


def test_matrix_effects_match_tasks_and_keep_missing_conditions_unavailable(tmp_path):
    def save(schedule, arm, outcomes):
        path = tmp_path / schedule / f"{arm}_s0" / "eval_valid_seen.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seed": 0, "protocol": {"regime": "inspection"},
                                    "per_task": [{"task": str(i), "won": w} for i, w in enumerate(outcomes)]}))
        return path
    save("stationary", "grpo", [0, 0])
    save("stationary", "grpo_cur_episodic", [0, 1])
    row, = matrix.matrix_effects(tmp_path, "grpo", [0])
    assert row["curiosity_effect"] == {"stationary": .5} and row["switch_minus_stationary_effect"] is None
    save("switch", "grpo", [0, 0])
    path = save("switch", "grpo_cur_episodic", [1, 1])
    row, = matrix.matrix_effects(tmp_path, "grpo", [0])
    assert row["switch_minus_stationary_effect"] == .5
    ev = json.loads(path.read_text())
    ev["per_task"][0]["task"] = "wrong"
    path.write_text(json.dumps(ev))
    with pytest.raises(ValueError, match="matching"):
        matrix.matrix_effects(tmp_path, "grpo", [0])


def test_estimate_charges_optional_probes_without_loading_policy(monkeypatch):
    from empg.envs import base
    class Bench:
        def tasks(self, split):
            return list(range(24))
        def task_type(self, task):
            return "Pick"
    monkeypatch.setattr(base, "make_benchmark", lambda *a, **kw: Bench())
    cfg = load("grpo_cur_episodic", "nonstationary_screen")
    enabled = runner.estimate(cfg)
    assert enabled["probe_seconds"] > 0
    cfg.nonstationary.probe_every = 0
    disabled = runner.estimate(cfg)
    assert disabled["probe_seconds"] == 0
    assert enabled["seconds"] - disabled["seconds"] == pytest.approx(enabled["probe_seconds"])


def test_matrix_passes_matched_conditions_and_estimate_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    monkeypatch.setattr(matrix, "estimate", lambda *a, **kw: {"seconds": 100., "training_seconds": 70.,
                                                            "probe_seconds": 20., "evaluation_seconds": 5.})
    commands = []
    monkeypatch.setattr(matrix, "execute_phase", lambda cmd, **kw: commands.append(cmd))
    assert matrix.main(["--estimate-only", "--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "out"),
                        "--", "nonstationary.change=rewards"]) == 0
    assert commands == []
    output = capsys.readouterr().out
    assert "4 runs" in output and "TOTAL remaining estimate" in output
    assert "switch / 0 / grpo" in output and "stationary / 0 / grpo_cur_episodic" in output
    assert not (tmp_path / "out").exists() and not (tmp_path / "runs").exists()


def test_matrix_checks_all_existing_conditions_before_starting_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    path = tmp_path / "runs/switch/grpo_cur_episodic_s0/config.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"old": "configuration"}')
    commands = []
    monkeypatch.setattr(matrix.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    with pytest.raises(ValueError, match="different configuration"):
        matrix.main(["--estimate-only", "--runs", str(tmp_path / "runs")])
    assert commands == []


def test_budget_fit_uses_one_fixed_workload_and_preserves_source_configs(monkeypatch):
    cfg = load("grpo", "nonstationary_overnight")
    monkeypatch.setattr(matrix, "estimate", lambda c: {"seconds": c.train.iterations * 100.})
    assert matrix.fit_budget([cfg] * 4, hours=1., fraction=.9) == 6
    assert cfg.train.iterations == 18 and cfg.nonstationary.switch_iteration == 10
    with pytest.raises(ValueError, match="minimum"):
        matrix.fit_budget([cfg] * 4, hours=.1)


def test_timed_out_job_does_not_consume_later_condition_allocations(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    monkeypatch.setattr(matrix, "estimate", lambda *a, **kw: {"seconds": 100.})
    elapsed, calls = [0.], []
    monkeypatch.setattr(matrix.time, "monotonic", lambda: elapsed[0])
    def execute(command, timeout=None):
        if timeout is not None:
            calls.append((command, timeout))
            if len(calls) == 1:
                elapsed[0] += timeout
                raise subprocess.TimeoutExpired(command, timeout)
            elapsed[0] += 5.
    monkeypatch.setattr(matrix, "execute_phase", execute)
    assert matrix.main(["--hours", ".1", "--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "out")]) == 124
    assert len(calls) == 4 and all(timeout > 0 for _, timeout in calls)
    assert "nonstationary.schedule=switch" in calls[0][0]
    assert "nonstationary.schedule=stationary" in calls[2][0]
    manifest = json.loads((tmp_path / "out/matrix_run.json").read_text())
    assert manifest["status"] == "deadline_reached"
    assert manifest["jobs"][0]["status"] == "deadline_reached"
    assert manifest["jobs"][-1]["status"] == "complete"


def test_failure_is_preserved_after_partial_reporting(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    monkeypatch.setattr(matrix, "estimate", lambda *a, **kw: {"seconds": 100.})
    def execute(command, timeout=None):
        if "benchmarks.train" in command:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(matrix, "execute_phase", execute)
    with pytest.raises(subprocess.CalledProcessError):
        matrix.main(["--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "out")])
    assert json.loads((tmp_path / "out/matrix_run.json").read_text())["status"] == "failed"


def test_matrix_interaction_intervals_use_matched_seed_task_effects(tmp_path):
    def save(schedule, arm, outcomes):
        path = tmp_path / schedule / f"{arm}_s0/eval_valid_seen.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seed": 0, "protocol": {"regime": "inspection"},
                                    "per_task": [{"task": str(i), "won": w} for i, w in enumerate(outcomes)]}))
    save("stationary", "grpo", [0, 0])
    save("stationary", "grpo_cur_episodic", [0, 1])
    save("switch", "grpo", [0, 0])
    save("switch", "grpo_cur_episodic", [1, 1])
    stats = matrix.matrix_statistics(tmp_path, "grpo", "grpo_cur_episodic", [0, 1], ["stationary", "switch"])
    assert stats["switch_minus_stationary_effect"]["delta"] == .5
    assert stats["switch"]["seeds"] == [0] and stats["switch"]["scope"] == "task variability only"


def test_report_does_not_invent_recovery_for_disabled_probes(tmp_path):
    from benchmarks.adaptation import write_report
    from empg.config import config_dict
    cfg = load("grpo", "nonstationary_screen", ["nonstationary.probe_every=0"])
    cfg.out_dir = str(tmp_path / "runs")
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "config.json").write_text(json.dumps(config_dict(cfg)))
    (cfg.run_dir / "metrics.jsonl").write_text(json.dumps({"iteration": 4, "episodes": 4, "n_steps": 20,
                                                          "env_return_sum": 1.}) + "\n")
    out = tmp_path / "out"
    out.mkdir()
    text = write_report(tmp_path / "runs", out)
    assert "unavailable" in text
    row, = json.loads((out / "adaptation_summary.json").read_text())
    assert row["recovery_updates"] is None and row["probe_count"] == 0
    assert row["post_boundary_env_return"] == 1.
