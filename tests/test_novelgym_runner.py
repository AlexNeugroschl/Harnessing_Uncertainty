import dataclasses
import json
from pathlib import Path

import pytest

import run_novelgym as runner
from empg.config import config_dict, config_fingerprint


def arguments(tmp_path, *options):
    return runner.parser().parse_args(["--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "results"), *options])


def test_matrix_matches_tasks_and_reuses_identical_stationary_controls(tmp_path):
    jobs = runner.plan(arguments(tmp_path))
    assert len(jobs) == 12
    aliases = [j for j in jobs if "alias_of" in j]
    assert len(aliases) == 2
    assert {j["treatment"] for j in aliases} == {"oracle_x3", "matched_x3"}
    assert all(j["schedule"] == "stationary" for j in aliases)
    switched = [j for j in jobs if j["schedule"] == "switch"]
    low = next(j for j in switched if j["treatment"] == "constant")["cfg"]
    matched = next(j for j in switched if j["treatment"] == "matched_x3")["cfg"]
    assert matched.advantage.curiosity.weight == pytest.approx(.0875)
    assert all(j["cfg"].seed == low.seed and j["cfg"].env == low.env for j in switched)


def test_recurring_and_short_manual_workloads_validate_before_model_loading(tmp_path):
    jobs = runner.plan(arguments(tmp_path, "--schedules", "recurring", "stationary", "--switches", "3", "6",
                                 "--treatments", "oracle", "--", "train.iterations=8"))
    nc = jobs[0]["cfg"].nonstationary
    assert nc.change_points() == [3, 6]
    jobs = runner.plan(arguments(tmp_path, "--switches", "2", "--", "train.iterations=3"))
    assert jobs[0]["cfg"].train.iterations == 3


def test_missing_results_stay_unavailable(tmp_path):
    args = arguments(tmp_path, "--treatments", "constant", "oracle")
    jobs = runner.plan(args)
    result = runner.report(jobs, Path(args.out))
    assert not result["comparisons"]
    assert all(not r["complete"] for r in result["runs"])
    assert "unavailable" in (Path(args.out) / "comparison.md").read_text()


def test_completed_pairs_require_matching_protocols_and_record_competence(tmp_path):
    from benchmarks.evaluate import evaluation_protocol, eval_path
    args = arguments(tmp_path, "--schedules", "switch", "--treatments", "constant", "oracle")
    jobs = runner.plan(args)
    for job in jobs:
        cfg = job["cfg"]
        cfg.run_dir.mkdir(parents=True)
        (cfg.run_dir / "config.json").write_text(json.dumps(config_dict(cfg)))
        (cfg.run_dir / "training_complete.json").write_text(json.dumps({"config_fingerprint": config_fingerprint(cfg)}))
        eval_path(cfg).write_text(json.dumps({"seed": cfg.seed, "all": .5, "protocol": evaluation_protocol(cfg),
            "config_fingerprint": config_fingerprint(cfg),
            "per_task": [{"task": "t1", "won": True}, {"task": "t2", "won": False}]}))
    result = runner.report(jobs, Path(args.out))
    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0]["degenerate"]
    assert all(not r["adaptation"]["adaptation_interpretable"] for r in result["runs"])
    cfg = jobs[1]["cfg"]
    data = json.loads(eval_path(cfg).read_text())
    data["protocol"]["benchmark"] = "different"
    eval_path(cfg).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="protocol"):
        runner.report(jobs, Path(args.out))


def test_changed_treatments_cannot_reuse_previous_fingerprints(tmp_path):
    jobs = runner.plan(arguments(tmp_path, "--treatments", "oracle"))
    cfg = jobs[0]["cfg"]
    before = config_fingerprint(cfg)
    cfg.advantage.curiosity.boost_factor = 7
    assert config_fingerprint(cfg) != before


def test_execution_manifest_marks_reused_conditions_complete(tmp_path, monkeypatch):
    from benchmarks import process
    from benchmarks.evaluate import eval_path, evaluation_protocol
    args = arguments(tmp_path, "--schedules", "stationary", "--treatments", "constant", "oracle", "matched")
    jobs = runner.plan(args)
    real = [j for j in jobs if "alias_of" not in j]
    assert len(real) == 1
    calls = []
    def execute(command, timeout=None):
        calls.append(command)
        cfg = real[0]["cfg"]
        cfg.run_dir.mkdir(parents=True)
        (cfg.run_dir / "adapter").mkdir()
        (cfg.run_dir / "config.json").write_text(json.dumps(config_dict(cfg)))
        (cfg.run_dir / "training_complete.json").write_text(json.dumps({"config_fingerprint": config_fingerprint(cfg)}))
        eval_path(cfg).write_text(json.dumps({"seed": 0, "all": 1., "protocol": evaluation_protocol(cfg),
            "config_fingerprint": config_fingerprint(cfg), "per_task": [{"task": "t", "won": True}]}))
    monkeypatch.setattr(process, "run_process", execute)
    assert runner.main(["--runs", args.runs, "--out", args.out,
                        "--schedules", "stationary", "--treatments", "constant", "oracle", "matched"]) == 0
    manifest = json.loads((Path(args.out) / "matrix_run.json").read_text())
    assert len(calls) == 1 and manifest["status"] == "complete"
    assert all(j["complete"] for j in manifest["jobs"])
    assert sum(j["status"] == "reused_complete" for j in manifest["jobs"]) == 2


def test_switch_stationary_interaction_preserves_four_cell_pairs(tmp_path):
    from benchmarks.evaluate import evaluation_protocol, eval_path
    args = arguments(tmp_path, "--treatments", "constant", "oracle")
    jobs = runner.plan(args)
    for job in jobs:
        if "alias_of" in job:
            continue
        cfg = job["cfg"]
        cfg.run_dir.mkdir(parents=True)
        (cfg.run_dir / "config.json").write_text(json.dumps(config_dict(cfg)))
        (cfg.run_dir / "training_complete.json").write_text(json.dumps({"config_fingerprint": config_fingerprint(cfg)}))
        won = job["treatment"].startswith("oracle") and job["schedule"] == "switch"
        eval_path(cfg).write_text(json.dumps({"seed": 0, "all": float(won), "protocol": evaluation_protocol(cfg),
            "config_fingerprint": config_fingerprint(cfg), "per_task": [{"task": "t", "won": won}]}))
    result = runner.report(jobs, Path(args.out))
    interactions = [r for r in result["comparisons"] if r.get("effect_type") == "paired interaction"]
    assert len(interactions) == 1 and interactions[0]["delta"] == 1
    assert interactions[0]["seeds"] == [0] and interactions[0]["n_tasks"] == 1


def test_time_admitted_replications_do_not_change_primary_estimate(tmp_path):
    from benchmarks.evaluate import evaluation_protocol, eval_path
    args = arguments(tmp_path, "--schedules", "switch", "--seeds", "0", "2", "--treatments", "constant", "oracle")
    jobs = runner.plan(args)
    for job in jobs:
        job["optional_reserve"] = job["seed"] == 2
        cfg = job["cfg"]
        cfg.run_dir.mkdir(parents=True)
        (cfg.run_dir / "config.json").write_text(json.dumps(config_dict(cfg)))
        (cfg.run_dir / "training_complete.json").write_text(json.dumps({"config_fingerprint": config_fingerprint(cfg)}))
        won = job["optional_reserve"] and job["treatment"].startswith("oracle")
        eval_path(cfg).write_text(json.dumps({"seed": cfg.seed, "all": float(won), "protocol": evaluation_protocol(cfg),
            "config_fingerprint": config_fingerprint(cfg), "per_task": [{"task": "t", "won": won}]}))
    result = runner.report(jobs, Path(args.out))
    cohorts = {r["cohort"]: r for r in result["comparisons"]}
    assert cohorts["primary"]["seeds"] == [0] and cohorts["primary"]["delta"] == 0.
    assert cohorts["reserve"]["seeds"] == [2] and cohorts["reserve"]["delta"] == 1.
