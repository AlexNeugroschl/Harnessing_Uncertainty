import json
from pathlib import Path
import subprocess

import pytest

import run_benchmarks as runner
from benchmarks.common import load


def prepare(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "estimate", lambda *a, **kw: {
        "seconds": 100., "low_seconds": 50., "high_seconds": 150.})


def test_shared_overrides_reach_suite_and_report_filters_seeds(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    commands = []

    def execute(cmd):
        commands.append(cmd)
        (tmp_path / "runs" / "smoke").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(runner, "execute", execute)
    assert runner.main(["--tier", "smoke", "--arms", "grpo", "empg", "--seeds", "2", "5",
                        "--", "train.iterations=1"]) == 0
    suite, report = commands
    assert suite[suite.index("--") + 1:] == ["train.iterations=1", "out_dir=runs/smoke"]
    assert report[report.index("--seeds") + 1:] == ["2", "5"]
    manifest = json.loads((tmp_path / "results/smoke/benchmark_run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["report_status"] == "complete"
    assert len(manifest["configs"]) == 4
    assert all(c["train"]["iterations"] == 1 for c in manifest["configs"])


def test_failure_still_generates_partial_report(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    commands = []
    (tmp_path / "runs/smoke").mkdir(parents=True)

    def execute(cmd):
        commands.append(cmd)
        if "benchmarks.suite" in cmd:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(runner, "execute", execute)
    with pytest.raises(subprocess.CalledProcessError):
        runner.main(["--tier", "smoke", "--arms", "grpo", "--seeds", "0"])
    assert "benchmarks.report" in commands[-1]
    manifest = json.loads((tmp_path / "results/smoke/benchmark_run.json").read_text())
    assert manifest["status"] == "interrupted_or_failed"


def test_all_existing_configs_checked_before_any_training(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    commands = []
    monkeypatch.setattr(runner, "execute", commands.append)
    cfg = load("empg", "smoke")
    d = tmp_path / cfg.run_dir
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"old": "configuration"}))
    with pytest.raises(ValueError, match="different configuration"):
        runner.main(["--tier", "smoke", "--arms", "grpo", "empg", "--seeds", "0", "--skip-base"])
    assert commands == []


def test_report_failure_is_not_marked_complete(monkeypatch, tmp_path):
    prepare(monkeypatch, tmp_path)
    (tmp_path / "runs/smoke").mkdir(parents=True)

    def execute(cmd):
        if "benchmarks.report" in cmd:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(runner, "execute", execute)
    with pytest.raises(subprocess.CalledProcessError):
        runner.main(["--tier", "smoke", "--arms", "grpo", "--seeds", "0"])
    manifest = json.loads((tmp_path / "results/smoke/benchmark_run.json").read_text())
    assert manifest["status"] == "report_failed"
    assert manifest["report_status"] == "failed"


def test_estimate_skips_completed_evaluations_and_writes_nothing(monkeypatch, tmp_path, capsys):
    from empg.config import config_fingerprint
    prepare(monkeypatch, tmp_path)
    cfg = load("grpo", "smoke", ["out_dir=runs/smoke"])
    d = tmp_path / cfg.run_dir
    d.mkdir(parents=True)
    (d / "eval_valid_unseen.json").write_text(json.dumps({"config_fingerprint": config_fingerprint(cfg)}))
    assert runner.main(["--tier", "smoke", "--arms", "grpo", "--seeds", "0",
                        "--skip-base", "--estimate-only"]) == 0
    assert "0 evaluations remaining" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


def test_windows_forwarding_keeps_spaces_and_overrides_as_arguments(monkeypatch):
    monkeypatch.setattr(runner, "ROOT", Path("C:/Project With Spaces/Harnessing_Uncertainty"))
    monkeypatch.setenv("EMPG_WSL_PYTHON", "/home/user/venvs/empg/bin/python")
    monkeypatch.setenv("EMPG_WSL_DISTRO", "Ubuntu-24.04")
    commands = []
    monkeypatch.setattr(runner.subprocess, "call", lambda cmd: commands.append(cmd) or 0)
    argv = ["--runs", "runs/a space", "--", "train.iterations=1"]
    assert runner.launch_in_wsl(argv) == 0
    cmd = commands[0]
    assert cmd[cmd.index("--cd") + 1] == "/mnt/c/Project With Spaces/Harnessing_Uncertainty"
    assert cmd[-len(argv):] == argv
