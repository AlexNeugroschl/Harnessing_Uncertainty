import dataclasses
import json

import numpy as np
import pytest

from benchmarks.common import load
from benchmarks.evaluate import select_tasks
from benchmarks.statistics import paired_comparison
from benchmarks.suite import _finished, _matching_eval
from empg.config import EvalConfig, config_fingerprint


class Types:
    def task_type(self, task):
        return task.split("_")[0]


def test_stratified_panel_is_fixed_and_covers_types():
    tasks = [f"{typ}_{i}" for typ, n in [("pick", 40), ("heat", 20), ("two", 3)]
             for i in range(n)]
    ec = EvalConfig(max_tasks=12, selection="stratified")
    a = select_tasks(tasks, Types(), ec)
    assert a == select_tasks(tasks, Types(), ec) and len(a) == len(set(a)) == 12
    assert {Types().task_type(t) for t in a} == {"pick", "heat", "two"}
    assert sum(t.startswith("two_") for t in a) == 1
    assert select_tasks(tasks, Types(), EvalConfig()) == tasks
    with pytest.raises(ValueError, match="at least one"):
        select_tasks(tasks, Types(), EvalConfig(max_tasks=2, selection="stratified"))


def test_legacy_random_panel_unchanged():
    tasks = [f"a_{i}" for i in range(30)]
    picks = sorted(np.random.default_rng(0).choice(30, 10, replace=False))
    assert select_tasks(tasks, Types(), EvalConfig(max_tasks=10)) == [tasks[i] for i in picks]


def panel(seed, outcomes):
    return dict(seed=seed, protocol={"max_steps": 50},
                per_task=[{"task": f"t{i}", "won": w} for i, w in enumerate(outcomes)])


def test_paired_interval_uses_pairing_and_reports_one_seed_limit():
    a = [panel(0, [0, 1, 0, 1])]
    b = [panel(0, [1, 1, 1, 1])]
    row = paired_comparison(a, b, bootstrap=1000)
    assert row["delta"] == 0.5 and row["discordant_seed_tasks"] == 2
    assert row["scope"] == "task variability only"
    assert not row["degenerate"]
    identity = paired_comparison(a, a, bootstrap=100)
    assert identity["ci95"] == [0.0, 0.0]
    assert identity["degenerate"]
    with pytest.raises(ValueError, match="paired seeds"):
        paired_comparison(a, [panel(1, [1, 1, 1, 1])])
    with pytest.raises(ValueError, match="task panel"):
        paired_comparison(a, [panel(0, [1])])
    mismatch = panel(0, [1, 1, 1, 1])
    mismatch["protocol"]["max_steps"] = 10
    with pytest.raises(ValueError, match="protocols"):
        paired_comparison(a, [mismatch])


def test_suite_rejects_stale_eval_and_training(tmp_path):
    cfg = load("empg", "pilot")
    cfg.out_dir = str(tmp_path)
    path = tmp_path / "eval.json"
    assert not _matching_eval(path, cfg)
    path.write_text(json.dumps({"config_fingerprint": config_fingerprint(cfg)}))
    assert _matching_eval(path, cfg)
    cfg.train.iterations += 1
    with pytest.raises(ValueError, match="configuration"):
        _matching_eval(path, cfg)
    cfg.run_dir.mkdir()
    old = dataclasses.asdict(cfg)
    old["train"]["max_steps"] = 2
    (cfg.run_dir / "config.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="different configuration"):
        _finished(cfg)


@pytest.mark.parametrize("tier", ["overnight", "paper_overnight"])
def test_budgeted_training_requires_completion_after_adapter_save(tmp_path, tier):
    cfg = load("grpo", tier)
    cfg.out_dir = str(tmp_path)
    cfg.run_dir.mkdir()
    (cfg.run_dir / "adapter").mkdir()
    (cfg.run_dir / "metrics.jsonl").write_text('{}\n' * cfg.train.iterations)
    assert not _finished(cfg)
    (cfg.run_dir / "training_complete.json").write_text(json.dumps({
        "config_fingerprint": config_fingerprint(cfg), "reason": "max_seconds"}))
    assert _finished(cfg)
