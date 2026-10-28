import json
import dataclasses
import pytest

from benchmarks.report import (comparisons, curiosity_comparisons, explore_table, load,
                               load_lengths, read_jsonl, table, write_run_comparison)
from benchmarks.report import validate_combined_runs
from benchmarks.common import load as load_config


def combined_seed(root, seed):
    directory = root / f"grpo_s{seed}"
    directory.mkdir(parents=True)
    cfg = load_config("grpo", "paper_overnight", seed=seed)
    ev = {"seed": seed, "all": .5, "protocol": {"temperature": .4},
          "per_task": [{"task": "task", "won": True}]}
    (directory / "config.json").write_text(json.dumps(dataclasses.asdict(cfg)))
    (directory / "eval_valid_unseen.json").write_text(json.dumps(ev))
    (directory / "metrics.jsonl").write_text('{"iteration": 12}\n')
    return directory, cfg


def test_existing_seeds_combine_without_retraining_and_keep_sources(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    directory, cfg = combined_seed(old, 0)
    directory, cfg = combined_seed(new, 1)
    cfg.policy.micro_batch_size, cfg.policy.score_batch_size = 4, 8
    cfg.policy.head_backend = "analytic"
    (directory / "config.json").write_text(json.dumps(dataclasses.asdict(cfg)))
    roots = [new, old]
    sources = validate_combined_runs(roots, "valid_unseen", planned=[cfg])
    assert set(sources) == {"grpo_s0", "grpo_s1"}
    evals, curves = load(roots, "valid_unseen", [0, 1])
    assert [e["seed"] for e in evals["grpo"]] == [1, 0]
    assert len(curves["grpo"]) == 2
    out = tmp_path / "out"
    out.mkdir()
    write_run_comparison(roots, "valid_unseen", out, ["grpo"], [0, 1, 2])
    assert "| grpo | 2 | not started |" in (out / "runs.md").read_text()


@pytest.mark.parametrize("change", ["budget", "lr", "panel", "protocol", "duplicate", "seed", "fingerprint"])
def test_combining_rejects_mismatched_experiments_and_duplicate_seeds(tmp_path, change):
    old, new = tmp_path / "old", tmp_path / "new"
    combined_seed(old, 0)
    directory, cfg = combined_seed(new, 0 if change == "duplicate" else 1)
    if change in ("budget", "lr"):
        if change == "budget":
            cfg.train.iterations += 1
        else:
            cfg.train.lr *= 10
        (directory / "config.json").write_text(json.dumps(dataclasses.asdict(cfg)))
    elif change in ("panel", "protocol", "seed", "fingerprint"):
        path = directory / "eval_valid_unseen.json"
        ev = json.loads(path.read_text())
        if change == "panel":
            ev["per_task"][0]["task"] = "different task"
        elif change == "protocol":
            ev["protocol"]["temperature"] = .8
        elif change == "seed":
            ev["seed"] = 0
        else:
            ev["config_fingerprint"] = "wrong"
        path.write_text(json.dumps(ev))
    with pytest.raises(ValueError, match="Different|Duplicate|seed|fingerprint"):
        validate_combined_runs([new, old], "valid_unseen")


def test_reuse_only_zero_shot_for_a_deeper_budget_requires_matching_eval_protocol(tmp_path, monkeypatch):
    from benchmarks.evaluate import evaluation_protocol
    from benchmarks import evaluate as evaluation
    from types import SimpleNamespace
    monkeypatch.setattr(evaluation, "make_benchmark", lambda *a, **kw: SimpleNamespace(tasks=lambda split: ["task"]))
    cfg = load_config("grpo", "paper_optimized")
    directory = tmp_path / "old/base"
    directory.mkdir(parents=True)
    (directory / "eval_valid_unseen.json").write_text(json.dumps({
        "protocol": evaluation_protocol(cfg), "all": .06,
        "per_task": [{"task": "task", "won": False}]}))
    sources = validate_combined_runs([tmp_path / "new", directory], "valid_unseen", planned=[cfg])
    assert set(sources) == {"base"}
    evals, _ = load([tmp_path / "new", directory], "valid_unseen")
    assert len(evals["base"]) == 1
    cfg.train.iterations = 100
    validate_combined_runs([directory], "valid_unseen", planned=[cfg])
    cfg.policy.model_name = "different-model"
    with pytest.raises(ValueError, match="Different evaluation protocol"):
        validate_combined_runs([directory], "valid_unseen", planned=[cfg])


def test_ablation_table_has_the_same_number_of_columns_in_every_row():
    text = table({"grpo": [{"all": 0.5}]}, [("grpo", "GRPO")], [])
    assert len({line.count("|") for line in text.splitlines()}) == 1


def test_all_failed_exploration_and_degenerate_comparison():
    text = explore_table({"grpo": [[{"success_len": float("nan"), "success": 0}]]})
    assert "GRPO" in text
    run = {"seed": 0, "per_task": [{"task": "t", "won": False}]}
    text, results = comparisons({"grpo": [run], "empg": [run]})
    assert results["empg"]["degenerate"]
    assert "(degenerate)" in text


def test_seed_filter_applies_to_scores_curves_lengths_and_csv(tmp_path):
    for seed in [0, 1]:
        d = tmp_path / f"grpo_s{seed}"
        d.mkdir()
        ev = {"seed": seed, "all": float(seed), "per_task": [{"task": "t", "won": bool(seed)}]}
        (d / "eval_valid_unseen.json").write_text(json.dumps(ev))
        (d / "metrics.jsonl").write_text(json.dumps({"iteration": 1, "seconds_total": 3600}) + "\n")
        lengths = {"cap": 96, "len": [10 + seed], "truncated": [False], "valid": [True]}
        (d / "resp_lengths_train.jsonl").write_text(json.dumps(lengths) + "\n")
    evals, curves = load(tmp_path, "valid_unseen", seeds=[1])
    assert [r["seed"] for r in evals["grpo"]] == [1]
    assert len(curves["grpo"]) == 1
    assert load_lengths(tmp_path, "valid_unseen", seeds=[1])["grpo"]["train"]["len"] == [11]
    out = tmp_path / "report"
    out.mkdir()
    text = write_run_comparison(tmp_path, "valid_unseen", out, ["grpo"], [1])
    assert "| grpo | 1 | evaluated | 100.0 | 1 | 1 |  | 1.00 |" in text
    assert len((out / "runs.csv").read_text().splitlines()) == 2


def test_partial_run_is_missing_success_not_zero(tmp_path):
    d = tmp_path / "empg_s0"
    d.mkdir()
    (d / "metrics.jsonl").write_text(json.dumps({"iteration": 2}) + "\n")
    out = tmp_path / "report"
    out.mkdir()
    text = write_run_comparison(tmp_path, "valid_unseen", out)
    assert "| empg | 0 | not evaluated | — |" in text


def test_interrupted_last_metric_preserves_earlier_iterations(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"iteration": 1}\n{"iteration":')
    assert read_jsonl(path) == [{"iteration": 1}]
    path.write_text('{"iteration": 1}\n{"iteration":\n')
    import pytest
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_curiosity_uses_matching_algorithm_baseline():
    def run(won):
        return [{"seed": 0, "per_task": [{"task": "t", "won": won}]}]
    text, comparisons = curiosity_comparisons({"grpo": run(False), "empg": run(True),
        "grpo_cur_group": run(True), "empg_cur_episodic": run(True)})
    assert comparisons["grpo_cur_group vs grpo"]["delta"] == 1
    assert comparisons["empg_cur_episodic vs empg"]["delta"] == 0
    assert "empg_cur_episodic vs empg" in text


def test_unstarted_methods_are_explicit_and_have_no_score(tmp_path):
    out = tmp_path / "report"
    out.mkdir()
    text = write_run_comparison(tmp_path, "valid_seen", out, ["grpo", "empg"], [0])
    assert "| grpo | 0 | not started | — |" in text
    assert "| empg | 0 | not started | — |" in text
