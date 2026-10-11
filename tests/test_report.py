import json

from benchmarks.report import comparisons, explore_table, load, load_lengths, table, write_run_comparison


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
