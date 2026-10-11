from benchmarks.report import comparisons, explore_table, table


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
