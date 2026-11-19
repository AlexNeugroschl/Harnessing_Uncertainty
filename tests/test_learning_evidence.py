import dataclasses
import json
from pathlib import Path

import pytest

from benchmarks.common import load
from benchmarks.nonstationary.artifacts import write_json
from benchmarks.nonstationary.evidence import file_sha256, inspect_run, load_learning_profile
from empg.config import adapter_fingerprint


@pytest.fixture
def evidence(tmp_path):
    adapter = tmp_path / "initial"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"baseline")
    records = []
    settings = {"train.lr": 3e-6, "train.entropy_coef": .01, "advantage.curiosity.key": "progress"}
    for seed in (101, 102):
        cfg = load("grpo", "novelgym_recovery", [f"train.initial_adapter={adapter}", f"out_dir={tmp_path}",
                                                   f"train.lr={settings['train.lr']}"], seed)
        cfg.run_dir.mkdir()
        metrics = [{"iteration": i, "regime": "baseline" if i <= 4 else "novelty", "success": .5, "episodes": 16}
                   for i in range(1, 17)]
        (cfg.run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
        probes = []
        for iteration in (0, 4, 16):
            for regime in ("baseline", "novelty"):
                wins = 8 if regime == "baseline" else 6 if iteration == 16 else 0
                probes.append({"iteration": iteration, "regime": regime, "all": wins / 8,
                               "protocol": {"split": "valid_seen", "regime": regime},
                               "per_task": [{"task": f"dev:{i}", "won": i < wins} for i in range(8)]})
        (cfg.run_dir / "adaptation_probes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in probes))
        artifacts = {str(cfg.run_dir / name): file_sha256(cfg.run_dir / name)
                     for name in ("metrics.jsonl", "adaptation_probes.jsonl")}
        records.append({"candidate": "chosen", "seed": seed, "config": dataclasses.asdict(cfg), "artifacts": artifacts})
    profile = {"schema": 1, "complete": True, "passed": True, "selected": "chosen", "settings": settings,
               "initial_adapter_sha256": adapter_fingerprint(adapter), "runs": records}
    path = tmp_path / "learning_profile.json"
    write_json(path, profile)
    return path, adapter, profile


def test_learning_profile_rechecks_recorded_development_outcomes(evidence):
    path, adapter, expected = evidence
    assert load_learning_profile(path, adapter)["settings"] == expected["settings"]


@pytest.mark.parametrize("mutation", ["incomplete", "one_seed", "different_checkpoint", "different_settings", "changed_artifact"])
def test_invalid_learning_evidence_cannot_unlock_a_long_run(evidence, mutation):
    path, adapter, profile = evidence
    if mutation == "incomplete":
        profile["complete"] = False
    elif mutation == "one_seed":
        profile["runs"].pop()
    elif mutation == "different_checkpoint":
        profile["initial_adapter_sha256"] = "wrong"
    elif mutation == "different_settings":
        profile["settings"]["train.lr"] = 1e-5
    else:
        Path(next(iter(profile["runs"][0]["artifacts"]))).write_text("changed")
    write_json(path, profile)
    with pytest.raises(ValueError):
        load_learning_profile(path, adapter)


def test_boolean_pass_flag_cannot_replace_actual_learning(evidence):
    path, adapter, profile = evidence
    for record in profile["runs"]:
        probes = Path(record["config"]["out_dir"]) / f"{record['config']['name']}_s{record['seed']}" / "adaptation_probes.jsonl"
        rows = [json.loads(line) for line in probes.read_text().splitlines()]
        for row in rows:
            if row["regime"] == "novelty":
                row["all"] = 0.
                for task in row["per_task"]:
                    task["won"] = False
        probes.write_text("".join(json.dumps(row) + "\n" for row in rows))
        record["artifacts"][str(probes)] = file_sha256(probes)
    write_json(path, profile)
    with pytest.raises(ValueError, match="do not pass"):
        load_learning_profile(path, adapter)
