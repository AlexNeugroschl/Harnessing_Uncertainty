"""Development-only learnability gates and content-verified shared settings."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .artifacts import read_jsonl


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate(initial, final, *, minimum_success=.5, minimum_gain=.125, minimum_tasks=8):
    """Same development tasks and sampling protocol before/after learning.

    This is an engineering gate, not a significance test or confirmation result.
    """
    if not 0 <= minimum_success <= 1 or not 0 < minimum_gain <= 1 or minimum_tasks < 1:
        raise ValueError("invalid learning gate thresholds")
    if initial["protocol"] != final["protocol"] or initial["protocol"].get("split") != "valid_seen":
        raise ValueError("learning gate requires identical evaluation protocols")
    before = {row["task"]: bool(row["won"]) for row in initial["per_task"]}
    after = {row["task"]: bool(row["won"]) for row in final["per_task"]}
    if (len(before) != len(initial["per_task"]) or len(after) != len(final["per_task"])
            or set(before) != set(after) or len(before) < minimum_tasks):
        raise ValueError("learning gate requires the same nonempty development task panel")
    start, finish = sum(before.values()) / len(before), sum(after.values()) / len(after)
    return {"passed": finish >= minimum_success and finish - start >= minimum_gain,
            "initial_success": start, "final_success": finish, "gain": finish - start,
            "tasks": len(before), "minimum_success": minimum_success, "minimum_gain": minimum_gain,
            "note": "development learnability screen; not a statistical claim"}


def inspect_run(cfg):
    probes = read_jsonl(cfg.run_dir / "adaptation_probes.jsonl")
    metrics = read_jsonl(cfg.run_dir / "metrics.jsonl")
    if [row["iteration"] for row in metrics] != list(range(1, cfg.train.iterations + 1)):
        raise ValueError("learning pilot did not complete the fixed training budget")
    def panel(iteration, regime):
        matches = [p for p in probes if p["iteration"] == iteration and p["regime"] == regime]
        if len(matches) != 1:
            raise ValueError("learning evidence requires one complete panel per checkpoint and regime")
        return matches[0]
    initial = panel(cfg.nonstationary.switch_iteration - 1, "novelty")
    final = panel(cfg.train.iterations, "novelty")
    result = gate(initial, final)
    old = panel(cfg.train.iterations, "baseline")["all"]
    pre = panel(cfg.nonstationary.switch_iteration - 1, "baseline")["all"]
    result.update(old_rule_retention=old, pre_change_success=pre,
                  initial_checkpoint_success=panel(0, "novelty")["all"],
                  successful_training_episodes=sum(round(r["success"] * r["episodes"]) for r in metrics
                                                   if r["regime"] == "novelty"))
    result["passed"] &= old >= .75 and pre >= .75 and result["successful_training_episodes"] > 0
    return result


def load_learning_profile(path, adapter):
    from empg.config import adapter_fingerprint
    data = json.loads(Path(path).read_text())
    if (data.get("schema") != 1 or not data.get("complete") or not data.get("passed")
            or data.get("initial_adapter_sha256") != adapter_fingerprint(adapter)):
        raise ValueError("a complete passing learning profile for this baseline checkpoint is required")
    settings = data.get("settings", {})
    if (set(settings) != {"train.lr", "train.entropy_coef", "advantage.curiosity.key"}
            or not math.isfinite(settings["train.lr"]) or settings["train.lr"] <= 0
            or not math.isfinite(settings["train.entropy_coef"]) or settings["train.entropy_coef"] < 0
            or settings["advantage.curiosity.key"] not in ("parsed", "progress")):
        raise ValueError("invalid shared learning settings")
    selected = [row for row in data["runs"] if row["candidate"] == data["selected"]]
    seeds = {row["seed"] for row in selected}
    if len(seeds) < 2 or len(selected) != len(seeds):
        raise ValueError("the learning gate requires at least two development training seeds")
    from empg.config import Config, _build
    for row in selected:
        for filename, expected in row["artifacts"].items():
            if file_sha256(filename) != expected:
                raise ValueError("learning evidence changed after the pilot")
        cfg = _build(Config, row["config"])
        if cfg.train.initial_adapter_sha256 != data["initial_adapter_sha256"]:
            raise ValueError("learning runs must start from the shared baseline-only checkpoint")
        if cfg.policy.temperature != 1. or cfg.policy.top_p != 1.:
            raise ValueError("learning evidence must use the scored on-policy training distribution")
        if not inspect_run(cfg)["passed"]:
            raise ValueError("saved development trajectories do not pass the learning gate")
        if cfg.env.get("suite_version") != 4 or cfg.nonstationary.probe_split != "valid_seen":
            raise ValueError("learning evidence must use suite-v4 development tasks")
        for key, value in settings.items():
            section, name = key.rsplit(".", 1)
            node = cfg.advantage.curiosity if section == "advantage.curiosity" else cfg.train
            if getattr(node, name) != value:
                raise ValueError("shared settings differ from the checked learning pilot")
    return data
