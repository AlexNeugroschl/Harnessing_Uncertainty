"""Fixed-panel probes using the loaded policy and reusable environment workers."""

import dataclasses
import csv
import json
import random
import time

import numpy as np

from benchmarks.evaluate import evaluate
from empg.envs.base import make_benchmark


class AdaptationProbes:
    def __init__(self, policy, cfg, log=print):
        nc = cfg.nonstationary
        self.policy, self.log = policy, log
        self.cfg = dataclasses.replace(cfg, eval=nc.probe_eval_config(cfg.eval))
        self.bench = make_benchmark(cfg.benchmark, **cfg.env)
        self.file = (cfg.run_dir / "adaptation_probes.jsonl").open("w")
        self.seconds = 0.0

    def run(self, iteration, **training_cost):
        start = time.perf_counter()
        # Evaluation must not perturb either legacy global RNG, including in
        # serial env mode. policy.sampling separately isolates torch/vLLM RNG.
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            # One wake/sleep (or HF merge/unmerge) for ALL regimes. Each evaluate
            # still resets its sampling stream, so regime order cannot alter it.
            with self.policy.sampling(self.cfg.nonstationary.probe_seed), self.policy.generation():
                for regime in self.cfg.nonstationary.probe_regimes:
                    self.log(f"  adaptation probe after update {iteration}: {regime}")
                    result, _ = evaluate(self.policy, self.cfg, self.log, regime=regime, benchmark=self.bench)
                    result.update(iteration=iteration, regime=regime, **training_cost)
                    self.file.write(json.dumps(result) + "\n")
                    self.file.flush()
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            self.seconds += time.perf_counter() - start

    def close(self):
        try:
            self.file.close()
        finally:
            if hasattr(self.bench, "close"):
                self.bench.close()


def summarize(rows, probes, nc):
    """Descriptive per-run adaptation, never treat missing recovery as zero.

    Recovery uses the latest pre-switch old-regime held-out success as its
    reference. It is first observed threshold crossing, not a convergence claim.
    """
    if nc.schedule == "recurring":
        events = []
        points = nc.change_points()
        for index, point in enumerate(points):
            end = points[index + 1] if index + 1 < len(points) else float("inf")
            event = dataclasses.replace(nc, schedule="switch", switch_iteration=point, switch_iterations=[],
                        initial_regime=nc.regime_at(point - 1), final_regime=nc.regime_at(point))
            events.append(summarize([r for r in rows if r["iteration"] < end],
                                    [p for p in probes if p["iteration"] < end], event))
        return {"schedule": "recurring", "change": nc.change, "events": events,
                "switch_iteration": nc.switch_iteration, "completed_iterations": rows[-1]["iteration"] if rows else 0,
                "shift_observed": any(e["shift_observed"] for e in events),
                "post_boundary_episodes": sum(r.get("episodes", 0) for r in rows if r["iteration"] >= nc.switch_iteration),
                "post_boundary_env_return": sum(r.get("env_return_sum", 0) for r in rows if r["iteration"] >= nc.switch_iteration),
                "recovery_updates": None,
                "adaptation_interpretable": bool(events) and all(e["adaptation_interpretable"] for e in events),
                "competence_status": "see per-change events", "probe_count": len(probes)}
    post = [r for r in rows if r["iteration"] >= nc.switch_iteration]
    result = {"schedule": nc.schedule, "change": nc.change,
              "switch_iteration": nc.switch_iteration,
              "shift_observed": nc.schedule == "switch" and bool(post),
              "completed_iterations": rows[-1]["iteration"] if rows else 0,
              "post_boundary_episodes": sum(r.get("episodes", 0) for r in post),
              "post_boundary_env_steps": sum(r.get("n_steps", 0) for r in post),
              "post_boundary_env_return": sum(r.get("env_return_sum", 0) for r in post),
              "recovery_target": None, "recovery_updates": None,
              "recovery_env_steps": None, "recovery_response_tokens": None,
              "recovery_training_seconds": None, "probe_count": len(probes)}
    result.update(adaptation_interpretable=False, competence_status="no pre-switch probe")
    if nc.schedule != "switch":
        result["competence_status"] = "not applicable (stationary)"
        return result
    before = [p for p in probes if p["iteration"] < nc.switch_iteration and p["regime"] == nc.initial_regime]
    after = [p for p in probes if p["iteration"] >= nc.switch_iteration and p["regime"] == nc.final_regime]
    if not before:
        return result
    reference = max(before, key=lambda p: p["iteration"])
    counterfactual = [p for p in probes if p["iteration"] == reference["iteration"] and p["regime"] == nc.final_regime]
    if counterfactual:
        result["immediate_shift_success"] = counterfactual[-1]["all"]
        result["immediate_shift_drop"] = reference["all"] - counterfactual[-1]["all"]
    result["pre_shift_success"] = reference["all"]
    successes = sum(bool(r["won"]) for r in reference["per_task"]) if "per_task" in reference else None
    result["pre_shift_successes"] = successes
    competent = (reference["all"] >= nc.competence_min_success and successes is not None and
                 successes >= nc.competence_min_successes)
    result["competence_status"] = ("established" if competent else "insufficient pre-switch success"
                                   if successes is not None else "missing per-task probe data")
    result["adaptation_interpretable"] = competent and bool(after) and result["shift_observed"]
    if after:
        result["post_shift_success"] = after[-1]["all"]
    if reference["all"] <= 0:
        return result
    target = reference["all"] * nc.recovery_fraction
    result["recovery_target"] = target
    result["heldout_success_shortfall"] = sum(max(0., target - p["all"]) for p in after)
    recovered = next((p for p in after if p["all"] >= target), None)
    boundary = next((r for r in rows if r["iteration"] == nc.switch_iteration - 1), {})
    if recovered:
        result["recovery_updates"] = recovered["iteration"] - nc.switch_iteration + 1
        for label, key in (("env_steps", "env_steps_total"), ("response_tokens", "response_tokens_total"),
                           ("training_seconds", "seconds_total")):
            if key in recovered and key in boundary:
                result[f"recovery_{label}"] = recovered[key] - boundary[key]
    return result


def write_report(runs, out, seeds=None, arms=None):
    from benchmarks.report import read_jsonl, run_dirs
    from empg.config import Config, _build
    summaries, panels = [], []
    for directory, arm in run_dirs(runs, seeds, arms):
        config_file = directory / "config.json"
        if arm == "base" or not config_file.exists():
            continue
        cfg = _build(Config, json.loads(config_file.read_text()))
        if not cfg.nonstationary.enabled:
            continue
        rows = read_jsonl(directory / "metrics.jsonl")
        probes = read_jsonl(directory / "adaptation_probes.jsonl")
        summary = summarize(rows, probes, cfg.nonstationary)
        summary.update(method=arm, seed=cfg.seed, curiosity_weight=cfg.advantage.curiosity.weight,
                       probe_seconds_total=rows[-1].get("probe_seconds_total", 0.) if rows else 0.)
        summaries.append(summary)
        panels.append((f"{arm} s{cfg.seed}", probes, cfg.nonstationary))
    if not summaries:
        return ""
    (out / "adaptation_summary.json").write_text(json.dumps(summaries, indent=2))
    fields = list(dict.fromkeys(k for r in summaries for k in r))
    with (out / "adaptation_runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(summaries)
    text = ("## Adaptation to environment changes\n\n"
            "Extrinsic returns exclude curiosity and formatting penalties. Post-boundary totals use the "
            "configured switch boundary even in stationary controls. Recovery is the first held-out probe "
            "reaching the configured fraction of pre-switch success in the old regime. "
            "Missing recovery, disabled probes, and a zero pre-switch reference are reported as unavailable. "
            "Shortfall is a sum over observed probes, not oracle regret.\n\n"
            "Pre-switch competence must meet both the configured success-rate and successful-task thresholds. "
            "Insufficient competence makes adaptation conclusions inconclusive, even if a descriptive recovery crossing exists.\n\n"
            "| Method | Seed | Schedule | Shift reached | Post-boundary episodes | Extrinsic return | Recovery updates | Competence |\n"
            "|---|---:|---|---|---:|---:|---:|---|\n")
    for r in summaries:
        recovery = r["recovery_updates"] if r["recovery_updates"] is not None else "unavailable"
        text += (f"| {r['method']} | {r['seed']} | {r['schedule']} | {r['shift_observed']} | "
                 f"{r['post_boundary_episodes']} | {r['post_boundary_env_return']:.3f} | {recovery} | {r['competence_status']} |\n")
    (out / "adaptation.md").write_text(text)
    if any(p for _, p, _ in panels):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, probes, nc in panels:
            for regime in nc.probe_regimes:
                values = [p for p in probes if p["regime"] == regime]
                if values:
                    ax.plot([p["env_steps_total"] for p in values], [p["all"] for p in values],
                            marker=".", linestyle="--" if regime == "baseline" else "-",
                            label=f"{label}, {regime}")
        ax.set(xlabel="Training environment steps (probes excluded)", ylabel="Held-out success", ylim=(-.02, 1.02))
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(out / "adaptation_curves.png", dpi=150)
        plt.close(fig)
    return text
