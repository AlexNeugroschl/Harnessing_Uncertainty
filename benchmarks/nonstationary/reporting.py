"""Paired effects, competence checks and learning curves."""

import json
from pathlib import Path
from .artifacts import write_json as _write, read_jsonl


def report(jobs, out):
    from benchmarks.adaptation import summarize
    from benchmarks.evaluate import eval_path
    from benchmarks.statistics import paired_comparison
    from empg.config import config_matches
    from benchmarks.suite import _finished, _matching_eval
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    completed, records, panels = {}, [], {}
    for job in jobs:
        cfg = jobs[job["alias_of"]]["cfg"] if "alias_of" in job else job["cfg"]
        identity = {k: job[k] for k in ("depth", "novelty", "schedule", "seed", "treatment")}
        saved = cfg.run_dir / "config.json"
        if saved.exists() and not config_matches(json.loads(saved.read_text()), cfg):
            raise ValueError(f"configuration mismatch: {saved}")
        rows = read_jsonl(cfg.run_dir / "metrics.jsonl")
        probes = read_jsonl(cfg.run_dir / "adaptation_probes.jsonl")
        record = {**identity, "adaptation": summarize(rows, probes, cfg.nonstationary),
                  "optional_reserve": bool(job.get("optional_reserve", False)),
                  "updates": len(rows), "complete": False,
                  "last_curiosity_weight": rows[-1].get("curiosity_weight") if rows else None,
                  "detector_triggers": sum(bool(r.get("curiosity_detector_triggered")) for r in rows),
                  "env_steps": sum(r.get("n_steps", 0) for r in rows),
                  "response_tokens": rows[-1].get("response_tokens_total", 0) if rows else 0,
                  "training_seconds": rows[-1].get("seconds_total", 0) if rows else 0,
                  "peak_vram_gib": max((r.get("gpu_peak_gb", 0) for r in rows), default=0),
                  "mean_truncation": sum(r.get("trunc_rate", 0) for r in rows) / max(1, len(rows))}
        if "alias_of" in job:
            record["reused_from"] = str(cfg.run_dir)
        # Final adapter/evaluation completion is required to enter comparisons.
        path = eval_path(cfg)
        if _finished(cfg) and _matching_eval(path, cfg):
            result = json.loads(path.read_text())
            cohort = "reserve" if job.get("optional_reserve") else "primary"
            key = (cohort, job["depth"], job["novelty"], job["schedule"], job["treatment"])
            completed.setdefault(key, []).append(result)
            record.update(complete=True, success=result["all"])
        records.append(record)
        panels[str(cfg.run_dir)] = {"metrics": rows, "probes": probes}
    comparisons = []
    for (cohort, depth, novelty, schedule, candidate), right in completed.items():
        if candidate in ("none", "constant"):
            continue
        controls = ["constant"]
        if candidate.startswith("oracle_"):
            suffix = candidate.removeprefix("oracle")
            controls += ["high" + suffix, "matched" + suffix]
        for control in controls:
            left = completed.get((cohort, depth, novelty, schedule, control), [])
            common = {r["seed"] for r in left} & {r["seed"] for r in right}
            if common:
                comparison = paired_comparison([r for r in left if r["seed"] in common],
                                               [r for r in right if r["seed"] in common])
                comparisons.append({"cohort": cohort, "depth": depth, "novelty": novelty, "schedule": schedule,
                                    "control": control, "candidate": candidate, **comparison})
    # Also report whether introducing curiosity helps at all.
    for (cohort, depth, novelty, schedule, candidate), right in completed.items():
        if candidate != "constant":
            continue
        left = completed.get((cohort, depth, novelty, schedule, "none"), [])
        common = {r["seed"] for r in left} & {r["seed"] for r in right}
        if common:
            comparisons.append({"cohort": cohort, "depth": depth, "novelty": novelty, "schedule": schedule,
                "control": "none", "candidate": candidate,
                **paired_comparison([r for r in left if r["seed"] in common], [r for r in right if r["seed"] in common])})
    # Does the treatment help more when rules actually change? Preserve the
    # seed/task pairing across all four cells; numeric rows below are effects,
    # not fabricated binary outcomes or additional observations.
    for (cohort, depth, novelty, schedule, candidate), switched_right in completed.items():
        if schedule != "switch" or candidate in ("none", "constant"):
            continue
        cells = [completed.get((cohort, depth, novelty, s, t), []) for s, t in
                 (("switch", "constant"), ("switch", candidate),
                  ("stationary", "constant"), ("stationary", candidate))]
        common = set.intersection(*({r["seed"] for r in cell} for cell in cells))
        controls, effects = [], []
        for seed in sorted(common):
            rows = [next(r for r in cell if r["seed"] == seed) for cell in cells]
            for left, right in ((rows[0], rows[1]), (rows[2], rows[3]), (rows[0], rows[2])):
                paired_comparison([left], [right], bootstrap=1)
            values = [{r["task"]: float(r["won"]) for r in cell["per_task"]} for cell in rows]
            metadata = {k: rows[0][k] for k in ("seed", "protocol")}
            controls.append({**metadata, "per_task": [{"task": t, "won": 0.} for t in values[0]]})
            effects.append({**metadata, "per_task": [{"task": t,
                "won": values[1][t] - values[0][t] - values[3][t] + values[2][t]} for t in values[0]]})
        if controls:
            comparisons.append({"cohort": cohort, "depth": depth, "novelty": novelty, "schedule": "switch-minus-stationary",
                "control": "constant", "candidate": candidate, "effect_type": "paired interaction",
                **paired_comparison(controls, effects)})
    payload = {"runs": records, "comparisons": comparisons}
    _write(out / "summary.json", payload)
    _write(out / "learning_curves.json", panels)
    text = ("# Curiosity after environment changes\n\n"
            "Custom text crafting suite on pinned NovelGridWorldsV2; these are not published NovelGym scores. "
            "Oracle boosts use privileged change timing; detected boosts use prior observed action failures. "
            "Matched controls match planned coefficient exposure, not realized intrinsic rewards. "
            "Only completed matched seed/task pairs enter comparisons. One-seed intervals describe task variability only. "
            "A degenerate interval is not evidence of equivalence. "
            "Time-admitted reserve replications are reported separately from the fixed primary study.\n\n"
            "| Cohort | Depth | Change | Schedule | Seed | Treatment | Success | Competence | Steps | Seconds |\n"
            "|---|---|---|---|---:|---|---:|---|---:|---:|\n")
    for row in records:
        success = f"{100 * row['success']:.1f}%" if row["complete"] else "unavailable"
        cohort = "reserve" if row["optional_reserve"] else "primary"
        text += (f"| {cohort} | {row['depth']} | {row['novelty']} | {row['schedule']} | {row['seed']} | {row['treatment']} | "
                 f"{success} | {row['adaptation']['competence_status']} | {row['env_steps']} | {row['training_seconds']:.0f} |\n")
    text += ("\n| Cohort | Depth | Change | Schedule | Candidate − control | Difference | 95% interval | Paired seeds |\n"
             "|---|---|---|---|---|---:|---|---|\n")
    for row in comparisons:
        lo, hi = row["ci95"]
        text += (f"| {row['cohort']} | {row['depth']} | {row['novelty']} | {row['schedule']} | {row['candidate']} − {row['control']} | "
                 f"{100 * row['delta']:+.1f} pp | [{100 * lo:+.1f}, {100 * hi:+.1f}] | {row['seeds']} |\n")
    text += "\nRecovery metrics and per-change competence are in summary.json; update/probe trajectories are in learning_curves.json.\n"
    (out / "comparison.md").write_text(text, encoding="utf-8")
    plot_curves(jobs, panels, out)
    return payload


def plot_curves(jobs, panels, out):
    if not any(p["metrics"] for p in panels.values()):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def group(job):
        return (job["depth"], job["novelty"], job["schedule"],
                f"reserve_s{job['seed']}" if job.get("optional_reserve") else "primary")
    groups = sorted({group(j) for j in jobs})
    for depth, novelty, schedule, cohort in groups:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for job in jobs:
            if group(job) != (depth, novelty, schedule, cohort) or "alias_of" in job:
                continue
            data = panels[str(job["cfg"].run_dir)]
            rows = data["metrics"]
            label = f"{job['treatment']} s{job['seed']}"
            x = [r["env_steps_total"] for r in rows]
            for axis, key, title in ((axes[0, 0], "success", "Training success"),
                                      (axes[1, 0], "curiosity_weight", "Effective curiosity coefficient"),
                                      (axes[1, 1], "action_failure_rate", "Observed action failure rate")):
                axis.plot(x, [r.get(key, float("nan")) for r in rows], label=label)
                axis.set(title=title, xlabel="Training environment steps")
            for regime in ("baseline", "novelty"):
                probe = [p for p in data["probes"] if p["regime"] == regime]
                axes[0, 1].plot([p["env_steps_total"] for p in probe], [p["all"] for p in probe],
                                marker=".", linestyle="--" if regime == "baseline" else "-",
                                label=f"{label} {regime}")
            axes[0, 1].set(title="Held-out adaptation probes", xlabel="Training environment steps")
        for axis in axes.flat:
            if axis.lines:
                axis.legend(fontsize=6)
            axis.grid(alpha=.2)
        fig.suptitle(f"Depth {depth} / {novelty} / {schedule} / {cohort}")
        fig.tight_layout()
        suffix = "" if cohort == "primary" else "_" + cohort
        fig.savefig(out / f"curves_depth_{depth}_{novelty}_{schedule}{suffix}.png", dpi=150)
        plt.close(fig)
