"""Curiosity x stationary/switch experiment, using existing optimized training."""

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_benchmarks import ROOT, duration, estimate, launch_in_wsl


def matrix_effects(root, backbone, seeds, curiosity=None, schedules=("stationary", "switch")):
    """Matched seed effects on the SAME final-regime held-out task panel."""
    effects = []
    curiosity = curiosity or f"{backbone}_cur_episodic"
    def delta(a, b):
        left = {r["task"]: float(r["won"]) for r in a["per_task"]}
        right = {r["task"]: float(r["won"]) for r in b["per_task"]}
        if (a["seed"] != b["seed"] or a.get("protocol") != b.get("protocol") or
                not left or left.keys() != right.keys() or len(left) != len(a["per_task"]) or len(right) != len(b["per_task"])):
            raise ValueError("matrix comparisons require matching seeds, protocols and unique task panels")
        return sum(right[t] - left[t] for t in left) / len(left)
    for seed in seeds:
        deltas, references = {}, {}
        for schedule in schedules:
            directory = root / schedule
            paths = []
            for arm in (backbone, curiosity):
                paths.append(sorted((directory / f"{arm}_s{seed}").glob("eval_*.json")))
            if any(len(p) != 1 for p in paths):
                continue
            a, b = (json.loads(p[0].read_text()) for p in paths)
            deltas[schedule] = delta(a, b)
            references[schedule] = a
        if references:
            reference = next(iter(references.values()))
            for candidate in references.values():
                delta(reference, candidate)  # verify cross-schedule comparability
        effects.append({"seed": seed, "curiosity_effect": deltas,
                        "switch_minus_stationary_effect": deltas["switch"] - deltas["stationary"]
                        if "switch" in deltas and "stationary" in deltas else None})
    return effects


def job_overrides(schedule, shared, template):
    overrides = [*shared, f"nonstationary.schedule={'switch' if schedule == 'switch' else 'stationary'}"]
    if schedule in ("stationary_initial", "stationary_final"):
        regime = template.nonstationary.initial_regime if schedule.endswith("initial") else template.nonstationary.final_regime
        overrides.append(f"nonstationary.stationary_regime={regime}")
    return overrides


def fit_budget(configs, hours, fraction=.9, max_iterations=200):
    """Choose one even iteration budget for ALL arms, without model loading.

    Only explicitly requested budget fitting changes iteration/switch/probe
    counts. Full-workload cost is used so resumes never expand an experiment.
    """
    if hours <= 0 or not 0 < fraction <= 1 or max_iterations < 6:
        raise ValueError("budget fitting needs positive hours, a fraction in (0, 1], and at least six iterations")
    target = hours * 3600 * fraction
    report_seconds = 60 + 30 * len({str(c.out_dir) for c in configs})
    def cost(iterations):
        trial = [dataclasses.replace(c, train=dataclasses.replace(c.train, iterations=iterations),
                 nonstationary=dataclasses.replace(c.nonstationary, switch_iteration=iterations // 2 + 1,
                                                   probe_every=iterations // 2 if c.nonstationary.probe_every else 0))
                 for c in configs]
        return sum(estimate(c)["seconds"] for c in trial) + report_seconds
    if cost(6) > target:
        raise ValueError("the minimum six-update comparison exceeds the fitting target; reduce seeds/conditions/panels or disable fitting")
    low, high, best = 3, max_iterations // 2, 3
    while low <= high:
        middle = (low + high) // 2
        if cost(2 * middle) <= target:
            best, low = middle, middle + 1
        else:
            high = middle - 1
    return 2 * best


def plan_jobs(args, selected, shared, runs):
    from benchmarks.common import load
    template = load(selected[0], args.tier, shared)
    planned = []
    # Complete on/off pairs within a seed before starting another condition.
    # Every job gets its own time allocation, including the last condition.
    for seed in args.seeds:
        for schedule in args.schedules:
            for arm in selected:
                overrides = [*job_overrides(schedule, shared, template), f"out_dir={runs / schedule}"]
                cfg = load(arm, args.tier, overrides, seed)
                planned.append({"schedule": schedule, "arm": arm, "seed": seed, "cfg": cfg,
                                "overrides": overrides})
    if args.fit_budget:
        iterations = fit_budget([p["cfg"] for p in planned], args.hours, args.budget_fraction, args.max_iterations)
        for job in planned:
            cfg = job["cfg"]
            job["overrides"] += [f"train.iterations={iterations}", f"nonstationary.switch_iteration={iterations // 2 + 1}",
                                 f"nonstationary.probe_every={iterations // 2 if cfg.nonstationary.probe_every else 0}"]
            job["cfg"] = load(job["arm"], args.tier, job["overrides"], job["seed"])
    return planned


def evaluation_pair(root, schedule, backbone, curiosity, seed):
    paths = [sorted((root / schedule / f"{arm}_s{seed}").glob("eval_*.json")) for arm in (backbone, curiosity)]
    return [json.loads(p[0].read_text()) for p in paths] if all(len(p) == 1 for p in paths) else None


def matrix_statistics(root, backbone, curiosity, seeds, schedules):
    """Crossed paired seed/task intervals; no invented scores for missing runs."""
    from benchmarks.statistics import paired_comparison
    pairs = {schedule: {seed: pair for seed in seeds if (pair := evaluation_pair(
             root, schedule, backbone, curiosity, seed)) is not None} for schedule in schedules}
    statistics = {}
    for schedule, values in pairs.items():
        if values:
            statistics[schedule] = paired_comparison([p[0] for p in values.values()], [p[1] for p in values.values()])
    if "switch" in pairs and "stationary" in pairs:
        controls, effects = [], []
        for seed in sorted(pairs["switch"].keys() & pairs["stationary"].keys()):
            stationary, switched = pairs["stationary"][seed], pairs["switch"][seed]
            # Reuse the existing paired bootstrap on task-level effect
            # differences. These synthetic numeric rows are effects, not wins.
            for left, right in (stationary, switched, (stationary[0], switched[0])):
                paired_comparison([left], [right], bootstrap=1)  # validate metadata/panels
            values = [{r["task"]: float(r["won"]) for r in p["per_task"]} for p in (*stationary, *switched)]
            metadata = {k: switched[0][k] for k in ("seed", "protocol")}
            tasks = sorted(values[0])
            controls.append({**metadata, "per_task": [{"task": t, "won": 0.} for t in tasks]})
            effects.append({**metadata, "per_task": [{"task": t, "won": values[3][t] - values[2][t] - values[1][t] + values[0][t]}
                                                       for t in tasks]})
        if controls:
            statistics["switch_minus_stationary_effect"] = paired_comparison(controls, effects)
    return statistics


def execute_phase(command, timeout=None):
    from benchmarks.process import run_process
    print("$", " ".join(command), flush=True)
    run_process(command, timeout=timeout)


def write_matrix_report(runs, out, args, curiosity):
    out.mkdir(parents=True, exist_ok=True)
    effects = matrix_effects(runs, args.backbone, args.seeds, curiosity, args.schedules)
    statistics = matrix_statistics(runs, args.backbone, curiosity, args.seeds, args.schedules)
    (out / "curiosity_matrix.json").write_text(json.dumps(effects, indent=2))
    (out / "curiosity_statistics.json").write_text(json.dumps(statistics, indent=2))
    text = ("# Curiosity under environment changes\n\n"
            "Final evaluation uses the inspection regime. Curiosity effects are paired on/off differences in held-out success. "
            "Constant-inspection training is the matched stationary control in the overnight preset; "
            "a constant-original-rule condition is a transfer control. Bootstrap intervals are exploratory. "
            "One seed describes task variability only and cannot establish reliability across training seeds. "
            "A zero-width interval does not establish equivalence; degenerate intervals are flagged in the JSON.\n\n"
            "| Condition/effect | Paired seeds | Difference | 95% interval | Variability |\n|---|---|---:|---|---|\n")
    for name, row in statistics.items():
        lo, hi = row["ci95"]
        text += (f"| {name} | {row['seeds']} | {100 * row['delta']:+.2f} pp | "
                 f"[{100 * lo:+.2f}, {100 * hi:+.2f}] pp | {row['scope']} |\n")
    if not statistics:
        text += "| No completed on/off pairs | | unavailable | unavailable | |\n"
    text += "\nMissing conditions remain unavailable. Pre-switch competence is required to interpret adaptation.\n\n"
    for schedule in args.schedules:
        condition_out = out / schedule
        condition_out.mkdir(parents=True, exist_ok=True)
        text += (f"- [{schedule}: adaptation and competence]({schedule}/adaptation.md)\n"
                 if (condition_out / "adaptation.md").exists() else f"- {schedule}: no adaptation data\n")
    (out / "curiosity_matrix.md").write_text(text)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if os.name == "nt":
        return launch_in_wsl(argv, "run_nonstationary.py")
    from benchmarks.common import tiers
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=tiers(), default="nonstationary_screen")
    ap.add_argument("--backbone", choices=("grpo", "empg"), default="grpo")
    ap.add_argument("--curiosity-arm", help="matching curiosity arm; default <backbone>_cur_episodic")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--schedules", nargs="+", choices=("stationary", "switch", "stationary_initial", "stationary_final"),
                    default=["switch", "stationary"])
    ap.add_argument("--runs")
    ap.add_argument("--out")
    ap.add_argument("--hours", type=float, help="deadline across the entire matrix; overnight default 8; 0 is unlimited")
    ap.add_argument("--fit-budget", action="store_true", help="fit one common iteration budget before starting")
    ap.add_argument("--budget-fraction", type=float, default=.9, help="fraction of --hours targeted by the central estimate")
    ap.add_argument("--max-iterations", type=int, default=200, help="upper limit when fitting a workload")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--estimate-only", action="store_true")
    modes.add_argument("--report-only", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    started = time.monotonic()
    if args.hours is None:
        args.hours = 8. if args.tier == "nonstationary_overnight" else 0.
    if not math.isfinite(args.hours) or args.hours < 0:
        ap.error("--hours must be finite and nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.schedules)) != len(args.schedules):
        ap.error("seeds and schedules must be unique")
    if args.fit_budget and any(o.split("=", 1)[0] in (
            "train.iterations", "nonstationary.switch_iteration", "nonstationary.probe_every") for o in args.overrides):
        ap.error("budget fitting controls iteration/switch/probe counts; omit --fit-budget for fixed manual counts")
    reserved = {"nonstationary.enabled", "nonstationary.schedule", "nonstationary.eval_regime", "name", "seed", "out_dir",
                "advantage.curiosity.weight", "advantage.method"}
    if any(o.split("=", 1)[0] in reserved for o in args.overrides):
        ap.error("matrix schedules/output/evaluation and treatment identities are controlled by matrix options; "
                 "edit the curiosity arm YAML or run individual conditions to change its weight")
    runs, out = Path(args.runs or f"runs/{args.tier}_matrix"), Path(args.out or f"results/{args.tier}_matrix")
    from benchmarks.common import arms
    curiosity = args.curiosity_arm or f"{args.backbone}_cur_episodic"
    if curiosity not in arms() or not curiosity.startswith(args.backbone + "_cur_"):
        ap.error("--curiosity-arm must be an installed curiosity arm matching --backbone")
    selected = [args.backbone, curiosity]
    shared = [*args.overrides, "nonstationary.enabled=true", "nonstationary.eval_regime=inspection"]
    os.chdir(ROOT)
    try:
        planned = plan_jobs(args, selected, shared, runs)
    except ValueError as error:
        ap.error(str(error))
    # Validate every destination and evaluation panel before loading a model.
    from benchmarks.evaluate import eval_path
    from benchmarks.suite import _finished, _matching_eval
    for job in planned:
        cfg = job["cfg"]
        job["complete"] = _matching_eval(eval_path(cfg), cfg)
        job["needs_train"] = not _finished(cfg) or not (cfg.run_dir / "adapter").exists()
        job["estimate"] = estimate(cfg, train=job["needs_train"], evaluate=True) if not job["complete"] else {"seconds": 0.}
    remaining = [p for p in planned if not p["complete"]]
    reporting = 60 + 30 * len(args.schedules)
    total = sum(p["estimate"]["seconds"] for p in remaining) + reporting
    print(f"\nComplete matrix: {len(selected)} methods x {len(args.schedules)} conditions x {len(args.seeds)} seeds = {len(planned)} runs.")
    print(f"TOTAL remaining estimate: {duration(total)} (scenario range {duration(total * .5)} to {duration(total * 1.5)}).")
    print(f"Common workload: {planned[0]['cfg'].train.iterations} updates; switch at update "
          f"{planned[0]['cfg'].nonstationary.switch_iteration}; {planned[0]['cfg'].train.max_steps}-turn horizon.")
    print(f"Stationary regime: {planned[0]['cfg'].nonstationary.stationary_regime}; "
          f"feedback: {planned[0]['cfg'].nonstationary.feedback}; backend: {planned[0]['cfg'].policy.backend}.")
    if args.hours:
        print(f"Whole-matrix ceiling: {args.hours:g} hours, including reserved reporting time. "
              "Actual runtime varies; the estimate is not a guarantee.")
        if total > args.hours * 3600:
            print("The estimate exceeds the ceiling. Use --fit-budget or reduce workload before starting.")
    print("\nCondition / seed / method / train / probes / final evaluation / estimated total")
    for p in planned:
        e = p["estimate"]
        print(f"{p['schedule']} / {p['seed']} / {p['arm']} / {duration(e.get('training_seconds', 0))} / "
              f"{duration(e.get('probe_seconds', 0))} / {duration(e.get('evaluation_seconds', 0))} / {duration(e['seconds'])}")
    if args.estimate_only:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"hours": args.hours, "estimate_seconds": total, "fit_budget": args.fit_budget,
                "jobs": [{k: v for k, v in p.items() if k != "cfg"} | {"config": dataclasses.asdict(p["cfg"])} for p in planned],
                "status": "running"}
    manifest_file = out / "matrix_run.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    deadline = started + max(0., args.hours * 3600 - reporting) if args.hours else None
    status = 0
    try:
        for index, job in enumerate(remaining if not args.report_only else []):
            available = deadline - time.monotonic() if deadline else None
            if available is not None and available <= 0:
                status = 124
                break
            future_cost = sum(p["estimate"]["seconds"] for p in remaining[index:])
            timeout = available * job["estimate"]["seconds"] / future_cost if available is not None else None
            module = "benchmarks.train" if job["needs_train"] else "benchmarks.evaluate"
            command = [sys.executable, "-u", "-m", module, "--config", job["arm"], "--tier", args.tier,
                       "--seed", str(job["seed"]), *(["--evaluate"] if job["needs_train"] else []), *job["overrides"]]
            try:
                execute_phase(command, timeout=timeout)
                job["status"], job["complete"] = "complete", True
            except subprocess.TimeoutExpired:
                job["status"], status = "deadline_reached", 124
                print("Job reached its fair time allocation; continuing with the reserved time for later conditions.", flush=True)
            except subprocess.CalledProcessError:
                job["status"], status = "failed", 1
                raise
            except KeyboardInterrupt:
                job["status"], status = "interrupted", 130
                raise
            finally:
                manifest["jobs"] = [{k: v for k, v in p.items() if k != "cfg"} | {"config": dataclasses.asdict(p["cfg"])} for p in planned]
                manifest_file.write_text(json.dumps(manifest, indent=2))
    finally:
        manifest["status"] = "reporting"
        manifest_file.write_text(json.dumps(manifest, indent=2))
        try:
            for schedule in args.schedules:
                command = [sys.executable, "-m", "benchmarks.report", "--runs", str(runs / schedule), "--out", str(out / schedule),
                           "--split", planned[0]["cfg"].eval.split, "--arms", *selected, "--seeds", *map(str, args.seeds)]
                execute_phase(command)
            write_matrix_report(runs, out, args, curiosity)
            manifest["status"] = ("deadline_reached" if status == 124 else "failed" if status else
                                  "reports_complete" if args.report_only else "complete")
        except BaseException:
            manifest["status"] = "report_failed"
            raise
        finally:
            manifest["elapsed_seconds"] = time.monotonic() - started
            manifest_file.write_text(json.dumps(manifest, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
