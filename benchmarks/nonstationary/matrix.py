"""Matched curiosity dose/timing experiments on native non-stationary crafting.

Windows forwards to WSL. Native calibration enables runtime forecasts.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_benchmarks import ROOT, launch_in_wsl
from .artifacts import write_json as _write
from .reporting import read_jsonl, report

TREATMENTS = ("none", "constant", "high", "matched", "oracle", "detected", "clock")


def treatments(args, nc, iterations):
    """Matched is equal planned coefficient exposure, not equal realized reward.

    Actual state novelty and episode lengths depend on the policy; realized
    intrinsic rewards are logged separately and cannot be matched in advance.
    """
    from empg.curiosity import CuriosityConfig, CuriosityController
    for kind in args.treatments:
        factors = [1.] if kind in ("none", "constant") else args.boost_factors
        for factor in factors:
            weight, boost = args.eta, "none"
            if kind == "none":
                weight = 0.
            elif kind == "high":
                weight *= factor
            elif kind == "matched":
                c = CuriosityConfig(boost="oracle_window", boost_factor=factor, boost_updates=args.boost_updates)
                controller = CuriosityController(c)
                weight *= sum(controller.multiplier(i, nc) for i in range(1, iterations + 1)) / iterations
            elif kind in ("oracle", "detected", "clock"):
                boost = {"oracle": "oracle_window", "detected": "observed_failure", "clock": "clock_window"}[kind]
            suffix = "" if kind in ("none", "constant") else "_x" + f"{factor:g}".replace(".", "p")
            yield kind + suffix, {"weight": weight, "boost": boost, "boost_factor": factor,
                                 "boost_updates": args.boost_updates}


def plan(args):
    from benchmarks.common import load
    from empg.envs.novelgym_env import nominal_solution_length
    jobs = []
    shared = list(args.overrides)
    if args.initial_adapter:
        if any(o.split("=", 1)[0] == "train.initial_adapter" for o in shared):
            raise ValueError("specify the initial adapter once")
        shared.append(f"train.initial_adapter={args.initial_adapter}")
    forbidden = ("benchmark", "name", "seed", "out_dir", "env.recipe_depth", "env.novelty",
                 "nonstationary.enabled", "nonstationary.schedule", "nonstationary.switch_iteration",
                 "nonstationary.switch_iterations", "nonstationary.eval_regime",
                 "advantage.curiosity.weight", "advantage.curiosity.boost",
                 "advantage.curiosity.boost_factor", "advantage.curiosity.boost_updates")
    if any(o.split("=", 1)[0] in forbidden for o in shared):
        raise ValueError("use the matrix options for environment, treatment, switch, name and output identities")
    template_overrides = [*shared, f"nonstationary.schedule={args.schedules[0]}"]
    if args.switches:
        template_overrides += [f"nonstationary.switch_iteration={args.switches[0]}",
                               f"nonstationary.switch_iterations={json.dumps(args.switches if len(args.switches) > 1 else [])}"]
    template = load(args.backbone, args.tier, template_overrides)
    if args.boost_updates is None:
        args.boost_updates = template.advantage.curiosity.boost_updates
    for depth in args.depths:
        for novelty in args.novelties:
            if template.nonstationary.change == "rewards" and novelty != "tool":
                raise ValueError("reward-only shifts require --novelties tool")
            for schedule in args.schedules:
                switches = args.switches or template.nonstationary.change_points()
                if schedule == "switch" and len(switches) != 1:
                    raise ValueError("use --schedules recurring stationary with multiple --switches")
                nc = dataclasses.replace(template.nonstationary, schedule=schedule,
                         switch_iteration=switches[0], switch_iterations=switches if len(switches) > 1 else [])
                nc.validate(template.train.iterations)
                directory = Path(args.runs) / f"depth_{depth}" / novelty / schedule
                # At least twice a constructive native solution plus error recovery.
                horizon = max(template.train.max_steps, 2 * nominal_solution_length(depth, novelty, "novelty") + 8)
                for seed in args.seeds:
                    for treatment, options in treatments(args, nc, template.train.iterations):
                        overrides = [*shared, f"env.recipe_depth={depth}", f"env.novelty={novelty}",
                            f"nonstationary.schedule={schedule}", f"nonstationary.switch_iteration={switches[0]}",
                            f"nonstationary.switch_iterations={json.dumps(nc.switch_iterations)}",
                            "nonstationary.eval_regime=novelty", f"train.max_steps={horizon}",
                            f"eval.max_steps={max(horizon, template.eval.max_steps)}",
                            f"name={args.backbone}_{treatment}", f"out_dir={directory}",
                            *(f"advantage.curiosity.{k}={v}" for k, v in options.items())]
                        cfg = load(args.backbone, args.tier, overrides, seed)
                        jobs.append({"depth": depth, "novelty": novelty, "schedule": schedule, "seed": seed,
                                     "treatment": treatment, "cfg": cfg, "overrides": overrides})
    if len({str(j["cfg"].run_dir) for j in jobs}) != len(jobs):
        raise ValueError("duplicate matrix run identities")
    # Stationary oracle/matched arms often have exactly the constant treatment's
    # coefficient schedule. Reuse that run explicitly instead of spending GPU
    # time training an identical policy; aliases never add independent seeds.
    known = {}
    for index, job in enumerate(jobs):
        key = execution_key(job["cfg"])
        if key is not None and key in known:
            job["alias_of"] = known[key]
        elif key is not None:
            known[key] = index
    return jobs


def execution_key(cfg):
    from empg.config import config_dict
    from empg.curiosity import CuriosityController, schedule_multiplier
    cur = cfg.advantage.curiosity
    if cur.boost == "observed_failure":
        return None
    controller = CuriosityController(cur)
    weights = [cur.weight * schedule_multiplier(cur, i, cfg.train.iterations) * controller.multiplier(i, cfg.nonstationary)
               for i in range(1, cfg.train.iterations + 1)]
    data = config_dict(cfg)
    data.pop("name")
    data.pop("out_dir")
    options = data["advantage"]["curiosity"]
    for key in ("boost", "boost_factor", "boost_updates", "failure_threshold", "failure_ema", "detector_warmup"):
        options.pop(key, None)
    options["weight"] = 0.
    return json.dumps([data, weights], sort_keys=True)
def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=["novelgym_screen", "novelgym", "novelgym_12h"], default="novelgym_screen")
    ap.add_argument("--backbone", choices=["grpo", "empg"], default="grpo")
    ap.add_argument("--depths", type=int, nargs="+", default=[4])
    ap.add_argument("--novelties", nargs="+", choices=["tool", "recipe", "resource"], default=["tool"])
    ap.add_argument("--schedules", nargs="+", choices=["switch", "stationary", "recurring"], default=["switch", "stationary"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--treatments", nargs="+", choices=TREATMENTS,
                    default=["none", "constant", "high", "matched", "oracle", "detected"])
    ap.add_argument("--eta", type=float, default=.05)
    ap.add_argument("--boost-factors", type=float, nargs="+", default=[3.])
    ap.add_argument("--boost-updates", type=int, help="defaults to the tier's boost window")
    ap.add_argument("--switches", type=int, nargs="+")
    ap.add_argument("--hours", type=float, default=0, help="whole-matrix ceiling, shared fairly across remaining jobs")
    ap.add_argument("--initial-adapter", help="common baseline competence checkpoint, content-fingerprinted for every arm")
    ap.add_argument("--calibration", help="matching native measurements for forecasts and weighted deadline allocation")
    ap.add_argument("--runs")
    ap.add_argument("--out")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("overrides", nargs="*")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.name == "nt":
        if not any(a == "--tier" or a.startswith("--tier=") for a in argv):
            argv = ["--tier", "novelgym_screen", *argv]
        return launch_in_wsl(argv, entrypoint="run_novelgym.py")
    ap = parser()
    args = ap.parse_args(argv)
    if not math.isfinite(args.hours) or args.hours < 0 or not math.isfinite(args.eta) or args.eta < 0:
        ap.error("hours and eta must be finite and nonnegative")
    if any(not math.isfinite(f) or f < 1 for f in args.boost_factors) or (args.boost_updates is not None and args.boost_updates < 1):
        ap.error("boost factors must be finite and >=1; boost updates must be positive")
    for field in ("seeds", "depths", "novelties", "schedules", "treatments", "boost_factors"):
        values = getattr(args, field)
        if len(values) != len(set(values)):
            ap.error(f"duplicate {field}")
    if any(s < 0 for s in args.seeds) or any(not 2 <= d <= 32 for d in args.depths):
        ap.error("seeds must be nonnegative; depths must be between 2 and 32")
    args.runs = args.runs or f"runs/{args.tier}_matrix"
    args.out = args.out or f"results/{args.tier}_matrix"
    os.chdir(ROOT)
    try:
        jobs = plan(args)
    except (ValueError, KeyError) as error:
        ap.error(str(error))
    from benchmarks.suite import _finished, _matching_eval
    from benchmarks.evaluate import eval_path
    for job in jobs:
        cfg = jobs[job["alias_of"]]["cfg"] if "alias_of" in job else job["cfg"]
        job["needs_train"] = not _finished(cfg) or not (cfg.run_dir / "adapter").exists()
        job["complete"] = _matching_eval(eval_path(cfg), cfg) and not job["needs_train"]
    remaining = [job for job in jobs if not job["complete"] and "alias_of" not in job]
    if args.calibration:
        from .budget import estimate, verify_identity
        profile = json.loads(Path(args.calibration).read_text())
        for job in remaining:
            verify_identity(job["cfg"], profile)
            job["estimate"] = estimate(job["cfg"], profile, train=job["needs_train"])
    print(f"{len(jobs)} conditions; {sum('alias_of' in j for j in jobs)} exact-equivalence reuses; "
          f"{len(remaining)} runs remaining. Backend: {jobs[0]['cfg'].policy.backend}.")
    training_steps = sum(j["cfg"].train.iterations * j["cfg"].train.games_per_iter * j["cfg"].train.group_size *
                         j["cfg"].train.max_steps for j in remaining if j["needs_train"])
    print(f"Training upper bound: {training_steps:,} environment turns. Early successes can reduce this.")
    if args.calibration:
        print(f"Measured remaining forecast: {sum(j['estimate']['seconds'] for j in remaining) / 3600:.2f} hours, including 10% cost headroom.")
    else:
        print("Workload-only preview: counts are not an hours estimate and do not reuse ALFWorld timings. Use native calibration for a measured forecast.")
    for j in jobs:
        c = j["cfg"]
        print(f"depth={j['depth']} {j['novelty']} {j['schedule']} seed={j['seed']} {j['treatment']}: "
              f"eta={c.advantage.curiosity.weight:g}, {c.train.iterations} updates, horizon={c.train.max_steps}" +
              (f" [reuse {jobs[j['alias_of']]['treatment']}]" if "alias_of" in j else " [complete]" if j["complete"] else ""))
    if args.estimate_only:
        return 0
    from benchmarks.process import run_process
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "hours": args.hours,
                "jobs": [{k: v for k, v in j.items() if k != "cfg"} | {"config": dataclasses.asdict(j["cfg"])} for j in jobs]}
    _write(out / "matrix_run.json", manifest)
    deadline = time.monotonic() + args.hours * 3600 if args.hours else None
    status = 0
    try:
        for index, job in enumerate([] if args.report_only else remaining):
            available = deadline - time.monotonic() if deadline else None
            if available is not None and available <= 0:
                status = 124
                break
            # Reserve report time and a fair allocation for EVERY remaining job.
            # Native final-regime runs can cost substantially more than switched
            # runs. Allocate by measured cost rather than starving later controls.
            future = remaining[index:]
            weight = job.get("estimate", {}).get("seconds", 1.)
            future_weight = sum(j.get("estimate", {}).get("seconds", 1.) for j in future)
            timeout = max(0., available - 90) * weight / future_weight if available is not None else None
            module = "benchmarks.train" if job["needs_train"] else "benchmarks.evaluate"
            cmd = [sys.executable, "-u", "-m", module, "--config", args.backbone,
                   "--tier", args.tier, "--seed", str(job["seed"]),
                   *(["--evaluate"] if job["needs_train"] else []), *job["overrides"]]
            print("$", " ".join(cmd), flush=True)
            try:
                run_process(cmd, timeout=timeout)
                if not _finished(job["cfg"]) or not _matching_eval(eval_path(job["cfg"]), job["cfg"]):
                    raise RuntimeError("phase exited successfully without a complete training/evaluation artifact")
                job["status"], job["complete"] = "complete", True
            except subprocess.TimeoutExpired:
                job["status"], status = "deadline_reached", 124
            except subprocess.CalledProcessError:
                job["status"], status = "failed", 1
                raise
            finally:
                manifest["jobs"][jobs.index(job)]["status"] = job.get("status", "interrupted")
                _write(out / "matrix_run.json", manifest)
    except KeyboardInterrupt:
        status = 130
    except BaseException:
        status = 1
        raise
    finally:
        try:
            report(jobs, out)
        except BaseException:
            manifest["status"] = "report_failed"
            _write(out / "matrix_run.json", manifest)
            raise
        for job in jobs:
            cfg = jobs[job["alias_of"]]["cfg"] if "alias_of" in job else job["cfg"]
            job["complete"] = _finished(cfg) and _matching_eval(eval_path(cfg), cfg)
            if "alias_of" in job:
                job["status"] = "reused_complete" if job["complete"] else "unavailable"
        manifest["jobs"] = [{k: v for k, v in j.items() if k != "cfg"} |
                            {"config": dataclasses.asdict(j["cfg"])} for j in jobs]
        manifest["status"] = "reports_complete" if args.report_only else "complete" if status == 0 and all(j["complete"] for j in jobs) else "partial"
        if not args.report_only and not all(j["complete"] for j in jobs):
            status = status or 1
        _write(out / "matrix_run.json", manifest)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
