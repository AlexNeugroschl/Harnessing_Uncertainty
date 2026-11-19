"""Competence-gated, calibrated non-stationary experiment with a 12-hour ceiling."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_benchmarks import ROOT, launch_in_wsl
from .artifacts import write_json
from . import matrix
from .budget import common_workload, estimate, validate_measurements, verify_identity


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/nonstationary/12h")
    ap.add_argument("--hours", type=float, default=12.)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--novelty", choices=["tool", "recipe", "resource"], default="tool")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--treatments", nargs="+", choices=matrix.TREATMENTS,
                    default=["constant", "high", "matched", "oracle", "detected"])
    ap.add_argument("--boost-factor", type=float, default=3.)
    ap.add_argument("--eta", type=float, default=.05)
    ap.add_argument("--minimum-updates", type=int, default=12)
    ap.add_argument("--maximum-updates", type=int, default=200)
    ap.add_argument("--initial-adapter")
    ap.add_argument("--calibration")
    ap.add_argument("--learning-profile", help="content-verified two-seed development learning evidence")
    ap.add_argument("--pre-change-updates", type=int, help="fixed baseline rounds, leaving the fitted remainder for adaptation")
    ap.add_argument("--warmup-hours", type=float, default=.5)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--prepare-only", action="store_true", help="prepare/validate/freeze the workload without starting the long matrix")
    ap.add_argument("--estimate-only", action="store_true", help="read-only; requires existing calibration for an hours forecast")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--reserve-seed", type=int, default=2, help="optional predeclared switched-arm replication if unused time fits")
    ap.add_argument("--reserve-count", type=int, default=32, help="maximum predeclared reserve seeds; admit only full blocks that fit")
    ap.add_argument("--no-reserve", action="store_true")
    ap.add_argument("--tier", default="novelgym_12h", help=argparse.SUPPRESS)
    return ap


def protocol(args):
    result = {k: getattr(args, k) for k in ("depth", "novelty", "seeds", "treatments", "boost_factor", "eta",
        "minimum_updates", "maximum_updates", "reserve_seed", "reserve_count", "no_reserve", "hours")}
    if args.tier != "novelgym_12h" or args.learning_profile or args.pre_change_updates is not None:
        result.update(tier=args.tier, learning_profile=str(Path(args.learning_profile).resolve()) if args.learning_profile else None,
                      pre_change_updates=args.pre_change_updates)
    return result


def matrix_argv(args, adapter, calibration, iterations, policy, *, reserve=False, hours=None, reserve_seed=None):
    root = Path(args.root)
    before = args.pre_change_updates if args.pre_change_updates is not None else iterations // 2
    if not 1 <= before < iterations:
        raise ValueError("pre-change updates must leave at least one adaptation update")
    argv = ["--tier", args.tier, "--depths", str(args.depth), "--novelties", args.novelty,
            "--seeds", *map(str, [args.reserve_seed if reserve_seed is None else reserve_seed] if reserve else args.seeds),
            "--treatments", *([t for t in ("constant", "oracle", "detected") if t in args.treatments] if reserve else args.treatments),
            "--schedules", *(["switch"] if reserve else ["switch", "stationary"]),
            "--eta", str(args.eta), "--boost-factors", str(args.boost_factor), "--boost-updates", str(max(2, iterations // 4)),
            "--switches", str(before + 1), "--initial-adapter", str(adapter),
            "--runs", str(root / "matrix"), "--out", str(root / ("reserve_results" if reserve else "results"))]
    if calibration:
        argv += ["--calibration", str(calibration)]
    if hours is not None:
        argv += ["--hours", str(hours)]
    argv += ["--", f"train.iterations={iterations}", f"nonstationary.probe_every={max(1, iterations // 4)}"]
    argv += [f"policy.{key}={json.dumps(value)}" for key, value in policy.items()]
    argv += [f"{key}={json.dumps(value)}" for key, value in getattr(args, "learning_settings", {}).items()]
    return argv


def build_jobs(args, adapter, calibration, iterations, policy, *, reserve=False, reserve_seed=None):
    parsed = matrix.parser().parse_args(matrix_argv(args, adapter, calibration, iterations, policy,
                                                  reserve=reserve, reserve_seed=reserve_seed))
    return matrix.plan(parsed)


def serialize_jobs(jobs):
    return [{k: v for k, v in job.items() if k != "cfg"} | {"config": dataclasses.asdict(job["cfg"])} for job in jobs]


def deserialize_jobs(rows):
    from empg.config import Config, _build
    return [{k: v for k, v in row.items() if k != "config"} | {"cfg": _build(Config, row["config"])} for row in rows]


def combined_jobs(frozen, *, include_unstarted=True):
    primary = deserialize_jobs(frozen["jobs"])
    reserve = deserialize_jobs(frozen["reserve_jobs"])
    for job in reserve:
        job["optional_reserve"] = True
        if "alias_of" in job:
            job["alias_of"] += len(primary)
    if include_unstarted:
        return primary + reserve
    started = {j["seed"] for j in reserve if (j["cfg"].run_dir / "config.json").exists()}
    included = [(i, j) for i, j in enumerate(primary + reserve) if i < len(primary) or j["seed"] in started]
    mapping = {old: new for new, (old, _) in enumerate(included)}
    for _, job in included:
        if "alias_of" in job:
            job["alias_of"] = mapping[job["alias_of"]]
    return [job for _, job in included]


def competence(profile, minimum=.75):
    baseline = next((r for r in profile.get("evaluation", []) if r["regime"] == "baseline"), None)
    if baseline is None or baseline["tasks"] < 8 or baseline["success"] < minimum:
        raise ValueError("initial checkpoint failed the held-out baseline competence gate (>=75% on at least 8 tasks)")


def freeze(args, adapter, calibration_path, available_seconds):
    learning = None
    if args.learning_profile:
        from .evidence import load_learning_profile
        learning = load_learning_profile(args.learning_profile, adapter)
        if learning["options"]["novelty"] != args.novelty:
            raise ValueError("learning evidence uses a different change family")
        args.learning_settings = learning["settings"]
    elif args.tier == "novelgym_recovery_12h":
        raise ValueError("recovery experiments require a passing development learning profile")
    profile = json.loads(Path(calibration_path).read_text())
    validate_measurements(profile)
    competence(profile)
    policy = profile["identity"]["policy"]
    builder = lambda n: build_jobs(args, adapter, calibration_path, n, policy)
    # Validate identity once before fitting; iteration/switch/probe counts do not
    # alter the measured GPU execution shape.
    verify_identity(builder(max(12, args.minimum_updates))[0]["cfg"], profile)
    workload = common_workload(builder, profile, available_seconds,
                               minimum=args.minimum_updates, maximum=args.maximum_updates)
    jobs = builder(workload["iterations"])
    reserve = []
    if not args.no_reserve and "constant" in args.treatments and any(t in args.treatments for t in ("oracle", "detected")):
        for seed in range(args.reserve_seed, args.reserve_seed + args.reserve_count):
            if seed in args.seeds:
                continue
            block = build_jobs(args, adapter, calibration_path, workload["iterations"], policy, reserve=True, reserve_seed=seed)
            for job in block:
                if "alias_of" in job:
                    job["alias_of"] += len(reserve)
            reserve.extend(block)
    frozen = {"schema": 1, "protocol": protocol(args), "initial_adapter": str(Path(adapter).resolve()),
            "calibration": str(Path(calibration_path).resolve()),
            "calibration_sha256": hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest(),
            "workload": workload, "policy": policy, "jobs": serialize_jobs(jobs), "reserve_jobs": serialize_jobs(reserve),
            "note": "Fixed primary workload; optional reserve replication depends on time, never on observed scores."}
    if learning:
        frozen.update(learning_profile=str(Path(args.learning_profile).resolve()),
                      learning_profile_sha256=hashlib.sha256(Path(args.learning_profile).read_bytes()).hexdigest(),
                      shared_settings=learning["settings"])
    return frozen


def _phase(command, deadline, *, cap=None):
    from benchmarks.process import run_process
    remaining = deadline - time.monotonic()
    if cap is not None:
        remaining = min(remaining, cap)
    print("$", " ".join(map(str, command)), flush=True)
    run_process(list(map(str, command)), timeout=max(0., remaining))


def reserve_cost(block, completed_candidates, profile):
    """Admit reserves using finished-run cost, with 25% cross-seed headroom."""
    from benchmarks.evaluate import eval_path
    from benchmarks.suite import _finished, _matching_eval
    from .artifacts import read_jsonl
    total = 0.
    rate = max(r["seconds"] / r["steps"] for r in profile["evaluation"])
    for job in block:
        if "alias_of" in job:
            continue
        samples = []
        for previous in completed_candidates:
            if (previous["treatment"] != job["treatment"] or previous["schedule"] != "switch"
                    or previous["depth"] != job["depth"] or previous["novelty"] != job["novelty"]):
                continue
            cfg = previous["cfg"]
            if not _finished(cfg) or not _matching_eval(eval_path(cfg), cfg):
                continue
            rows = read_jsonl(cfg.run_dir / "metrics.jsonl")
            score = json.loads(eval_path(cfg).read_text())
            if rows:
                samples.append(rows[-1].get("seconds_total", 0.) + rows[-1].get("probe_seconds_total", 0.) +
                    sum(r["steps"] for r in score["per_task"]) * rate + profile.get("startup_seconds", 30.) +
                    profile.get("shutdown_seconds", 5.) + profile.get("evaluation_session_overhead_seconds", 0.))
        total += 1.25 * max(samples) if samples else estimate(job["cfg"], profile)["seconds"]
    return total


def _prepared_adapter(root):
    from empg.config import adapter_fingerprint
    for complete in sorted((root / "preparation").glob("warmup_*/complete.json"), reverse=True):
        marker = json.loads(complete.read_text())
        if marker.get("ready") and adapter_fingerprint(complete.parent / "adapter") == marker.get("adapter_sha256"):
            return complete.parent / "adapter"
    return None


def prepare(args, root, deadline, state):
    adapter = Path(args.initial_adapter).resolve() if args.initial_adapter else _prepared_adapter(root)
    if adapter is None:
        attempts = list((root / "preparation").glob("warmup_*"))
        directory = root / "preparation" / f"warmup_{len(attempts) + 1:03d}"
        depths = sorted({2, 4, args.depth})
        recovery = args.tier == "novelgym_recovery_12h"
        extra = (["--tier", "novelgym_recovery", "--tool-probability", .05, "--min-tool-probability", .005,
                  "--max-tool-probability", .25, "--validate-every", 1, "--lr", .00003] if recovery else [])
        command = [sys.executable, "-u", "run_novelgym_warmup.py", "--out", directory,
                   "--depths", *depths, "--epochs", 8 if recovery else 12,
                   "--games", 4 if recovery else 8, "--require-competence", *extra]
        _phase(command, deadline, cap=args.warmup_hours * 3600)
        adapter = directory / "adapter"
    if args.tier == "novelgym_recovery_12h":
        if not args.learning_profile:
            attempts = list((root / "preparation").glob("pilot_*"))
            directory = root / "preparation" / f"pilot_{len(attempts) + 1:03d}"
            _phase([sys.executable, "-u", "run_novelgym_pilot.py", "--initial-adapter", adapter,
                    "--novelty", args.novelty, "--out", directory, "--hours", 1.], deadline)
            args.learning_profile = str(directory / "learning_profile.json")
        from .evidence import load_learning_profile
        args.learning_settings = load_learning_profile(args.learning_profile, adapter)["settings"]
        state["learning_profile"] = str(args.learning_profile)
    if args.calibration:
        calibration = Path(args.calibration).resolve()
        if not calibration.is_file():
            raise ValueError("the explicitly supplied calibration file does not exist")
    else:
        from empg.config import adapter_fingerprint
        candidates = sorted((root / "preparation").glob("calibration_*/calibration.json"), reverse=True)
        expected = adapter_fingerprint(adapter)
        calibration = None
        for candidate in candidates:
            data = json.loads(candidate.read_text())
            if data.get("complete") and data.get("identity", {}).get("initial_adapter_sha256") == expected:
                calibration = candidate
                break
        if calibration is None:
            directory = root / "preparation" / f"calibration_{len(list((root / 'preparation').glob('calibration_*'))) + 1:03d}"
            calibration = directory / "calibration.json"
    if not calibration.is_file():
        _phase([sys.executable, "-u", "run_novelgym_calibrate.py", "--initial-adapter", adapter,
            "--depth", args.depth, "--novelty", args.novelty, "--repeats", 2,
            "--out", calibration.parent, "--tier", args.tier,
            *(["--learning-profile", args.learning_profile] if args.learning_profile else []),
            *([] if args.no_tune else ["--tune"])], deadline)
    state["initial_adapter"], state["calibration"] = str(adapter), str(calibration)
    return adapter, calibration


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if os.name == "nt":
        return launch_in_wsl(["--tier", "novelgym_12h", *argv], entrypoint="run_novelgym_12h.py")
    ap = parser()
    args = ap.parse_args(argv)
    if (not math.isfinite(args.hours) or args.hours <= 0 or not math.isfinite(args.warmup_hours) or args.warmup_hours <= 0
            or not 2 <= args.depth <= 32 or args.minimum_updates < 4 or args.maximum_updates < args.minimum_updates or not 0 <= args.reserve_count <= 64
            or not math.isfinite(args.eta) or args.eta < 0 or not math.isfinite(args.boost_factor) or args.boost_factor < 1
            or min(args.seeds + [args.reserve_seed]) < 0 or (not args.no_reserve and args.reserve_seed in args.seeds)
            or len(set(args.seeds)) != len(args.seeds) or len(set(args.treatments)) != len(args.treatments)):
        ap.error("invalid budget, depths, treatments, seeds or coefficients")
    if args.pre_change_updates is not None and not 1 <= args.pre_change_updates < args.minimum_updates:
        ap.error("pre-change rounds must be positive and below the minimum fitted workload")
    os.chdir(ROOT)
    root = Path(args.root).resolve()
    plan_path = root / "plan.json"
    started = time.monotonic()
    deadline = started + args.hours * 3600
    frozen = None
    state = {"status": "preparing", "hours": args.hours}
    if plan_path.exists():
        frozen = json.loads(plan_path.read_text())
        explicit = {value.split("=", 1)[0] for value in argv if value.startswith("--")}
        for key, value in frozen["protocol"].items():
            if "--" + key.replace("_", "-") not in explicit:
                setattr(args, key, value)
        deadline = started + args.hours * 3600
        state["hours"] = args.hours
        if frozen["protocol"] != protocol(args):
            ap.error("frozen experiment options differ; use a fresh --root to change the protocol")
        if args.initial_adapter and Path(args.initial_adapter).resolve() != Path(frozen["initial_adapter"]):
            ap.error("initial adapter differs from the frozen plan")
        if args.calibration and Path(args.calibration).resolve() != Path(frozen["calibration"]):
            ap.error("calibration differs from the frozen plan")
    if args.estimate_only and frozen is None:
        if not args.calibration or not args.initial_adapter:
            print("Native preparation is required. --prepare-only trains/checks baseline skills, measures GPU costs, and freezes the matched 12-hour workload.")
            return 0
        frozen = freeze(args, args.initial_adapter, args.calibration, args.hours * 3600)
    if args.report_only:
        if frozen is None:
            ap.error("no frozen plan exists")
        matrix.report(combined_jobs(frozen, include_unstarted=False), root / "results")
        return 0
    if args.estimate_only:
        workload = frozen["workload"]
        print(json.dumps(workload, indent=2))
        print(f"Primary: depth {args.depth}, {len(args.seeds)} seeds, {workload['unique_runs']} distinct runs; "
              f"{workload['iterations']} matched updates each. " +
              ("Reserve replication disabled." if args.no_reserve else
               f"Up to {args.reserve_count} predeclared reserve seeds from {args.reserve_seed}."))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "system_run.json", state)
    status = 0
    try:
        if frozen is None:
            adapter, calibration = prepare(args, root, deadline, state)
            available = args.hours * 3600 if args.prepare_only else deadline - time.monotonic()
            frozen = freeze(args, adapter, calibration, available)
            write_json(plan_path, frozen)
        else:
            if "learning_profile" in frozen:
                from .evidence import file_sha256, load_learning_profile
                if file_sha256(frozen["learning_profile"]) != frozen["learning_profile_sha256"]:
                    raise ValueError("learning evidence changed after the plan was frozen")
                learned = load_learning_profile(frozen["learning_profile"], frozen["initial_adapter"])
                args.learning_settings = learned["settings"]
                if args.learning_settings != frozen["shared_settings"]:
                    raise ValueError("frozen shared learning settings differ")
            payload = Path(frozen["calibration"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != frozen["calibration_sha256"]:
                raise ValueError("calibration changed after the plan was frozen")
            profile = json.loads(payload)
            for job in deserialize_jobs(frozen["jobs"]):
                verify_identity(job["cfg"], profile)
        state.update(status="prepared", workload=frozen["workload"])
        write_json(root / "system_run.json", state)
        if args.prepare_only:
            print(json.dumps(frozen["workload"], indent=2))
            return 0
        profile = json.loads(Path(frozen["calibration"]).read_text())
        remaining = deadline - time.monotonic() - 90
        state["status"] = "primary_matrix"
        write_json(root / "system_run.json", state)
        command = [sys.executable, "-u", "run_novelgym.py", *matrix_argv(args, frozen["initial_adapter"],
            frozen["calibration"], frozen["workload"]["iterations"], frozen["policy"], hours=max(0., remaining) / 3600)]
        _phase(command, deadline - 60)
        reserve_jobs = deserialize_jobs(frozen["reserve_jobs"])
        for seed in sorted({j["seed"] for j in reserve_jobs}):
            block = [j for j in reserve_jobs if j["seed"] == seed]
            cost = reserve_cost(block, combined_jobs(frozen), profile)
            if deadline - time.monotonic() <= cost + 120:
                break
            state["status"] = "reserve_replication"
            state["reserve_seed"] = seed
            write_json(root / "system_run.json", state)
            command = [sys.executable, "-u", "run_novelgym.py", *matrix_argv(args, frozen["initial_adapter"],
                frozen["calibration"], frozen["workload"]["iterations"], frozen["policy"], reserve=True,
                hours=max(0., deadline - time.monotonic() - 90) / 3600, reserve_seed=seed)]
            _phase(command, deadline - 60)
            state.setdefault("reserve_completed_seeds", []).append(seed)
        state["status"] = "complete"
    except subprocess.TimeoutExpired:
        status, state["status"] = 124, "deadline_reached"
    except subprocess.CalledProcessError as error:
        status = error.returncode or 1
        state["status"] = "deadline_reached" if status == 124 else "competence_failed" if status == 2 else "phase_failed"
    except KeyboardInterrupt:
        status, state["status"] = 130, "interrupted"
    except BaseException as error:
        state.update(status="failed", error=str(error))
        status = 1
        raise
    finally:
        if frozen is not None and not args.prepare_only:
            try:
                matrix.report(combined_jobs(frozen, include_unstarted=False), root / "results")
            except BaseException as error:
                state.update(status="report_failed", error=str(error))
                status = status or 1
        state.update(seconds=time.monotonic() - started, exit_code=status)
        write_json(root / "system_run.json", state)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
