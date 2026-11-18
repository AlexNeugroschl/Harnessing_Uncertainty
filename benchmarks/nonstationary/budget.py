"""Measured workload budgeting. All treatments keep the same fixed update count."""
from __future__ import annotations

import math
import statistics


def validate_measurements(profile):
    if not profile.get("complete") or profile.get("schema") != 1:
        raise ValueError("a complete native calibration is required")
    for regime in ("baseline", "novelty"):
        rows = [r for r in profile.get("training", []) if r["regime"] == regime]
        if not rows or any(not all(math.isfinite(r[k]) and r[k] > 0 for k in
                                  ("steps", "rollout_seconds", "update_seconds")) for r in rows):
            raise ValueError(f"missing or invalid {regime} training measurements")
    evaluation = profile.get("evaluation", [])
    if {r["regime"] for r in evaluation} != {"baseline", "novelty"} or any(
            not all(math.isfinite(r[k]) and r[k] > 0 for k in ("steps", "seconds", "tasks")) for r in evaluation):
        raise ValueError("missing or invalid evaluation measurements")
    if any(not math.isfinite(r["success"]) or not 0 <= r["success"] <= 1 for r in evaluation):
        raise ValueError("invalid competence measurements")
    if "diversity" in profile and not all(math.isfinite(profile["diversity"][k]) and profile["diversity"][k] > 0
                                          for k in ("update_seconds", "gradient_rows", "steps")):
        raise ValueError("invalid diversity measurements")


def verify_identity(cfg, profile):
    from .calibration import runtime_identity
    validate_measurements(profile)
    if runtime_identity(cfg) != profile["identity"]:
        raise ValueError("calibration differs in hardware, code, checkpoint, environment or execution settings; recalibrate")


def estimate(cfg, profile, *, train=True, evaluate=True):
    """Hedge competent baseline episodes toward the full cap; use measured regimes.

    New policies may change episode length. Budgeting is a forecast, backed by a
    whole-process deadline, and never changes iteration budgets while runs execute.
    """
    validate_measurements(profile)
    full_batch = cfg.train.games_per_iter * cfg.train.group_size * cfg.train.max_steps
    batches = {}
    for regime in ("baseline", "novelty"):
        rows = [r for r in profile["training"] if r["regime"] == regime]
        observed = max(r["steps"] for r in rows)
        hedge = (.8 * observed + .2 * full_batch) / observed
        rollout = statistics.median(r["rollout_seconds"] for r in rows) * hedge
        update = statistics.median(r["update_seconds"] for r in rows) * hedge
        if "diversity" in profile:
            stress = profile["diversity"]
            rate = stress["update_seconds"] / stress["gradient_rows"]
            # A predeclared hedge against learning more diverse trajectories:
            # at least 35% of future response rows are budgeted as distinct.
            fraction = max(.35, max(r["gradient_rows"] / r["steps"] for r in rows))
            update = max(update, rate * observed * hedge * fraction)
        batches[regime] = rollout + update
    training = sum(batches[cfg.nonstationary.regime_at(i)] for i in range(1, cfg.train.iterations + 1)) if train else 0.
    eval_rate = max(r["seconds"] / r["steps"] for r in profile["evaluation"])
    overhead = profile.get("evaluation_session_overhead_seconds", 0.)
    available = cfg.env.get("tasks_per_split", 64)
    tasks = min(cfg.eval.max_tasks or available, available)
    probe_tasks = min(cfg.nonstationary.probe_tasks or available, available)
    final = overhead + tasks * cfg.eval.max_steps * eval_rate if evaluate else 0.
    probes = (len(cfg.nonstationary.probe_iterations(cfg.train.iterations)) *
              (overhead + probe_tasks * cfg.eval.max_steps * eval_rate * len(cfg.nonstationary.probe_regimes))) if train else 0.
    startup = profile.get("startup_seconds", 30.) + profile.get("shutdown_seconds", 5.)
    return {"seconds": 1.10 * (startup + training + probes + final),
            "training_seconds": training, "probe_seconds": probes, "evaluation_seconds": final,
            "startup_seconds": startup, "safety_multiplier": 1.10}


def common_workload(build_jobs, profile, seconds, *, minimum=12, maximum=200, fraction=.92):
    if (not math.isfinite(seconds) or seconds <= 0 or not 0 < fraction <= 1
            or minimum < 4 or maximum < minimum):
        raise ValueError("invalid fitting budget or update bounds")
    target = seconds * fraction
    best = None
    # At most 50 candidates; no model or environments are created here.
    for iterations in range(4 * math.ceil(minimum / 4), maximum + 1, 4):
        jobs = build_jobs(iterations)
        total = 90 + sum(estimate(j["cfg"], profile)["seconds"] for j in jobs if "alias_of" not in j)
        if total > target:
            break
        best = {"iterations": iterations, "estimated_seconds": total,
                "target_seconds": target, "available_seconds": seconds,
                "safety_fraction": fraction, "unique_runs": sum("alias_of" not in j for j in jobs)}
    if best is None:
        raise ValueError("the minimum matched experiment does not fit; reduce conditions/seeds or increase hours")
    return best
