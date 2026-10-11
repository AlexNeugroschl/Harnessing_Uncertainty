"""Paired task/seed comparisons without treating repeated tasks as independent."""

import numpy as np


def paired_comparison(baseline, candidate, bootstrap=10000, seed=0):
    """Runs must contain seed + per_task[{task, won}]. Resample paired seeds and
    tasks independently, reusing the task draw across seeds (crossed bootstrap).
    With one seed the interval describes task variability only.
    """
    if bootstrap < 1:
        raise ValueError("bootstrap count must be positive")

    def index(runs):
        out = {}
        for run in runs:
            if "seed" not in run or "per_task" not in run:
                raise ValueError("per-task evaluations with explicit seeds are required")
            if run["seed"] in out:
                raise ValueError("duplicate seed in comparison")
            rows = run["per_task"]
            values = {row["task"]: float(row["won"]) for row in rows}
            if len(values) != len(rows):
                raise ValueError("duplicate evaluation task")
            out[run["seed"]] = (values, run.get("protocol"))
        return out

    left, right = index(baseline), index(candidate)
    if set(left) != set(right) or not left:
        raise ValueError("arms must have the same nonempty set of paired seeds")
    seeds = sorted(left)
    tasks = sorted(left[seeds[0]][0])
    if not tasks:
        raise ValueError("empty task panel")
    for s in seeds:
        if set(left[s][0]) != set(tasks) or set(right[s][0]) != set(tasks):
            raise ValueError("arms/seeds must use the identical task panel")
        if left[s][1] != right[s][1]:
            raise ValueError("evaluation protocols differ")
    delta = np.array([[right[s][0][t] - left[s][0][t] for t in tasks] for s in seeds])
    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap)
    # Small blocks keep the bootstrap bounded even on much larger task panels.
    for start in range(0, bootstrap, 128):
        n = min(128, bootstrap - start)
        sidx = rng.integers(len(seeds), size=(n, len(seeds), 1))
        tidx = rng.integers(len(tasks), size=(n, 1, len(tasks)))
        draws[start:start + n] = delta[sidx, tidx].mean(axis=(1, 2))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"delta": float(delta.mean()), "ci95": [float(lo), float(hi)],
            "degenerate": bool(np.ptp(delta) == 0),
            "seeds": seeds, "n_tasks": len(tasks), "discordant_seed_tasks": int((delta != 0).sum()),
            "seed_deltas": delta.mean(axis=1).tolist(),
            "scope": "task variability only" if len(seeds) == 1 else "paired seed/task variability",
            "note": "Exploratory percentile bootstrap; few seeds give uncertain intervals. "
                    "Constant observed differences give a degenerate interval, not evidence of equivalence."}
