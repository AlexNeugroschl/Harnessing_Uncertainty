"""Aggregate runs into paper-style tables and training curves.

    python -m benchmarks.report [--runs runs/reduced] [--split valid_unseen] [--out results] [--arms ...]

Writes:
    results/table_alfworld.md   per-subtask success (paper Table 1 layout)
    results/ablation.md         GRPO vs g-only vs f-only vs full EMPG (paper ablation)
    results/exploration.md      exploration metrics from training (curiosity plan Sec. 6)
    results/response_lengths.md response lengths vs the max_new_tokens cap (is a lower cap safe?)
    results/training_curves.png train success, step entropy, revisit rate, P(group has a success)
    results/summary.json        all numbers above
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from empg.rollout import length_stats
from benchmarks.statistics import paired_comparison

SUBTASKS = ["Pick", "Look", "Clean", "Heat", "Cool", "Pick2"]
ARMS = [("grpo", "GRPO"), ("empg", "GRPO + EMPG"),
        ("empg_g_only", "+ gradient scaling g only"), ("empg_f_only", "+ clarity bonus f only")]
LABELS = {
    "grpo_memory": "GRPO + visited-locations memory",
    "grpo_cur_episodic": "GRPO + curiosity (episodic)",
    "grpo_cur_group": "GRPO + curiosity (group)",
    "grpo_cur_oracle": "GRPO + curiosity (oracle state)",
    "empg_cur_episodic": "EMPG + curiosity (episodic)",
    "empg_seek_uncertainty": "EMPG, clarity sign flipped",
    **dict(ARMS),
}
# Fixed categorical identities for all ten configured arms.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
           "#795548", "#0097a7"]
COLORS = {arm: color for arm, color in zip(
    [a for a, _ in ARMS] + [a for a in LABELS if a not in dict(ARMS)], PALETTE)}
EXPLORE_COLS = [("success", "train success"), ("revisit_rate", "revisit rate"), ("noop_rate", "no-op rate"),
                ("unique_states", "unique states / ep"), ("group_coverage", "group coverage"),
                ("any_success_frac", "P(group ≥1 success)"), ("no_signal_frac", "no-signal groups"),
                ("success_len", "steps to success"), ("intrinsic_share", "intrinsic share")]


def run_dirs(runs: Path, seeds=None):
    """Use the same seed selection for every artifact, including token lengths."""
    if not runs.exists():
        return
    for d in sorted(runs.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "base":
            yield d, "base"
            continue
        arm, separator, seed = d.name.rpartition("_s")
        if seeds is not None and (not separator or seed not in {str(s) for s in seeds}):
            continue
        yield d, arm if separator else d.name


def load(runs: Path, split: str, seeds=None):
    evals, curves = defaultdict(list), defaultdict(list)
    for d, arm in run_dirs(runs, seeds):
        e = d / f"eval_{split}.json"
        if e.exists():
            evals[arm].append(json.loads(e.read_text()))
        m = d / "metrics.jsonl"
        if m.exists():
            curves[arm].append([json.loads(line) for line in m.open()])
    return evals, curves


def cell(vals) -> str:
    if not vals:
        return "–"
    v = 100 * np.array(vals)
    return f"{v.mean():.1f}" if len(v) == 1 else f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"


def with_extra_arms(rows, found) -> list:
    """Known arms in their fixed order, then any other run names found (e.g. custom configs)."""
    known = {a for a, _ in rows} | {a for a, _ in ARMS} | {"base"}
    return rows + [(a, LABELS.get(a, a)) for a in sorted(found) if a not in known]


def table(evals, rows, cols) -> str:
    names = ["Method", *cols, "All", "seeds"]
    head = "| " + " | ".join(names) + " |\n" + "|---" * len(names) + "|\n"
    body = ""
    for arm, label in rows:
        rs = evals.get(arm, [])
        if not rs:
            continue
        cells = [cell([r[c] for r in rs if c in r]) for c in cols]
        body += "| " + " | ".join([label, *cells, cell([r['all'] for r in rs]), str(len(rs))]) + " |\n"
    return head + body


def smooth(x, w=5):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), c])


def plot(curves, out: Path, x_key="iteration"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("success", "Training success rate (5-iter moving avg)"),
              ("entropy_mean", "Mean step entropy H_t (nats)"),
              ("revisit_rate", "Revisit rate (5-iter moving avg)"),
              ("any_success_frac", "P(group has ≥1 success) (5-iter moving avg)")]
    smoothed = {"success", "revisit_rate", "any_success_frac"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    axes = axes.ravel()
    arms = [(a, l) for a, l in with_extra_arms(list(ARMS), curves) if a in curves]
    spare = iter(c for c in PALETTE if c not in {COLORS.get(a) for a, _ in arms})
    colors = {a: COLORS.get(a) or next(spare, "#52514e") for a, _ in arms}
    for ax, (key, title) in zip(axes, panels):
        for arm, label in arms:
            runs = [r for r in curves[arm] if r and key in r[0]
                    and all(x_key in m for m in r)]
            if not runs:
                continue
            lo = max(r[0][x_key] for r in runs)
            hi = min(r[-1][x_key] for r in runs)
            if lo > hi:
                continue
            it = np.linspace(lo, hi, min(len(r) for r in runs))
            y = np.array([np.interp(it, [m[x_key] for m in r],
                                    [m[key] for m in r]) for r in runs])
            mean = smooth(y.mean(0)) if key in smoothed else y.mean(0)
            ax.plot(it, mean, color=colors[arm], lw=2, label=f"{label} (n={len(runs)})")
            if len(runs) > 1:
                sd = y.std(0) / np.sqrt(len(runs))
                sd = smooth(sd) if key in smoothed else sd
                ax.fill_between(it, mean - sd, mean + sd, color=colors[arm], alpha=0.12, lw=0)
        ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b")
        ax.set_xlabel({"iteration": "iteration", "env_steps_total": "environment steps",
                       "response_tokens_total": "generated response tokens",
                       "seconds_total": "training seconds"}[x_key], color="#52514e")
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b5b4ad")
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.xaxis.get_major_locator().set_params(integer=True)
        if key in smoothed:
            ax.set_ylim(0, 1)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def comparisons(evals, baseline="grpo"):
    head = ("### Paired evaluation differences\n\n"
            "Intervals resample paired tasks and seeds; one seed measures task variability only. "
            "Few seeds give uncertain intervals. Degenerate intervals cannot establish equivalence. "
            "Screening and selected variants need fresh confirmation.\n\n"
            "| Candidate vs baseline | delta (pp) | 95% interval (pp) | seeds | tasks |\n"
            "|---|---:|---|---:|---:|\n")
    summary = {}
    if baseline not in evals:
        return head + f"\nNo {baseline} evaluations available.\n", summary
    for arm in sorted(evals):
        if arm in ("base", baseline):
            continue
        try:
            row = paired_comparison(evals[baseline], evals[arm])
            summary[arm] = row
            lo, hi = row["ci95"]
            qualifier = " (degenerate)" if row["degenerate"] else ""
            head += (f"| {arm} vs {baseline} | {100 * row['delta']:+.1f} | "
                     f"[{100 * lo:+.1f}, {100 * hi:+.1f}]{qualifier} | {len(row['seeds'])} | {row['n_tasks']} |\n")
        except ValueError as exc:
            summary[arm] = {"unavailable": str(exc)}
            head += f"| {arm} vs {baseline} | unavailable: {exc} | | | |\n"
    return head, summary


def explore_table(curves, last: int = 5) -> str:
    """Mean of each exploration metric over the last ``last`` iterations, averaged over seeds."""
    names = ["Method", *(label for _, label in EXPLORE_COLS)]
    out = "| " + " | ".join(names) + " |\n" + "|---" * len(names) + "|\n"
    for arm, label in with_extra_arms(list(ARMS), curves):
        runs = curves.get(arm)
        if not runs:
            continue
        cells = []
        for key, _ in EXPLORE_COLS:
            vals = []
            for r in runs:
                values = np.array([m.get(key, np.nan) for m in r[-last:]])
                finite = values[np.isfinite(values)]
                if len(finite):
                    vals.append(finite.mean())
            cells.append(f"{np.mean(vals):.2f}" if vals else "–")
        out += f"| {label} | " + " | ".join(cells) + " |\n"
    return out


def load_lengths(runs: Path, split: str, seeds=None) -> dict:
    """{arm: {phase: raw lengths}} from the resp_lengths_* files, pooled over seeds."""
    pooled = defaultdict(lambda: defaultdict(
        lambda: {"len": [], "truncated": [], "valid": [], "cap": set()}))

    def add(arm, phase, rec):
        d = pooled[arm][phase]
        for k in ("len", "truncated", "valid"):
            d[k] += rec[k]
        d["cap"].add(rec["cap"])

    for d, arm in run_dirs(runs, seeds):
        t = d / "resp_lengths_train.jsonl"
        if t.exists():
            recs = [json.loads(line) for line in t.open()]
            for rec in recs:
                add(arm, "train", rec)
            for rec in recs[-5:]:
                add(arm, "train, last 5 it", rec)
        e = d / f"resp_lengths_eval_{split}.json"
        if e.exists():
            add(arm, "eval", json.loads(e.read_text()))
    return pooled


def lengths_table(pooled: dict) -> str:
    names = ["Method", "phase", "responses", "mean", "p50", "p90", "p99", "p99.9", "max", "cap",
             "% truncated", "% of truncated invalid"]
    out = "| " + " | ".join(names) + " |\n" + "|---" * len(names) + "|\n"
    rows = with_extra_arms([("base", "Qwen2.5-1.5B-Instruct (zero-shot)"), *ARMS], pooled)
    for arm, label in rows:
        for phase, raw in pooled.get(arm, {}).items():
            st = length_stats(raw)
            if not st:
                continue
            cap = "/".join(str(c) for c in sorted(raw["cap"]))
            out += (f"| {label} | {phase} | {st['n']} | {st['mean']:.0f} | {st['p50']:.0f} "
                    f"| {st['p90']:.0f} | {st['p99']:.0f} | {st['p99.9']:.0f} | {st['max']} "
                    f"| {cap} | {100 * st['trunc_rate']:.2f} | {100 * st['trunc_invalid']:.0f} |\n")
    return out


def write_run_comparison(runs: Path, split: str, out: Path, arms=None, seeds=None):
    """One row per seed, with observed training cost (startup/evaluation excluded)."""
    rows = []
    for d, arm in run_dirs(runs, seeds):
        if arms and arm not in arms and arm != "base":
            continue
        evaluation = d / f"eval_{split}.json"
        metrics = d / "metrics.jsonl"
        ev = json.loads(evaluation.read_text()) if evaluation.exists() else {}
        records = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()] if metrics.exists() else []
        if not ev and not records:
            continue
        last = records[-1] if records else {}
        row = {"method": arm, "seed": ev.get("seed", d.name.rsplit("_s", 1)[-1]) if arm != "base" else "",
               "status": "evaluated" if ev else "not evaluated",
               "success_pct": 100 * ev["all"] if "all" in ev else "",
               "eval_tasks": len(ev.get("per_task", [])) or sum(ev.get("n", {}).values()) or "",
               "iterations": last.get("iteration", ""),
               "env_steps": last.get("env_steps_total", ""),
               "response_tokens": last.get("response_tokens_total", ""),
               "training_seconds": last.get("seconds_total", "")}
        rows.append(row)
    fields = ["method", "seed", "status", "success_pct", "eval_tasks", "iterations",
              "env_steps", "response_tokens", "training_seconds"]
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    text = ("### Individual runs\n\nTraining time includes rollout and updates, excluding model startup, "
            "checkpoint saves and evaluation. Missing evaluations are not zero successes.\n\n"
            "| Method | Seed | Status | Success (%) | Eval tasks | Iterations | Env steps | Training (h) |\n"
            "|---|---:|---|---:|---:|---:|---:|---:|\n")
    for r in rows:
        success = f"{r['success_pct']:.1f}" if r["success_pct"] != "" else "—"
        hours = f"{r['training_seconds'] / 3600:.2f}" if r["training_seconds"] != "" else "—"
        text += (f"| {r['method']} | {r['seed']} | {r['status']} | {success} | {r['eval_tasks']} | "
                 f"{r['iterations']} | {r['env_steps']} | {hours} |\n")
    (out / "runs.md").write_text(text, encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/reduced")
    ap.add_argument("--split", default="valid_unseen")
    ap.add_argument("--out", default="results")
    ap.add_argument("--arms", nargs="+", help="only these methods (plus base)")
    ap.add_argument("--seeds", nargs="+", type=int, help="only these training seeds (plus base)")
    ap.add_argument("--baseline", default="grpo", help="baseline for paired task/seed comparisons")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    evals, curves = load(Path(args.runs), args.split, args.seeds)
    if args.arms:
        evals = {a: v for a, v in evals.items() if a in args.arms or a == "base"}
        curves = {a: v for a, v in curves.items() if a in args.arms}

    main_rows = with_extra_arms([("base", "Qwen2.5-1.5B-Instruct (zero-shot)"), ARMS[0], ARMS[1]],
                                evals)
    t1 = f"### ALFWorld `{args.split}` success rate (%)\n\n" + table(evals, main_rows, SUBTASKS)
    t2 = f"### Ablation, ALFWorld `{args.split}` success rate (%)\n\n" + table(evals, list(ARMS), [])
    (out / "table_alfworld.md").write_text(t1)
    (out / "ablation.md").write_text(t2)
    t3 = "### Exploration during training (mean of last 5 iterations)\n\n" + explore_table(curves)
    (out / "exploration.md").write_text(t3)
    lengths = load_lengths(Path(args.runs), args.split, args.seeds)
    if args.arms:
        lengths = {a: v for a, v in lengths.items() if a in args.arms or a == "base"}
    t4 = ("### Response lengths (tokens) vs the `max_new_tokens` cap\n\n"
          "A cap is free only if (almost) no response reaches it. While `% truncated` is near 0, "
          "p99 and p99.9 show where a lower cap would start cutting responses off.\n\n"
          + lengths_table(lengths))
    (out / "response_lengths.md").write_text(t4)
    (out / "summary.json").write_text(json.dumps(evals, indent=2))
    paired_md, paired = comparisons(evals, args.baseline)
    (out / "paired_comparisons.md").write_text(paired_md)
    (out / "paired_comparisons.json").write_text(json.dumps(paired, indent=2))
    all_rows = with_extra_arms([("base", "Zero-shot base"), *ARMS], evals)
    individual = write_run_comparison(Path(args.runs), args.split, out, args.arms, args.seeds)
    combined = (f"# ALFWorld `{args.split}` benchmark comparison\n\n"
                "Success rates are percentages; ± is sample standard deviation across training seeds. "
                "Paired intervals below are unadjusted for multiple comparisons. "
                "Selecting a winner among many methods needs fresh confirmation.\n\n"
                + table(evals, all_rows, SUBTASKS) + "\n" + paired_md + "\n" + individual)
    (out / "comparison.md").write_text(combined, encoding="utf-8")
    if curves:
        plot(curves, out / "training_curves.png")
        for x in ("env_steps_total", "response_tokens_total", "seconds_total"):
            if any(r and x in r[0] for runs in curves.values() for r in runs):
                plot(curves, out / f"training_curves_{x}.png", x_key=x)
    print(t1 + "\n" + t2 + "\n" + t3 + "\n" + t4)


if __name__ == "__main__":
    main()
