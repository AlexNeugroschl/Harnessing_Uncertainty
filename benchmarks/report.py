"""Aggregate runs into paper-style tables and training curves.

    python -m benchmarks.report [--runs runs] [--split valid_unseen] [--out results]

Writes:
    results/table_alfworld.md   per-subtask success (paper Table 1 layout)
    results/ablation.md         GRPO vs g-only vs f-only vs full EMPG (paper ablation)
    results/training_curves.png train success + mean step entropy per iteration
    results/summary.json        all numbers above
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SUBTASKS = ["Pick", "Look", "Clean", "Heat", "Cool", "Pick2"]
ARMS = [("grpo", "GRPO"), ("empg", "GRPO + EMPG"),
        ("empg_g_only", "+ gradient scaling g only"), ("empg_f_only", "+ clarity bonus f only")]
# Fixed categorical order (identity follows the arm, never its rank).
COLORS = {"grpo": "#2a78d6", "empg": "#eb6834", "empg_g_only": "#1baf7a", "empg_f_only": "#eda100"}


def load(runs: Path, split: str):
    evals, curves = defaultdict(list), defaultdict(list)
    for d in sorted(runs.iterdir()):
        if not d.is_dir():
            continue
        arm = "base" if d.name == "base" else d.name.rsplit("_s", 1)[0]
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
    return rows + [(a, a) for a in sorted(found) if a not in known]


def table(evals, rows, cols) -> str:
    names = ["Method", *cols, "All", "seeds"]
    head = "| " + " | ".join(names) + " |\n" + "|---" * len(names) + "|\n"
    body = ""
    for arm, label in rows:
        rs = evals.get(arm, [])
        if not rs:
            continue
        cells = [cell([r[c] for r in rs if c in r]) for c in cols]
        body += f"| {label} | " + " | ".join(cells) + f" | {cell([r['all'] for r in rs])} | {len(rs)} |\n"
    return head + body


def smooth(x, w=5):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), c])


def plot(curves, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("success", "Training success rate (5-iter moving avg)"),
              ("entropy_mean", "Mean step entropy H_t (nats)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    extra_colors = iter(["#e87ba4", "#008300", "#4a3aa7", "#e34948"])  # slots 5-8
    arms = with_extra_arms(list(ARMS), curves)
    colors = {**COLORS, **{a: next(extra_colors, "#52514e") for a, _ in arms if a not in COLORS}}
    for ax, (key, title) in zip(axes, panels):
        for arm, label in arms:
            runs = curves.get(arm)
            if not runs:
                continue
            n = min(len(r) for r in runs)
            y = np.array([[m[key] for m in r[:n]] for r in runs])
            mean = smooth(y.mean(0)) if key == "success" else y.mean(0)
            it = np.arange(1, n + 1)
            ax.plot(it, mean, color=colors[arm], lw=2, label=f"{label} (n={len(runs)})")
            if len(runs) > 1:
                sd = y.std(0) / np.sqrt(len(runs))
                sd = smooth(sd) if key == "success" else sd
                ax.fill_between(it, mean - sd, mean + sd, color=colors[arm], alpha=0.12, lw=0)
        ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b")
        ax.set_xlabel("iteration", color="#52514e")
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b5b4ad")
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.xaxis.get_major_locator().set_params(integer=True)
        if key == "success":
            ax.set_ylim(0, 1)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split", default="valid_unseen")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    evals, curves = load(Path(args.runs), args.split)

    main_rows = with_extra_arms([("base", "Qwen2.5-1.5B-Instruct (zero-shot)"), ARMS[0], ARMS[1]],
                                evals)
    t1 = f"### ALFWorld `{args.split}` success rate (%)\n\n" + table(evals, main_rows, SUBTASKS)
    t2 = f"### Ablation, ALFWorld `{args.split}` success rate (%)\n\n" + table(evals, list(ARMS), [])
    (out / "table_alfworld.md").write_text(t1)
    (out / "ablation.md").write_text(t2)
    (out / "summary.json").write_text(json.dumps(evals, indent=2))
    if curves:
        plot(curves, out / "training_curves.png")
    print(t1 + "\n" + t2)


if __name__ == "__main__":
    main()
