"""Run the EMPG demo: train a GRPO arm and an EMPG arm, compare, plot.

    python run_demo.py                    # GRPO vs EMPG
    python run_demo.py --ablation         # + g-only and f-only (paper Table 2)
    python run_demo.py --seeds 3          # average over 3 seeds, report spread

Both arms share seeds, rollout caching, features and optimiser, so the only
difference between them is the advantage weighting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from empg.llm import OllamaClient
from empg.train import TrainConfig, evaluate, train

RESULTS = Path("results")

LABELS = {
    "grpo": "GRPO (baseline)",
    "empg": "EMPG (full)",
    "empg_g": "EMPG: scaling only",
    "empg_f": "EMPG: clarity only",
}


def plot(curves: dict[str, list[list[float]]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"grpo": "#888888", "empg": "#2f6fdb",
              "empg_g": "#d08c1e", "empg_f": "#43976a"}

    for method, runs in curves.items():
        arr = np.array(runs)                      # (n_seeds, n_iters)
        mean = arr.mean(axis=0)
        x = np.arange(1, len(mean) + 1)
        # Draw the baseline dashed and on top so it stays legible where the
        # curves overlap - which, on this small task, is most of the run.
        is_base = method == "grpo"
        ax.plot(x, mean, label=LABELS.get(method, method),
                color=colors.get(method), linewidth=2.2 if is_base else 1.8,
                linestyle="--" if is_base else "-",
                marker="o", markersize=3.5, zorder=5 if is_base else 3)
        if arr.shape[0] > 1:
            se = arr.std(axis=0) / np.sqrt(arr.shape[0])
            ax.fill_between(x, mean - se, mean + se,
                            color=colors.get(method), alpha=0.15, linewidth=0)

    ax.set_xlabel("training iteration")
    ax.set_ylabel("success rate")
    ax.set_title("Sparse-reward text household task\n"
                 "entropy measured from a real LLM (llama3.2 via Ollama)",
                 fontsize=10)
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


def plot_entropy(client: OllamaClient, cfg: TrainConfig, path: Path) -> None:
    """Show the measured entropy distribution and the g(H) mapping it drives."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from empg.agent import PolicyHead
    from empg.empg import gradient_scale_g, normalize_entropy
    from empg.train import collect_group, flatten

    head = PolicyHead()
    steps_by_kind: dict[str, list[float]] = {}
    all_ent: list[float] = []
    from empg.env import TASKS
    for task in TASKS:
        group = collect_group(client, head, task.name, cfg)
        for traj in group:
            for s in traj.steps:
                steps_by_kind.setdefault(s.kind, []).append(s.entropy)
                all_ent.append(s.entropy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    kinds = sorted(steps_by_kind, key=lambda k: np.mean(steps_by_kind[k]))
    # matplotlib renamed `labels` to `tick_labels` in 3.9; support both.
    try:
        ax1.boxplot([steps_by_kind[k] for k in kinds], tick_labels=kinds)
    except TypeError:
        ax1.boxplot([steps_by_kind[k] for k in kinds], labels=kinds)
    ax1.set_title("measured step entropy $H_t$ by phase", fontsize=10)
    ax1.set_ylabel("$H_t$ (nats)")
    ax1.grid(alpha=0.25, axis="y", linewidth=0.6)

    ent = np.array(all_ent)
    h_norm = normalize_entropy(ent)
    g = gradient_scale_g(h_norm)
    order = np.argsort(h_norm)
    ax2.plot(h_norm[order], g[order], color="#2f6fdb", linewidth=2)
    ax2.axhline(1.0, color="#888", linestyle="--", linewidth=1)
    ax2.set_xlabel("$H_{norm}$  (Eq. 12)")
    ax2.set_ylabel("$g(H)$  (Eq. 10)")
    ax2.set_title("confident steps amplified, uncertain damped", fontsize=10)
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.text(0.03, max(g) * 0.95, "amplified", fontsize=8, color="#2f6fdb")
    ax2.text(0.62, min(g) * 1.05, "damped", fontsize=8, color="#888")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--ablation", action="store_true",
                    help="also run scaling-only and clarity-only (paper Table 2)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    client = OllamaClient(**({"model": args.model} if args.model else {}))
    if not client.is_available():
        raise SystemExit(
            "Ollama is not reachable at localhost:11434.\n"
            "Start it with `ollama serve` and pull the model:\n"
            "  ollama pull llama3.2"
        )

    methods = ["grpo", "empg"]
    if args.ablation:
        methods = ["grpo", "empg_g", "empg_f", "empg"]

    RESULTS.mkdir(exist_ok=True)
    curves: dict[str, list[list[float]]] = {m: [] for m in methods}
    finals: dict[str, list[float]] = {m: [] for m in methods}
    evals: dict[str, list[float]] = {m: [] for m in methods}

    for seed in range(args.seeds):
        cfg = TrainConfig(iterations=args.iterations, group_size=args.group, seed=seed)
        print(f"\n{'=' * 66}\nseed {seed}\n{'=' * 66}")
        for method in methods:
            head, log = train(method, client, cfg)
            curves[method].append(log.success_rate)
            # Average the last 3 iterations: single-iteration endpoints are
            # noisy at this group size.
            finals[method].append(float(np.mean(log.success_rate[-3:])))
            evals[method].append(evaluate(head, client, cfg))

    # --- report -------------------------------------------------------------
    print(f"\n{'=' * 66}\nRESULTS  ({args.seeds} seed(s), {args.iterations} iterations)\n{'=' * 66}")
    print(f"{'method':<22} {'final train':>14} {'greedy eval':>14}")
    print("-" * 66)
    base = None
    for method in methods:
        ft, ev = np.mean(finals[method]), np.mean(evals[method])
        if base is None:
            base = ft
        spread = ""
        if args.seeds > 1:
            spread = f" +/-{np.std(finals[method]):.1%}"
        delta = "" if method == "grpo" else f"   ({ft - base:+.1%} vs GRPO)"
        print(f"{LABELS.get(method, method):<22} {ft:>13.1%}{spread} {ev:>13.1%}{delta}")

    # Seed variance dominates here, so the fair comparison is paired: same
    # seed, both arms. An unpaired mean +/- sd can easily hide a real effect
    # (or manufacture one) at this sample size.
    if args.seeds > 1 and "grpo" in finals:
        base_runs = np.array(finals["grpo"])
        print("\npaired vs GRPO (same seed):")
        for method in methods:
            if method == "grpo":
                continue
            d = np.array(finals[method]) - base_runs
            sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            t = d.mean() / sem if sem > 0 else float("nan")
            print(
                f"  {LABELS.get(method, method):<22} {d.mean():+.1%}  "
                f"t={t:+.2f}  wins {(d > 0).sum()}/{len(d)}"
            )

    print("\nNote: this is a small policy head on a hand-built task. It")
    print("demonstrates the EMPG *mechanism* on real measured entropy; it is")
    print("not a reproduction of the paper's benchmark numbers. Below ~20")
    print("seeds the ablation ordering is not stable.")

    (RESULTS / "results.json").write_text(
        json.dumps(
            {"curves": curves, "finals": finals, "evals": evals,
             "iterations": args.iterations, "group_size": args.group,
             "seeds": args.seeds, "model": client.model},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  wrote {RESULTS / 'results.json'}")

    if not args.no_plot:
        plot(curves, RESULTS / "learning_curve.png")
        plot_entropy(client, TrainConfig(group_size=args.group),
                     RESULTS / "entropy_analysis.png")

    print(f"\nLLM calls: {client.n_calls} (cache hits: {client.n_cache_hits})")


if __name__ == "__main__":
    main()
