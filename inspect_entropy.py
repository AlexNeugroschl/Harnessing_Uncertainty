"""Show, step by step, how EMPG re-weights credit within a trajectory.

This is the most direct view of the paper's mechanism: one group of rollouts,
one table, with the raw measured entropy alongside every intermediate quantity
from Eq. 8-13.

    python inspect_entropy.py [--task clean_mug] [--group 8]
"""

from __future__ import annotations

import argparse

import numpy as np

from empg.empg import (
    DEFAULT_K,
    DEFAULT_ZETA,
    broadcast_advantages,
    clarity_bonus_f,
    finalize_advantages,
    gradient_scale_g,
    group_advantages,
    normalize_entropy,
    successor_entropy,
)
from empg.llm import OllamaClient
from empg.train import TrainConfig, collect_group, flatten, train
from empg.agent import PolicyHead


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="clean_mug")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=6,
                    help="train this many iterations first, so the group "
                         "contains a mix of successes and failures")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    client = OllamaClient(**({"model": args.model} if args.model else {}))
    if not client.is_available():
        raise SystemExit("Ollama is not reachable. Start it with `ollama serve`.")

    cfg = TrainConfig(group_size=args.group, iterations=args.warmup)
    if args.warmup:
        # An untrained head fails nearly every episode, and a group where all
        # rewards are equal gives A_i = 0 everywhere (Eq. 4) - interesting in
        # its own right, but it hides the A_i * g(H) term. A short warm-up
        # produces a group with both successes and failures.
        print(f"Warming up for {args.warmup} iterations ...")
        head, _ = train("empg", client, cfg, tasks=[args.task], verbose=False)
    else:
        head = PolicyHead()

    print(f"\nRolling out {args.group} episodes of '{args.task}' ...")
    group = collect_group(client, head, args.task, cfg)

    entropies, done, lengths = flatten(group)
    rewards = np.array([t.reward for t in group])

    traj_adv = group_advantages(rewards)             # Eq. 4
    step_adv = broadcast_advantages(traj_adv, lengths)
    h_norm = normalize_entropy(entropies)            # Eq. 12
    h_next = successor_entropy(h_norm, done)
    g = gradient_scale_g(h_norm, k=DEFAULT_K)        # Eq. 10
    f = clarity_bonus_f(h_next)                      # Eq. 11
    a_empg = finalize_advantages(step_adv * g + DEFAULT_ZETA * f)   # Eq. 8 + 13
    a_grpo = finalize_advantages(step_adv)

    print(
        f"\nGroup: {int(rewards.sum())}/{len(rewards)} succeeded  "
        f"(reward is sparse: +1 only at task completion)\n"
    )
    header = (
        f"{'traj':>4} {'st':>3} {'phase':<7} {'ok':<3} {'H_t':>6} {'Hnorm':>6} "
        f"{'g(H)':>6} {'f(H+1)':>7} {'A_GRPO':>7} {'A_EMPG':>7}  action"
    )
    print(header)
    print("-" * len(header))

    idx = 0
    for ti, traj in enumerate(group):
        for si, step in enumerate(traj.steps):
            note = ""
            if g[idx] > 1.15:
                note = "  <- confident: amplified"
            elif g[idx] < 0.85:
                note = "  <- uncertain: damped"
            print(
                f"{ti:>4} {si:>3} {step.kind:<7} {'Y' if step.correct else 'n':<3} "
                f"{entropies[idx]:>6.3f} {h_norm[idx]:>6.3f} {g[idx]:>6.3f} "
                f"{f[idx]:>7.3f} {a_grpo[idx]:>+7.3f} {a_empg[idx]:>+7.3f}  "
                f"{step.action_text}{note}"
            )
            idx += 1
        print("-" * len(header))

    # --- what the table is meant to show -----------------------------------
    print("\nSummary")
    print(f"  mean g(H) over batch      = {g.mean():.6f}   (Eq. 10 forces this to 1)")
    print(f"  entropy range             = {entropies.min():.3f} .. {entropies.max():.3f} nats")

    by_kind: dict[str, list[float]] = {}
    for traj in group:
        for step in traj.steps:
            by_kind.setdefault(step.kind, []).append(step.entropy)
    print("  mean H by phase:")
    for kind, vals in sorted(by_kind.items(), key=lambda kv: np.mean(kv[1])):
        print(f"    {kind:<8} {np.mean(vals):.3f}  (n={len(vals)})")

    succeeded = np.repeat(rewards > 0, lengths)
    if succeeded.any() and (~succeeded).any():
        print(
            f"\n  Confident steps in FAILED trajectories are pushed further "
            f"negative than uncertain ones - that is the "
            f"'penalise confident errors' term:"
        )
        fail_conf = a_empg[(~succeeded) & (h_norm < 0.35)]
        fail_unc = a_empg[(~succeeded) & (h_norm > 0.65)]
        if len(fail_conf) and len(fail_unc):
            print(f"    mean A_EMPG, confident+failed = {fail_conf.mean():+.3f}")
            print(f"    mean A_EMPG, uncertain+failed = {fail_unc.mean():+.3f}")
    else:
        print(
            "\n  (All trajectories share the same outcome, so Eq. 4 gives "
            "A_i = 0 and only the clarity bonus differentiates steps - "
            "exactly the case where GRPO alone learns nothing.)"
        )


if __name__ == "__main__":
    main()
