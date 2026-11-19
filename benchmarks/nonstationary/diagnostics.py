"""Exact sequence probabilities on a finite catalog of native action responses.

Catalog probabilities are conditional on emitting one of these exact responses.
They are diagnostic only; training still samples the unrestricted policy at T=1.
"""
from __future__ import annotations

import numpy as np


def catalog_distribution(log_probabilities):
    values = np.asarray(log_probabilities, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty sequence log probabilities are required")
    maximum = values.max()
    normalizer = maximum + np.log(np.exp(values - maximum).sum())
    conditional = np.exp(values - normalizer)
    return {"catalog_mass": float(np.exp(normalizer)),
            "conditional": conditional.tolist(),
            "conditional_entropy": float(-(conditional * (values - normalizer)).sum())}


def decision_prompts(bench, tasks, *, regime="baseline"):
    # Make new native environments without modifying the caller's intervention.
    original = bench.intervention
    prompts = []
    try:
        bench.intervention = {"regime": regime, "change": "rules", "feedback": "generic"}
        for task in tasks:
            env = bench.make_env(task)
            try:
                obs, info = env.reset()
                history = []
                for action in ("approach_log", "craft_planks" if regime == "baseline" else "break_block"):
                    previous = obs
                    obs, _, done, info = env.step(action)
                    history.append((previous, action))
                    if done:
                        raise RuntimeError("decision probe unexpectedly completed the task")
                prompts.append((task, bench.prompt(info["task"], obs, info["admissible"], history), info["admissible"]))
            finally:
                env.close()
    finally:
        bench.intervention = original
    return prompts


def action_diagnostics(policy, bench, tasks, *, regime="baseline"):
    if not tasks:
        raise ValueError("a nonempty decision-probe task panel is required")
    pairs, metadata = [], []
    for task, prompt, actions in decision_prompts(bench, tasks, regime=regime):
        prompt_ids = policy.encode_prompt(prompt)
        metadata.append((task, list(actions), len(pairs)))
        for action in actions:
            response = policy.tokenizer("action: " + action, add_special_tokens=False)["input_ids"]
            pairs.append((prompt_ids, response + [policy.tokenizer.eos_token_id]))
    scores, _ = policy.score([p for p, _ in pairs], [r for _, r in pairs], with_entropy=False)
    logps = [float(lp.double().sum()) for lp in scores]
    records = []
    for task, actions, offset in metadata:
        values = logps[offset:offset + len(actions)]
        distribution = catalog_distribution(values)
        probabilities = dict(zip(actions, distribution.pop("conditional")))
        records.append({"task": task, "regime": regime, **distribution,
                        "conditional_action_probabilities": probabilities,
                        "sequence_log_probabilities": dict(zip(actions, values))})
    return {"definition": "P(exact action response including EOS | one listed response emitted); not the unrestricted action distribution",
            "regime": regime, "tasks": records,
            "mean_tool_probability": float(np.mean([r["conditional_action_probabilities"]["select_axe"] for r in records])),
            "mean_catalog_mass": float(np.mean([r["catalog_mass"] for r in records])),
            "mean_conditional_entropy": float(np.mean([r["conditional_entropy"] for r in records]))}
