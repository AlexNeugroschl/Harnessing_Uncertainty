"""Observable behavior summaries, independent of reward and regime labels."""
from collections import Counter


def behavior_metrics(trajectories):
    steps = [s for tr in trajectories for s in tr.steps]
    if not any(s.diagnostics for s in steps):
        return {}
    actions, failures = Counter(), Counter()
    repeated = 0
    for tr in trajectories:
        previous_failed = None
        for step in tr.steps:
            actions[step.action] += 1
            failed = step.diagnostics.get("failed", False)
            if failed:
                failures[step.action] += 1
                repeated += previous_failed == step.action
            previous_failed = step.action if failed else None
    return {"action_counts": dict(actions), "failed_action_counts": dict(failures),
        "repeated_failed_actions": repeated,
        "tool_selection_episodes": sum(any(s.diagnostics.get("tool_selected") for s in tr.steps) for tr in trajectories),
        "harvesting_episodes": sum(any(s.diagnostics.get("successful_harvest") for s in tr.steps) for tr in trajectories),
        "harvested_items": sum(s.diagnostics.get("harvested_items", 0) for s in steps),
        "crafted_items": sum(s.diagnostics.get("crafted_items", 0) for s in steps),
        "max_component_depth": max((s.diagnostics.get("component_depth", 0) for s in steps), default=0)}


def behavior_trace(trajectory):
    return {"task": trajectory.task, "won": trajectory.won,
            "steps": [{"action": s.action, "valid": s.valid, "observation": s.obs,
                       "diagnostics": s.diagnostics} for s in trajectory.steps]}
