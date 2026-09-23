"""ALFWorld prompt and action parsing, following verl-agent (the paper's framework).

Each turn the agent reasons inside <think></think> and then acts inside
<action></action>. That whole response is one "step" whose mean token entropy is
H_t (paper Eq. 7).
"""

from __future__ import annotations

import re

ALFWORLD_TEMPLATE_NO_HIS = """You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {observation}
Your admissible actions of the current situation are: [{admissible}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

ALFWORLD_TEMPLATE = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {history}
You are now at step {current_step} and your current observation is: {observation}
Your admissible actions of the current situation are: [{admissible}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def alfworld_prompt(task: str, observation: str, admissible: list[str],
                    history: list[tuple[str, str]], history_length: int = 2) -> str:
    """Build the user prompt. ``history`` is [(observation, action), ...] so far."""
    adm = ", ".join(f"'{a}'" for a in admissible if a not in ("help",))
    if not history:
        return ALFWORLD_TEMPLATE_NO_HIS.format(observation=observation, admissible=adm)
    recent = history[-history_length:]
    start = len(history) - len(recent)
    lines = "\n".join(
        f"[Observation {start + i + 1}: '{o}', Action {start + i + 1}: '{a}']"
        for i, (o, a) in enumerate(recent)
    )
    return ALFWORLD_TEMPLATE.format(
        task=task, step_count=len(history), history_length=len(recent), history=lines,
        current_step=len(history) + 1, observation=observation, admissible=adm,
    )


def parse_action(response: str) -> tuple[str, bool]:
    """Return (action, valid). Valid means one <think> block and one <action> block.

    Invalid responses still yield a best-effort action string (the env answers
    "Nothing happens."), and the step incurs the invalid-action penalty.
    """
    actions = _ACTION_RE.findall(response)
    valid = len(actions) == 1 and _THINK_RE.search(response) is not None
    if actions:
        action = actions[-1]
    else:
        action = response[-40:]
    return action.strip().lower(), valid
