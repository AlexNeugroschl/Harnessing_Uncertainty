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
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def visited_locations(history: list[tuple[str, str]], observation: str) -> list[str]:
    """Locations reached by a successful ``go to`` so far, in first-visit order.

    ``history[i]`` is (observation before action i, action i); the result of
    action i is the next entry's observation, or ``observation`` for the last.
    """
    visited = []
    for i, (_, action) in enumerate(history):
        result = history[i + 1][0] if i + 1 < len(history) else observation
        if action.startswith("go to ") and result.startswith("You arrive"):
            loc = action[len("go to "):]
            if loc not in visited:
                visited.append(loc)
    return visited


def alfworld_prompt(task: str, observation: str, admissible: list[str],
                    history: list[tuple[str, str]], history_length: int = 2,
                    visited: list[str] | None = None) -> str:
    """Build the user prompt. ``history`` is [(observation, action), ...] so far.

    ``visited`` (memory baseline, not in verl-agent) adds a line listing every
    location visited this episode, which the 2-step history otherwise forgets.
    """
    adm = ", ".join(f"'{a}'" for a in admissible if a not in ("help",))
    if visited:
        observation = f"{observation}\nLocations you have already visited: {', '.join(visited)}."
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


def parse_action(response: str, mode: str = "strict") -> tuple[str, bool]:
    """Return (action, valid). Valid means one <think> block and one <action> block.

    Invalid responses still yield a best-effort action string (the env answers
    "Nothing happens."), and the step incurs the invalid-action penalty.
    """
    if mode == "verl":
        # Behavior of the Apache-2.0 Verl-Agent ALFWorld projection, pinned to
        # 29e3e8b3e3412aff53b1e4a461dac77a8eec19bd (2025-09-03).
        lowered = response.lower()
        start, end = lowered.find("<action>"), lowered.find("</action>")
        if start < 0 or end < 0:
            return lowered[-30:], False
        action = lowered[start + len("<action>"):end].strip()
        valid = ("<think>" in response and "</think>" in response
                 and _CJK_RE.search(response) is None)
        return action, valid
    if mode != "strict":
        raise ValueError("action parser must be 'strict' or 'verl'")
    actions = _ACTION_RE.findall(response)
    valid = len(actions) == 1 and _THINK_RE.search(response) is not None
    if actions:
        action = actions[-1]
    else:
        action = response[-40:]
    return action.strip().lower(), valid
