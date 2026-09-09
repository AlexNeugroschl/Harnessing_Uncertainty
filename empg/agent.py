"""The agent: an LLM scores the options, a small trainable head decides.

Why the split? EMPG is a *training* method, but Ollama serves a frozen model
and cannot backpropagate. So the demo separates the two roles:

* **Ollama (frozen)** reads the observation and scores the candidate actions.
  We present them as a labelled multiple choice ("A) ... B) ...") and read the
  probability distribution over the *label token*. That gives two things at
  once: a per-action prior, and the step entropy H_t (Eq. 7) of the decision
  itself. All uncertainty in this demo is measured from a real LLM.
* **A small torch head (trainable)** combines the LLM prior with features of
  the action and observation, and is the thing the policy gradient updates.

Measuring entropy over the *choice* rather than over free-form prose matters.
Probing both on this setup, prose entropy was dominated by phrasing noise
(the 100%-accurate `take` step scored the *highest* entropy), whereas
choice entropy tracks competence the way EMPG assumes:

    final  100% acc, H=0.25     take   100% acc, H=0.28
    goto    40% acc, H=0.77     search  67% acc, H=1.08

Confident steps are the correct ones, and the genuinely ambiguous
"which container?" step carries ~4x the entropy. That spread is what Eq. 12
normalises and Eq. 10 turns into a gradient weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .env import TextHouseholdEnv
from .llm import ChoiceResult, OllamaClient

SYSTEM_PROMPT = (
    "You are an expert agent in a text-based household game. "
    "Answer with a single letter only."
)


def build_choice_prompt(goal: str, observation: str, actions: list[str]) -> list[dict]:
    listing = "\n".join(f"{chr(65 + i)}) {a}" for i, a in enumerate(actions))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {goal}\n"
                f"Observation: {observation}\n"
                f"Options:\n{listing}\n\n"
                "Best action? Reply with one letter."
            ),
        },
    ]


# --- features ---------------------------------------------------------------

VOCAB = [
    "go", "open", "take", "put", "clean", "heat", "unlock", "close", "drop",
    "shake", "look", "counter", "sink", "desk", "fridge", "cabinet", "drawer",
    "trunk", "shelf", "box", "key", "mug", "bowl", "table", "pantry",
    "microwave", "rug", "with", "on", "in", "under", "1", "2", "3",
]
VOCAB_INDEX = {w: i for i, w in enumerate(VOCAB)}
# bag-of-words(action) + bag-of-words(observation) + [llm_prior, llm_is_top]
FEATURE_DIM = len(VOCAB) * 2 + 2


def featurize(
    action: str, observation: str, llm_prior: float, is_top: bool
) -> list[float]:
    """Hand-rolled features: no transformers dependency, fully inspectable.

    The LLM's probability for this action is passed in as a feature, so the
    head can learn *how much* to trust the frozen model rather than being
    forced to either obey it or ignore it.
    """
    vec = [0.0] * FEATURE_DIM
    for w in action.lower().split():
        if w in VOCAB_INDEX:
            vec[VOCAB_INDEX[w]] = 1.0
    for w in observation.lower().split():
        w = w.strip(".,")
        if w in VOCAB_INDEX:
            vec[len(VOCAB) + VOCAB_INDEX[w]] = 1.0
    vec[-2] = float(llm_prior)
    vec[-1] = 1.0 if is_top else 0.0
    return vec


class PolicyHead(nn.Module):
    """Scores each candidate action; a softmax over those scores is the policy.

    Tiny on purpose (~a few thousand parameters) so it trains on CPU in
    seconds and the learning curve reflects the advantage weighting rather
    than optimiser exotica.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Mild warm start toward the LLM prior. Calibration matters here, and
        # both extremes destroy the experiment:
        #   * no warm start at all -> the head is near-uniform, tasks almost
        #     never complete, every rollout in a group scores 0, and Eq. 4
        #     yields A_i = 0 for the whole group: no signal to learn from.
        #   * too strong a warm start -> every rollout succeeds, A_i = 0 again
        #     for the opposite reason, and there is no headroom to improve.
        # Measured on this setup, this setting starts training around 60-70%
        # success, which leaves both variance within a group and room to climb.
        with torch.no_grad():
            self.net[0].weight[:, -2] += 0.35

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (n_actions, feature_dim) -> logits: (n_actions,)"""
        return self.net(features).squeeze(-1)

    def action_log_probs(self, features: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.forward(features), dim=-1)


@dataclass
class Step:
    """One reason-then-act cycle, with everything EMPG needs to weight it."""

    features: list[list[float]]   # (n_actions, feature_dim)
    action_index: int
    entropy: float                # H_t, measured from the LLM (Eq. 7)
    correct: bool
    kind: str
    action_text: str
    llm_probs: list[float]


@dataclass
class Trajectory:
    steps: list[Step]
    reward: float                 # sparse: 1.0 on success, else 0.0
    success: bool
    task: str


def rollout(
    env: TextHouseholdEnv,
    client: OllamaClient,
    head: PolicyHead | None = None,
    temperature: float = 1.0,
    greedy: bool = False,
) -> Trajectory:
    """Run one episode, recording per-step entropy and features.

    Stochasticity comes from sampling the head's action distribution, not from
    the frozen LLM (whose per-state scores are cached and identical for both
    training arms).
    """
    env.reset()
    steps: list[Step] = []
    total_reward = 0.0

    while not env.done:
        actions = env.valid_actions()
        observation = env.observation()

        scored: ChoiceResult = client.score_choices(
            build_choice_prompt(env.task.goal, observation, actions),
            n_options=len(actions),
        )
        priors = scored.probs
        top = max(range(len(actions)), key=lambda i: priors[i])

        feats = [
            featurize(a, observation, priors[i], is_top=(i == top))
            for i, a in enumerate(actions)
        ]

        if head is None:
            choice = top
        else:
            with torch.no_grad():
                logits = head(torch.tensor(feats, dtype=torch.float32))
                if greedy:
                    choice = int(torch.argmax(logits).item())
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    choice = int(torch.multinomial(probs, 1).item())

        _, reward, _, info = env.step(actions[choice])
        total_reward += reward

        steps.append(
            Step(
                features=feats,
                action_index=choice,
                entropy=scored.entropy,
                correct=bool(info.get("correct", False)),
                kind=info.get("kind", "?"),
                action_text=actions[choice],
                llm_probs=priors,
            )
        )

    return Trajectory(
        steps=steps, reward=total_reward, success=env.success, task=env.task.name
    )
