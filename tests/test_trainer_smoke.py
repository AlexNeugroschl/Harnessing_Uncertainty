"""End-to-end check of rollout -> entropy -> advantages -> update, on CPU.

Uses a scripted fake benchmark (no ALFWorld) and a tiny random Qwen2 model.
"""

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from empg.config import load_config  # noqa: E402
from empg.policy import LLMPolicy  # noqa: E402
from empg.rollout import run_episodes  # noqa: E402
from empg.trainer import update  # noqa: E402

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


class FakeEnv:
    """Won if the agent ever emits the action 'open door'; otherwise times out."""

    def __init__(self, task):
        self.task = task

    def reset(self):
        return "You see a door.", {"admissible": ["open door", "look"], "task": self.task}

    def step(self, action):
        won = action == "open door"
        return "ok", float(won), won, {"admissible": ["open door", "look"], "won": won}

    def close(self):
        pass


class FakeBench:
    name = "fake"

    def tasks(self, split):
        return ["open the door"]

    def make_env(self, task):
        return FakeEnv(task)

    def task_type(self, task):
        return "door"

    def prompt(self, task_desc, observation, admissible, history):
        return f"{task_desc}. {observation} Actions: {admissible}"


@pytest.fixture(scope="module")
def policy():
    cfg = load_config(None, [f"policy.model_name={TINY}", "policy.max_new_tokens=8",
                             "policy.gradient_checkpointing=false", "policy.lora_r=4"])
    try:
        return LLMPolicy(cfg.policy, device="cpu"), cfg
    except OSError as e:  # offline / model unavailable
        pytest.skip(f"cannot load {TINY}: {e}")


def test_score_matches_forward(policy):
    pol, _ = policy
    p, r, _ = pol.generate(["hello there"], greedy=True)
    lp, ent = pol.score(p, r)
    assert lp[0].shape == ent[0].shape == (len(r[0]),)
    assert (lp[0] <= 0).all() and (ent[0] >= 0).all()
    ref, _ = pol.score(p, r, with_entropy=False, reference=True)
    # Fresh LoRA (B = 0) is the identity, so policy == reference. (Runs before
    # the update tests, which share this module-scoped policy.)
    torch.testing.assert_close(ref[0], lp[0], atol=1e-4, rtol=0)


@pytest.mark.parametrize("method,curiosity", [("grpo", 0.0), ("empg", 0.0), ("empg", 0.1)])
def test_one_update(policy, method, curiosity):
    pol, cfg = policy
    cfg.advantage.method = method
    cfg.advantage.curiosity.weight = curiosity  # FakeEnv has no state keys -> "obs" fallback
    cfg.train.mini_batches = 2
    torch.manual_seed(0)
    trajs = run_episodes(pol, FakeBench(), ["open the door"] * 4, [0, 0, 1, 1], max_steps=3)
    assert all(1 <= len(t.steps) <= 3 for t in trajs)
    # Force a mixed-outcome group so GRPO has signal too.
    trajs[0].reward, trajs[1].reward = 10.0, 0.0

    before = [p.detach().clone() for p in pol.parameters()]
    stats = update(pol, torch.optim.AdamW(pol.parameters(), lr=1e-3), trajs, cfg,
                   np.random.default_rng(0))
    after = pol.parameters()

    assert all(np.isfinite(v) for k, v in stats.items() if k != "success_len")
    assert (stats["intrinsic_share"] > 0) == (curiosity > 0)
    assert stats["entropy_mean"] > 0
    assert all(s.entropy > 0 for t in trajs for s in t.steps)
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
