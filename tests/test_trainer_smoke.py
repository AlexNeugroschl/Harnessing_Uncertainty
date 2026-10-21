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


def test_chunked_policy_matches_dense_with_unequal_lengths(policy):
    pol, _ = policy
    prompts = [pol.encode_prompt("hello"), pol.encode_prompt("a longer prompt about a door")]
    responses = [[1, 2, 3], [4, 5]]
    params = pol.parameters()
    pol.model.eval()
    saved = pol.cfg.chunked_head
    try:
        pol.cfg.chunked_head = False
        lp, ent, mask = pol.forward_logprobs(prompts, responses, with_entropy=True)
        dense_lp, dense_ent = lp.detach(), ent.detach()
        expected = torch.autograd.grad(lp[mask].sum(), params, allow_unused=True)
        pol.cfg.chunked_head = True
        lp, ent, actual_mask = pol.forward_logprobs(prompts, responses, with_entropy=True)
        actual = torch.autograd.grad(lp[actual_mask].sum(), params, allow_unused=True)
        assert torch.equal(mask, actual_mask)
        torch.testing.assert_close(lp[mask], dense_lp[mask], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ent[mask], dense_ent[mask], atol=1e-5, rtol=1e-5)
        for a, b in zip(actual, expected):
            if b is not None:
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    finally:
        pol.cfg.chunked_head = saved


def test_generation_config_cannot_change_the_sampled_distribution(policy):
    """Rollouts must sample from the policy that is scored. Qwen's generation_config
    ships repetition_penalty=1.1, which generate() would otherwise apply."""
    pol, _ = policy
    gc = pol.model.generation_config
    old = gc.repetition_penalty
    try:
        gc.repetition_penalty = 1.0
        _, r_plain, _ = pol.generate(["hello hello hello", "open the door"], greedy=True)
        gc.repetition_penalty = 5.0
        _, r_pen, _ = pol.generate(["hello hello hello", "open the door"], greedy=True)
    finally:
        gc.repetition_penalty = old
    assert r_pen == r_plain


def test_merged_generation_matches_and_restores_weights(policy):
    pol, _ = policy
    lora = [p for n, p in pol.model.named_parameters() if "lora_B" in n]
    saved = [p.detach().clone() for p in lora]
    torch.manual_seed(0)
    for p in lora:  # non-zero adapter, so merging actually changes the weights
        p.data.normal_(0, 0.05)
    base = {n: p.detach().clone() for n, p in pol._lora_base_weights().items()}
    pol.cfg.merge_lora = False
    _, r_plain, _ = pol.generate(["hello there", "open the door"], greedy=True)
    pol.cfg.merge_lora = True
    _, r_merged, _ = pol.generate(["hello there", "open the door"], greedy=True)
    assert r_merged == r_plain
    for n, p in pol._lora_base_weights().items():
        assert torch.equal(p, base[n])  # bit-exact restore: the KL reference cannot drift
    assert not any(getattr(m, "merged", False) for m in pol.model.modules())
    for p, s in zip(lora, saved):
        p.data.copy_(s)


@pytest.mark.parametrize("method,curiosity", [("grpo", 0.0), ("empg", 0.0), ("empg", 0.1)])
@pytest.mark.parametrize("reward_mode", ["trajectory", "step"])
def test_one_update(policy, method, curiosity, reward_mode):
    pol, cfg = policy
    cfg.advantage.method = method
    cfg.advantage.curiosity.weight = curiosity  # FakeEnv has no state keys -> "obs" fallback
    cfg.train.mini_batches = 2
    cfg.train.reward_mode = reward_mode
    cfg.train.entropy_coef = .001 if reward_mode == "step" else 0.
    cfg.train.dual_clip = 3. if reward_mode == "step" else 0.
    torch.manual_seed(0)
    trajs = run_episodes(pol, FakeBench(), ["open the door"] * 4, [0, 0, 1, 1], max_steps=3)
    assert all(1 <= len(t.steps) <= 3 for t in trajs)
    # Force a mixed-outcome group so GRPO has signal too.
    trajs[0].reward, trajs[1].reward = 10.0, 0.0
    trajs[0].won = True

    before = [p.detach().clone() for p in pol.parameters()]
    stats = update(pol, torch.optim.AdamW(pol.parameters(), lr=1e-3), trajs, cfg,
                   np.random.default_rng(0))
    after = pol.parameters()

    assert all(np.isfinite(v) for k, v in stats.items() if k != "success_len")
    assert (stats["intrinsic_share"] > 0) == (curiosity > 0)
    assert stats["entropy_mean"] > 0
    assert all(s.entropy > 0 for t in trajs for s in t.steps)
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    assert stats["resp_len_max"] <= pol.cfg.max_new_tokens
    assert 0 <= stats["trunc_rate"] <= 1


def test_length_stats():
    from empg.rollout import length_stats
    st = length_stats({"len": [10, 20, 256, 256], "truncated": [0, 0, 1, 1],
                       "valid": [1, 1, 0, 1]}, cap=256)
    assert st["n"] == 4 and st["max"] == 256 and st["cap"] == 256
    assert st["trunc_rate"] == 0.5 and st["trunc_invalid"] == 0.5
    assert length_stats({"len": [], "truncated": [], "valid": []}) == {}


def test_truncated_flag(policy):
    pol, _ = policy
    cap, eos = pol.cfg.max_new_tokens, pol.tokenizer.eos_token_id
    assert pol.truncated([1] * cap)
    assert not pol.truncated([1] * (cap - 1) + [eos])
    assert not pol.truncated([1, eos])


def test_training_rejects_transformed_sampling(policy):
    pol, cfg = policy
    saved = pol.cfg.temperature
    try:
        pol.cfg.temperature = 0.7
        with pytest.raises(ValueError, match="on-policy"):
            update(pol, None, [], cfg, np.random.default_rng(0))
    finally:
        pol.cfg.temperature = saved


def test_adapter_save_is_complete_without_a_hub_config_probe(policy, tmp_path, monkeypatch):
    from peft import get_peft_model_state_dict
    from peft.utils import load_peft_weights, save_and_load

    pol, _ = policy

    def unexpected_network(*args, **kwargs):
        raise AssertionError("adapter saving must not probe the Hub")

    monkeypatch.setattr(save_and_load, "check_file_exists_on_hf_hub", unexpected_network)
    expected = get_peft_model_state_dict(pol.model, save_embedding_layers=False)
    pol.save(str(tmp_path))
    actual = load_peft_weights(str(tmp_path), device="cpu")
    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
