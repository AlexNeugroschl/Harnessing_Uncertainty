"""Execution changes must preserve scores, masking, optimizer work and gradients."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from empg.config import Config
from empg.policy import LLMPolicy, PolicyConfig, token_microbatches
from empg.rollout import Step, Trajectory
from empg.trainer import update


def tiny_policy():
    torch.manual_seed(3)
    policy = LLMPolicy.__new__(LLMPolicy)
    policy.model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=37, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, pad_token_id=0))
    policy.cfg = PolicyConfig(lora_r=0, micro_batch_size=2, gradient_checkpointing=False)
    policy.device, policy.use_lora, policy._plain_head = "cpu", False, True
    policy.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    policy.model.eval()
    return policy


def test_larger_score_batches_preserve_response_order_padding_and_entropy():
    policy = tiny_policy()
    prompts = [[1, 2], [1, 2, 3, 4], [2], [1, 2, 3]]
    responses = [[3, 4, 5], [6], [7, 8], [9, 10, 11, 12]]
    expected_lp, expected_ent = policy.score(prompts, responses)
    policy.cfg.score_batch_size = 3
    actual_lp, actual_ent = policy.score(prompts, responses)
    for a, b in zip(actual_lp + actual_ent, expected_lp + expected_ent):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("method", ["grpo", "empg"])
def test_analytic_head_and_larger_microbatches_preserve_full_update(method):
    policy = tiny_policy()
    cfg = Config(policy=policy.cfg)
    cfg.advantage.method = method
    cfg.train.kl_coef = 0.
    cfg.train.entropy_coef = .001
    cfg.train.reward_mode = "step"
    cfg.train.mini_batch_size = 3
    cfg.train.shuffle = False
    cfg.train.dual_clip = 3.
    trajs = [Trajectory("task", 0, [Step([1, 2], [3, 4], "", "look", True),
                                  Step([2, 3, 4], [5], "", "look", False)]),
             Trajectory("task", 0, [Step([1], [6, 7, 8], "", "look", True),
                                  Step([1, 2, 3, 4], [9, 10], "", "look", True)])]
    trajs[0].won = True
    for traj in trajs:
        traj.init_keys = {"obs": "initial"}
        for i, step in enumerate(traj.steps):
            step.state_keys = {"obs": f"state {i}"}
    initial = {k: v.detach().clone() for k, v in policy.model.state_dict().items()}
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    reference = update(policy, optimizer, trajs, cfg, np.random.default_rng(1))
    expected = {k: v.detach().clone() for k, v in policy.model.state_dict().items()}
    policy.model.load_state_dict(initial)
    pc = replace(policy.cfg, score_batch_size=4, micro_batch_size=3, micro_batch_tokens=12,
                 head_backend="analytic")
    policy.cfg = pc
    actual = update(policy, torch.optim.AdamW(policy.parameters(), lr=1e-4), trajs,
                    replace(cfg, policy=pc), np.random.default_rng(1))
    assert reference["optimizer_steps"] == actual["optimizer_steps"] == 2
    for k, v in policy.model.state_dict().items():
        torch.testing.assert_close(v, expected[k], atol=2e-6, rtol=1e-4)
    for key in ["pg_loss", "kl", "entropy_mean", "adv_std"]:
        assert actual[key] == pytest.approx(reference[key], abs=1e-5)


def test_token_budget_bounds_padding_without_dropping_or_reordering_rows():
    lengths = [3, 4, 9, 10, 11]
    batches = list(token_microbatches(range(5), lengths, 4, 20))
    assert batches == [[0, 1], [2, 3], [4]]
    assert [i for batch in batches for i in batch] == list(range(5))
    assert all(len(batch) * max(lengths[i] for i in batch) <= 20 for batch in batches)
    with pytest.raises(ValueError, match="sequence length"):
        list(token_microbatches([0], [21], 4, 20))


def test_selective_checkpointing_preserves_gradients_without_adding_hooks():
    policy = tiny_policy()
    policy.cfg.gradient_checkpointing = True
    policy.model.gradient_checkpointing_enable()
    prompts, responses = [[1, 2], [1, 2, 3]], [[4, 5], [6]]
    policy.model.train()
    lp, ent, mask = policy.forward_logprobs(prompts, responses, True)
    expected = torch.autograd.grad((lp + .001 * ent)[mask].sum(), policy.parameters())
    hooks = sum(len(module._forward_hooks) for module in policy.model.modules())
    for threshold, checkpointed in [(20, False), (1, True), (20, False)]:
        policy.cfg.checkpoint_min_tokens = threshold
        lp, ent, actual_mask = policy.forward_logprobs(prompts, responses, True)
        actual = torch.autograd.grad((lp + .001 * ent)[actual_mask].sum(), policy.parameters())
        assert policy.model.is_gradient_checkpointing == checkpointed
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        assert sum(len(module._forward_hooks) for module in policy.model.modules()) == hooks
