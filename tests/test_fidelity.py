"""Publication-era reward semantics, entropy gradients, and artifact compatibility."""
from contextlib import nullcontext
import dataclasses
import hashlib
import json

import numpy as np
import pytest
import torch

from benchmarks.common import load
from benchmarks import evaluate as evaluation
from empg.advantages import AdvantageConfig, compute_advantages
from empg.config import config_dict, config_fingerprint, config_matches
from empg.loss import policy_loss
from empg.prompts import parse_action
from empg.rollout import Step, Trajectory


def test_step_penalty_is_local_instead_of_spreading_over_the_episode():
    cfg = AdvantageConfig(method="grpo")
    mapping = [0, 0, 1, 1]
    # All failed; only the first response of episode zero was invalid.
    legacy = compute_advantages([-.1, 0.], [0, 0], mapping, [1.] * 4, cfg)
    actual = compute_advantages([-.1, 0.], [0, 0], mapping, [1.] * 4, cfg,
                                step_rewards=[-.1, 0., 0., 0.])
    assert legacy[0] == legacy[1] < 0
    np.testing.assert_allclose(actual, [-1.5, .5, .5, .5], rtol=1e-4)


def test_step_normalization_weights_actual_response_rows():
    # Successful episode: one response; failed episode: three responses.
    actual = compute_advantages([10., 0.], [0, 0], [0, 1, 1, 1], [1.] * 4,
                                AdvantageConfig(method="grpo"), step_rewards=[10., 0., 0., 0.])
    np.testing.assert_allclose(actual, [1.5, -.5, -.5, -.5], rtol=1e-5)
    with pytest.raises(ValueError, match="one score"):
        compute_advantages([0.], [0], [0, 0], [1., 1.], AdvantageConfig(), step_rewards=[0.])


def test_actor_minibatches_use_response_count_and_preserve_sequential_order():
    from empg.trainer import optimizer_minibatches

    tc = load("empg", "paper_overnight").train
    rng = np.random.default_rng(42)
    batches = optimizer_minibatches(601, tc, rng)
    assert [len(mb) for mb in batches] == [256, 256, 89]
    np.testing.assert_array_equal(np.concatenate(batches), np.arange(601))
    assert len(optimizer_minibatches(4, tc, rng)) == 1
    legacy = load("empg", "overnight_full").train
    batches = optimizer_minibatches(23, legacy, rng)
    assert len(batches) == legacy.mini_batches
    assert sorted(np.concatenate(batches)) == list(range(23))


def test_entropy_bonus_has_the_analytic_gradient_and_masks_padding():
    torch.manual_seed(13)
    logits = torch.randn(2, 3, 7, requires_grad=True)
    log_all = logits.log_softmax(-1)
    probabilities = log_all.exp()
    entropy = -(probabilities * log_all).sum(-1)
    logp = log_all[..., 0]
    mask = torch.tensor([[True, True, False], [True, False, False]])
    coef = .001
    loss, _ = policy_loss(logp, logp.detach(), None, torch.zeros_like(logp), mask,
                          kl_coef=0., entropy=entropy, entropy_coef=coef)
    gradient, = torch.autograd.grad(loss, logits)
    expected = coef * probabilities.detach() * (
        log_all.detach() + entropy.detach().unsqueeze(-1)) * mask.unsqueeze(-1)
    torch.testing.assert_close(gradient, expected, atol=1e-9, rtol=1e-5)
    assert loss.item() == pytest.approx(-coef * entropy.detach()[mask].sum().item())
    with pytest.raises(ValueError, match="differentiable"):
        policy_loss(logp, logp.detach(), None, torch.zeros_like(logp), mask,
                    kl_coef=0., entropy_coef=coef)


def test_reference_dual_clip_only_bounds_negative_advantage_tails():
    # Negative-A ratio 5 is bounded at 3; ratio 2 keeps its ordinary penalty.
    # Positive-A ratio 5 retains the ordinary upper PPO clip of 1.2.
    logp = torch.tensor([[5., 2., 5.]]).log().requires_grad_()
    old = torch.zeros_like(logp)
    advantage = torch.tensor([[-1., -1., 1.]])
    mask = torch.ones_like(logp, dtype=torch.bool)
    loss, _ = policy_loss(logp, old, None, advantage, mask, kl_coef=0., dual_clip=3.)
    assert loss.item() == pytest.approx(3. + 2. - 1.2)
    gradient, = torch.autograd.grad(loss, logp)
    torch.testing.assert_close(gradient, torch.tensor([[0., 2., 0.]]))
    legacy, _ = policy_loss(logp, old, None, advantage, mask, kl_coef=0.)
    assert legacy.item() == pytest.approx(5. + 2. - 1.2)


def test_new_options_do_not_invalidate_disabled_legacy_configurations():
    cfg = load("grpo", "overnight_full")
    legacy = dataclasses.asdict(cfg)
    for section, keys in {"train": ["entropy_coef", "reward_mode", "mini_batch_size", "shuffle", "dual_clip"],
                          "policy": ["amp", "vllm_max_num_seqs", "max_prompt_tokens",
                                     "score_batch_size", "micro_batch_tokens", "cuda_memory_fraction",
                                     "head_backend", "head_checkpointing", "checkpoint_min_tokens"],
                          "eval": ["temperature", "sampling_seed"],
                          "advantage": ["entropy_eps", "scale_eps"]}.items():
        for key in keys:
            legacy[section].pop(key)
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
    assert config_fingerprint(cfg) == expected
    assert config_matches(legacy, cfg)
    assert config_matches(dataclasses.asdict(cfg), cfg)
    assert config_dict(cfg) == legacy
    cfg.train.entropy_coef = .001
    assert config_fingerprint(cfg) != expected
    assert not config_matches(legacy, cfg)


def test_paper_lora_preset_is_explicit_and_does_not_replace_overnight():
    cfg = load("empg", "paper_lora")
    assert cfg.train.entropy_coef == .001 and cfg.train.lr == 1e-6
    assert cfg.train.weight_decay == .01
    assert cfg.train.mini_batch_size == 256 and not cfg.train.shuffle
    assert cfg.train.dual_clip == 3.
    assert cfg.train.reward_mode == "step"
    assert cfg.train.iterations == 150 and cfg.train.group_size == 8
    assert cfg.train.games_per_iter == 2  # documented local compromise
    assert cfg.policy.lora_r == 16 and cfg.policy.amp
    assert cfg.policy.max_new_tokens == 512
    assert cfg.policy.max_prompt_tokens == 2048
    assert cfg.advantage.entropy_eps == cfg.advantage.scale_eps == 1e-8
    assert cfg.eval.sampling_seed == 0
    assert cfg.eval.max_tasks == 0 and cfg.eval.split == "valid_unseen"
    assert not cfg.eval.greedy and cfg.eval.temperature == .4
    assert cfg.env["action_parser"] == "verl"
    assert load("empg", "overnight_full").train.entropy_coef == 0.
    fast = load("empg", "paper_lora_fast")
    assert fast.train == cfg.train and fast.eval == cfg.eval and fast.env == cfg.env
    assert fast.policy.backend == "vllm" and not fast.policy.vllm_enforce_eager
    assert fast.policy.vllm_max_num_seqs == 32
    assert fast.policy.vllm_max_model_len == 2560


def test_verl_projection_selects_first_action_and_rejects_cjk():
    response = "<think>x</think><action>LOOK</action><action>inventory</action>"
    assert parse_action(response, mode="verl") == ("look", True)
    assert not parse_action(response)[1]  # strict legacy parser is preserved
    assert parse_action("<think>中文</think><action>look</action>", mode="verl") == ("look", False)
    incomplete = "x" * 40 + "<think>missing an action"
    assert parse_action(incomplete, mode="verl") == (incomplete.lower()[-30:], False)


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_sampling_override_is_recorded_and_restored(monkeypatch, fail):
    cfg = load("grpo", "smoke")
    cfg.eval.greedy, cfg.eval.temperature = False, .4
    original = cfg.policy
    fingerprint = config_fingerprint(cfg)

    class Bench:
        closed = False

        def tasks(self, split):
            return ["task"]

        def task_type(self, task):
            return "Pick"

        def close(self):
            self.closed = True

    class Policy:
        def __init__(self):
            self.cfg = cfg.policy

        def generation(self):
            assert self.cfg.temperature == .4
            return nullcontext()

    bench, policy = Bench(), Policy()

    def episodes(*args, **kwargs):
        assert not kwargs["greedy"] and policy.cfg.temperature == .4
        if fail:
            raise RuntimeError("test failure")
        return [Trajectory("task", 0, [Step([1], [2], "", "look", True)])]

    monkeypatch.setattr(evaluation, "make_benchmark", lambda *a, **kw: bench)
    monkeypatch.setattr(evaluation, "run_episodes", episodes)
    if fail:
        with pytest.raises(RuntimeError, match="test failure"):
            evaluation.evaluate(policy, cfg)
    else:
        result, _ = evaluation.evaluate(policy, cfg)
        assert result["protocol"]["temperature"] == .4
    assert policy.cfg is original and cfg.policy.temperature == 1.
    assert config_fingerprint(cfg) == fingerprint and bench.closed
