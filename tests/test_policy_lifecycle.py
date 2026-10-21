"""Prompt budgets, evaluation RNG isolation, and inference worker lifecycle."""
from types import SimpleNamespace
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from benchmarks import suite
from empg.policy import LLMPolicy, PolicyConfig


def bare_policy():
    policy = LLMPolicy.__new__(LLMPolicy)
    policy.cfg, policy.device = PolicyConfig(), "cpu"
    policy._sampling_seed, policy._sampling_calls = None, 0
    policy._closed, policy._in_generation, policy._engine_asleep = False, False, False
    policy._llm, policy._adapter_tmp, policy._adapter_dir = None, None, None
    return policy


def test_prompt_limit_rejects_without_truncation():
    policy = bare_policy()
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "chat"

        def __call__(self, *args, **kwargs):
            return {"input_ids": list(range(5))}

    policy.tokenizer = Tokenizer()
    policy.cfg.max_prompt_tokens = 5
    assert policy.encode_prompt("task") == list(range(5))
    policy.cfg.max_prompt_tokens = 4
    with pytest.raises(ValueError, match="No prompt was silently truncated"):
        policy.encode_prompt("task")


def test_eval_sampling_restores_training_rng_and_generation_state():
    policy = bare_policy()
    policy._sampling_seed, policy._sampling_calls = 99, 7
    torch.manual_seed(13)
    before = torch.random.get_rng_state()
    with policy.sampling(0):
        first = torch.rand(5)
        assert policy._sampling_seed == 0 and policy._sampling_calls == 0
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(20)  # training consumption must not alter evaluation samples
    with policy.sampling(0):
        torch.testing.assert_close(torch.rand(5), first)
    assert (policy._sampling_seed, policy._sampling_calls) == (99, 7)


def test_vllm_sampling_seeds_reset_per_evaluation(monkeypatch):
    policy = bare_policy()
    policy.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0,
                                       decode=lambda *a, **kw: "look")
    policy.encode_prompt = lambda prompt: [1]
    calls = []

    def generate(prompts, params, **kwargs):
        calls.append(params)
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[2], finish_reason="stop")])]

    policy._llm = SimpleNamespace(generate=generate)
    policy._lora_request = None
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    with policy.sampling(17):
        policy._vllm_generate(["task"], False)
        policy._vllm_generate(["task"], False)
    with policy.sampling(17):
        policy._vllm_generate(["task"], False)
    assert [row["seed"] for row in calls] == [17, 18, 17]
    assert all(row["temperature"] == 1. for row in calls)


def test_close_frees_hf_memory_before_wake_and_shutdown(monkeypatch):
    policy = bare_policy()
    calls = []
    policy.model = SimpleNamespace(to=lambda device: calls.append(device))
    policy._engine_asleep = True
    policy._llm = SimpleNamespace(
        wake_up=lambda: calls.append("wake"),
        llm_engine=SimpleNamespace(engine_core=SimpleNamespace(
            shutdown=lambda **kw: calls.append(("shutdown", kw["timeout"])))))
    policy._adapter_tmp = tempfile.TemporaryDirectory(prefix="empg_test_")
    path = Path(policy._adapter_tmp.name)
    policy._adapter_dir = str(path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    policy.close()
    policy.close()
    assert calls == ["cpu", "wake", ("shutdown", 10)]
    assert not path.exists() and policy._llm is None
    with pytest.raises(RuntimeError, match="closed"):
        with policy.generation():
            pass


@pytest.mark.parametrize("tier,reuse", [("paper_overnight", True), ("overnight_full", False)])
def test_suite_reuses_engine_only_with_an_explicit_evaluation_rng(monkeypatch, tier, reuse):
    args = SimpleNamespace(skip_base=True, seeds=[0], tier=tier, overrides=[])
    calls = []
    monkeypatch.setattr(suite, "_matching_eval", lambda *a: False)
    monkeypatch.setattr(suite, "_finished", lambda *a: False)
    monkeypatch.setattr(suite, "run", lambda *a, **kw: calls.append(a))
    suite.run_suite(args, ["grpo"])
    assert len(calls) == (1 if reuse else 2)
    assert ("--evaluate" in calls[0]) == reuse
