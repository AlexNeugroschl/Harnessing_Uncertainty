import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from run_adaexplore import LocalBatchClient


@dataclass
class Response:
    content: str
    model: str
    usage: dict
    raw: dict
    finish_reason: str


class Policy:
    cfg = SimpleNamespace(model_name="local")

    def __init__(self):
        self.calls = []

    def generate(self, prompts, greedy):
        self.calls.append(prompts)
        return [[1, 2] for _ in prompts], [[3] for _ in prompts], ['{"tool":"request_next_batch","arguments":{}}'] * len(prompts)

    def truncated(self, response):
        return False


def test_independent_official_conversations_share_one_gpu_batch():
    policy = Policy()
    client = LocalBatchClient(policy, Response)
    messages = [[{"role": "system", "content": "rule"}, {"role": "user", "content": str(i)}] for i in range(4)]
    async def work():
        return await asyncio.gather(*(client.generate(request=SimpleNamespace(protocol="deepseek_chat", messages=m),
                                      schema={}, trace_context={}) for m in messages))
    results = asyncio.run(work())
    assert policy.calls == [messages]
    assert all(r.usage["total_tokens"] == 3 and r.finish_reason == "stop" for r in results)


def test_generation_failure_reaches_every_waiting_episode_without_hanging():
    class Failing(Policy):
        def generate(self, prompts, greedy):
            raise ValueError("prompt limit")
    client = LocalBatchClient(Failing(), Response)
    async def work():
        request = SimpleNamespace(protocol="deepseek_chat", messages=[{"role": "user", "content": "hello"}])
        return await asyncio.gather(*(client.generate(request=request, schema={}) for _ in range(3)),
                                    return_exceptions=True)
    results = asyncio.run(work())
    assert all(isinstance(r, ValueError) for r in results)
