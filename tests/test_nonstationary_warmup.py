from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.nonstationary.warmup import baseline_examples, supervised_epoch


@pytest.mark.parametrize("depth", [2, 4, 8, 16])
def test_recovery_demonstrations_use_unchanged_baseline_states_and_complete(depth):
    pytest.importorskip("gym_novel_gridworlds2")
    from empg.envs.novelgym_env import NovelGymBenchmark
    bench = NovelGymBenchmark(recipe_depth=depth, action_parser="native")
    samples = list(baseline_examples(bench, bench.tasks("train")[0], 1., np.random.default_rng(0)))
    assert len(samples) == 8 * depth
    assert all(samples[i][1] == samples[i + 1][1] for i in range(0, len(samples), 2))
    assert all("Operation failed." in samples[i][0] for i in range(1, len(samples), 2))


def test_supervised_accumulation_matches_full_token_mean_objective():
    class Policy:
        device = "cpu"
        cfg = SimpleNamespace(micro_batch_size=1, micro_batch_tokens=32)
        def __init__(self):
            self.model = torch.nn.Linear(1, 3, bias=False)
            self.model.weight.data.zero_()
        def parameters(self):
            return list(self.model.parameters())
        def forward_logprobs(self, prompts, responses):
            logits = self.model(torch.ones(1)).log_softmax(0)
            lp = torch.stack([logits[torch.tensor(r)] for r in responses])
            return lp, None, torch.ones_like(lp, dtype=torch.bool)
    policy = Policy()
    opt = torch.optim.SGD(policy.parameters(), lr=.1)
    data = [([0], [1]), ([0, 0], [1, 1, 1]), ([0], [2])]
    nll = supervised_epoch(policy, opt, data, np.arange(3), 3)
    expected = -.1 * (torch.full((3,), 1 / 3) - torch.tensor([0., .8, .2]))
    torch.testing.assert_close(policy.model.weight[:, 0], expected)
    assert nll == pytest.approx(np.log(3))
