import pytest
import torch
import torch.nn.functional as F

from empg.scoring import chunked_scores


@pytest.mark.parametrize("chunk", [1, 7, 64])
@pytest.mark.parametrize("train_head", [False, True])
def test_chunked_scores_values_entropy_and_gradients(chunk, train_head):
    torch.manual_seed(12)
    head = torch.nn.Linear(11, 37, bias=True)
    head.requires_grad_(train_head)
    hidden = torch.randn(19, 11, requires_grad=True)
    targets = torch.randint(37, (19,))
    weights = torch.randn(19)
    logits = head(hidden)
    dense_lp = F.log_softmax(logits, -1).gather(-1, targets[:, None]).squeeze(-1)
    dense_ent = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    params = [hidden, *head.parameters()] if train_head else [hidden]
    expected = torch.autograd.grad((weights * dense_lp + 0.02 * dense_ent).sum(), params)
    lp, ent = chunked_scores(hidden, head, targets, chunk, with_entropy=True)
    actual = torch.autograd.grad((weights * lp + 0.02 * ent).sum(), params)
    torch.testing.assert_close(lp, dense_lp)
    torch.testing.assert_close(ent, dense_ent)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_invalid_chunk_size():
    with pytest.raises(ValueError, match="positive"):
        chunked_scores(torch.randn(2, 3), torch.nn.Linear(3, 5), torch.tensor([0, 1]), 0)
