import pytest
import torch
import torch.nn.functional as F

from empg.scoring import chunked_scores, analytic_scores


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


@pytest.mark.parametrize("entropy", [False, True])
@pytest.mark.parametrize("train_head", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_analytic_scores_preserve_values_and_arbitrary_upstream_gradients(entropy, train_head, bias):
    torch.manual_seed(31)
    head = torch.nn.Linear(11, 37, bias=bias).requires_grad_(train_head)
    h = torch.randn(19, 11, requires_grad=True)
    y = torch.randint(37, (19,))
    a, b = torch.randn(19), torch.randn(19)
    params = [h, *head.parameters()] if train_head else [h]
    expected_lp, expected_ent = chunked_scores(h, head, y, 7, entropy)
    expected_loss = (a * expected_lp).sum()
    if entropy:
        expected_loss += (b * expected_ent).sum()
    expected = torch.autograd.grad(expected_loss, params)
    lp, ent = analytic_scores(h, head, y, 7, entropy)
    actual_loss = (a * lp).sum()
    if entropy:
        actual_loss += (b * ent).sum()
        torch.testing.assert_close(ent, expected_ent)
    else:
        assert ent is None
    actual = torch.autograd.grad(actual_loss, params)
    torch.testing.assert_close(lp, expected_lp)
    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)


def test_analytic_entropy_only_backward_and_transformed_head_fallback():
    torch.manual_seed(7)
    h = torch.randn(5, 3, requires_grad=True)
    y = torch.tensor([0, 1, 2, 0, 1])
    head = torch.nn.Linear(3, 5)
    _, ent = analytic_scores(h, head, y, 2, True)
    actual = torch.autograd.grad(ent.sum(), h)[0]
    _, expected_ent = chunked_scores(h, head, y, 2, True)
    expected = torch.autograd.grad(expected_ent.sum(), h)[0]
    torch.testing.assert_close(actual, expected)
    transformed = torch.nn.Sequential(head, torch.nn.Tanh())
    lp, ent = analytic_scores(h, transformed, y, 2, True)
    expected_lp, expected_ent = chunked_scores(h, transformed, y, 2, True)
    torch.testing.assert_close(lp, expected_lp)
    torch.testing.assert_close(ent, expected_ent)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP equivalence")
def test_analytic_cuda_autocast_preserves_entropy_and_hidden_gradients():
    torch.manual_seed(23)
    head = torch.nn.Linear(64, 1024, bias=False, device="cuda", dtype=torch.bfloat16)
    head.requires_grad_(False)
    h = torch.randn(17, 64, device="cuda", requires_grad=True)
    y = torch.randint(1024, (17,), device="cuda")
    a, b = torch.randn(17, device="cuda"), torch.randn(17, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lp, ent = chunked_scores(h, head, y, 7, True)
        expected = torch.autograd.grad((a * lp + b * ent).sum(), h)[0]
        actual_lp, actual_ent = analytic_scores(h, head, y, 7, True)
        actual = torch.autograd.grad((a * actual_lp + b * actual_ent).sum(), h)[0]
    torch.testing.assert_close(actual_lp, lp)
    torch.testing.assert_close(actual_ent, ent)
    # Separate bf16 matmuls/reductions can round differently in backward.
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
