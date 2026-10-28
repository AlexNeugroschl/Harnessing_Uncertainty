"""Exact token scores with bounded vocabulary-projection memory.

Checkpoint only the output head during training: its vocabulary-sized activations
are recomputed a chunk at a time in backward instead of kept for the whole batch.
No vocabulary approximation or sampled entropy is used.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_scores(hidden, head, targets, chunk_size=128, with_entropy=False, *, checkpoint_head=True):
    """Score flat, selected response-token hidden states; return [N] tensors."""
    if chunk_size < 1:
        raise ValueError("head_chunk_size must be positive")

    def project(h, y):
        logits = head(h).float()
        if not with_entropy:
            return -F.cross_entropy(logits, y, reduction="none")
        log_all = F.log_softmax(logits, dim=-1)
        logp = log_all.gather(-1, y[:, None]).squeeze(-1)
        entropy = -(log_all.exp() * log_all).sum(-1)
        return logp, entropy

    logps, entropies = [], []
    for start in range(0, len(targets), chunk_size):
        h, y = hidden[start:start + chunk_size], targets[start:start + chunk_size]
        if torch.is_grad_enabled() and checkpoint_head:
            out = checkpoint(project, h, y, use_reentrant=False)
        else:
            out = project(h, y)
        if with_entropy:
            lp, ent = out
            entropies.append(ent)
        else:
            lp = out
        logps.append(lp)
    return torch.cat(logps), torch.cat(entropies) if with_entropy else None


class _AnalyticScores(torch.autograd.Function):
    """Exact linear-head scores, retaining no vocabulary-sized activations.

    Backward recomputes each chunk once and uses the analytic derivatives of
    log-softmax and entropy. Unlike separate checkpoint nodes, it releases a
    chunk before projecting the next, including during backward. No sampled or
    top-k entropy, clipping, or changes to the downstream PPO loss are involved.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, hidden, weight, bias, targets, chunk_size, with_entropy):
        lp, ent = [], []
        for start in range(0, len(targets), chunk_size):
            logits = F.linear(hidden[start:start + chunk_size], weight, bias).float()
            log_all = F.log_softmax(logits, -1)
            lp.append(log_all.gather(-1, targets[start:start + chunk_size, None]).squeeze(-1))
            if with_entropy:
                ent.append(-(log_all.exp() * log_all).sum(-1))
        logp = torch.cat(lp)
        entropy = torch.cat(ent) if with_entropy else logp.new_empty(0)
        ctx.save_for_backward(hidden, weight, bias, targets)
        ctx.chunk_size, ctx.with_entropy = chunk_size, with_entropy
        ctx.set_materialize_grads(False)
        return logp, entropy

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_logp, grad_entropy):
        hidden, weight, bias, targets = ctx.saved_tensors
        dh = torch.empty_like(hidden) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        db = torch.zeros_like(bias) if bias is not None and ctx.needs_input_grad[2] else None
        for start in range(0, len(targets), ctx.chunk_size):
            stop = start + ctx.chunk_size
            h = hidden[start:stop]
            projected = F.linear(h, weight, bias)
            log_all = F.log_softmax(projected.float(), -1)
            p = log_all.exp()
            dl = torch.zeros_like(p)
            if grad_logp is not None:
                g = grad_logp[start:stop, None]
                dl = -p * g
                dl.scatter_add_(-1, targets[start:stop, None], g)
            if ctx.with_entropy and grad_entropy is not None:
                entropy = -(p * log_all).sum(-1, keepdim=True)
                dl.add_(-grad_entropy[start:stop, None] * p * (log_all + entropy))
            dl = dl.to(projected.dtype)
            if dh is not None:
                dh[start:stop] = (dl @ weight).to(hidden.dtype)
            if dw is not None:
                dw.add_((dl.T @ h).to(weight.dtype))
            if db is not None:
                db.add_(dl.sum(0).to(bias.dtype))
            del projected, log_all, p, dl
        return dh, dw, db, None, None, None


def analytic_scores(hidden, head, targets, chunk_size=128, with_entropy=False):
    """Bounded exact scores for a plain linear head; other heads use the reference.

    In particular, do not bypass a PEFT adapter or a model-specific output-head
    transformation by reading its base weights directly.
    """
    if chunk_size < 1:
        raise ValueError("head_chunk_size must be positive")
    if type(head) is not torch.nn.Linear:
        return chunked_scores(hidden, head, targets, chunk_size, with_entropy)
    lp, ent = _AnalyticScores.apply(hidden, head.weight, head.bias, targets, chunk_size, with_entropy)
    return lp, ent if with_entropy else None
