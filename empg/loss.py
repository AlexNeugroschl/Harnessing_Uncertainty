"""Clipped policy-gradient loss with a KL penalty to the reference policy (GRPO style)."""

from __future__ import annotations

import torch


def policy_loss(logp, old_logp, ref_logp, adv, mask, clip_low=0.2, clip_high=0.2,
                kl_coef=0.01, tensor_stats=False, *, entropy=None, entropy_coef=0.0):
    """Per-token PPO-clip surrogate + k3 KL (verl's ``low_var_kl``).

    All tensors are [B, T]. ``adv`` holds each step's advantage broadcast to all
    of its response tokens. Returns the *sum* over valid tokens (the caller
    divides by the mini-batch token count, giving verl's "token-mean"), and a
    dict of summed stats.
    """
    ratio = torch.exp(logp - old_logp)
    pg = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * adv)
    per_tok = pg
    kl = torch.zeros_like(logp)
    if kl_coef > 0 and ref_logp is not None:
        d = (ref_logp - logp).clamp(-20, 20)
        kl = (torch.exp(d) - d - 1).clamp(max=10)
        per_tok = pg + kl_coef * kl
    if entropy_coef:
        if entropy is None:
            raise ValueError("entropy_coef requires differentiable token entropy")
        per_tok = per_tok - entropy_coef * entropy
    m = mask.float()
    clipped = ((ratio < 1 - clip_low) | (ratio > 1 + clip_high)).float()
    stats = {"pg": (pg * m).sum().detach(), "kl": (kl * m).sum().detach(),
             "clipfrac": (clipped * m).sum().detach()}
    if not tensor_stats:
        stats = {key: value.item() for key, value in stats.items()}
    return (per_tok * m).sum(), stats
