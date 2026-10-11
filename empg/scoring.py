"""Exact token scores with bounded vocabulary-projection memory.

Checkpoint only the output head during training: its vocabulary-sized activations
are recomputed a chunk at a time in backward instead of kept for the whole batch.
No vocabulary approximation or sampled entropy is used.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_scores(hidden, head, targets, chunk_size=128, with_entropy=False):
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
        if torch.is_grad_enabled():
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
