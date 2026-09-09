"""Entropy-Modulated Policy Gradients (EMPG).

Direct transcription of the method section of
"Harnessing Uncertainty: Entropy-Modulated Policy Gradients for
Long-Horizon LLM Agents" (Wang et al., arXiv:2509.09265).

Every function here is pure (numpy in, numpy out) so the math can be
unit-tested without an LLM or a training loop in the way.

Equation map
------------
Eq. 4  group_advantages   - GRPO group-relative advantage (outcome reward only)
Eq. 7  step_entropy       - lives in llm.py (needs token logprobs)
Eq. 8  modulated_advantage- A_i * g(H_t) + zeta * f(H_{t+1})
Eq. 10 gradient_scale_g   - self-calibrating scale, batch-mean normalised to 1
Eq. 11 clarity_bonus_f    - future-clarity bonus on the *successor* step
Eq. 12 normalize_entropy  - batch min-max scaling of raw entropies
Eq. 13 finalize_advantages- zero-mean shift across the mini-batch
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8

# Paper defaults (Sec. 4 / experimental setup).
DEFAULT_K = 1.0       # k  in Eq. 10
DEFAULT_K_PRIME = 1.0  # k' in Eq. 11
DEFAULT_ZETA = 0.05   # zeta in Eq. 8 (0.05 WebShop/ALFWorld, 0.1 Deep Search)


def group_advantages(rewards: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Eq. 4 - GRPO group-relative advantage.

    A_i = (r_i - mean_k r_k) / (std_k r_k + eps)

    ``rewards`` holds one *outcome* reward per trajectory in the group. This is
    the sparse signal the paper starts from: every step of a trajectory
    inherits this single scalar, which is exactly the credit-assignment
    problem EMPG sets out to fix.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def normalize_entropy(entropies: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Eq. 12 - batch-level min-max normalisation of step entropies.

    H_norm = (H - min_batch H) / (max_batch H - min_batch H + eps)

    Normalising per batch is what makes the scheme self-calibrating: it is the
    *relative* confidence of a step among its peers that matters, not the
    absolute nats, which drift as the policy trains.
    """
    entropies = np.asarray(entropies, dtype=np.float64)
    lo = entropies.min()
    hi = entropies.max()
    return (entropies - lo) / (hi - lo + eps)


def gradient_scale_g(h_norm: np.ndarray, k: float = DEFAULT_K) -> np.ndarray:
    """Eq. 10 - self-calibrating gradient scaling.

    g(H_t) = exp(-k * H_norm,t) / mean_batch[exp(-k * H_norm)]

    The denominator forces ``g.mean() == 1`` over the batch, so EMPG
    *redistributes* gradient mass between steps rather than inflating or
    shrinking the overall learning rate.

    Confident step (H below batch average) -> g > 1 -> amplified.
    Uncertain step (H above batch average) -> g < 1 -> damped.

    Note this multiplies the advantage, so it acts on the *magnitude* and
    keeps the sign: a confident-but-wrong step (A < 0) gets a proportionally
    stronger corrective penalty. That is the paper's answer to "hallucinated
    confidence", and it falls out of the same single term.
    """
    h_norm = np.asarray(h_norm, dtype=np.float64)
    raw = np.exp(-k * h_norm)
    # raw is a sum of exponentials and so is strictly positive; no epsilon is
    # needed here, and adding one would bias g.mean() off 1.
    return raw / raw.mean()


def clarity_bonus_f(
    h_norm_next: np.ndarray, k_prime: float = DEFAULT_K_PRIME
) -> np.ndarray:
    """Eq. 11 - future-clarity bonus.

    f(H_{t+1}) = exp(-k' * H_norm,t+1)

    Scored on the entropy of the state the action *leads to*. Actions that
    land the agent somewhere it feels sure about score higher, nudging the
    policy toward predictable solution paths. Unlike ``g`` this is an additive
    term, so it supplies a dense per-step signal even when the outcome reward
    is identical across a group (A_i == 0) and GRPO would learn nothing.
    """
    h_norm_next = np.asarray(h_norm_next, dtype=np.float64)
    return np.exp(-k_prime * h_norm_next)


def successor_entropy(h_norm: np.ndarray, done: np.ndarray) -> np.ndarray:
    """Align each step with its successor's normalised entropy, for Eq. 11.

    ``h_norm`` and ``done`` are flat, step-ordered arrays covering a batch of
    concatenated trajectories; ``done[t]`` marks the final step of one.

    A terminal step has no successor. We assign it ``h_norm = 0`` so that
    ``f = exp(0) = 1``, the maximum bonus: reaching a terminal state *is*
    perfect clarity, and this avoids leaking entropy across the trajectory
    boundary into the next trajectory's first step.
    """
    h_norm = np.asarray(h_norm, dtype=np.float64)
    done = np.asarray(done, dtype=bool)

    nxt = np.empty_like(h_norm)
    nxt[:-1] = h_norm[1:]
    nxt[-1] = 0.0
    nxt[done] = 0.0
    return nxt


def modulated_advantage(
    advantages: np.ndarray,
    h_norm: np.ndarray,
    h_norm_next: np.ndarray,
    k: float = DEFAULT_K,
    k_prime: float = DEFAULT_K_PRIME,
    zeta: float = DEFAULT_ZETA,
    use_scaling: bool = True,
    use_clarity: bool = True,
) -> np.ndarray:
    """Eq. 8 - A_mod(i,t) = A_i * g(H_t) + zeta * f(H_{t+1}).

    ``advantages`` is the per-step broadcast of each trajectory's Eq. 4 scalar.
    ``use_scaling`` / ``use_clarity`` toggle the two mechanisms independently
    so the ablation (paper Table 2) can be run from the same code path.
    """
    advantages = np.asarray(advantages, dtype=np.float64)

    scaled = advantages * gradient_scale_g(h_norm, k) if use_scaling else advantages
    if use_clarity:
        scaled = scaled + zeta * clarity_bonus_f(h_norm_next, k_prime)
    return scaled


def finalize_advantages(a_mod: np.ndarray) -> np.ndarray:
    """Eq. 13 - zero-mean shift across the mini-batch.

    A_final(i,t) = A_mod(i,t) - mean_batch[A_mod]

    The clarity bonus is strictly positive, so without this every step would
    carry a positive offset that acts as an indiscriminate push toward *any*
    action taken. Re-centring keeps the bonus purely comparative.
    """
    a_mod = np.asarray(a_mod, dtype=np.float64)
    return a_mod - a_mod.mean()


def compute_empg_advantages(
    rewards_per_step: np.ndarray,
    entropies: np.ndarray,
    done: np.ndarray,
    k: float = DEFAULT_K,
    k_prime: float = DEFAULT_K_PRIME,
    zeta: float = DEFAULT_ZETA,
    use_scaling: bool = True,
    use_clarity: bool = True,
) -> np.ndarray:
    """Full EMPG pipeline: Eq. 12 -> 10/11 -> 8 -> 13.

    ``rewards_per_step`` is the already-broadcast Eq. 4 advantage for each step
    (all steps of a trajectory share their trajectory's scalar).
    ``entropies`` are the *raw* measured step entropies in nats.
    """
    h_norm = normalize_entropy(entropies)
    h_norm_next = successor_entropy(h_norm, done)
    a_mod = modulated_advantage(
        rewards_per_step, h_norm, h_norm_next,
        k=k, k_prime=k_prime, zeta=zeta,
        use_scaling=use_scaling, use_clarity=use_clarity,
    )
    return finalize_advantages(a_mod)


def broadcast_advantages(traj_advantages: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Repeat each trajectory's Eq. 4 scalar across its steps.

    This *is* the GRPO baseline's per-step credit: one outcome number smeared
    uniformly over every step, good and bad alike.
    """
    return np.repeat(np.asarray(traj_advantages, dtype=np.float64),
                     np.asarray(lengths, dtype=int))
