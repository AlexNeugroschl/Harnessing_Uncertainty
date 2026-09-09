"""Ollama client that measures real step-level entropy (Eq. 7).

Two things worth knowing, both established by probing the local install:

1. Ollama's *native* ``/api/generate`` rejects an integer ``logprobs`` field.
   Per-token logprobs are only available on the OpenAI-compatible endpoint
   ``/v1/chat/completions`` with ``logprobs: true, top_logprobs: N``.

2. ``top_logprobs`` is capped at 20, so we see a truncated distribution rather
   than the full vocabulary. Measured on this setup the top 20 carry ~99.6% of
   the probability mass, so renormalising over them is a sound estimate of the
   true entropy. It is a slight *under*-estimate (the missing tail would add a
   little), but it is a consistent one across steps, and EMPG only ever uses
   entropy comparatively (Eq. 12 min-max normalises it per batch).
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
# llama3.2 answers with a clean single action line. Avoid qwen3.5:2b as the
# actor: it spends its whole token budget inside a <think> block and returns
# empty `content`.
DEFAULT_MODEL = "llama3.2:latest"
TOP_LOGPROBS = 20


CHOICE_LABELS = "ABCDEFGH"


@dataclass
class ChoiceResult:
    """The LLM's distribution over a labelled set of candidate actions.

    ``probs`` is a proper distribution over the options (renormalised over the
    label tokens), and ``entropy`` is its Shannon entropy - the uncertainty of
    the *decision*, which is what EMPG's H_t should mean for a step.
    """

    probs: list[float]
    entropy: float
    label_mass: float = 1.0  # how much probability landed on valid labels
    cached: bool = False


class OllamaClient:
    """Chat client with on-disk response caching.

    Caching is not just a speed optimisation. Rollouts cost ~1-3s per step, and
    caching by prompt hash means the GRPO and EMPG arms train on *identical*
    trajectories and identical entropies. Without it the two arms would see
    different samples from a stochastic policy and any performance gap would be
    partly sampling noise rather than the algorithm.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str | Path = ".cache",
        temperature: float = 0.7,
        seed: int | None = 0,
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_calls = 0
        self.n_cache_hits = 0

    def _cache_key(self, messages: list[dict], tag: str = "") -> Path:
        payload = json.dumps(
            [self.model, messages, self.temperature, self.seed, tag], sort_keys=True
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"

    def score_choices(self, messages: list[dict], n_options: int) -> ChoiceResult:
        """Read the LLM's distribution over labelled options A), B), C)...

        Only one token is generated. We take its top-k distribution, keep the
        mass on valid option labels, and renormalise. The resulting entropy is
        the uncertainty of the decision itself rather than of English phrasing
        - measured on this setup, that distinction is what makes entropy track
        step competence instead of surface wording.

        There is deliberately no nonce here. The LLM is frozen, so its
        distribution over a given (task, observation, options) state is a fixed
        property of the model - asking again just re-derives the same numbers.
        The environment has only ~32 reachable states, so the whole demo costs
        ~32 LLM calls and every later rollout is served from cache. Rollout
        stochasticity comes from the policy head's sampling instead, and both
        training arms therefore see identical entropies.
        """
        cache_path = self._cache_key(messages, tag=f"choice{n_options}")
        if cache_path.exists():
            self.n_cache_hits += 1
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return ChoiceResult(**{**data, "cached": True})

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            body["seed"] = self.seed

        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {OLLAMA_URL}. Is `ollama serve` "
                f"running and `{self.model}` pulled?  ({exc})"
            ) from exc

        self.n_calls += 1
        tokens = (raw["choices"][0].get("logprobs") or {}).get("content") or []
        valid = CHOICE_LABELS[:n_options]
        probs = [0.0] * n_options

        if tokens:
            for cand in tokens[0].get("top_logprobs") or []:
                label = cand["token"].strip().upper()
                if len(label) == 1 and label in valid:
                    probs[valid.index(label)] += math.exp(cand["logprob"])

        label_mass = sum(probs)
        if label_mass <= 0:
            # The model produced no recognisable label; fall back to uniform so
            # the episode continues rather than crashing.
            probs = [1.0 / n_options] * n_options
        else:
            probs = [p / label_mass for p in probs]

        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        result = ChoiceResult(probs=probs, entropy=entropy, label_mass=label_mass)
        cache_path.write_text(
            json.dumps(
                {
                    "probs": result.probs,
                    "entropy": result.entropy,
                    "label_mass": result.label_mass,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        return result

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5):
                return True
        except Exception:
            return False
