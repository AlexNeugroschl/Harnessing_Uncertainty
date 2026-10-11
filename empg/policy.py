"""LLM policy: a causal LM + LoRA with batched generation and scoring.

Token entropy is computed in a no-grad forward pass *after* generation, rather
than from logits stored during generate(): keeping a 151k-vocab logit tensor
for every generated token would not fit in 8GB. The parameters are unchanged
between rollout and scoring, so the result is the same distribution.

Generation happens inside a ``generation()`` session (one per rollout), which
sets up the fast path and undoes it afterwards:
  - ``backend: hf``: LoRA is merged into the base weights while decoding, so each
    layer does one matmul per token instead of three. The frozen base weights are
    restored bit-exactly afterwards (a bf16 unmerge alone would leave rounding
    drift, and that would move the KL reference).
  - ``backend: vllm``: the HF model goes to CPU and a vLLM engine (continuous
    batching, prefix caching) samples with the current adapter, then sleeps and
    frees the GPU for the update. Scoring, entropy and training always use the
    HF model, so the PPO ratio and EMPG's H_t come from the model being trained.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from empg.scoring import chunked_scores


@dataclass
class PolicyConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"
    lora_r: int = 16             # 0 disables LoRA (full fine-tuning)
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: list[str] = field(default_factory=lambda: ["all-linear"])
    gradient_checkpointing: bool = True
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    gen_batch_size: int = 32     # prompts per generate() call
    micro_batch_size: int = 4    # sequences per forward pass when scoring / training
    head_chunk_size: int = 128   # response tokens per exact vocabulary projection
    chunked_head: bool = True   # false: dense reference path for equivalence/profiling
    backend: str = "hf"          # generation engine: hf | vllm (scoring/training are always HF)
    merge_lora: bool = True      # hf: merge LoRA into the weights while generating
    vllm_gpu_memory_utilization: float = 0.8  # share of VRAM vLLM may take (HF model is on CPU then)
    vllm_max_model_len: int = 8192
    # No CUDA graphs: asleep, vLLM then keeps ~1.2 GiB instead of ~2.3 GiB, which on an
    # 8GB card is what lets the update fit. Rollouts are env-bound, so it costs ~nothing.
    vllm_enforce_eager: bool = True


class LLMPolicy:
    def __init__(self, cfg: PolicyConfig, device: str | None = None, seed: int = 0):
        if cfg.backend not in ("hf", "vllm"):
            raise ValueError(f"unknown policy.backend {cfg.backend!r}")
        self.cfg = cfg
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = getattr(torch, cfg.dtype) if self.device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=dtype)
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        self.use_lora = cfg.lora_r > 0
        if self.use_lora:
            from peft import LoraConfig, get_peft_model
            targets = cfg.lora_targets
            model = get_peft_model(model, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=targets[0] if targets == ["all-linear"] else targets,
                task_type="CAUSAL_LM",
            ))
        self.model = model.to(self.device)
        # Models with architecture-specific output transforms (e.g. logit
        # soft-capping) must keep their causal-LM forward path.
        self._plain_head = model.config.model_type in ("qwen2", "qwen3", "llama", "mistral")
        if cfg.chunked_head and not self._plain_head:
            warnings.warn(f"chunked head unsupported for {model.config.model_type}; using dense scores")
        self.model.eval()
        self._in_generation = False
        self._base_snapshot = None   # hf merge: CPU copy of the frozen weights LoRA wraps
        self._llm = None             # vllm engine, created on first use
        self._lora_request = None
        self._lora_id = 0
        self._adapter_dir = None

    # ------------------------------------------------------------------ io
    def encode_prompt(self, prompt: str) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def save(self, path: str):
        self.model.save_pretrained(path)

    def load_adapter(self, path: str):
        from peft import set_peft_model_state_dict
        from peft.utils import load_peft_weights
        set_peft_model_state_dict(self.model, load_peft_weights(path, device=self.device))

    # ------------------------------------------------------------ generate
    @contextmanager
    def generation(self):
        """Wraps one rollout's generate() calls. Nested sessions are no-ops."""
        if self._in_generation:
            yield
            return
        enter, leave = ((self._vllm_wake, self._vllm_sleep) if self.cfg.backend == "vllm"
                        else (self._merge_lora, self._unmerge_lora))
        enter()
        self._in_generation = True
        try:
            yield
        finally:
            self._in_generation = False
            leave()

    def truncated(self, response_ids: list[int]) -> bool:
        """True if the response hit max_new_tokens before emitting EOS."""
        return (len(response_ids) >= self.cfg.max_new_tokens
                and response_ids[-1] != self.tokenizer.eos_token_id)

    @torch.no_grad()
    def generate(self, prompts: list[str], greedy: bool = False):
        """Returns (prompt_ids, response_ids, texts), one per prompt (ids as lists)."""
        if not self._in_generation:
            with self.generation():
                return self.generate(prompts, greedy)
        if self.cfg.backend == "vllm":
            return self._vllm_generate(prompts, greedy)
        self.model.eval()
        all_p, all_r, all_t = [], [], []
        pad, eos = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        for s in range(0, len(prompts), self.cfg.gen_batch_size):
            p_ids = [self.encode_prompt(p) for p in prompts[s:s + self.cfg.gen_batch_size]]
            batch = self._left_pad(p_ids, pad)
            out = self.model.generate(
                **batch, max_new_tokens=self.cfg.max_new_tokens,
                do_sample=not greedy, temperature=self.cfg.temperature if not greedy else None,
                top_p=self.cfg.top_p if not greedy else None, top_k=0 if not greedy else None,
                # Qwen's generation_config sets repetition_penalty 1.1; left in, rollouts
                # would sample from a distribution other than the policy being scored.
                repetition_penalty=1.0, pad_token_id=pad,
            )
            gen = out[:, batch["input_ids"].shape[1]:].tolist()
            for p, r in zip(p_ids, gen):
                r = self._trim(r, eos, pad)
                all_p.append(p)
                all_r.append(r)
                all_t.append(self.tokenizer.decode(r, skip_special_tokens=True))
        return all_p, all_r, all_t

    # hf backend: merged-LoRA decoding
    def _lora_base_weights(self) -> dict:
        return {n: p for n, p in self.model.named_parameters() if n.endswith("base_layer.weight")}

    @torch.no_grad()
    def _merge_lora(self):
        if not (self.use_lora and self.cfg.merge_lora):
            return
        if self._base_snapshot is None:  # frozen, so one snapshot serves the whole run
            pin = torch.cuda.is_available()
            self._base_snapshot = {
                n: p.detach().cpu().pin_memory() if pin else p.detach().cpu().clone()
                for n, p in self._lora_base_weights().items()}
        self.model.merge_adapter()

    @torch.no_grad()
    def _unmerge_lora(self):
        if not (self.use_lora and self.cfg.merge_lora):
            return
        self.model.unmerge_adapter()  # resets PEFT's merged state; the copy below fixes rounding
        for n, p in self._lora_base_weights().items():
            p.copy_(self._base_snapshot[n], non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # vllm backend
    def _vllm_wake(self):
        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._llm is None:
            self._llm = self._make_llm()
        else:
            self._llm.wake_up()
        if self.use_lora:
            # A fresh id every session: the adapter may have changed since the last
            # one, and a woken engine must not reuse LoRA slots it had before sleeping.
            from vllm.lora.request import LoRARequest
            self._adapter_dir = self._adapter_dir or tempfile.mkdtemp(prefix="empg_adapter_")
            self.save(self._adapter_dir)
            self._lora_id += 1
            self._lora_request = LoRARequest(f"policy_{self._lora_id}", self._lora_id,
                                             self._adapter_dir)

    def _vllm_sleep(self):
        self._llm.sleep(level=1)  # weights to CPU, KV cache freed
        self.model.to(self.device)

    def _make_llm(self):
        _vllm_env()
        from vllm import LLM
        c = self.cfg
        lora = {}
        if self.use_lora:
            rank = 8
            while rank < c.lora_r:
                rank *= 2
            lora = dict(enable_lora=True, max_lora_rank=rank, max_loras=1)
        return LLM(model=c.model_name, dtype=c.dtype, seed=self.seed,
                   gpu_memory_utilization=c.vllm_gpu_memory_utilization,
                   max_model_len=c.vllm_max_model_len, enforce_eager=c.vllm_enforce_eager,
                   enable_prefix_caching=True, enable_sleep_mode=True, **lora)

    def _vllm_generate(self, prompts: list[str], greedy: bool):
        from vllm import SamplingParams
        c = self.cfg
        params = SamplingParams(max_tokens=c.max_new_tokens,
                                temperature=0.0 if greedy else c.temperature,
                                top_p=1.0 if greedy else c.top_p)
        p_ids = [self.encode_prompt(p) for p in prompts]
        outs = self._llm.generate([{"prompt_token_ids": p} for p in p_ids], params,
                                  lora_request=self._lora_request, use_tqdm=False)
        pad, eos = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        all_r, all_t = [], []
        for o in outs:
            out = o.outputs[0]
            ids = list(out.token_ids)
            if out.finish_reason == "stop" and (not ids or ids[-1] not in (eos, pad)):
                ids.append(eos)  # keep the EOS, as the HF path does, so the model learns to stop
            r = self._trim(ids, eos, pad)
            all_r.append(r)
            all_t.append(self.tokenizer.decode(r, skip_special_tokens=True))
        return p_ids, all_r, all_t

    @staticmethod
    def _trim(ids: list[int], eos: int, pad: int) -> list[int]:
        """Cut after the first EOS (kept, so the model learns to stop)."""
        for i, t in enumerate(ids):
            if t == eos:
                return ids[: i + 1]
        while ids and ids[-1] == pad:
            ids = ids[:-1]
        return ids or [eos]

    def _left_pad(self, seqs: list[list[int]], pad: int) -> dict:
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, n - len(s):] = torch.tensor(s)
            mask[i, n - len(s):] = 1
        return {"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device)}

    # --------------------------------------------------------------- score
    def forward_logprobs(self, prompt_ids, response_ids, with_entropy=False):
        """Per-token logprob (and entropy) of each response under the current model.

        Returns padded tensors [B, R] plus a response mask. Gradients flow unless
        the caller wraps this in no_grad.
        """
        seqs = [p + r for p, r in zip(prompt_ids, response_ids)]
        pad = self.tokenizer.pad_token_id
        batch = self._left_pad(seqs, pad)
        n = batch["input_ids"].shape[1]
        R = max(len(r) for r in response_ids)
        # Logits at position j predict token j+1. Responses are right-aligned, so
        # only the last R+1 positions matter; the final one predicts past the end.
        # Explicit positions: with left padding, a plain forward would count pads as
        # positions, so scores would depend on batch padding and not match generate().
        pos = (batch["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        targets = batch["input_ids"][:, n - R:]
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for i, r in enumerate(response_ids):
            mask[i, R - len(r):] = True
        if self.cfg.chunked_head and self._plain_head:
            causal = self.model.get_base_model() if self.use_lora else self.model
            # LoRA remains active inside the transformer. Project only actual
            # response positions, excluding prompts and response padding.
            hidden = causal.base_model(**batch, position_ids=pos, use_cache=False).last_hidden_state
            selected = hidden[:, n - R - 1:n - 1][mask]
            lp, entropy = chunked_scores(selected, causal.get_output_embeddings(), targets[mask],
                                         self.cfg.head_chunk_size, with_entropy)
            logp = lp.new_zeros(mask.shape).masked_scatter(mask, lp)
            ent = (entropy.new_zeros(mask.shape).masked_scatter(mask, entropy)
                   if with_entropy else None)
            return logp, ent, mask
        logits = self.model(**batch, position_ids=pos, use_cache=False,
                            logits_to_keep=R + 1).logits[:, :-1].float()
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ent = -(logp_all.exp() * logp_all).sum(-1) if with_entropy else None
        return logp, ent, mask

    @torch.no_grad()
    def score(self, prompt_ids, response_ids, with_entropy=True, reference=False):
        """No-grad chunked scoring. Returns lists of 1-D CPU tensors (one per sample)."""
        self.model.eval()
        logps, ents = [None] * len(prompt_ids), [None] * len(prompt_ids) if with_entropy else []
        mb = self.cfg.micro_batch_size
        order = sorted(range(len(prompt_ids)), key=lambda i: len(prompt_ids[i]) + len(response_ids[i]))
        ctx = self.model.disable_adapter() if (reference and self.use_lora) else _null()
        with ctx:
            for s in range(0, len(prompt_ids), mb):
                idx = order[s:s + mb]
                lp, ent, mask = self.forward_logprobs(
                    [prompt_ids[i] for i in idx], [response_ids[i] for i in idx], with_entropy)
                for i in range(lp.shape[0]):
                    logps[idx[i]] = lp[i][mask[i]].cpu()
                    if with_entropy:
                        ents[idx[i]] = ent[i][mask[i]].cpu()
        return logps, ents


def _vllm_env():
    """Environment fixes for vLLM from pip (no system CUDA toolkit), set before the
    engine process is spawned so it inherits them."""
    # FlashInfer JIT-compiles its sampler and needs full CUDA headers. PyTorch's
    # sampler draws from the same distribution (top_p = 1, no top-k).
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # torch.compile calls `nvcc`. The pip CUDA wheels ship one that is not on PATH.
    if shutil.which("nvcc") is None:
        spec = importlib.util.find_spec("nvidia")
        for root in (spec.submodule_search_locations if spec else []):
            hits = sorted(Path(root).glob("*/bin/nvcc"))
            if hits:
                os.environ["PATH"] = f"{hits[-1].parent}{os.pathsep}{os.environ.get('PATH', '')}"
                break


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
