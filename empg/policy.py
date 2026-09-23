"""LLM policy: a causal LM + LoRA with batched generation and scoring.

Token entropy is computed in a no-grad forward pass *after* generation, rather
than from logits stored during generate(): keeping a 151k-vocab logit tensor
for every generated token would not fit in 8GB. The parameters are unchanged
between rollout and scoring, so the result is the same distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


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


class LLMPolicy:
    def __init__(self, cfg: PolicyConfig, device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = getattr(torch, cfg.dtype) if self.device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype)
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
        self.model.eval()

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
    @torch.no_grad()
    def generate(self, prompts: list[str], greedy: bool = False):
        """Returns (prompt_ids, response_ids, texts), one per prompt (ids as lists)."""
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
                pad_token_id=pad,
            )
            gen = out[:, batch["input_ids"].shape[1]:].tolist()
            for p, r in zip(p_ids, gen):
                r = self._trim(r, eos, pad)
                all_p.append(p)
                all_r.append(r)
                all_t.append(self.tokenizer.decode(r, skip_special_tokens=True))
        return all_p, all_r, all_t

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
        logits = self.model(**batch, position_ids=pos, use_cache=False,
                            logits_to_keep=R + 1).logits
        logits = logits[:, :-1].float()
        targets = batch["input_ids"][:, n - R:]
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for i, r in enumerate(response_ids):
            mask[i, R - len(r):] = True
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ent = -(logp_all.exp() * logp_all).sum(-1) if with_entropy else None
        return logp, ent, mask

    @torch.no_grad()
    def score(self, prompt_ids, response_ids, with_entropy=True, reference=False):
        """No-grad chunked scoring. Returns lists of 1-D CPU tensors (one per sample)."""
        self.model.eval()
        logps, ents = [], []
        mb = self.cfg.micro_batch_size
        ctx = self.model.disable_adapter() if (reference and self.use_lora) else _null()
        with ctx:
            for s in range(0, len(prompt_ids), mb):
                lp, ent, mask = self.forward_logprobs(
                    prompt_ids[s:s + mb], response_ids[s:s + mb], with_entropy)
                for i in range(lp.shape[0]):
                    logps.append(lp[i][mask[i]].cpu())
                    if with_entropy:
                        ents.append(ent[i][mask[i]].cpu())
        return logps, ents


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
