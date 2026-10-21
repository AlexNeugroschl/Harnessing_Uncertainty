"""Experiment config: nested dataclasses loaded from YAML with dotted CLI overrides.

A YAML file may name a parent with ``inherit: other.yaml`` (path relative to it).
Overrides look like ``advantage.zeta=0.1`` or ``train.iterations=20``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from empg.advantages import AdvantageConfig
from empg.policy import PolicyConfig


@dataclass
class TrainConfig:
    iterations: int = 100
    games_per_iter: int = 4      # distinct tasks per iteration
    group_size: int = 8          # rollouts per task (GRPO group)
    max_steps: int = 30          # env turns per episode
    success_reward: float = 10.0
    invalid_penalty: float = 0.1
    lr: float = 1e-5
    weight_decay: float = 0.0
    kl_coef: float = 0.01
    entropy_coef: float = 0.0    # differentiable entropy bonus; distinct from EMPG's detached H_t
    reward_mode: str = "trajectory"  # legacy episode penalties | verl step scores/group normalization
    clip_low: float = 0.2
    clip_high: float = 0.2       # > clip_low gives DAPO's clip-higher
    dual_clip: float = 0.0       # 0: legacy PPO; pinned Verl actor bounds negative advantages at 3
    ppo_epochs: int = 1
    mini_batches: int = 4        # optimizer steps per epoch
    mini_batch_size: int = 0     # response rows per optimizer step; 0 uses legacy mini_batches
    shuffle: bool = True        # legacy random rows; pinned actor uses sequential minibatches
    max_grad_norm: float = 1.0
    save_every: int = 25
    max_env_steps: int = 0       # stop at a batch boundary; 0 = iterations only
    max_seconds: float = 0.0     # rollout+update budget; may overshoot by one batch


@dataclass
class EvalConfig:
    split: str = "valid_unseen"  # ALFWorld: valid_seen | valid_unseen
    max_tasks: int = 0           # 0 = all
    max_steps: int = 50
    greedy: bool = True
    temperature: float | None = None  # stochastic eval override; None preserves policy temperature
    sampling_seed: int | None = None  # explicit eval RNG, independent of training consumption
    batch_envs: int = 32
    selection: str = "random"   # random | stratified (proportional task-type allocation)
    selection_seed: int = 0     # independent of training seed, shared by all arms


@dataclass
class Config:
    name: str = "run"
    seed: int = 0
    out_dir: str = "runs"
    benchmark: str = "alfworld"
    env: dict = field(default_factory=dict)  # benchmark-specific kwargs
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / f"{self.name}_s{self.seed}"


def _optional_defaults(data: dict) -> dict:
    """Keep legacy artifact fingerprints valid when new options are disabled."""
    result = dict(data)
    for section, defaults in {
        "train": {"entropy_coef": 0.0, "reward_mode": "trajectory", "mini_batch_size": 0,
                  "shuffle": True, "dual_clip": 0.0},
        "policy": {"amp": False, "vllm_max_num_seqs": 256, "max_prompt_tokens": 0},
        "eval": {"temperature": None, "sampling_seed": None},
        "advantage": {"entropy_eps": 1e-6, "scale_eps": 0.0},
    }.items():
        if section in result:
            result[section] = dict(result[section])
            for key, default in defaults.items():
                if result[section].get(key) == default:
                    result[section].pop(key, None)
    return result


def config_dict(cfg: Config) -> dict:
    return _optional_defaults(dataclasses.asdict(cfg))


def config_matches(saved: dict, cfg: Config) -> bool:
    return _optional_defaults(saved) == config_dict(cfg)


def config_fingerprint(cfg: Config) -> str:
    return hashlib.sha256(json.dumps(config_dict(cfg), sort_keys=True).encode()).hexdigest()


def _read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("inherit", None)
    if parent:
        base = _read_yaml(path.parent / parent)
        data = _merge(base, data)
    return data


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _build(cls, data: dict):
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for k, v in data.items():
        if k not in fields:
            raise KeyError(f"unknown config key {cls.__name__}.{k}")
        sub = fields[k].default_factory  # the nested dataclass type, if any
        kwargs[k] = _build(sub, v) if dataclasses.is_dataclass(sub) and isinstance(v, dict) else v
    return cls(**kwargs)


def load_config(path: str | None = None, overrides: list[str] | None = None,
                layers: list[str] | None = None) -> Config:
    """``path`` YAML, then each ``layers`` YAML merged on top (e.g. a size tier),
    then dotted ``overrides``."""
    data = _read_yaml(Path(path)) if path else {}
    for layer in layers or []:
        data = _merge(data, _read_yaml(Path(layer)))
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = data
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return _build(Config, data)
