"""Experiment config: nested dataclasses loaded from YAML with dotted CLI overrides.

A YAML file may name a parent with ``inherit: other.yaml`` (path relative to it).
Overrides look like ``advantage.zeta=0.1`` or ``train.iterations=20``.
"""

from __future__ import annotations

import dataclasses
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
    clip_low: float = 0.2
    clip_high: float = 0.2       # > clip_low gives DAPO's clip-higher
    ppo_epochs: int = 1
    mini_batches: int = 4        # optimizer steps per epoch
    max_grad_norm: float = 1.0
    save_every: int = 25


@dataclass
class EvalConfig:
    split: str = "valid_unseen"  # ALFWorld: valid_seen | valid_unseen
    max_tasks: int = 0           # 0 = all
    max_steps: int = 50
    greedy: bool = True
    batch_envs: int = 32


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


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    data = _read_yaml(Path(path)) if path else {}
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = data
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return _build(Config, data)
