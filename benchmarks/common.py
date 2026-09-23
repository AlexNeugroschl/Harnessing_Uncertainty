"""Shared CLI plumbing: arm configs x size tiers."""

from __future__ import annotations

import argparse
from pathlib import Path

from empg.config import Config, load_config

CONFIGS = Path(__file__).parent / "configs"
BASE = CONFIGS / "base.yaml"


def arms() -> dict[str, Path]:
    return {p.stem: p for p in sorted((CONFIGS / "arms").glob("*.yaml"))}


def tiers() -> list[str]:
    return sorted(p.stem for p in (CONFIGS / "tiers").glob("*.yaml"))


def resolve_config(config: str) -> Path:
    """An arm name (``empg``) or a path to a YAML file."""
    return arms().get(config) or Path(config)


def load(config: str | Path, tier: str | None = None, overrides=None, seed: int = 0) -> Config:
    layers = [str(CONFIGS / "tiers" / f"{tier}.yaml")] if tier else []
    cfg = load_config(str(resolve_config(str(config))), overrides, layers)
    cfg.seed = seed
    return cfg


def add_args(ap: argparse.ArgumentParser, config_required: bool = True):
    ap.add_argument("--config", required=config_required,
                    help=f"arm name ({', '.join(arms())}) or YAML path")
    ap.add_argument("--tier", choices=tiers(), help="run size: " + ", ".join(tiers()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. advantage.zeta=0.1")
