"""Train one arm.

    python -m benchmarks.train --config benchmarks/configs/alfworld_empg.yaml --seed 0
    python -m benchmarks.train --config benchmarks/configs/alfworld_grpo.yaml train.iterations=50
"""

import argparse

from empg.config import load_config
from empg.trainer import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. advantage.zeta=0.1")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg.seed = args.seed
    train(cfg)


if __name__ == "__main__":
    main()
