"""Train one arm.

    python -m benchmarks.train --config empg --tier smoke
    python -m benchmarks.train --config grpo_cur_group --tier small --seed 1
    python -m benchmarks.train --config empg --tier small advantage.curiosity.weight=0.1 name=empg_eta01
"""

import argparse

from benchmarks.common import add_args, load
from empg.trainer import train


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--evaluate", action="store_true", help="evaluate with the same loaded model/engine")
    args = ap.parse_args()
    cfg = load(args.config, args.tier, args.overrides, args.seed)
    policy = train(cfg)
    try:
        if args.evaluate:
            from benchmarks.evaluate import save_evaluation
            save_evaluation(policy, cfg)
    finally:
        policy.close()


if __name__ == "__main__":
    main()
