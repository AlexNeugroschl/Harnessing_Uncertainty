"""Run the benchmark suite: base model + every arm x seeds, train then evaluate.

    python -m benchmarks.suite --seeds 0 1 2
    python -m benchmarks.suite --arms grpo empg --seeds 0 train.iterations=30

Finished runs (eval json present) are skipped, so the suite can be resumed.
Each run is a subprocess so GPU memory is fully released between runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from benchmarks.evaluate import eval_path
from empg.config import load_config

CONFIGS = Path(__file__).parent / "configs"
ARMS = {
    "grpo": CONFIGS / "alfworld_grpo.yaml",
    "empg": CONFIGS / "alfworld_empg.yaml",
    "empg_g_only": CONFIGS / "ablation_g_only.yaml",
    "empg_f_only": CONFIGS / "ablation_f_only.yaml",
}


def run(module: str, config: Path, seed: int, overrides: list[str], *extra: str):
    cmd = [sys.executable, "-m", module, "--config", str(config), "--seed", str(seed),
           *extra, *overrides]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("overrides", nargs="*", help="dotted overrides applied to every run")
    args = ap.parse_args()

    if not args.skip_base:
        cfg = load_config(str(CONFIGS / "base.yaml"), args.overrides)
        if not eval_path(cfg, base=True).exists():
            run("benchmarks.evaluate", CONFIGS / "base.yaml", 0, args.overrides, "--base")

    for seed in args.seeds:
        for arm in args.arms:
            cfg = load_config(str(ARMS[arm]), args.overrides)
            cfg.seed = seed
            if eval_path(cfg).exists():
                print(f"skip {arm} s{seed} (done)")
                continue
            if not (cfg.run_dir / "adapter").exists() or not _finished(cfg):
                run("benchmarks.train", ARMS[arm], seed, args.overrides)
            run("benchmarks.evaluate", ARMS[arm], seed, args.overrides)


def _finished(cfg) -> bool:
    m = cfg.run_dir / "metrics.jsonl"
    return m.exists() and sum(1 for _ in m.open()) >= cfg.train.iterations


if __name__ == "__main__":
    main()
