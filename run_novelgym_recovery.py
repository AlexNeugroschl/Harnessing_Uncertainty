"""Run the development-gated recovery study; preserves the previous experiment."""
import sys
from benchmarks.nonstationary.overnight import main

if __name__ == "__main__":
    defaults = ["--tier", "novelgym_recovery_12h", "--root", "runs/nonstationary/recovery_12h",
                "--minimum-updates", "16", "--maximum-updates", "128", "--pre-change-updates", "4", "--no-reserve",
                "--treatments", "constant", "oracle", "detected"]
    raise SystemExit(main([*defaults, *sys.argv[1:]]))
