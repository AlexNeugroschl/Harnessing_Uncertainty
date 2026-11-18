"""Compatibility entry point; implementation is in benchmarks.nonstationary.adaexplore."""
from benchmarks.nonstationary.adaexplore import main, LocalBatchClient, official_runtime, evaluate_episodes

if __name__ == "__main__":
    raise SystemExit(main())
