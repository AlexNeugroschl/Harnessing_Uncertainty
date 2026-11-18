"""Run the non-stationary curiosity experiment matrix."""
from benchmarks.nonstationary.matrix import main, parser, plan, treatments, execution_key
from benchmarks.nonstationary.reporting import report, read_jsonl, plot_curves

if __name__ == "__main__":
    raise SystemExit(main())
