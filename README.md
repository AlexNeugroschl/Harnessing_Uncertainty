# Harnessing Uncertainty

Research on whether increasing curiosity after an environment changes improves
adaptation on long-horizon tasks. The main experiment uses a controlled text
crafting suite backed by the pinned NovelGridWorldsV2 engine. Tasks require
multi-stage plans; tool, recipe, resource, or reward rules can change during
training. These are custom experiments, not published NovelGym benchmark scores.

The 12-hour experiment compares constant curiosity, high constant
curiosity, an average-coefficient control, and temporary boosts triggered by
either known changes or observed failures. It includes stationary controls,
paired training seeds, baseline competence checks, and held-out evaluations.

## Run the prepared experiment

The recommended follow-up to the completed all-zero v3 experiment is the
[development-gated recovery protocol](docs/recovery_protocol.md):

```powershell
python run_novelgym_recovery.py --prepare-only
python run_novelgym_recovery.py --estimate-only
python run_novelgym_recovery.py
```

This uses varied baseline demonstrations, observable action/progress diagnostics,
a two-seed short-task learning gate, inventory/selection novelty, and a detector
that rearms after a stable baseline. The new root is
`runs/nonstationary/recovery_12h/`. The historical commands and artifacts below
retain the previous experiment.

## Historical v3 commands

Inspect or regenerate the completed v3 experiment from PowerShell:

```powershell
python run_novelgym_12h.py --estimate-only
python run_novelgym_12h.py --report-only
```

The v3 run completed in 11 hours 43 minutes with no changed-rule successes.
The first command inspects that frozen plan and the second rebuilds its reports.
Windows forwards to the existing Ubuntu-24.04 WSL GPU environment. Keep the
machine awake and plugged in while it runs.

Outputs live in `runs/nonstationary/12h/`: `plan.json` freezes the protocol,
`system_run.json` records progress, `matrix/` holds adapters and metrics, and
`results/` contains comparisons, paired intervals, and learning curves.

See [the 12-hour workflow](docs/nonstationary_12h.md) for preparation on another
machine, exact controls, budget fitting, optional replications, and interruption
behavior. [Validation](docs/nonstationary_validation.md) records real GPU checks
and their limits. A short integration run establishes functionality; the long run
is needed to measure whether curiosity boosts help.

## Setup and custom experiments

Inside the existing WSL environment:

```bash
source ~/venvs/empg-vllm/bin/activate
python -m pip install -r requirements-novelgym.txt
pytest tests -q
```

For a new installation, use `scripts/setup_wsl.sh` with `EMPG_VLLM=1` and
`EMPG_VENV=~/venvs/empg-vllm`, then install the native dependency above.

```powershell
python run_novelgym.py --estimate-only
python run_novelgym_warmup.py
python run_novelgym_calibrate.py --initial-adapter runs/novelgym_warmup/adapter --tune
```

[The research protocol](docs/nonstationary_research.md) describes longer
dependency chains, recurring changes, dose sweeps, reward shifts, and the optional
official AdaExploreBench evaluation through `run_adaexplore.py`.

## Code organization

```text
run_novelgym_12h.py          prepare, validate, freeze, and execute the bounded study
run_novelgym_recovery.py     v4 development-gated study with more adaptation rounds
run_novelgym_pilot.py        short-task learning gate with paired development probes
run_novelgym_diagnose.py     exact action-catalog and shared-rollout novelty diagnostics
run_novelgym.py              configurable non-stationary experiment matrix
run_novelgym_warmup.py       shared baseline-only skill initialization
run_novelgym_calibrate.py    native timing, competence, and memory checks
run_adaexplore.py            complementary official static/dynamic evaluation
benchmarks/nonstationary/
  overnight.py              whole-run deadline and optional seed replications
  matrix.py                 treatments, configuration, execution, and exact reuse
  budget.py calibration.py  measured workload fitting and GPU selection
  warmup.py                 recovery demonstrations and held-out competence gates
  reporting.py              paired comparisons and adaptation curves
  adaexplore.py             upstream runtime and batched local inference
benchmarks/configs/tiers/   native screen, longer suite, and 12-hour presets
empg/                       shared policy, rollout, curiosity, advantages, and PPO
empg/envs/novelgym_env.py    real native crafting transitions and text interface
tests/                      CPU invariants, native environments, and integration checks
docs/                       protocol, validation, and historical research notes
```

GPU execution uses batched vLLM generation, prefix caching, sleeping inference
workers during LoRA updates, bf16, bounded microbatches, and exact response-only
vocabulary scoring. Repeated identical responses share computation with their
original multiplicity, advantage, and optimizer boundaries preserved. Calibration
rejects settings that fail memory stress checks.

The [original ALFWorld replication guide](docs/replication_guide.md),
[paper fidelity review](docs/paper_fidelity_review.md), and historical code and
artifacts remain available. `run_benchmarks.py` retains the replication workflow;
the `run_novelgym*` commands are the main research entry points.
