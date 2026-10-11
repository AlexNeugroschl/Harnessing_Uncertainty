#!/usr/bin/env bash
# One-time environment setup. Run inside WSL (Ubuntu) from the repo root:
#   bash scripts/setup_wsl.sh
# The venv lives on the Linux filesystem (much faster than /mnt/c).
set -euo pipefail

VENV="${EMPG_VENV:-$HOME/venvs/empg}"
PY="${EMPG_PYTHON:-python3}"

if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

if [ "${EMPG_VLLM:-0}" = "1" ]; then
  # vLLM pins its own torch build, so it gets its own venv, e.g.
  #   EMPG_VLLM=1 EMPG_VENV=~/venvs/empg-vllm bash scripts/setup_wsl.sh
  pip install vllm
else
  # RTX 50xx (Blackwell, sm_120) needs CUDA >= 12.8 builds of torch.
  pip install torch --index-url https://download.pytorch.org/whl/cu128
fi
pip install -r requirements.txt

# ALFWorld game files + PDDL (to ~/.cache/alfworld by default).
alfworld-download

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
echo "Done. Activate with: source $VENV/bin/activate"
