#!/usr/bin/env bash
#SBATCH --job-name=stagek-env-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${STAGE_K_PROJECT_DIR:?}"
: "${STAGE_K_GRIDSFM_ENV_PREFIX:?}"
: "${STAGE_K_GRIDSFM_ROOT:?}"
: "${STAGE_K_GRIDSFM_CHECKPOINT_DIR:?}"

module load anaconda3/2023.03 cuda/12.6.1 >/dev/null 2>&1
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$STAGE_K_PROJECT_DIR"
mkdir -p logs
bash experiments/test/wildfire_tests/stage_k_case_study/environment/setup_gridsfm_a100.sh
"$STAGE_K_GRIDSFM_ENV_PREFIX/bin/python" -m pip check
"$STAGE_K_GRIDSFM_ENV_PREFIX/bin/python" - <<'PY'
import platform
import numpy
import scipy
import torch
import torch_geometric
import huggingface_hub
import gridsfm

print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_geometric": torch_geometric.__version__,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "gridsfm": gridsfm.__version__,
})
PY
