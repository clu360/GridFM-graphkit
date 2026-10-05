#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONTRACT="$HERE/GRIDSFM_ENVIRONMENT_CONTRACT.json"
ENVIRONMENT="$HERE/gridsfm_a100_environment.yml"
CONSTRAINTS="$HERE/gridsfm_a100_constraints.txt"

: "${STAGE_K_GRIDSFM_ENV_PREFIX:?Set the absolute Conda environment prefix}"
: "${STAGE_K_GRIDSFM_ROOT:?Set the absolute official GridSFM checkout path}"
: "${STAGE_K_GRIDSFM_CHECKPOINT_DIR:?Set the durable checkpoint cache directory}"

CONDA_EXE=${CONDA_EXE:-conda}
GIT_REPOSITORY=$(python -c "import json; print(json.load(open('$CONTRACT'))['source']['git_repository'])")
GIT_COMMIT=$(python -c "import json; print(json.load(open('$CONTRACT'))['source']['git_commit'])")

if [[ ! -d "$STAGE_K_GRIDSFM_ROOT/.git" ]]; then
  git clone "$GIT_REPOSITORY" "$STAGE_K_GRIDSFM_ROOT"
fi
if [[ "$(git -C "$STAGE_K_GRIDSFM_ROOT" remote get-url origin)" != "$GIT_REPOSITORY" ]]; then
  echo "GridSFM origin does not match the frozen Microsoft repository" >&2
  exit 1
fi
if [[ -n "$(git -C "$STAGE_K_GRIDSFM_ROOT" status --porcelain)" ]]; then
  echo "GridSFM checkout is modified; refusing to replace local work" >&2
  exit 1
fi
if ! git -C "$STAGE_K_GRIDSFM_ROOT" cat-file -e "$GIT_COMMIT^{commit}" 2>/dev/null; then
  git -C "$STAGE_K_GRIDSFM_ROOT" fetch origin "$GIT_COMMIT"
fi
git -C "$STAGE_K_GRIDSFM_ROOT" checkout --detach "$GIT_COMMIT"

if [[ ! -x "$STAGE_K_GRIDSFM_ENV_PREFIX/bin/python" ]]; then
  "$CONDA_EXE" env create --prefix "$STAGE_K_GRIDSFM_ENV_PREFIX" --file "$ENVIRONMENT"
fi
PYTHON="$STAGE_K_GRIDSFM_ENV_PREFIX/bin/python"
if [[ "$("$PYTHON" -c 'import platform; print(platform.python_version())')" != "3.11.9" ]]; then
  echo "The Stage K GridSFM environment must use Python 3.11.9" >&2
  exit 1
fi
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu126 \
  "torch==2.7.1"
"$PYTHON" -m pip install --constraint "$CONSTRAINTS" \
  "$STAGE_K_GRIDSFM_ROOT/model" \
  pandas==2.2.3 pyarrow==19.0.1 PyYAML==6.0.2 matplotlib==3.10.1 lightning==2.6.6 pytest==8.3.5
"$PYTHON" -m pip check
"$PYTHON" "$HERE/download_gridsfm_checkpoint.py" \
  --checkpoint-dir "$STAGE_K_GRIDSFM_CHECKPOINT_DIR"

printf 'Environment: %s\n' "$STAGE_K_GRIDSFM_ENV_PREFIX"
printf 'GridSFM:    %s @ %s\n' "$STAGE_K_GRIDSFM_ROOT" "$GIT_COMMIT"
printf 'Checkpoint: %s/%s\n' "$STAGE_K_GRIDSFM_CHECKPOINT_DIR" \
  "$("$PYTHON" -c "import json; print(json.load(open('$CONTRACT'))['checkpoint']['filename'])")"
