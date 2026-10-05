#!/usr/bin/env bash
#SBATCH --job-name=stagek-gpu-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${STAGE_K_PROJECT_DIR:?Set STAGE_K_PROJECT_DIR}"
: "${STAGE_K_RUN_DIR:?Set STAGE_K_RUN_DIR}"
: "${STAGE_K_PYTHON:?Set STAGE_K_PYTHON}"
: "${STAGE_K_GRIDSFM_ROOT:?Set STAGE_K_GRIDSFM_ROOT}"
: "${STAGE_K_GRIDSFM_CHECKPOINT:?Set STAGE_K_GRIDSFM_CHECKPOINT}"

test "${SLURM_JOB_PARTITION:-}" = gpu-a100
cd "$STAGE_K_PROJECT_DIR"
mkdir -p "$STAGE_K_RUN_DIR/preflight/gpu" "$STAGE_K_RUN_DIR/runtime" logs

nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

/usr/bin/time -v -o "$STAGE_K_RUN_DIR/runtime/gpu_preflight_job_time.txt" \
"$STAGE_K_PYTHON" experiments/test/wildfire_tests/stage_k_case_study/scripts/preflight.py \
  --scope gpu \
  --config experiments/test/wildfire_tests/stage_k_case_study/config/smoke.yaml \
  --prepared-dir "$STAGE_K_RUN_DIR/prepared" \
  --output-dir "$STAGE_K_RUN_DIR/preflight/gpu" \
  --gridsfm-root "$STAGE_K_GRIDSFM_ROOT" \
  --checkpoint "$STAGE_K_GRIDSFM_CHECKPOINT" \
  --device cuda:0
