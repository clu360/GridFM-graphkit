#!/usr/bin/env bash
#SBATCH --job-name=stagek-prod-a100-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${STAGE_K_PROJECT_DIR:?}"
: "${STAGE_K_RUN_DIR:?}"
: "${STAGE_K_PYTHON:?}"
: "${STAGE_K_GRIDSFM_ROOT:?}"
: "${STAGE_K_GRIDSFM_CHECKPOINT:?}"
: "${STAGE_K_GPU_PARTITION:?}"
: "${STAGE_K_GRIDSFM_ENV_CONTRACT:?}"
test "${SLURM_JOB_PARTITION:-}" = "$STAGE_K_GPU_PARTITION"
cd "$STAGE_K_PROJECT_DIR"
mkdir -p "$STAGE_K_RUN_DIR/preflight/gpu" "$STAGE_K_RUN_DIR/runtime" logs
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv
/usr/bin/time -v -o "$STAGE_K_RUN_DIR/runtime/gpu_preflight_job_time.txt" \
"$STAGE_K_PYTHON" experiments/test/wildfire_tests/stage_k_case_study/scripts/preflight.py \
  --scope gpu --config experiments/test/wildfire_tests/stage_k_case_study/config/full.yaml \
  --prepared-dir "$STAGE_K_RUN_DIR/prepared" --output-dir "$STAGE_K_RUN_DIR/preflight/gpu" \
  --gridsfm-root "$STAGE_K_GRIDSFM_ROOT" --checkpoint "$STAGE_K_GRIDSFM_CHECKPOINT" \
  --environment-contract "$STAGE_K_GRIDSFM_ENV_CONTRACT" \
  --device cuda:0
