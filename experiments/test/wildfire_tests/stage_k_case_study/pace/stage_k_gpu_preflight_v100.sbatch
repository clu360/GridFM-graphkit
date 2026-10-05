#!/usr/bin/env bash
#SBATCH --job-name=stagek-v100-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${STAGE_K_PROJECT_DIR:?}"
: "${STAGE_K_RUN_DIR:?}"
: "${STAGE_K_PYTHON:?}"
: "${STAGE_K_GRIDSFM_ROOT:?}"
: "${STAGE_K_GRIDSFM_CHECKPOINT:?}"
test "${SLURM_JOB_PARTITION:-}" = gpu-v100
cd "$STAGE_K_PROJECT_DIR"
mkdir -p "$STAGE_K_RUN_DIR/preflight/gpu_v100" "$STAGE_K_RUN_DIR/runtime" logs
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/time -v -o "$STAGE_K_RUN_DIR/runtime/gpu_v100_preflight_job_time.txt" \
"$STAGE_K_PYTHON" experiments/test/wildfire_tests/stage_k_case_study/scripts/preflight.py \
  --scope gpu --config experiments/test/wildfire_tests/stage_k_case_study/config/smoke.yaml \
  --prepared-dir "$STAGE_K_RUN_DIR/prepared" --output-dir "$STAGE_K_RUN_DIR/preflight/gpu_v100" \
  --gridsfm-root "$STAGE_K_GRIDSFM_ROOT" --checkpoint "$STAGE_K_GRIDSFM_CHECKPOINT" \
  --environment-contract experiments/test/wildfire_tests/stage_k_case_study/environment/GRIDSFM_SMOKE_V100_CONTRACT.json \
  --device cuda:0
