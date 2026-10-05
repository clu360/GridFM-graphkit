#!/usr/bin/env bash
#SBATCH --job-name=stagek-prod-gsfm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
: "${STAGE_K_PROJECT_DIR:?}"
: "${STAGE_K_RUN_DIR:?}"
: "${STAGE_K_PYTHON:?}"
: "${STAGE_K_GRIDSFM_ROOT:?}"
: "${STAGE_K_GRIDSFM_CHECKPOINT:?}"
: "${STAGE_K_GRIDSFM_ENV_CONTRACT:?}"
: "${STAGE_K_GRIDSFM_ENV_MANIFEST:?}"
: "${STAGE_K_GPU_PARTITION:?}"
test "${SLURM_JOB_PARTITION:-}" = "$STAGE_K_GPU_PARTITION"
test -f "$STAGE_K_GRIDSFM_ENV_MANIFEST"
lambdas=(0.0 0.2 0.5 0.8 1.0)
slugs=(lambda_0 lambda_0.2 lambda_0.5 lambda_0.8 lambda_1)
lambda_r="${lambdas[$SLURM_ARRAY_TASK_ID]}"
slug="${slugs[$SLURM_ARRAY_TASK_ID]}"
out="$STAGE_K_RUN_DIR/chunks/gridsfm/$slug"
cd "$STAGE_K_PROJECT_DIR"
mkdir -p "$out" "$STAGE_K_RUN_DIR/runtime" logs
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/time -v -o "$STAGE_K_RUN_DIR/runtime/gridsfm_${slug}_job_time.txt" \
"$STAGE_K_PYTHON" -m experiments.test.wildfire_tests.stage_k_case_study.src.evaluator_runner \
  --evaluator gridsfm --lambda-r "$lambda_r" \
  --config experiments/test/wildfire_tests/stage_k_case_study/config/full.yaml \
  --prepared-dir "$STAGE_K_RUN_DIR/prepared" --output-dir "$out" \
  --gridsfm-root "$STAGE_K_GRIDSFM_ROOT" --checkpoint "$STAGE_K_GRIDSFM_CHECKPOINT" \
  --environment-manifest "$STAGE_K_GRIDSFM_ENV_MANIFEST" \
  --environment-contract "$STAGE_K_GRIDSFM_ENV_CONTRACT" --device cuda:0
