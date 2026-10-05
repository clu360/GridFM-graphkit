#!/usr/bin/env bash
#SBATCH --job-name=stagek-prod-ref
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
evaluators=(gridsfm dc ac)
lambdas=(0.0 0.2 0.5 0.8 1.0)
slugs=(lambda_0 lambda_0.2 lambda_0.5 lambda_0.8 lambda_1)
evaluator="${evaluators[$((SLURM_ARRAY_TASK_ID / 5))]}"
lambda_r="${lambdas[$((SLURM_ARRAY_TASK_ID % 5))]}"
slug="${slugs[$((SLURM_ARRAY_TASK_ID % 5))]}"
out="$STAGE_K_RUN_DIR/reference_tasks/$evaluator/$slug"
cd "$STAGE_K_PROJECT_DIR"
mkdir -p "$out" "$STAGE_K_RUN_DIR/runtime" logs
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/time -v -o "$STAGE_K_RUN_DIR/runtime/reference_${evaluator}_${slug}_job_time.txt" \
"$STAGE_K_PYTHON" -m experiments.test.wildfire_tests.stage_k_case_study.src.reference_runner \
  --evaluator "$evaluator" --lambda-r "$lambda_r" \
  --config experiments/test/wildfire_tests/stage_k_case_study/config/full.yaml \
  --prepared-dir "$STAGE_K_RUN_DIR/prepared" --evaluator-dir "$STAGE_K_RUN_DIR/evaluators" \
  --output-dir "$out"
