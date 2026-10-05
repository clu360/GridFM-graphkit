#!/usr/bin/env bash
#SBATCH --job-name=stagek-refs
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
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
cd "$STAGE_K_PROJECT_DIR"
mkdir -p "$STAGE_K_RUN_DIR/references" "$STAGE_K_RUN_DIR/report" "$STAGE_K_RUN_DIR/runtime" logs

/usr/bin/time -v -o "$STAGE_K_RUN_DIR/runtime/references_job_time.txt" \
"$STAGE_K_PYTHON" -m experiments.test.wildfire_tests.stage_k_case_study.src.reference_runner \
  --config experiments/test/wildfire_tests/stage_k_case_study/config/smoke.yaml \
  --prepared-dir "$STAGE_K_RUN_DIR/prepared" \
  --evaluator-dir "$STAGE_K_RUN_DIR/evaluators" \
  --output-dir "$STAGE_K_RUN_DIR/references"
