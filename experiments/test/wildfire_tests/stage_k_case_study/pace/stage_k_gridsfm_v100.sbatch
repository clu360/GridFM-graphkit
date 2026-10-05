#!/usr/bin/env bash
#SBATCH --job-name=stagek-gsfm-v100
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
: "${STAGE_K_GRIDSFM_ENV_MANIFEST:?}"
test "${SLURM_JOB_PARTITION:-}" = gpu-v100
test -f "$STAGE_K_GRIDSFM_ENV_MANIFEST"
cd "$STAGE_K_PROJECT_DIR"
mkdir -p "$STAGE_K_RUN_DIR/evaluators" "$STAGE_K_RUN_DIR/runtime" logs
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/usr/bin/time -v -o "$STAGE_K_RUN_DIR/runtime/gridsfm_job_time.txt" \
"$STAGE_K_PYTHON" -m experiments.test.wildfire_tests.stage_k_case_study.src.evaluator_runner \
  --evaluator gridsfm --config experiments/test/wildfire_tests/stage_k_case_study/config/smoke.yaml \
  --prepared-dir "$STAGE_K_RUN_DIR/prepared" --output-dir "$STAGE_K_RUN_DIR/evaluators" \
  --gridsfm-root "$STAGE_K_GRIDSFM_ROOT" --checkpoint "$STAGE_K_GRIDSFM_CHECKPOINT" \
  --environment-manifest "$STAGE_K_GRIDSFM_ENV_MANIFEST" \
  --environment-contract experiments/test/wildfire_tests/stage_k_case_study/environment/GRIDSFM_SMOKE_V100_CONTRACT.json \
  --device cuda:0
