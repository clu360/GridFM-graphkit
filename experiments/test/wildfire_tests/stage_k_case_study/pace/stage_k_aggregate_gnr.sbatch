#!/usr/bin/env bash
#SBATCH --job-name=stagek-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${STAGE_K_PROJECT_DIR:?Set STAGE_K_PROJECT_DIR}"
: "${STAGE_K_RUN_DIR:?Set STAGE_K_RUN_DIR}"
: "${STAGE_K_PYTHON:?Set STAGE_K_PYTHON}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
cd "$STAGE_K_PROJECT_DIR"
mkdir -p "$STAGE_K_RUN_DIR/report" "$STAGE_K_RUN_DIR/runtime" logs

if [[ -f "$STAGE_K_RUN_DIR/submitted_job_ids.csv" ]]; then
  job_ids=$(tail -n +2 "$STAGE_K_RUN_DIR/submitted_job_ids.csv" | cut -d, -f2 | paste -sd, -)
  sacct -j "$job_ids" --parsable2 \
    --format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,AllocCPUS,NodeList \
    > "$STAGE_K_RUN_DIR/runtime/slurm_sacct.psv" || true
fi

"$STAGE_K_PYTHON" -u -m experiments.test.wildfire_tests.stage_k_case_study.src.aggregate \
  --run-dir "$STAGE_K_RUN_DIR/evaluators" \
  --output-dir "$STAGE_K_RUN_DIR/report"
