#!/usr/bin/env bash
#SBATCH --job-name=stagek-prod-join
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${STAGE_K_PROJECT_DIR:?}"
: "${STAGE_K_RUN_DIR:?}"
: "${STAGE_K_PYTHON:?}"
cd "$STAGE_K_PROJECT_DIR"
mkdir -p "$STAGE_K_RUN_DIR/evaluators" logs
"$STAGE_K_PYTHON" -m experiments.test.wildfire_tests.stage_k_case_study.src.production_join \
  --config experiments/test/wildfire_tests/stage_k_case_study/config/full.yaml \
  --prepared-dir "$STAGE_K_RUN_DIR/prepared" --chunk-root "$STAGE_K_RUN_DIR/chunks" \
  --output-dir "$STAGE_K_RUN_DIR/evaluators"

