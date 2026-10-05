#!/usr/bin/env bash
#SBATCH --job-name=stagek-prod-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${STAGE_K_PRIVATE_ENV:?}"
source "$STAGE_K_PRIVATE_ENV"
mkdir -p "$STAGE_K_RUN_DIR/preflight/deployment" logs
for file in experiments/test/wildfire_tests/stage_k_case_study/pace/*.sh \
  experiments/test/wildfire_tests/stage_k_case_study/pace/*.sbatch; do
  bash -n "$file"
done
python -m pytest experiments/test/wildfire_tests/stage_k_case_study/tests -q \
  | tee "$STAGE_K_RUN_DIR/preflight/deployment/pytest.txt"
bash experiments/test/wildfire_tests/stage_k_case_study/pace/submit_production.sh --dry-run \
  | tee "$STAGE_K_RUN_DIR/preflight/deployment/production_dry_run.txt"
