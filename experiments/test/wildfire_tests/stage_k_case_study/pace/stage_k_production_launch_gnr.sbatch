#!/usr/bin/env bash
#SBATCH --job-name=stagek-prod-launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

: "${STAGE_K_PRIVATE_ENV:?}"
source "$STAGE_K_PRIVATE_ENV"
bash experiments/test/wildfire_tests/stage_k_case_study/pace/submit_production.sh --submit
