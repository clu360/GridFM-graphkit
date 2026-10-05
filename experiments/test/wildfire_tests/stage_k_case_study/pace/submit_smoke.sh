#!/usr/bin/env bash
set -euo pipefail

SUBMIT=false
if [[ "${1:-}" == "--submit" ]]; then
  SUBMIT=true
elif [[ -n "${1:-}" && "${1:-}" != "--dry-run" ]]; then
  echo "Usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi

: "${STAGE_K_PROJECT_DIR:?Set STAGE_K_PROJECT_DIR}"
: "${STAGE_K_RUN_DIR:?Set STAGE_K_RUN_DIR}"
: "${STAGE_K_CPU_RESOURCE_ARGS:?Set complete discovered GNR sbatch resource arguments}"
: "${STAGE_K_GPU_RESOURCE_ARGS:?Set complete discovered A100 sbatch resource arguments, including one GPU}"

PACE_DIR="$STAGE_K_PROJECT_DIR/experiments/test/wildfire_tests/stage_k_case_study/pace"
ACCOUNT_ARGS=()
if [[ -n "${PACE_ACCOUNT:-}" ]]; then ACCOUNT_ARGS=(--account="$PACE_ACCOUNT"); fi

read -r -a CPU_RESOURCE_ARGS <<< "$STAGE_K_CPU_RESOURCE_ARGS"
read -r -a GPU_RESOURCE_ARGS <<< "$STAGE_K_GPU_RESOURCE_ARGS"
grid_cmd=(sbatch --parsable "${ACCOUNT_ARGS[@]}" "${GPU_RESOURCE_ARGS[@]}" "$PACE_DIR/stage_k_gridsfm_v100.sbatch")
dc_cmd=(sbatch --parsable "${ACCOUNT_ARGS[@]}" "${CPU_RESOURCE_ARGS[@]}" "$PACE_DIR/stage_k_dc_gnr.sbatch")
ac_cmd=(sbatch --parsable "${ACCOUNT_ARGS[@]}" "${CPU_RESOURCE_ARGS[@]}" "$PACE_DIR/stage_k_ac_gnr.sbatch")

if [[ "$SUBMIT" != true ]]; then
  printf 'DRY RUN: '; printf '%q ' "${grid_cmd[@]}"; printf '\n'
  printf 'DRY RUN: '; printf '%q ' "${dc_cmd[@]}"; printf '\n'
  printf 'DRY RUN: '; printf '%q ' "${ac_cmd[@]}"; printf '\n'
  echo "DRY RUN: reference job uses afterok:<gridsfm>:<dc>:<ac> with $STAGE_K_CPU_RESOURCE_ARGS"
  echo "DRY RUN: aggregate job uses afterok:<references> with $STAGE_K_CPU_RESOURCE_ARGS"
  exit 0
fi

grid_id=$("${grid_cmd[@]}")
dc_id=$("${dc_cmd[@]}")
ac_id=$("${ac_cmd[@]}")
ref_id=$(sbatch --parsable "${ACCOUNT_ARGS[@]}" "${CPU_RESOURCE_ARGS[@]}" \
  --dependency="afterok:${grid_id}:${dc_id}:${ac_id}" "$PACE_DIR/stage_k_references_gnr.sbatch")
aggregate_id=$(sbatch --parsable "${ACCOUNT_ARGS[@]}" "${CPU_RESOURCE_ARGS[@]}" \
  --dependency="afterok:${ref_id}" "$PACE_DIR/stage_k_aggregate_gnr.sbatch")
mkdir -p "$STAGE_K_RUN_DIR"
printf 'job,job_id\ngridsfm,%s\ndc,%s\nac,%s\nreferences,%s\naggregate,%s\n' \
  "$grid_id" "$dc_id" "$ac_id" "$ref_id" "$aggregate_id" > "$STAGE_K_RUN_DIR/submitted_job_ids.csv"
printf 'GridSFM=%s\nDC=%s\nAC=%s\nReferences=%s\nAggregate=%s\n' \
  "$grid_id" "$dc_id" "$ac_id" "$ref_id" "$aggregate_id"
