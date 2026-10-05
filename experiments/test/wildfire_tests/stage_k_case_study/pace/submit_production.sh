#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--preflight-submit" && "$MODE" != "--submit" ]]; then
  echo "Usage: $0 [--dry-run|--preflight-submit|--submit]" >&2
  exit 2
fi

: "${STAGE_K_PROJECT_DIR:?}"
: "${STAGE_K_RUN_DIR:?}"
: "${STAGE_K_PYTHON:?}"
: "${STAGE_K_CPU_RESOURCE_ARGS:?}"
: "${STAGE_K_AC_RESOURCE_ARGS:?}"
: "${STAGE_K_REFERENCE_RESOURCE_ARGS:?}"
: "${STAGE_K_GPU_RESOURCE_ARGS:?}"
: "${STAGE_K_GPU_PARTITION:?}"
: "${STAGE_K_GRIDSFM_ENV_CONTRACT:?}"
: "${STAGE_K_GRIDSFM_ENV_MANIFEST:?}"

PACE_DIR="$STAGE_K_PROJECT_DIR/experiments/test/wildfire_tests/stage_k_case_study/pace"
GRID_THROTTLE="${STAGE_K_GRID_THROTTLE:-5}"
DC_THROTTLE="${STAGE_K_DC_THROTTLE:-5}"
AC_THROTTLE="${STAGE_K_AC_THROTTLE:-5}"
REFERENCE_THROTTLE="${STAGE_K_REFERENCE_THROTTLE:-5}"
for value in "$GRID_THROTTLE" "$DC_THROTTLE" "$AC_THROTTLE" "$REFERENCE_THROTTLE"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "array throttles must be positive integers" >&2; exit 2; }
done

read -r -a CPU_ARGS <<< "$STAGE_K_CPU_RESOURCE_ARGS"
read -r -a AC_ARGS <<< "$STAGE_K_AC_RESOURCE_ARGS"
read -r -a REF_ARGS <<< "$STAGE_K_REFERENCE_RESOURCE_ARGS"
read -r -a GPU_ARGS <<< "$STAGE_K_GPU_RESOURCE_ARGS"

if [[ "$MODE" == "--dry-run" ]]; then
  cat <<EOF
PRODUCTION DRY RUN (two gated submissions)
config=full.yaml lambdas=0.0,0.2,0.5,0.8,1.0
phase_1_command=submit_production.sh --preflight-submit
prepare: sbatch ${CPU_ARGS[*]} stage_k_production_prepare_gnr.sbatch
GPU sanity ($STAGE_K_GPU_PARTITION): afterok:<prepare> sbatch ${GPU_ARGS[*]} stage_k_production_gpu_preflight_a100.sbatch
HARD GATE: prepared input and GPU environment manifests must both report PASS
phase_2_command=submit_production.sh --submit
GridSFM: array=0-4%${GRID_THROTTLE} ${GPU_ARGS[*]}
DC: array=0-4%${DC_THROTTLE} ${CPU_ARGS[*]}
AC: array=0-4%${AC_THROTTLE} ${AC_ARGS[*]}
join/seal: afterok:<gridsfm>:<dc>:<ac> ${CPU_ARGS[*]}
references: afterok:<join> array=0-14%${REFERENCE_THROTTLE} ${REF_ARGS[*]}
reference join: afterok:<references> ${CPU_ARGS[*]}
aggregation: afterok:<reference-join> ${CPU_ARGS[*]}
run_dir=$STAGE_K_RUN_DIR
environment_manifest=$STAGE_K_GRIDSFM_ENV_MANIFEST
EOF
  exit 0
fi

mkdir -p "$STAGE_K_RUN_DIR"
if [[ -e "$STAGE_K_RUN_DIR/RUN_COMPLETE.json" ]]; then
  echo "completed production run is immutable: $STAGE_K_RUN_DIR" >&2
  exit 1
fi

if [[ "$MODE" == "--preflight-submit" ]]; then
  if [[ -e "$STAGE_K_RUN_DIR/preflight_job_ids.csv" ]]; then
    echo "production preflight was already submitted: $STAGE_K_RUN_DIR/preflight_job_ids.csv" >&2
    exit 1
  fi
  prepare_id=$(sbatch --parsable "${CPU_ARGS[@]}" "$PACE_DIR/stage_k_production_prepare_gnr.sbatch")
  gpu_preflight_id=$(sbatch --parsable "${GPU_ARGS[@]}" --dependency="afterok:${prepare_id}" \
    "$PACE_DIR/stage_k_production_gpu_preflight_a100.sbatch")
  printf 'job,job_id\nprepare,%s\ngpu_preflight,%s\n' "$prepare_id" "$gpu_preflight_id" \
    > "$STAGE_K_RUN_DIR/preflight_job_ids.csv"
  printf 'Prepare=%s\nGPUPreflight=%s\n' "$prepare_id" "$gpu_preflight_id"
  exit 0
fi

if [[ -e "$STAGE_K_RUN_DIR/submitted_job_ids.csv" ]]; then
  echo "production jobs were already submitted: $STAGE_K_RUN_DIR/submitted_job_ids.csv" >&2
  exit 1
fi
"$STAGE_K_PYTHON" - "$STAGE_K_RUN_DIR/prepared/input_manifest.json" \
  "$STAGE_K_GRIDSFM_ENV_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

for label, raw_path in (("prepared input", sys.argv[1]), ("GPU environment", sys.argv[2])):
    path = Path(raw_path)
    if not path.is_file():
        raise SystemExit(f"production hard gate failed: missing {label} manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise SystemExit(f"production hard gate failed: {label} status={manifest.get('status')!r}")
    checks = manifest.get("checks", {})
    if checks and not all(checks.values()):
        raise SystemExit(f"production hard gate failed: {label} contains failed checks")
PY

grid_id=$(sbatch --parsable "${GPU_ARGS[@]}" --array="0-4%${GRID_THROTTLE}" \
  "$PACE_DIR/stage_k_production_gridsfm_a100.sbatch")
dc_id=$(sbatch --parsable "${CPU_ARGS[@]}" --array="0-4%${DC_THROTTLE}" \
  --export=ALL,STAGE_K_EVALUATOR=dc \
  "$PACE_DIR/stage_k_production_native_gnr.sbatch")
ac_id=$(sbatch --parsable "${AC_ARGS[@]}" --array="0-4%${AC_THROTTLE}" \
  --export=ALL,STAGE_K_EVALUATOR=ac \
  "$PACE_DIR/stage_k_production_native_gnr.sbatch")
join_id=$(sbatch --parsable "${CPU_ARGS[@]}" --dependency="afterok:${grid_id}:${dc_id}:${ac_id}" \
  "$PACE_DIR/stage_k_production_join_gnr.sbatch")
references_id=$(sbatch --parsable "${REF_ARGS[@]}" --array="0-14%${REFERENCE_THROTTLE}" \
  --dependency="afterok:${join_id}" "$PACE_DIR/stage_k_production_reference_gnr.sbatch")
reference_join_id=$(sbatch --parsable "${CPU_ARGS[@]}" --dependency="afterok:${references_id}" \
  "$PACE_DIR/stage_k_production_reference_join_gnr.sbatch")
aggregate_id=$(sbatch --parsable "${CPU_ARGS[@]}" --dependency="afterok:${reference_join_id}" \
  "$PACE_DIR/stage_k_production_aggregate_gnr.sbatch")

printf 'job,job_id\ngridsfm_array,%s\ndc_array,%s\nac_array,%s\njoin,%s\nreferences_array,%s\nreference_join,%s\naggregate,%s\n' \
  "$grid_id" "$dc_id" "$ac_id" "$join_id" \
  "$references_id" "$reference_join_id" "$aggregate_id" > "$STAGE_K_RUN_DIR/submitted_job_ids.csv"
printf 'GridSFM=%s\nDC=%s\nAC=%s\nJoin=%s\nReferences=%s\nReferenceJoin=%s\nAggregate=%s\n' \
  "$grid_id" "$dc_id" "$ac_id" "$join_id" \
  "$references_id" "$reference_join_id" "$aggregate_id"
