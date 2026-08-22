# Phase 1A Package Manifest Reconciliation Report

```yaml
artifact_id: PHASE1A-RECON-0001
artifact_version: v001
created_utc: 2026-07-30T23:36:58+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: FROZEN
sha256_when_frozen: recorded_in_phase1a_freeze_ledger
```


## Summary

Original manifest row count: `442`

Classification totals reconcile to: `442`

| Classification | Count |
| --- | ---: |
| ARCHIVE_OMISSION | 13 |
| DIRECTORY_NOT_FILE | 0 |
| INDEXED_NOT_COPIED | 11 |
| INVALID_PACKAGE_PATH | 0 |
| MISSING_UNEXPECTEDLY | 343 |
| ORIGINAL_PATH_ONLY | 0 |
| PATH_NORMALIZATION_FAILURE | 2 |
| PRESENT_HASH_MATCH | 72 |
| PRESENT_HASH_MISMATCH | 0 |
| SELF_REFERENTIAL_MANIFEST | 1 |
| UNKNOWN | 0 |

## Root Cause

Two root causes were found.

1. Freeze ordering / integrity metadata circularity: the original manifest includes `PACKAGE_FREEZE.json`, while `PACKAGE_FREEZE.json` stores hashes for the manifest files. The current manifest hashes therefore differ from the hashes recorded inside the freeze file.
2. Archive omissions: many rows marked as package-copied artifacts do not resolve as readable package files. Some are recoverable from active originals with matching hashes; others are no longer available at either the package path or original path and are recorded as missing materials.

## Manifest Hash Comparison

| File | Current SHA-256 | Freeze-recorded SHA-256 | Match |
| --- | --- | --- | --- |
| `PACKAGE_MANIFEST.csv` | `024406ed7092fcae9aaeff058e097da96a4430197f8cfc002daf2b6a294f646e` | `8f542d6f2fabc21c67810b9324c8e0013edaf992af3c9a4efdfcf121d5bb4489` | `False` |
| `PACKAGE_MANIFEST.json` | `a4ca0819899322a414e514ebbac632d5749cbb22891f854dd6f892bb7ff8b988` | `2edaf2906b449a5a5bb4bb256601a895b54166b31fca0190b9d0a029ced8c64c` | `False` |

## Direct Answers

1. Manifest hashes differ because the original freeze/manifest design was not finalized in a non-self-referential order. The manifest includes the freeze record, and the freeze record hashes manifests.
2. The 369 unverified paths are not true content mismatches. They are rows whose package paths could not be resolved/read as copied files during the original package audit.
3. The issue is a combination of freeze ordering, archive omissions, intentionally indexed external artifacts, and missing original/generated result files. It is not primarily path normalization or package movement.
4. True copied-file content-hash mismatches found: `0`.

## Representative Examples

### ARCHIVE_OMISSION

- `experiments/test/wildfire_tests/stage_i_workflow_review_package/01_authoritative_state/other_current_state_or_handoff_files/STAGE_H_DC_COMPARISON_PROGRESS.md`
  - package copy missing but original path exists with expected hash
- `experiments/test/wildfire_tests/stage_i_workflow_review_package/02_planning_and_intent/codex_session_handoffs/STAGE_H_STAGE_IA_PROXY_INNER_LAMBDA_FINDINGS.md`
  - package copy missing but original path exists with expected hash
- `experiments/test/wildfire_tests/stage_i_workflow_review_package/03_stage_i_source/stage_i_modules/run_proxy_inner_stage_ia_summary_figures.py`
  - package copy missing but original path exists with expected hash

### INDEXED_NOT_COPIED

- `experiments/test/wildfire_tests/stage_i_workflow_review_package/05_results/large_artifact_summaries/main_results_r11__all_evaluated_stage_h_points.json`
  - manifest row represents intentionally omitted large external artifact
- `experiments/test/wildfire_tests/stage_i_workflow_review_package/05_results/large_artifact_summaries/main_results_r11__all_evaluated_stage_h_points_with_miqp_pool.json`
  - manifest row represents intentionally omitted large external artifact
- `experiments/test/wildfire_tests/stage_i_workflow_review_package/05_results/large_artifact_summaries/main_results_r11__dc_recourse_results.json`
  - manifest row represents intentionally omitted large external artifact

### MISSING_UNEXPECTEDLY

- `experiments/test/wildfire_tests/stage_i_workflow_review_package/05_results/primary_complete_run/plots/per_rho/rho0/S1/common_operational_diagnostic_by_lambda.png`
  - package copy missing and original path unavailable
- `experiments/test/wildfire_tests/stage_i_workflow_review_package/05_results/primary_complete_run/plots/per_rho/rho0/S1/effective_load_shedding_by_lambda.png`
  - package copy missing and original path unavailable
- `experiments/test/wildfire_tests/stage_i_workflow_review_package/05_results/primary_complete_run/plots/per_rho/rho0/S1/expected_vs_selected_shutoff_lines.png`
  - package copy missing and original path unavailable

### PATH_NORMALIZATION_FAILURE

- `experiments/test/wildfire_tests/stage_i_workflow_review_package/05_results/companion_runs/proxy_inner_lambda_sweep_r2/tables/dc_branch_model_audit.json`
  - same filename/hash found elsewhere: experiments/test/wildfire_tests/stage_i_workflow_review_package/05_results/companion_runs/MLD_r5/tables/dc_branch_model_audit.json
- `experiments/test/wildfire_tests/stage_i_workflow_review_package/05_results/companion_runs/proxy_inner_lambda_sweep_r2/tables/p_env_by_scenario.csv`
  - same filename/hash found elsewhere: experiments/test/wildfire_tests/stage_i_workflow_review_package/05_results/companion_runs/MLD_r5/tables/p_env_by_scenario.csv

### PRESENT_HASH_MATCH

- `experiments/test/wildfire_tests/stage_i_workflow_review_package/01_authoritative_state/CURRENT_STATE_SUMMARY.md`
- `experiments/test/wildfire_tests/stage_i_workflow_review_package/01_authoritative_state/HISTORY.md`
- `experiments/test/wildfire_tests/stage_i_workflow_review_package/01_authoritative_state/README.md`

### SELF_REFERENTIAL_MANIFEST

- `experiments/test/wildfire_tests/stage_i_workflow_review_package/PACKAGE_FREEZE.json`
  - integrity metadata row participates in freeze/manifest circularity
