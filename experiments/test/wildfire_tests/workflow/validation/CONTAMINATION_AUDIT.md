# Contamination Audit

```yaml
artifact_id: CONTAMINATION-0001
artifact_version: v001
created_utc: 2026-07-30T03:32:04+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: FROZEN
sha256_when_frozen: recorded_in_workflow/validation/FROZEN_ARTIFACTS.csv
```


## Write Boundary

All Phase 1 generated artifacts were written under:

```text
experiments/test/wildfire_tests/workflow/
```

The frozen package, Stage I source, tests, official results, `CURRENT_STATE_SUMMARY.md`, and `HISTORY.md` were read-only inputs.

## Dirty Working Tree Classification

| Git Status | Path | Classification |
| --- | --- | --- |
| `M` | `experiments/test/wildfire_tests/stage_i_dc_comparison/STAGE_H_DC_COMPARISON_PROGRESS.md` | `tracked_modifications` |
| `??` | `6.0.0` | `ignored_or_local_only_files` |
| `??` | `experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/plots/summary/` | `generated_result_artifacts` |
| `??` | `experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/best_j_true_by_method_lambda_rho_metadata.json` | `generated_result_artifacts` |
| `??` | `experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/best_j_true_by_method_lambda_rho_summary.csv` | `generated_result_artifacts` |
| `??` | `experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/stage_e_k2_load_shed_discrepancy_main_results.csv` | `generated_result_artifacts` |
| `??` | `experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/stage_e_k2_load_shed_discrepancy_metadata.json` | `generated_result_artifacts` |
| `??` | `experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/stage_e_k2_load_shed_discrepancy_summary_by_rho_lambda.csv` | `generated_result_artifacts` |
| `??` | `experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/stage_e_k2_load_shed_discrepancy_summary_overall.csv` | `generated_result_artifacts` |
| `??` | `experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/proxy_inner_lambda_sweep/r2/tables/stage_e_continuous_objective_call_trace.csv` | `generated_result_artifacts` |
| `??` | `experiments/test/wildfire_tests/stage_i_dc_comparison/run_stage_e_load_shed_discrepancy_summary.py` | `untracked_source_or_docs` |
| `??` | `experiments/test/wildfire_tests/stage_i_dc_comparison/run_stage_h_best_j_true_lambda_summary.py` | `untracked_source_or_docs` |
| `??` | `experiments/test/wildfire_tests/stage_i_workflow_review_package/` | `untracked_workflow_package_files` |
| `??` | `experiments/test/wildfire_tests/workflow/` | `untracked_workflow_files` |
| `??` | `tmp/` | `ignored_or_local_only_files` |

## Role Separation

Role separation is procedural and auditable. It is not OS-level access prevention.

## Contamination Result

No monitored active source, test, result, or frozen package artifact changed during Phase 1 generation.
