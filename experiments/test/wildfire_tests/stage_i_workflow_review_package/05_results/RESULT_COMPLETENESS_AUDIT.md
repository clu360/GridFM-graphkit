# Result Completeness Audit

This package records factual completeness evidence only. It does not certify the
scientific correctness of the implementation.

## Expected Main Run Structure

- `main_results/r11`: expected 300 best rows = 5 scenarios x 5 lambda values x 2 rho values x 6 methods.
- `MLD/r5`: expected 60 best rows = 5 scenarios x 1 lambda value x 2 rho values x 6 methods.
- `proxy_inner_lambda_sweep/r2`: expected 250 best rows = 5 scenarios x 5 proxy lambdas x 5 inner lambdas x 2 methods.

## Checks Represented By Saved Tables

- Methodology checks: `methodology_fidelity_checks.csv`
- AC projections: `ac_projection_distances.csv` where available.
- Projection cache: `ac_projection_cache.csv` where available.
- Stage I-b solver diagnostics: `stage_i_b_solution_pool*.csv`, `best_by_rho_scenario_lambda_stage.csv`
- Decision quality: `expected_vs_selected_by_rho.csv`
- Runtime: `dc_runtime_comparison_stage_i_a_vs_i_b.csv`

## Known Limitations

- AC projection failures are logged and nonfatal; finite distances are not present for every finalist.
- Proxy-inner sweep does not include AC projection distances.
- Large trace/provenance CSVs are indexed rather than duplicated.
- This audit does not judge whether the results are fair or physically correct.
