# Stage J J7.5 Full-Alpha Smoke Status

created_utc: 2026-08-16

status: J7_5_FULL_ALPHA_SMOKE_COMPLETE_J8_BLOCKED_PENDING_ALPHA_SEARCH_DECISION

## Scope

J7.5 implemented and smoke-tested the missing full per-load alpha optimizer before starting full J8 S1-S3 experiments.

The locked architecture remains:

```text
outer topology proposal z
-> full per-load alpha optimizer
-> repeated fixed-(z, alpha) electrical evaluations
-> best alpha for that topology
-> topology result / no-good cut remains topology-only
```

GridSFM is not embedded inside Gurobi, JuMP, IPOPT, autograd, or any differentiable optimization loop. The alpha optimizer is derivative-free and backend-neutral.

## Implemented Files

```text
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/alpha_optimizer.py
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/run_j7_5_guided_dc_alpha_smoke.py
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/run_j7_5_gridsfm_alpha_smoke.py
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/export_gridsfm_alpha_candidate_state.py
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/measure_gridsfm_candidate_repeatability.py
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/reference_a_fixed_alpha_ac_opf.jl
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/write_j7_5_full_alpha_report.py
tests/test_wildfire_stage_j_gridsfm_goc500.py
```

## Smoke Configuration

```text
scenario_id = J-S1
lambda_R = 0.5
lambda_R_proxy = 0.5 conceptually, fixed topology smoke used topology {285, 473}
topology = {285, 473}
K <= 2
seed alpha values = [1.00, 0.98, 0.95]
Delta schedule = [0.10, 0.05, 0.02, 0.01]
B_alpha = 300
rho_phys = 2
w_op = 1
w_AC = 1
w_model = 0
```

The smoke used the same full-alpha optimizer configuration for Guided-DC and Guided-GridSFM.

## Main Artifacts

```text
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\j7_5_full_alpha\J-S1_lambda0p5_topology_285_473\
```

Key files:

```text
J7_5_FULL_ALPHA_OPTIMIZER_REPORT.md
J7_5_FULL_ALPHA_OPTIMIZER_REPORT.json
guided_dc/j7_5_guided_dc_alpha_trace.csv
guided_dc/j7_5_guided_dc_alpha_summary.json
gridsfm/j7_5_gridsfm_alpha_trace.csv
gridsfm/j7_5_gridsfm_alpha_summary.json
gridsfm/j7_5_gridsfm_best_alpha_requested.csv
gridsfm/j7_5_gridsfm_best_alpha_effective.csv
gridsfm_finalist_state/gridsfm_finalist_branch_state.csv
gridsfm_finalist_state/gridsfm_finalist_bus_state.csv
reference_a/reference_a_summary.json
reference_a/reference_a_branch_loading.csv
reference_a/reference_a_bus_state.csv
reference_a/reference_a_gen_dispatch.csv
repeatability/gridsfm_repeatability_summary.json
repeatability/gridsfm_repeatability_rows.csv
```

## Smoke Results

Guided-DC:

```text
actual_evaluation_count = 300
termination_reason = budget_exhausted
completed_sweeps = 1
accepted_moves = 1
best changed load = 202 -> alpha 0.9
best J_trade = 0.1874964855
best R_norm = 0.3743761863
best L_shed_total = 0.0006167847
```

Guided-GridSFM:

```text
actual_evaluation_count = 300
termination_reason = budget_exhausted
completed_sweeps = 1
accepted_moves = 1
best changed load under J_total = 157 -> alpha 0.9
best J_total = 0.2454604583
best J_trade = 0.2439834494
best R_norm = 0.4870372602
best L_shed_total = 0.0009296386
best PAC_total = 0.0007385044
max predicted loading = 1.403987486
num predicted loading > 1 = 6
```

The GridSFM smoke found different alpha candidates for:

```text
argmin J_trade: load 246 -> alpha 0.9
argmin J_total: load 157 -> alpha 0.9
```

This confirms that `rho_phys * PAC_total` affects GridSFM candidate selection.

## Reference A

Reference A fixed-`z,alpha` economic AC-OPF was attempted for the selected GridSFM finalist:

```text
status = LOCALLY_SOLVED
objective = 456772.0234
runtime_seconds = 7.564
max_ac_loading = 1.000000005
num_ac_loading_gt_1 = 2
```

Reference A now consumes the selected GridSFM `alpha_effective` CSV, not the requested-alpha CSV. This matters for future source-less island cases. The current smoke has:

```text
source_less_load_ids = []
```

GridSFM-vs-Reference-A comparison:

```text
common_branch_count = 726
common_bus_count = 500
D_flow_normalized_mse = 0.002638988927
D_bus_vm_va_mse = 0.0008371538989
D_state_to_AC_flow_voltage_v1 = 0.003476142826
mean_abs_loading_delta = 0.02992972052
max_abs_loading_delta = 0.5778618517
generator_distance_status = NOT_COMPUTED_CANONICAL_GEN_ID_MAPPING_UNRESOLVED
```

The current `D_state_to_AC` value is a v1 flow+voltage distance only. Generator dispatch distance is not yet computed because the GridSFM dense generator output index and PowerModels generator IDs were not proven to be a safe canonical match.

## Repeatability / Epsilon

The selected GridSFM finalist was re-evaluated three times:

```text
repetitions = 3
max objective range across J_trade/J_total = 0
recommended_epsilon_abs_floor = 1e-9
```

This supports the provisional `epsilon_abs = 1e-9` for this smoke candidate. A broader J8 tolerance should still be frozen before full S1-S3 sweeps if a different alpha-search budget or batching strategy is approved.

## Tests

Local focused tests:

```text
pytest tests/test_wildfire_stage_j_gridsfm_goc500.py -q
18 passed, 2 skipped
```

Smoke commands completed:

```text
Guided-DC J7.5 full-alpha smoke: PASS
Guided-GridSFM J7.5 full-alpha smoke: PASS
Reference A fixed-z-alpha AC audit: LOCALLY_SOLVED
GridSFM finalist state export: PASS
J7.5 consolidated report generation: PASS
```

Workflow auditor findings addressed during this pass:

```text
Reference A effective-alpha artifact: ADDRESSED
cache namespace provenance in trace rows: PARTIALLY ADDRESSED
GridSFM repeatability artifact: ADDRESSED_FOR_SMOKE_CANDIDATE
report recommendation hard-coding: ADDRESSED
```

Remaining J8 limitations:

```text
persistent cross-run cache is not implemented yet
generator component of D_state_to_AC remains unresolved
```

## Recommendation

```text
J8_BLOCKED_BY_ALPHA_SEARCH
```

Reason:

The optimizer and smoke execution work, but both Guided-DC and Guided-GridSFM exhausted `B_alpha = 300` after only one completed full coordinate sweep and one accepted move. Full J8 should not start until Caleb approves one of:

```text
larger B_alpha / time budget
modified stopping or checkpoint policy
screened coordinate subset
zonal/block/global alpha approximation
other approved alpha-search approximation
```

This is not a software blocker. It is a methodological/runtime decision before scaling from one topology smoke to full S1-S3 experiments.
