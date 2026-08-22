# Stage J J5-J7 Smoke Status

## Status

`J5_PAC_CALIBRATION_COMPLETE`

`J6_INPUT_SCENARIOS_PREPARED`

`J7_ONE_SCENARIO_SMOKE_COMPLETE_WITH_LIMITATION`

## Execution Mode

`single_context_role_simulation`

Reviewer independence remains procedural and auditable, not OS-level isolation.

## J4 Baseline Loading Export

Exact intact economic AC-OPF baseline branch loading was exported from `pglib_opf_case500_goc.m` using PowerModels/IPOPT.

External artifacts:

```text
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\baseline\case500_goc_ac_baseline_loading.csv
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\baseline\case500_goc_ac_baseline_loading_summary.json
```

Observed summary:

```text
termination_status = LOCALLY_SOLVED
exported_active_branch_count = 728
max_baseline_loading = 1.0000000049826396
num_baseline_loading_gt_1 = 1
```

The one value above 1.0 is at solver tolerance scale and should be treated as an audit item, not as a GridFM-style outlier.

## J6 Scenario And Proxy Artifacts

Prepared input directory:

```text
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\inputs\case500_goc_e0
```

Artifacts:

```text
stage_j_input_manifest.json
stage_j_branch_identity.csv
stage_j_unit_compatibility_report.json
stage_j_scenario_register.csv
stage_j_candidate_line_scores.csv
stage_j_p_env_by_scenario.csv
stage_j_scenario_summary.json
```

Input preparation summary:

```text
branch_count = 728
risk_branch_count = 536
candidate_branch_count = 536
load_count = 281
c_l_backend = connectivity_source_less_single_outage
unit_compatibility = true
max_rate_a_abs_delta = 0.0
endpoint_mismatch_branch_ids = []
```

Diagnostic scenario targets:

```text
J-S1 target_branch_ids = [473]
J-S2 target_branch_ids = [238]
J-S3 target_branch_ids = [285]
```

This uses the shared connectivity/source-less single-outage `c_l` proxy for both Guided-DC and Guided-GridSFM. DC-MLD impact proxy remains a later upgrade, not the active smoke backend.

## J5 PAC Calibration

PAC calibration used only intact plus selected N-1/N-2 smoke states before broader S1-S3 comparison.

External artifacts:

```text
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\pac_calibration\v001\PAC_WEIGHT_FREEZE.json
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\pac_calibration\v001\pac_calibration_smoke_rows.csv
```

The calibration script now hard-fails instead of writing `PASS` if any smoke state lacks objective/PAC components or if no finite `rho_phys * PAC_total / J_trade` ratios are available.

Frozen weights:

```text
rho_phys = 2.0
w_op = 1.0
w_AC = 1.0
w_model = 0.0
```

Calibration result:

```text
calibration_rule = identity_weights_kept_because_median_rho_pac_over_j_trade_within_threshold
median_rho_pac_over_j_trade_provisional = 0.0004989643264865505
max_rho_pac_over_j_trade_provisional = 0.005454876091730332
num_smoke_states = 5
```

`PAC_model` is zero-weighted in v1 because the official GridSFM implementation computes branch flows from predicted voltage/angle internally; no independent branch-flow output head was found.

## J5 Alpha Integration Smoke

Official GOC-500 GridSFM candidate-evaluation smokes completed for:

```text
alpha_i = 1.0 for every load
alpha_i = 0.98 for every load
alpha_i = 1.0 for every load except load_id 1 set to 0.5
```

External artifacts:

```text
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\smoke\gridsfm_candidate_eval_J-S1.json
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\smoke\gridsfm_candidate_eval_J-S1_uniform_alpha_0p98.json
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\smoke\gridsfm_candidate_eval_J-S1_target_load_1_alpha_0p5.json
```

Observed alpha effects:

```text
uniform alpha 0.98:
  L_shed_total = 0.020000000000000018
  R_norm = 0.4604782206777216
  max_loading = 1.3014879518246598

target load 1 alpha 0.5:
  L_shed_total = 0.0010020531946726423
  R_norm = 0.4719360280837406
  max_loading = 1.318543614073421
```

These are alpha-integration smokes only. They do not constitute a full per-load alpha optimizer.

## J7 One-Scenario Smoke

Smoke setting:

```text
scenario_id = J-S1
lambda_R_proxy = 0.5
lambda_R = 0.5
K <= 2
pool_size = 5
alpha_strategy = smoke_fixed_full_vector_alpha_ones_not_full_alpha_search
```

External output directory:

```text
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\j7_smoke\J-S1_lambda0p5
```

Artifacts:

```text
j7_topology_pool.csv
j7_guided_dc_results.csv
j7_gridsfm_results.csv
j7_proxy_and_dc_summary.json
j7_gridsfm_summary.json
```

Shared proxy proposals:

```text
rank 1: 285;473
rank 2: 290;473
rank 3: 345;473
rank 4: 276;473
rank 5: 473;505
TH top-1: 473
TH top-2: 473;285
```

The no-good-cut pool generated distinct guided proposals and the TH top-2 topology was correctly detected as a duplicate of guided rank 1 after canonical sorting.

Guided-DC smoke:

```text
rows = 5
all evaluation_status = ok
all L_shed_total = 0
all num_loading_gt_1 = 0
best J_trade among smoke rows = 0.18028337165548766
```

Guided-GridSFM / TH-GridSFM smoke:

```text
rows = 7
unique topology evaluations = 6
all D_input = 0
all L_shed_total = 0
all evaluation_status = model_output_penalized
num_loading_gt_1 ranged from 6 to 9
max_loading ranged from approximately 1.316 to 1.822
```

This confirms the GridSFM path runs through official preprocessing/inference and exposes physically implausible predicted overloads through `PAC_total`, rather than rejecting successful model outputs as hard failures.

## Verification

Tests:

```text
python -m pytest tests/test_wildfire_stage_j_gridsfm_goc500.py -q
16 passed, 2 skipped

python -m pytest tests/test_wildfire_stage_j_gridsfm_goc500.py -q --basetemp=tmp\pytest-stage-j-user
18 passed
```

Smoke commands completed:

```text
export_pglib_case500_ac_baseline_loading.jl
prepare_stage_j_inputs.py
calibrate_gridsfm_pac_weights.py
bootstrap_gridsfm_candidate_eval_smoke.py
bootstrap_guided_dc_candidate_eval_smoke.py
run_j7_proxy_and_dc_smoke.py
run_j7_gridsfm_pool_smoke.py
```

## Major Remaining Flag Before J8

The locked formulation has one continuous `alpha_i` decision for every load bus/load row. The J7 smoke intentionally used a full-length alpha vector with `alpha_i = 1` for every load to validate plumbing, not to solve the high-dimensional alpha optimization problem.

Starting J8 full S1-S3 comparison would require one of:

```text
1. an approved full per-load alpha optimizer/search strategy;
2. an approved algorithmic approximation such as global, zonal, block, or selected-load alpha;
3. a Caleb-approved decision to run topology-only / alpha=1 as a diagnostic baseline.
```

Per the locked plan, replacing full per-load alpha search with a lower-dimensional alpha policy is `APPROXIMATION REQUIRED` and should be approved before full experimental results are generated.
