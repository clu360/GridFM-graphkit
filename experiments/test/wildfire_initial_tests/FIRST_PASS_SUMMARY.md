# Wildfire First Pass Summary

## Current Optimization Problem

The active experiment path is `experiments/test/wildfire_initial_tests`.

The first-pass optimizer solves a reduced, fixed-topology predict-then-optimize
problem over selected generator redispatch and selected load-service fractions:

```text
minimize_u  J(u)

u = [Delta_Pg_selected, alpha_selected]
```

The current scalar objective is:

```text
J(u) = lambda_R * R_norm(u) + lambda_L * L_norm(u)
```

with:

```text
R_norm(u) = R_group(u) / R_group(u_base)
w_n       = P_D,n / sum_m P_D,m
L_shed(u) = sum_n w_n * (1 - alpha_n)
L_norm(u) = L_shed(u)
```

So the implemented objective is:

```text
J(u) = lambda_R * (R_group / R_baseline)
     + lambda_L * sum_n (P_D,n / sum_m P_D,m) * (1 - alpha_n)
```

`R_baseline` is grouped wildfire risk at
`u_base = [0, ..., 0, 1, ..., 1]`. The demand-weighted load-shedding term is
already normalized because the weights sum to 1 across total positive baseline
demand, so `load_shedding_normalizer = 1.0` in current generated runs.

The active tradeoff settings all use convex weights:

```text
risk:     lambda_R = 0.999001, lambda_L = 0.000999
balanced: lambda_R = 0.5,      lambda_L = 0.5
shed:     lambda_R = 0.000999, lambda_L = 0.999001
```

Generator movement is recorded as a diagnostic:

```text
sum_i (Delta_Pg_i / 5 MW)^2
```

but its objective weight is `0.0`, so it does not affect `J(u)`.

## Decision Variables And Constraints

The decision vector is:

```text
u = [Delta_Pg_1, ..., Delta_Pg_m, alpha_1, ..., alpha_q]
```

Current auto-selected control dimensions:

```text
m = 3 selected generator buses
q = 5 selected load buses
```

Latest GPS run selected:

```text
selected_generator_buses = [1, 10, 12]
selected_load_buses = [6, 20, 11, 29, 18]
```

The explicit optimizer bounds are:

```text
-5 MW <= Delta_Pg_i <= +5 MW
0.0   <= alpha_j    <= 1.0
```

Fixed assumptions:

```text
y_n = 1 for all buses
z_l = 1 for all lines
Qg fixed at baseline
no bus shutoff
no line shutoff as an optimizer decision
no mixed-integer variables
no hard AC power-flow constraints inside the optimizer
```

Only selected load buses can shed. Non-selected buses always have
`alpha_n = 1`. `Pd` and `Qd` are scaled by the same selected `alpha_j`.

## Wildfire Risk Term

The risk term follows the probability-times-consequence structure:

```text
R_group(u) = sum_k group_weight_k * sum_{l in G_k}
             z_l * p_env_l * loading_l(u)^2 * I_l(u)
```

Line energization is not optimized yet, so `z_l = 1` for all lines. The
relative-flow term `loading_l` is reconstructed from predicted `Vm` and `Va`
and normalized by the line rating/proxy rating before squaring.

The consequence term `I_l(u)` is computed by a counterfactual GridFM line
outage:

```text
I_l(u) = max(0, S_current(u) - S_outage_l(u)) / max(S_current(u), eps)
```

`S` is an equal-weight bus service estimate inferred from predicted `Pd` as
clipped `Pd_pred / Pd_base`, with zero-load buses counted as fully served.

## Current Simplifications

- No de-energization decision variables: line removal only happens inside the
  counterfactual `I_l(u)` calculation, not as an optimizer action.
- Synthetic environmental ignition probability: `p_env_l = 1.0` on selected
  corridor lines and `0.1` elsewhere.
- Single wildfire group: `high_risk_corridor`, `G_1 = [23, 32, 26]`.
- Reduced decision vector: only selected `Delta_Pg` and selected `alpha` are
  optimized.
- Load shedding is demand-weighted by baseline MW demand in the scalar
  objective. The old equal-bus shedding sum and unserved MW are retained as
  diagnostics in objective components/traces.
- Consequence score is still equal-bus service loss, not MW-weighted
  consequence.
- Counterfactual outage is surrogate-based through GridFM, not a validated AC
  power-flow or contingency solver.
- No hard AC feasibility constraints are enforced in the optimizer.
- Static single scenario: one IEEE-30 operating scenario.
- Risk normalization is baseline-relative.

## Latest Demand-Weighted Runs

On May 29, 2026, the optimized load-shedding cost was changed from the
equal-bus sum divided by `N_bus` to the demand-weighted fraction:

```text
L_shed_weighted = sum_n (P_D,n / sum_m P_D,m) * (1 - alpha_n)
```

This keeps the load term normalized on a 0-to-1 scale and ensures that a 50%
curtailment at a 100 MW bus costs more than a 50% curtailment at a 5 MW bus.
The current generated `objective_normalizers.json` files therefore record:

```text
load_shedding_normalizer = 1.0
load_shedding_metric = demand_weighted_fraction
```

Baseline-started connected-corridor runs:

```text
results/demand_weighted/connected_corridor/risk/gnn/risk_gnn_20260529_134532/
results/demand_weighted/connected_corridor/risk/gps/risk_gps_20260529_134541/
results/demand_weighted/connected_corridor/balanced/gnn/balanced_gnn_20260529_134543/
results/demand_weighted/connected_corridor/balanced/gps/balanced_gps_20260529_134550/
results/demand_weighted/connected_corridor/shed/gnn/shed_gnn_20260529_134552/
results/demand_weighted/connected_corridor/shed/gps/shed_gps_20260529_134553/
```

Baseline-started summary:

```text
risk/gnn:
  objective:      0.999001 -> 0.9989932074556757
  group risk:     9.276730045998226 -> 9.276657378702414
  mean alpha:     0.9999863345024738
  min alpha:      0.9995908031440097

risk/gps:
  objective:      0.999001 -> 0.999001
  group risk:     2.6937669151682497 -> 2.6937669151682497
  mean alpha:     1.0

balanced/gnn:
  objective:      0.5 -> 0.5
  group risk:     9.276730045998226 -> 9.276730045998226
  note: optimizer reported ABNORMAL and returned baseline

balanced/gps:
  objective:      0.5 -> 0.5
  group risk:     2.6937669151682497 -> 2.6937669151682497

shed/gnn:
  objective:      0.000999 -> 0.000999
  group risk:     9.276730045998226 -> 9.276730045998226

shed/gps:
  objective:      0.000999 -> 0.000999
  group risk:     2.6937669151682497 -> 2.6937669151682497
```

Grid-seeded multistart runs:

```text
results/demand_weighted/multistart/risk/gnn/multistart_gnn_20260529_134608/
results/demand_weighted/multistart/risk/gps/multistart_gps_20260529_134657/
results/demand_weighted/multistart/balanced/gnn/multistart_gnn_20260529_134755/
results/demand_weighted/multistart/balanced/gps/multistart_gps_20260529_134825/
results/demand_weighted/multistart/shed/gnn/multistart_gnn_20260529_134854/
results/demand_weighted/multistart/shed/gps/multistart_gps_20260529_134901/
```

Multistart summary:

```text
risk/gnn:
  objective:                  0.999001 -> 0.9953593075274626
  group risk:                 9.24251512077115
  demand-weighted shedding:   0.042918644954574224
  equal-bus shedding:         1.0817346814920368
  unserved demand:            6.847095500765371 MW

risk/gps:
  objective:                  0.999001 -> 0.8961469098945768
  group risk:                 2.4163346027562054
  demand-weighted shedding:   0.0335228777710078
  equal-bus shedding:         1.0000237157820664
  unserved demand:            5.348126572996836 MW

balanced/gnn:
  objective:                  0.5 -> 0.5
  group risk:                 9.276730045998226
  demand-weighted shedding:   0.0
  note: optimizer reported ABNORMAL and returned baseline

balanced/gps:
  objective:                  0.5 -> 0.4652724237474569
  group risk:                 2.4163717542193073
  demand-weighted shedding:   0.03352152279186194
  equal-bus shedding:         1.0
  unserved demand:            5.347910404205322 MW

shed/gnn:
  objective:                  0.000999 -> 0.000999
  group risk:                 9.276730045998226
  demand-weighted shedding:   0.0

shed/gps:
  objective:                  0.000999 -> 0.000999
  group risk:                 2.6937669151682497
  demand-weighted shedding:   0.0
```

Machine-readable summaries:

```text
results/demand_weighted/connected_corridor/connected_corridor_tradeoff_summary.csv
results/demand_weighted/connected_corridor/connected_corridor_tradeoff_summary.json
results/demand_weighted/multistart/multistart_tradeoff_summary.csv
results/demand_weighted/multistart/multistart_tradeoff_summary.json
```

## Automatic Figures

Every child run writes:

```text
figures/optimization_behavior.png
figures/ieee30_network_changes.png
visualization_summary.json
```

`optimization_behavior.png` shows objective, grouped wildfire risk, the
demand-weighted load-shedding metric, and configured objective weights.
`ieee30_network_changes.png` shows the topology-connected corridor and
selected control buses. The network layout is topology-accurate, not
geographic.

## Stage B.1/B.2 Automatic Multi-Group Sensitivity

Stage B.1/B.2 has been added without replacing the manual connected-corridor
workflow. The new selection mode is:

```yaml
wildfire:
  selection_method: automatic_risk_components
```

Automatic ranking uses a uniform synthetic environmental probability for every
candidate line before selection. Current automatic configs set
`risk_score.candidate_p_env = 1.0`, avoiding the circular manual-corridor
convention where selected lines had hazard `1.0` and non-selected lines had
hazard `0.1`.

The automatic baseline score is:

```text
score_l = p_env * loading_l(base)^2 * I_l(base)
```

Lines are selected by top fraction using `ceil(num_lines * top_fraction)`, so
outputs include both `requested_top_fraction` and
`realized_selected_fraction = num_selected_lines / num_lines`. Selected lines
are grouped by connected components and written as `G_1`, `G_2`, ...

The Stage B threshold runner is:

```powershell
python experiments/test/wildfire_initial_tests/run_multi_group_threshold_sensitivity.py --top-fractions 0.10 0.125 0.15 0.175 0.20
```

It defaults to grid-seeded multistart for `gps` and `gnn` over `risk`,
`balanced`, and `shed`, using near-0/near-1 lambda cases:

```text
risk:     lambda_R = 0.999001, lambda_L = 0.000999
balanced: lambda_R = 0.5,      lambda_L = 0.5
shed:     lambda_R = 0.000999, lambda_L = 0.999001
```

Outputs are under:

```text
results/multi_group/threshold_0p10/
results/multi_group/threshold_0p125/
results/multi_group/threshold_0p15/
results/multi_group/threshold_0p175/
results/multi_group/threshold_0p20/
```

Each automatic run writes `automatic_line_risk_scores.csv`,
`automatic_wildfire_groups.json`, `automatic_group_summary.csv`, the standard
multistart artifacts, and `figures/ieee30_network_changes.png` with automatic
group colors/legend entries. Aggregate outputs are:

```text
results/multi_group/multi_group_threshold_sensitivity_summary.csv
results/multi_group/multi_group_threshold_sensitivity_summary.json
```

Verified smoke run:

```powershell
python experiments/test/wildfire_initial_tests/run_multi_group_threshold_sensitivity.py --top-fractions 0.10 --models gps --tradeoff-cases risk --num-seed-points 3 --max-seeds 2
```

Latest smoke output:

```text
results/multi_group/threshold_0p10/risk/gps/multistart_gps_20260531_191421/
```

The smoke selected 11 of 110 directed scenario edges
(`realized_selected_fraction = 0.10`), produced 5 automatic groups, did not
collapse to a single group, and wrote both automatic audit artifacts and both
figures. The full 30-run default sweep has not yet been run in full because it
is materially heavier than the smoke check.

Stage B.3/B.4 deferred: distinct seeded grouping and manual multi-region stress
testing are intentionally not implemented yet. They should only be considered
after manual review of Stage B.1/B.2 results under `results/multi_group/`.

## Stage C PSPS Threshold Baseline

Stage C has been added as a deterministic PSPS-only comparator. It does not
optimize `z_l`, `y_n`, or continuous controls after PSPS. Stage D optimized
de-energization remains deferred.

Stage C uses two thresholds:

```text
grouping_top_fraction = 0.30
psps_top_fraction = 0.10
```

The 30% threshold builds the automatic multi-group scenario. Only those
candidate lines are eligible for PSPS de-energization. The 10% PSPS threshold
then de-energizes:

```text
num_psps_lines = max(1, ceil(psps_top_fraction * num_candidate_lines))
```

Stage C replaces the dynamic first-pass consequence `I_l(u)` with a fixed
baseline demand-weighted consequence:

```text
I_l = max(0, S_D_base - S_D_outage_l) / S_D_base
S_D_base = sum_n P_D,n
```

For the post-PSPS diagnostic, demand-weighted load shed is inferred from the
PSPS GridFM prediction:

```text
service_fraction_n = clipped(Pd_pred,n / Pd_base,n, 0, 1)
L_shed = sum_n P_D,n * (1 - service_fraction_n) / sum_n P_D,n
```

Each environmental case assigns `p_env_l` before PSPS ranking, so different
cases can de-energize different lines:

```text
baseline_psps_risk_l = p_env_l * loading_l(base)^2 * I_l
```

Implemented cases:

```text
auto_env
largest_group_high
```

Stage C uses the near-boundary risk-emphasized weights, matching the
optimization-behavior convention used elsewhere:

```text
lambda_R = 0.999001
lambda_L = 0.000999
```

Runner:

```powershell
python experiments/test/wildfire_initial_tests/run_stage_c_psps_baseline.py --grouping-top-fraction 0.30 --psps-top-fraction 0.10 --models gps --cases auto_env largest_group_high
```

The planned long folder names were shortened internally to avoid Windows
OneDrive path-length failures. Outputs are under:

```text
results/stage_c_psps/t0p30/p0p10/r/gps/auto/run_20260531_205156/
results/stage_c_psps/t0p30/p0p10/r/gps/lgh/run_20260531_205158/
```

Aggregate outputs:

```text
results/stage_c_psps/stage_c_psps_summary.csv
results/stage_c_psps/stage_c_psps_summary.json
```

GPS results:

```text
auto_env:
  candidate lines:              33
  PSPS lines:                   4
  realized PSPS fraction:       0.12121212121212122
  de-energized line IDs:        [23, 18, 27, 101]
  baseline all-energized risk:  48.347018605732885
  post-PSPS risk:               2.2689103960556243
  risk reduction fraction:      0.9530703141271553
  demand-weighted load shed:    0.5495700231022472
  objective:                    0.999001 -> 0.047431823569736915

largest_group_high:
  candidate lines:              33
  PSPS lines:                   4
  realized PSPS fraction:       0.12121212121212122
  largest/manual group:         G_1
  de-energized line IDs:        [23, 18, 27, 101]
  baseline all-energized risk:  48.105301041506934
  post-PSPS risk:               2.172713688167144
  risk reduction fraction:      0.9548342149175525
  demand-weighted load shed:    0.5495700231022472
  objective:                    0.999001 -> 0.045669684916229365
```

Each run writes fixed consequence scores, PSPS ranking/decision CSVs, a
two-row `objective_trace.csv`, automatic group artifacts, standard summaries,
and figures. `ieee30_network_changes.png` now overlays PSPS de-energized lines
when the Stage C artifacts are present.

## Stage D Limited Enumerated De-Energization

Stage D v1 adds an interpretable enumerated `z_l` study. It is not a
mixed-integer optimizer and does not optimize continuous controls after
choosing topology. For each model/environmental case it builds the same
automatic `grouping_top_fraction = 0.30` candidate set, computes:

```text
num_candidate_lines = len(candidate_line_ids)
num_evaluated_subsets = 1 + num_candidate_lines + C(num_candidate_lines, 2)
```

and evaluates all candidate de-energization subsets with 0, 1, or 2 lines.
The empty subset is included as the all-energized baseline candidate.

Stage D uses the Stage C fixed demand-weighted consequence score:

```text
risk_l = z_l * p_env_l * loading_l^2 * I_l
R_norm = R_group / R_group_baseline
L_norm = demand-weighted load shed from the post-topology GridFM prediction
J = lambda_R * R_norm + lambda_L * L_norm
```

`R_group_baseline` is the all-energized baseline under the same model,
environmental case, grouping threshold, fixed `I_l`, and candidate groups. It
is computed once per model/environmental case and reused across lambda cases.
The topology evaluations are also shared across lambda cases; only the scalar
objective and chosen best subset change with lambda.

Lambda cases:

```text
risk_leaning:    lambda_R = 0.8, lambda_L = 0.2
balanced:        lambda_R = 0.5, lambda_L = 0.5
service_leaning: lambda_R = 0.2, lambda_L = 0.8
```

Runner:

```powershell
python experiments/test/wildfire_initial_tests/run_stage_d_deenergization.py --grouping-top-fraction 0.30 --models gps --cases auto_env largest_group_high --evaluation-mode limited_enumerated_z_only --max-deenergized-lines 2
```

Outputs are under:

```text
results/stage_d_deenergization/t0p30/gps/auto/<risk|bal|svc>/run_20260531_2121xx/
results/stage_d_deenergization/t0p30/gps/lgh/<risk|bal|svc>/run_20260531_2121xx/
results/stage_d_deenergization/stage_d_deenergization_summary.csv
results/stage_d_deenergization/stage_d_deenergization_summary.json
```

GPS result highlights:

```text
candidate lines:         33
evaluated subsets:       562 per environmental case

auto_env:
  risk_leaning best:      [23, 27], objective 0.1740339143005827
  balanced best:          [23, 27], objective 0.27244685090395315
  service_leaning best:   [],       objective 0.2

largest_group_high:
  risk_leaning best:      [23, 27], objective 0.1714800049354558
  balanced best:          [23, 27], objective 0.27085065755074883
  service_leaning best:   [],       objective 0.2
```

When Stage C outputs are available, Stage D loads the fixed Stage C PSPS
topology for the matching model/environmental case and re-scores its scalar
objective under each Stage D lambda for apples-to-apples comparison. The Stage
C topology itself remains independent of Stage D lambda.

Each Stage D run writes `candidate_line_risk_scores.csv`,
`deenergization_candidate_evaluations.csv`,
`optimized_deenergization_decisions.csv`, automatic group artifacts,
`objective_trace.csv`, `optimization_summary.json`, `visualization_summary.json`,
and figures. `ieee30_network_changes.png` now labels Stage D optimized
de-energized lines separately from PSPS de-energized lines.

Deferred: full mixed-integer topology control, relaxed continuous `z_l`,
optimized `y_n`, AC feasibility enforcement, and continuous-control
optimization after selecting `z_l` remain out of scope.

## Separate AC-OPF Hard-Constraint Experiment

`ac_opf_experiment.py` is an isolated exploratory path for testing hard AC-OPF
constraints separately from the main GridFM connected-corridor workflow. The
latest run on May 28, 2026 did not satisfy hard AC constraints, so those
outputs should be treated as evidence that the local `ScenarioData`
approximation is not yet a reliable hard AC-OPF model.

## Tests And Commands Run

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_objective_analysis.py tests/test_wildfire_first_pass_multistart.py -q
python experiments/test/wildfire_initial_tests/run_connected_corridor_tradeoffs.py
python experiments/test/wildfire_initial_tests/run_multistart_optimization.py --tradeoff-sets --num-seed-points 11 --max-seeds 5
```

Latest focused test result:

```text
16 passed, 3 external deprecation warnings
```

## Limitations And Next Steps

This is not a full wildfire-resilience-aware OPF. The demand-weighted
load-shedding term fixes the previous equal-bus cost issue, but the first pass
still uses a GridFM surrogate consequence score and lacks hard AC feasibility
constraints. Stage D now evaluates limited enumerated line de-energization, but
does not yet introduce a full topology-control optimizer.
