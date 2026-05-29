# Wildfire First Pass Summary

## Current Optimization Problem

The active experiment path is `experiments/test/wildfire_initial_tests`.

The first-pass optimizer solves a reduced, fixed-topology predict-then-optimize
problem over selected generator redispatch and selected load-service fractions:

```text
minimize_u  J(u)

u = [Delta_Pg_selected, alpha_selected]
```

The current scalar objective value is:

```text
J(u) = lambda_R * R_norm(u) + lambda_L * L_norm(u)
```

with:

```text
R_norm(u) = R_group(u) / R_group(u_base)
L_norm(u) = L_shed(u) / N_bus
L_shed(u) = sum_n (1 - alpha_n)
```

So the implemented objective is:

```text
J(u) = lambda_R * (R_group / R_baseline)
     + lambda_L * (L_shed / N_bus)
```

`R_baseline` is grouped wildfire risk at
`u_base = [0, ..., 0, 1, ..., 1]`. `N_bus = 30` for the current IEEE-30 case
and is written as `load_shedding_normalizer` in each run.

The active tradeoff settings all use convex weights:

```text
risk:     lambda_R = 0.999001, lambda_L = 0.000999
balanced: lambda_R = 0.5,      lambda_L = 0.5
shed:     lambda_R = 0.000999, lambda_L = 0.999001
```

## Decision Variables

The current decision vector is:

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

The decision vector modifies GridFM input node features as:

```text
Pg_n(u) = Pg_base,n + Delta_Pg_n       for selected generator buses
Pd_n(u) = Pd_base,n * alpha_n          for selected load buses
Qd_n(u) = Qd_base,n * alpha_n          for selected load buses
```

Non-selected buses keep their baseline generator dispatch and `alpha_n = 1`.

Generator movement is recorded as a diagnostic:

```text
sum_i (Delta_Pg_i / 5 MW)^2
```

but its objective weight is currently `0.0`, so it does not affect `J(u)`.

## Constraints

The explicit optimizer bounds are:

```text
-5 MW <= Delta_Pg_i <= +5 MW
0.0   <= alpha_j    <= 1.0
```

The fixed-topology constraints and assumptions are:

```text
y_n = 1 for all buses
z_l = 1 for all lines
Qg fixed at baseline
no bus shutoff
no line shutoff as an optimizer decision
no mixed-integer variables
no hard AC power-flow feasibility constraints inside the optimizer
```

The load-service constraints are:

```text
alpha_n = 1 for every non-selected bus
alpha_j in [0, 1] only for selected load buses
Pd and Qd are scaled by the same alpha_j at selected load buses
```

The wildfire scenario constraints are:

```text
line group: high_risk_corridor
manual connected line IDs: [23, 32, 26]
group_weight = 1.0
p_env_l = 1.0 on selected corridor lines
p_env_l = 0.1 on non-selected lines
standard_rate_a_mva = 100.0 as the branch-rating proxy
```

The prediction and validity constraints are:

```text
GridFM is called in memory for every candidate u
Vm and Va from GridFM are used to reconstruct relative line loading
NaN or inf predictions are invalid
invalid predictions receive objective penalty 1.0e12 when strict mode is false
```

The optimizer settings are:

```text
method = L-BFGS-B
maxiter = 80
ftol = 1.0e-9
gtol = 1.0e-6
eps = 1.0e-4
```

There are no additional equality or inequality constraints beyond these
bounds, fixed-variable assumptions, and invalid-prediction penalties in the
current first pass.

## Wildfire Risk Term

The risk term follows the probability-times-consequence structure:

```text
R_group(u) = sum_k group_weight_k * sum_{l in G_k}
             z_l * p_env_l * loading_l(u)^2 * I_l(u)
```

Line energization is not optimized yet, so `z_l = 1` for all lines. The
relative-flow term `loading_l` is reconstructed from predicted `Vm` and `Va`
and normalized by the line rating/proxy rating before squaring. `p_env_l` is
currently a synthetic weather/environment ignition probability proxy. On
May 28, 2026, the selected-corridor value was changed from `5.0` to `1.0` so
the selected-line ignition probability factor is probability-like rather than
a large hazard multiplier.

The consequence term `I_l(u)` is computed by a counterfactual GridFM line
outage:

```text
I_l(u) = max(0, S_current(u) - S_outage_l(u)) / max(S_current(u), eps)
```

`S` is an equal-weight bus service estimate inferred from predicted `Pd` as
clipped `Pd_pred / Pd_base`, with zero-load buses counted as fully served. This
replaces the previous constant `impact_l = 1.0` behavior.

## Latest Connected-Corridor Runs

The existing `results/connected_corridor` directory was deleted on
May 28, 2026 before regeneration. Fresh connected-corridor runs were then
generated with selected-corridor `p_env_l = 1.0`.

Latest objective values:

```text
risk/gnn:      0.999001 -> 0.999001
risk/gps:      0.999001 -> 0.999001
balanced/gnn:  0.500000 -> 0.500000
balanced/gps:  0.500000 -> 0.500000
shed/gnn:      0.000999 -> 0.000999
shed/gps:      0.000999 -> 0.000999
```

Latest grouped wildfire risk values:

```text
risk/gnn:      9.276730 -> 9.276730
risk/gps:      2.693767 -> 2.693767
balanced/gnn:  9.276730 -> 9.276730
balanced/gps:  2.693767 -> 2.693767
shed/gnn:      9.276730 -> 9.276730
shed/gps:      2.693767 -> 2.693767
```

All six regenerated tradeoff runs returned the baseline decision. The GNN
balanced run reported `ABNORMAL` but also returned the baseline decision.

Machine-readable summary:

```text
results/connected_corridor/connected_corridor_tradeoff_summary.csv
results/connected_corridor/connected_corridor_tradeoff_summary.json
```

## Current Simplifications

The current first pass intentionally keeps several pieces of the full
wildfire-resilience formulation simplified:

- No de-energization decision variables: `z_l = 1` and `y_n = 1` for all
  lines and buses. Line removal only happens inside the counterfactual
  `I_l(u)` calculation, not as an optimizer action.
- Synthetic environmental ignition probability: `p_env_l` is currently a
  configured weather/environment probability proxy, with `1.0` on selected
  corridor lines and `0.1` elsewhere. It is not yet calibrated from weather,
  vegetation, or fire-probability data.
- Single wildfire group: the current experiment optimizes one
  `high_risk_corridor`, `G_1 = [23, 32, 26]`, with `group_weight = 1.0`.
- Reduced decision vector: only selected `Delta_Pg` and selected `alpha` are
  optimized. `Qg`, voltage setpoints, topology, reserves, and other OPF
  controls are fixed or absent.
- Only selected load buses can shed: non-selected buses always have
  `alpha_n = 1`.
- Load shedding is equal-bus/fraction based:
  `L_shed = sum_n (1 - alpha_n)`, not MW-weighted or criticality-weighted.
- Consequence score is equal-bus service loss: `I_l(u)` uses equal-weight
  service inferred from predicted `Pd`, not MW-weighted demand loss or
  critical-load consequence.
- Counterfactual outage is surrogate-based: line outage impact is computed by
  removing one edge in the GridFM graph and rerunning GridFM, not by a
  validated AC power-flow or contingency solver.
- Relative loading uses reconstructed/proxy line loading: `loading_l` comes
  from predicted `Vm`, `Va`, and branch reconstruction, using
  `standard_rate_a_mva = 100.0` when ratings are unavailable.
- No hard AC feasibility constraints: voltage limits, line thermal limits,
  power balance, generator limits beyond the local `Delta_Pg` bound, and
  islanding feasibility are not enforced as optimizer constraints.
- No full wildfire physics: no weather spread model, ignition propagation,
  time dynamics, or fire suppression/evolution.
- Static single scenario: one IEEE-30 operating scenario, not a distribution of
  load, weather, or topology scenarios.
- Optimization is local and smooth-ish: L-BFGS-B with finite-difference steps,
  even though the surrogate/counterfactual impact may not be perfectly smooth.
- Risk normalization is baseline-relative: `R_group / R_baseline`, so
  conclusions are relative to this specific baseline and synthetic corridor
  setup.
- Current result interpretation is limited: after correcting `I_l(u)` and
  setting selected-corridor `p_env_l = 1.0`, all tradeoff settings still
  returned baseline, so we have formulation plumbing but not yet a strong
  demonstrated operational improvement.

## Automatic Figures

Every child run writes:

```text
figures/optimization_behavior.png
figures/ieee30_network_changes.png
visualization_summary.json
```

`optimization_behavior.png` shows objective, grouped wildfire risk, load
shedding, and configured objective weights. `ieee30_network_changes.png` shows
the topology-connected corridor and selected control buses. The network layout
is topology-accurate, not geographic.

## Objective Analysis

`objective_analysis.py` provides two distinct sensitivity workflows:

- frozen consequence sweep: freeze one GridFM evaluation and manually vary
  exactly one line consequence value without rerunning GridFM or invoking the
  optimizer
- decision-variable sweep: vary one actual decision variable at a time, rerun
  GridFM and counterfactual impacts at each point, and do not invoke the
  optimizer

The frozen-consequence callable entry point is:

```text
run_line_impact_objective_sweep(config_path, line_id, impact_values, output_root)
```

The decision-variable callable entry point is:

```text
run_decision_variable_objective_sweeps(config_path, num_points, output_root)
```

`run_multistart_optimization.py` provides a separate grid-seeded multistart
optimization diagnostic under `results/multistart/`. It does not change the
objective or constraints. It first evaluates a one-variable grid around the
baseline, selects the best unique seed decisions, then runs the same L-BFGS-B
optimizer from each seed and keeps the best final result.

The CLI form used for the latest analysis was:

```powershell
python experiments/test/wildfire_initial_tests/objective_analysis.py --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml --line-id 23 --num-points 101
```

Latest output directory:

```text
results/objective_analysis/line_23_impact_sweep_20260528_220439/
```

This run froze the baseline GPS decision and varied only `I_23(u)` from `0.0`
to `1.0`. The frozen baseline values were:

```text
frozen grouped risk: 2.6937669151682497
risk normalizer:     2.6937669151682497
load normalizer:     30.0
lambda_R:            0.999001
lambda_L:            0.000999
base I_23(u):        0.03000268142041778
```

Generated artifacts:

```text
objective_vs_line_impact.csv
objective_vs_line_impact.png
analysis_summary.json
frozen_line_risk.csv
frozen_group_risk.csv
config.yaml
```

Latest decision-variable sweep commands:

```powershell
python experiments/test/wildfire_initial_tests/objective_analysis.py --mode decision --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml --num-points 21
python experiments/test/wildfire_initial_tests/objective_analysis.py --mode decision --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gnn.yaml --num-points 21
```

Latest decision sweep outputs:

```text
results/objective_analysis/decision_sweeps/decision_sweeps_gps_20260528_235031/
results/objective_analysis/decision_sweeps/decision_sweeps_gnn_20260528_235108/
```

In `decision_variable_sweeps.png`, each subplot corresponds to one decision
variable while all other decision variables remain fixed at the baseline
decision. The x-axis is the actual value assigned to that one variable:

```text
Delta_Pg variables: -5.0 MW to +5.0 MW
alpha variables:     0.0 to 1.0
```

The left y-axis is the scalar objective `J(u)`, plotted as the black curve.
The right y-axis is raw grouped wildfire risk `R_group(u)`, plotted as the red
curve. The CSV also records normalized risk, normalized load shedding, the
weighted risk objective term, the weighted load-shedding term, and the three
corridor line impacts.

GPS decision sweep result:

```text
baseline objective: 0.999001
best one-variable point: alpha_7_bus18 = 0.0
best objective: 0.8961604984016736
best grouped risk: 2.4163717542193073
best load shedding: 1.0
```

For `alpha_7_bus18`, the objective changes mainly because the GridFM-predicted
counterfactual line impacts and grouped wildfire risk change as bus 18 service
is varied. At the best one-variable point:

```text
alpha_7_bus18:                  1.0 -> 0.0
R_group:                        2.6937669151682497 -> 2.4163717542193073
normalized R_group:             1.0 -> 0.8970233247030519
L_shed:                         0.0 -> 1.0
normalized L_shed:              0.0 -> 0.03333333333333333
risk objective term:            0.999001 -> 0.8961271984016735
load-shedding objective term:   0.0 -> 0.0000333
total objective:                0.999001 -> 0.8961604984016736
```

The total objective is therefore almost entirely driven by the normalized risk
term in the current GPS risk-prioritized config, because
`lambda_R = 0.999001` and `lambda_L = 0.000999`. Shedding one selected bus
completely only adds `0.000999 * (1/30) = 0.0000333` to the objective, so the
load-shedding penalty is tiny compared with a roughly 10.3% normalized-risk
drop.

GNN decision sweep result:

```text
baseline objective: 0.999001
best one-variable point: alpha_5_bus11 = 0.0
best objective: 0.9953966468056175
best grouped risk: 9.24295077382007
best load shedding: 1.0
```

Interpretation: unlike the local L-BFGS-B optimization runs, one-dimensional
decision sweeps do show objective-improving points, especially for GPS. This
suggests the no-movement optimizer result is likely an optimizer/local-search
or finite-difference issue, not simply an absence of objective signal. The
GPS `alpha_7_bus18` sweep is also nonmonotonic: near the baseline
`alpha = 1.0`, small reductions in alpha make the objective worse, while larger
reductions eventually improve it. This explains why the local optimizer can
return the baseline even though a farther one-variable sweep point is better.

Latest grid-seeded multistart command for the full three-tradeoff, two-model
result set:

```powershell
python experiments/test/wildfire_initial_tests/run_multistart_optimization.py --tradeoff-sets --num-seed-points 11 --max-seeds 5
```

The older nested multistart output under `results/objective_analysis/` was
removed. The canonical multistart output root is now:

```text
results/multistart/
```

The latest six-run output layout is:

```text
results/multistart/risk/gnn/multistart_gnn_20260529_001357/
results/multistart/risk/gps/multistart_gps_20260529_001442/
results/multistart/balanced/gnn/multistart_gnn_20260529_001543/
results/multistart/balanced/gps/multistart_gps_20260529_001624/
results/multistart/shed/gnn/multistart_gnn_20260529_001700/
results/multistart/shed/gps/multistart_gps_20260529_001708/
```

Aggregate summary files:

```text
results/multistart/multistart_tradeoff_summary.csv
results/multistart/multistart_tradeoff_summary.json
```

Each child run writes:

```text
analysis_summary.json
optimization_summary.json
selected_starts.csv
multistart_results.csv
objective_trace.csv
best_objective_trace.csv
best_decision_vector.csv
figures/optimization_behavior.png
```

Latest multistart tradeoff summary:

```text
risk/gnn:
  objective:      0.999001 -> 0.9950359605419952
  group risk:     9.239211240283675
  load shedding:  2.2618633086571864

risk/gps:
  objective:      0.999001 -> 0.8961482348645832
  group risk:     2.416338685293566
  load shedding:  1.0000086888091182

balanced/gnn:
  objective:      0.5 -> 0.5
  group risk:     9.276730045998226
  load shedding:  0.0
  note: optimizer reported ABNORMAL and returned baseline

balanced/gps:
  objective:      0.5 -> 0.4651783290181926
  group risk:     2.4163717542193073
  load shedding:  1.0

shed/gnn:
  objective:      0.000999 -> 0.000999
  group risk:     9.276730045998226
  load shedding:  0.0

shed/gps:
  objective:      0.000999 -> 0.000999
  group risk:     2.6937669151682497
  load shedding:  0.0
```

The risk/GPS case again shows the clearest nonlocal improvement: the grid seed
moves away from baseline, and L-BFGS-B improves the modified start further. In
the shed-preferred cases, the baseline remains optimal under the current
objective because `L_shed = 0` is already the minimum load-shedding value and
load shedding is heavily penalized.

## Separate AC-OPF Hard-Constraint Experiment

`ac_opf_experiment.py` is an isolated exploratory path for testing hard AC-OPF
constraints separately from the main GridFM connected-corridor workflow. It can
be deleted by removing:

```text
experiments/test/wildfire_initial_tests/ac_opf_experiment.py
experiments/test/wildfire_initial_tests/results/ac_opf_connected_corridor/
```

The experiment follows the standard AC-OPF constraint structure used by
MATPOWER/PowerModels-style formulations:

- AC active and reactive nodal power-balance equality constraints
- branch apparent-flow thermal limit inequalities
- generator active and reactive power bounds
- voltage magnitude bounds
- reference-bus angle constraint
- selected load-service fraction bounds

Latest command:

```powershell
python experiments/test/wildfire_initial_tests/ac_opf_experiment.py --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml
```

Latest output directory:

```text
results/ac_opf_connected_corridor/ac_opf_20260528_223144/
```

Latest result summary:

```text
risk:     success=False, max |balance|=7767.498436 MVA, min thermal margin=-2884.620086 MVA
balanced: success=False, max |balance|=7767.498467 MVA, min thermal margin=-2884.680958 MVA
shed:     success=False, max |balance|=7767.498435 MVA, min thermal margin=-2884.616218 MVA
```

Interpretation: these are not promising AC-OPF results. They should be treated
as evidence that the local `ScenarioData` approximation used here does not yet
provide a reliable hard AC-OPF model. In particular, the current isolated
experiment uses the local edge admittance approximation, broad fallback Q
bounds, and configured proxy branch ratings rather than a fully recovered
MATPOWER case with original generator cost/bounds/rating data.

## Tests And Commands Run

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py -q
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_objective_analysis.py -q
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_objective_analysis.py tests/test_wildfire_first_pass_multistart.py -q
python experiments/test/wildfire_initial_tests/run_connected_corridor_tradeoffs.py
python experiments/test/wildfire_initial_tests/objective_analysis.py --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml --line-id 23 --num-points 101
python experiments/test/wildfire_initial_tests/objective_analysis.py --mode decision --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml --num-points 21
python experiments/test/wildfire_initial_tests/objective_analysis.py --mode decision --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gnn.yaml --num-points 21
python experiments/test/wildfire_initial_tests/run_multistart_optimization.py --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml --num-seed-points 11 --max-seeds 5
python experiments/test/wildfire_initial_tests/run_multistart_optimization.py --tradeoff-sets --num-seed-points 11 --max-seeds 5
python experiments/test/wildfire_initial_tests/ac_opf_experiment.py --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml
```

Latest focused test result:

```text
16 passed, 3 external deprecation warnings
```

## Limitations And Next Steps

This is not a full wildfire-resilience-aware OPF. The counterfactual line
outage impact is represented through GridFM predictions with one scenario edge
removed, but this should still be treated as a first-pass surrogate consequence
score. The May 28 baseline-started tradeoff regeneration returned the baseline
decision for all six runs, but the May 29 GPS decision sweep and grid-seeded
multistart run found a lower risk-preferred objective. This means the corrected
risk term has usable signal, while the baseline-started local optimizer can
miss nonlocal improvements. Next steps are to decide whether multistart should
become the default optimizer mode for risk-preferred studies, run the same
diagnostic for other tradeoff/model settings, tighten feasibility diagnostics,
and decide whether the line-outage counterfactual should be backed by a
physical solver.
