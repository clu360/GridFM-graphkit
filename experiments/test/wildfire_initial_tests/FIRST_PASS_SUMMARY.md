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
still uses a GridFM surrogate consequence score, lacks hard AC feasibility
constraints, and has no topology-control decision variables. Next steps are to
inspect the demand-weighted traces, decide whether multistart should become the
default for risk-preferred studies, add voltage/thermal diagnostics, and later
evaluate de-energization as a separate control pathway.
