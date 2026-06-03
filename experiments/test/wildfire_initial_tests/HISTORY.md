# Wildfire First Pass History And Future Context

This file is the durable handoff for future Codex sessions. It should be read before changing the wildfire first-pass experiment.

## Working Protocol

After every meaningful implementation or experiment change, update this file so
a new session can recover the current research state without needing the prior
chat. Keep this file aligned with the code, configs, tests, latest verified
commands, and current next steps. If a run or test changes the evidence base,
record the command and outcome here after verification.

As of May 28, 2026, the active state is the connected-corridor first pass:

- target workflow: `configs/connected_corridor_gnn.yaml` and
  `configs/connected_corridor_gps.yaml`
- target result layout for demand-weighted runs:
  `results/demand_weighted/connected_corridor/...`
- high-risk corridor: the manually selected, connected line IDs `[23, 32, 26]`
- visualization: `figures/ieee30_network_changes.png` plots the IEEE-30
  topology and highlights the synthetic high-risk corridor
- topology controls are not active; buses and lines stay energized
- wildfire risk now uses probability-times-consequence form with `z_l = 1`,
  synthetic `p_env_l`, relative loading squared, and a GridFM counterfactual
  line-outage consequence score
- load shedding term: modeled as a demand-weighted shed fraction,
  `sum_n (P_D,n / sum_m P_D,m) * (1 - alpha_n)`
- selected-load alpha is relaxed to `[0.0, 1.0]`
- connected-corridor outputs are generated as three tradeoff sets:
  `risk`, `balanced`, and `shed`

Important wording: the current network visual is topology-accurate, not
geographic. The selected lines are adjacent/connected in the IEEE-30 graph, but
the plotted coordinates come from a deterministic spring layout rather than
real geographic bus coordinates.

As of May 29, 2026, the simplified fixed-energization implementation has been
polished into a coherent first-pass pipeline: the corrected probability-times-
consequence wildfire risk term is implemented, decision-variable and frozen
consequence objective analyses are separated, baseline-started and grid-seeded
multistart optimizer behavior can be compared, three lambda tradeoff cases are
generated for both GNN and GPS, figures and machine-readable summaries are
written consistently, and focused tests pass. The next research phase should
evaluate how the optimization behavior changes when de-energization is included
as an explicit decision/control pathway rather than only as a counterfactual
line-outage consequence calculation.

Later on May 29, 2026, the optimized load-shedding term was changed from an
equal-bus shed-fraction sum divided by `N_bus` to a demand-weighted shed
fraction:

```text
w_n = P_D,n / sum_m P_D,m
L_shed_weighted = sum_n w_n * (1 - alpha_n)
```

This term is already normalized because the weights sum to 1 across positive
baseline demand, so current generated runs use
`load_shedding_normalizer = 1.0`. The previous equal-bus shedding sum and
unserved MW are retained as diagnostics.

## Why This Folder Exists

The research goal is to test a limited GridFM predict-then-optimize loop before claiming a full wildfire-resilience-aware OPF result.

The first research claim is:

```text
In a fixed IEEE-30 operating scenario with a synthetic high-risk wildfire corridor,
a GridFM-based predict-then-optimize loop can produce stable and interpretable
operating changes that reduce aggregate wildfire exposure while preserving most
served demand.
```

This is intentionally not a full ACOPF, topology-control, or wildfire physics claim.

## Current Active Folder

The active implementation is:

```text
experiments/test/wildfire_initial_tests/
```

The active support files kept at `experiments/test/` are:

```text
experiments/test/__init__.py
experiments/test/pipeline_utils.py
experiments/test/scenario_data.py
experiments/test/neural_solver.py
experiments/test/pv_dispatch.py
experiments/test/overload_penalty.py
```

These support files are still needed for scenario loading, checkpoint loading, in-memory GridFM inference, and branch-loading reconstruction. Do not remove them unless `wildfire_initial_tests` is refactored to replace their functionality.

## Cleanup Completed

The following were removed from `experiments/test` because they belonged to earlier workflows and were no longer required by the current first-pass formulation:

```text
experiments/test/improved_optimization/
experiments/test/example_optimization.py
experiments/test/example_optimization_with_shedding.py
experiments/test/extended_dispatch_spec.py
experiments/test/load_shedding_spec.py
experiments/test/optimization.py
experiments/test/validation.py
experiments/test/wildfire_penalty.py
experiments/test/test_pipeline.py
experiments/test/test_pipeline_ieee30.py
experiments/test/test_gnn_vs_gps_shedding.py
experiments/test/ieee30_optimization_validation.ipynb
experiments/test/pf_node.csv
experiments/test/pf_edge.csv
```

The first-pass code was also refactored so it no longer imports `experiments.test.improved_optimization.wildfire_metrics`. Branch loading now goes through `experiments.test.overload_penalty.OverloadPenaltyEvaluator`.

## Current Reduced Formulation

The full formulation in the research notes uses:

```text
u = (Pg, Qg, alpha_n, y_n, z_l)
```

The first-pass implementation uses:

```text
u_first = [Delta_Pg_selected, alpha_selected]
```

Fixed assumptions:

- `y_n = 1` for all buses
- `z_l = 1` for all lines
- `Qg` fixed at baseline
- no line shutoff
- no bus shutoff
- no mixed-integer optimization
- no hard AC feasibility constraints
- no external wildfire physics solver

Bounds:

- `Delta_Pg` in `[-5 MW, +5 MW]`
- `alpha` in `[0.00, 1.00]`

Objective, aligned with the current reduced formulation:

```text
J(u) = lambda_R * (R_group / R_baseline)
     + lambda_L * L_shed_weighted
```

Generator redispatch remains available as a control variable. Its normalized
movement is recorded in objective components and traces for interpretability,
but it is not part of the optimized scalar objective. This keeps the first pass
focused on the intended tradeoff between grouped wildfire exposure and unmet
demand service.

The grouped wildfire risk now follows the full formulation structure while
keeping line energization fixed:

```text
R_group(u) = sum_k group_weight_k * sum_{l in G_k}
             z_l * p_env_l * loading_l(u)^2 * I_l(u)
```

Current simplifications:

- `z_l = 1` for all lines; line de-energization is not a decision variable
- `p_env_l` is the configured synthetic environmental ignition score
- `loading_l` is reconstructed relative line loading, normalized by rating or
  the configured proxy rating before squaring
- `I_l(u)` is a counterfactual consequence score from removing line `l` from
  the GridFM graph under the current decision vector

The current consequence score is equal-weight service loss, not MW-weighted
load loss:

```text
I_l(u) = max(0, S_current(u) - S_outage_l(u)) / max(S_current(u), eps)
```

`S` is inferred from predicted `Pd` as clipped `Pd_pred / Pd_base`, summed with
equal weight per bus, and zero-load buses counted as fully served. This replaced
the previous constant `impact_l = 1.0` behavior on May 28, 2026.

`L_shed_weighted` is currently the demand-weighted load-shedding fraction over
the full bus vector:

```text
w_n = P_D,n / sum_m P_D,m
L_shed_weighted = sum_n w_n * (1 - alpha_n)
```

Non-selected buses have `alpha_n = 1`, so they contribute zero. This replaced
the equal-bus shed-fraction objective on May 29, 2026 so that shedding 50% of a
large-demand bus is more expensive than shedding 50% of a small-demand bus. The
term is already normalized by total baseline demand, so current generated runs
write `load_shedding_normalizer = 1.0` and
`load_shedding_metric = demand_weighted_fraction`. The old equal-bus sum and
unserved MW remain diagnostic outputs.

Current connected-corridor work uses three convex normalized objective tradeoff
sets:

```text
risk:     lambda_R = 0.999001, lambda_L = 0.000999
balanced: lambda_R = 0.5,      lambda_L = 0.5
shed:     lambda_R = 0.000999, lambda_L = 0.999001
```

The `risk` set is the 0-to-1 equivalent of the previous 1000:1
risk-prioritized preference. The normalizers are written per run in
`objective_normalizers.json`.

## Risk Scenario

The risk scenario is synthetic:

- one line group named `high_risk_corridor`
- line group is manually configured as one connected corridor
- current connected corridor line IDs are `[23, 32, 26]`
- those scenario edges correspond to bus 6--8, bus 8--28, and bus 6--28
- selected group lines get weather/environment ignition probability proxy `1.0`
- non-selected lines get hazard `0.1`
- impact is now computed per evaluated decision as counterfactual equal-weight
  service loss under a one-line GridFM outage
- group weight is currently `1.0`

The previous `top_loaded` method selected the top 3 baseline-loaded line IDs,
which could produce a high-risk line group that was not physically connected in
the topology. The current default configs use `selection_method:
manual_connected` and validate that the selected line IDs form one connected
component before writing results.

Line risk:

```text
risk_l = z_l * p_env_l * loading_l^2 * I_l(u)
```

`z_l` is fixed to `1` in the current first pass.

Grouped risk:

```text
R_group = sum_k group_weight_k * sum_{l in G_k} risk_l
```

## Current Files In wildfire_initial_tests

Core modules:

```text
config.py
scenario.py
decision_vector.py
gridfm_runner.py
state_extraction.py
wildfire_scenario.py
wildfire_risk.py
objective.py
optimization_problem.py
validation.py
reporting.py
objective_analysis.py
ac_opf_experiment.py
```

Entry points:

```text
run_basic_case.py
run_stability_sweep.py
run_connected_corridor_tradeoffs.py
plot_optimization_behavior.py
plot_network_changes.py
objective_analysis.py
ac_opf_experiment.py
run_multistart_optimization.py
```

Configs:

```text
configs/basic_gps.yaml
configs/basic_gnn.yaml
configs/connected_corridor_gps.yaml
configs/connected_corridor_gnn.yaml
configs/stability_sweep.yaml
```

Documentation and artifacts:

```text
FIRST_PASS_SUMMARY.md
HISTORY.md
results/
```

## Tests

Focused tests live outside `experiments/test`:

```text
tests/test_wildfire_first_pass_scenario.py
tests/test_wildfire_first_pass_risk.py
tests/test_wildfire_first_pass_decision_vector.py
tests/test_wildfire_first_pass_objective.py
tests/test_wildfire_first_pass_basic_run.py
tests/test_wildfire_first_pass_objective_analysis.py
tests/test_wildfire_first_pass_multistart.py
```

Latest post-cleanup result:

```text
16 passed
```

Latest verification on May 29, 2026:

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_objective_analysis.py -q
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_objective_analysis.py tests/test_wildfire_first_pass_multistart.py -q
```

Result:

```text
16 passed, 3 external deprecation warnings
```

This check includes the counterfactual line-impact unit test, the frozen
objective-analysis sweep test, and the grid-seeded multistart seed-selection
test.

## Latest Connected-Corridor Runs

Fresh connected-corridor runs were produced on May 28, 2026 after changing the
wildfire consequence term from constant `impact_l = 1.0` to a counterfactual
equal-weight served-load-loss score from one-line GridFM outage predictions.

An attempted `--clear` regeneration could not remove one older OneDrive-locked
figure directory, so the successful May 28 regeneration was run without
`--clear`. The aggregate CSV/JSON summary points to the new May 28 run
directories, while at least one older locked result folder may remain on disk.

The three result sets are under:

```text
results/connected_corridor/risk/
results/connected_corridor/balanced/
results/connected_corridor/shed/
```

Aggregate machine-readable outputs:

```text
results/connected_corridor/connected_corridor_tradeoff_summary.csv
results/connected_corridor/connected_corridor_tradeoff_summary.json
```

Latest run summary after deleting and regenerating `results/connected_corridor`
with selected-corridor `p_env_l = 1.0`:

```text
risk/gnn:
  objective:      0.999001 -> 0.999001
  group risk:     9.276730 -> 9.276730
  load shedding:  0.000000
  mean alpha:     1.000000

risk/gps:
  objective:      0.999001 -> 0.999001
  group risk:     2.693767 -> 2.693767
  load shedding:  0.000000
  mean alpha:     1.000000

balanced/gnn:
  objective:      0.500000 -> 0.500000
  group risk:     9.276730 -> 9.276730
  load shedding:  0.000000
  note: optimizer reported ABNORMAL but returned the baseline decision

balanced/gps:
  objective:      0.500000 -> 0.500000
  group risk:     2.693767 -> 2.693767
  load shedding:  0.000000

shed/gnn:
  objective:      0.000999 -> 0.000999
  group risk:     9.276730 -> 9.276730
  load shedding:  0.000000

shed/gps:
  objective:      0.000999 -> 0.000999
  group risk:     2.693767 -> 2.693767
  load shedding:  0.000000
```

On May 15, 2026, the implementation was corrected to remove the previous
generator movement cost from the optimized scalar objective. On May 18, 2026,
the load penalty was changed from MW-weighted unserved demand to unweighted
load-shedding fraction sum, the objective weights were converted from
`1000.0`/`1.0` to convex tradeoff weights, and `alpha_min` was relaxed to
`0.0`. Generator movement is still recorded as a diagnostic, but the scalar
objective is now exactly the two-term wildfire-risk versus load-shedding
tradeoff.

On May 19, 2026, the optimization behavior plot methodology was clarified:
the fourth panel now plots the configured weights `lambda_R` and `lambda_L`,
not the realized weighted normalized terms. The realized terms remain in
`objective_trace.csv` for auditability.

On May 28, 2026, `GridFMRunner.predict_with_line_outage` and
`compute_counterfactual_line_impacts` were added so `impact` in the risk CSVs
is a decision-dependent counterfactual consequence score. The May 28 runs
returned the baseline decision for all tradeoff sets, which should be treated
as evidence that the current corrected risk term and selected controls do not
yet produce an accepted improvement, not as a final physical conclusion.

Also on May 28, 2026, `FIRST_PASS_SUMMARY.md` was updated to include the
current optimization problem, objective value definition, latest objective
values, decision variables, explicit bounds, fixed topology assumptions,
wildfire scenario constraints, prediction validity rules, and optimizer
settings.

Later on May 28, 2026, the selected-corridor weather/environment ignition
factor was changed from `5.0` to `1.0` in the first-pass configs so the
selected-line probability factor is probability-like rather than a large hazard
multiplier. The existing `results/connected_corridor` directory was deleted
successfully, and `run_connected_corridor_tradeoffs.py` regenerated the three
tradeoff sets. Group risks dropped by the expected factor, but all six runs
still returned the baseline decision.

Also on May 28, 2026, `objective_analysis.py` was added for frozen objective
sensitivity analysis. It performs one GridFM evaluation at a fixed decision,
freezes the prediction, decision vector, reconstructed loading, normalizers,
and all line impacts, then manually varies exactly one line consequence value
without rerunning GridFM or invoking the optimizer. The first analysis swept
`I_23(u)` from `0.0` to `1.0` using
`configs/connected_corridor_gps.yaml` and wrote:

```text
results/objective_analysis/line_23_impact_sweep_20260528_220439/
```

The frozen baseline values were:

```text
frozen grouped risk: 2.6937669151682497
risk normalizer:     2.6937669151682497
load normalizer:     30.0
lambda_R:            0.999001
lambda_L:            0.000999
base I_23(u):        0.03000268142041778
```

`FIRST_PASS_SUMMARY.md` was also expanded with a dedicated simplifications
section covering fixed energization, synthetic environmental probability,
single wildfire group, reduced decision vector, selected-load-only shedding,
equal-bus shedding/consequence metrics, surrogate line-outage impact,
reconstructed/proxy loading, missing hard AC feasibility constraints, absence
of full wildfire physics, static single-scenario scope, local optimizer
limitations, baseline-relative normalization, and limited current result
interpretation.

Later on May 28, 2026, `objective_analysis.py` was extended with a second,
separate decision-variable sweep mode. This is different from the frozen
consequence sweep above:

- the frozen consequence sweep freezes one GridFM evaluation and manually
  changes one line impact value, such as `I_23`, without rerunning GridFM
- the decision-variable sweep varies one actual optimizer variable at a time,
  reruns GridFM and the counterfactual line-impact calculations at each point,
  and still does not invoke the optimizer

The latest decision-variable sweep commands were:

```powershell
python experiments/test/wildfire_initial_tests/objective_analysis.py --mode decision --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml --num-points 21
python experiments/test/wildfire_initial_tests/objective_analysis.py --mode decision --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gnn.yaml --num-points 21
```

They wrote:

```text
results/objective_analysis/decision_sweeps/decision_sweeps_gps_20260528_235031/
results/objective_analysis/decision_sweeps/decision_sweeps_gnn_20260528_235108/
```

In the decision-sweep figure, each subplot corresponds to one actual decision
variable while all other decision variables remain fixed at baseline. The
x-axis is the decision value: `Delta_Pg` sweeps from `-5 MW` to `+5 MW`, and
`alpha` sweeps from `0.0` to `1.0`. The left y-axis is scalar objective `J(u)`
as the black curve. The right y-axis is raw grouped wildfire risk `R_group(u)`
as the red curve. The CSV carries the audit columns needed to reconstruct the
objective: normalized risk, normalized load shedding, weighted risk term,
weighted load-shedding term, and corridor line impacts.

GPS decision sweep result:

```text
baseline objective: 0.999001
best one-variable point: alpha_7_bus18 = 0.0
best objective: 0.8961604984016736
best grouped risk: 2.4163717542193073
best load shedding: 1.0
```

For this best GPS point, the objective change decomposes as:

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

The objective changes only modestly in many plots because the current GPS
risk-prioritized config heavily favors risk:
`lambda_R = 0.999001`, `lambda_L = 0.000999`. Completely shedding one selected
bus adds only `0.000999 * (1/30) = 0.0000333` to the scalar objective. Thus,
unless the candidate decision meaningfully changes normalized grouped risk,
the total objective will stay visually close to baseline.

GNN decision sweep result:

```text
baseline objective: 0.999001
best one-variable point: alpha_5_bus11 = 0.0
best objective: 0.9953966468056175
best grouped risk: 9.24295077382007
best load shedding: 1.0
```

Interpretation: the decision sweeps show objective-improving one-variable
points, especially for GPS, even though the local L-BFGS-B tradeoff runs
returned the baseline decision. This suggests the no-movement result is more
likely due to local-search, finite-difference, or nonsmooth surrogate behavior
than due to a total absence of objective signal. The GPS `alpha_7_bus18` sweep
is also nonmonotonic: small reductions from `alpha = 1.0` initially make the
objective worse, while larger reductions eventually improve it, so a local
optimizer initialized at baseline can rationally stay near baseline even though
a farther one-variable point is better.

On May 29, 2026, `run_multistart_optimization.py` was added as a separate
grid-seeded multistart diagnostic under `results/multistart/`. It does not
change the objective or constraints. It evaluates a one-variable grid around
the baseline, selects the best unique seed decisions, runs the same L-BFGS-B
optimizer from each seed, and keeps the best final result. The optimizer class now has
`optimize_multistart(starts)` while preserving the original single-start
`optimize(...)` path.

Command:

```powershell
python experiments/test/wildfire_initial_tests/run_multistart_optimization.py --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml --num-seed-points 11 --max-seeds 5
```

Output:

```text
results/multistart/multistart_gps_20260529_000941/
```

Selected starts:

```text
alpha_7_bus18 = 0.0, 0.1, 0.2, 0.3, 0.4
```

Result:

```text
baseline objective:          0.999001
best seed objective:         0.9092841653972372
best final objective:        0.8961482348645832
best final R_group:          2.416338685293566
best final L_shed:           1.0000086888091182
best start index:            1
optimizer success:           True
optimizer message:           CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
```

All five starts converged successfully to nearly the same objective region.
The final decision keeps all generator redispatch at `0.0`, sets
`alpha_7_bus18 = 0.0`, and makes only tiny numerical changes to a few other
selected alpha values. This confirms that the risk-preferred GPS formulation
has a lower-objective point that the baseline-started optimizer was not
finding. The objective moves in two stages: the grid-seeded start moves the
candidate from the baseline objective `0.999001` to `0.9092841653972372`, and
then L-BFGS-B improves that modified start to `0.8961482348645832`.

The multistart runner now writes both multistart-specific artifacts and the
standard optimization-behavior plot for the best start:

```text
results/multistart/multistart_gps_20260529_000941/selected_starts.csv
results/multistart/multistart_gps_20260529_000941/multistart_results.csv
results/multistart/multistart_gps_20260529_000941/objective_trace.csv
results/multistart/multistart_gps_20260529_000941/figures/optimization_behavior.png
```

Later on May 29, 2026, the older nested multistart output under
`results/objective_analysis/multistart/` was deleted so multistart is no longer
mixed with the objective-analysis sensitivity folders. The multistart runner
was extended with `--tradeoff-sets`, which generates the three standard lambda
cases (`risk`, `balanced`, `shed`) for both `gnn` and `gps` under:

```text
results/multistart/<tradeoff_set>/<model_type>/<timestamped_run>/
```

Command:

```powershell
python experiments/test/wildfire_initial_tests/run_multistart_optimization.py --tradeoff-sets --num-seed-points 11 --max-seeds 5
```

An attempted `--clear` run could not delete one older direct
`results/multistart/multistart_gps_20260529_000941/figures` directory because
OneDrive/Windows reported an access-denied lock. This did not affect the
requested cleanup of the nested objective-analysis folder, and the six fresh
tradeoff runs were generated without `--clear`.

Latest multistart tradeoff outputs:

```text
results/multistart/risk/gnn/multistart_gnn_20260529_001357/
results/multistart/risk/gps/multistart_gps_20260529_001442/
results/multistart/balanced/gnn/multistart_gnn_20260529_001543/
results/multistart/balanced/gps/multistart_gps_20260529_001624/
results/multistart/shed/gnn/multistart_gnn_20260529_001700/
results/multistart/shed/gps/multistart_gps_20260529_001708/
```

Aggregate summary:

```text
results/multistart/multistart_tradeoff_summary.csv
results/multistart/multistart_tradeoff_summary.json
```

Each run includes `figures/optimization_behavior.png`, `selected_starts.csv`,
`multistart_results.csv`, `objective_trace.csv`, `best_objective_trace.csv`,
`best_decision_vector.csv`, `analysis_summary.json`, and
`optimization_summary.json`.

Latest multistart tradeoff result summary:

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

Interpretation: risk-preferred GPS remains the clearest improvement, balanced
GPS also improves, risk-preferred GNN improves slightly, and both
shed-preferred cases correctly stay at baseline because the objective strongly
penalizes any load shedding.

Later on May 29, 2026, after switching the optimized load-shedding cost to the
demand-weighted fraction, the three connected-corridor tradeoff cases were
regenerated for both GNN and GPS without multistart:

```powershell
python experiments/test/wildfire_initial_tests/run_connected_corridor_tradeoffs.py
```

Latest demand-weighted baseline-started outputs:

```text
results/demand_weighted/connected_corridor/risk/gnn/risk_gnn_20260529_134532/
results/demand_weighted/connected_corridor/risk/gps/risk_gps_20260529_134541/
results/demand_weighted/connected_corridor/balanced/gnn/balanced_gnn_20260529_134543/
results/demand_weighted/connected_corridor/balanced/gps/balanced_gps_20260529_134550/
results/demand_weighted/connected_corridor/shed/gnn/shed_gnn_20260529_134552/
results/demand_weighted/connected_corridor/shed/gps/shed_gps_20260529_134553/
```

Demand-weighted baseline-started summary:

```text
risk/gnn:
  objective:  0.999001 -> 0.9989932074556757
  group risk: 9.276730045998226 -> 9.276657378702414
  min alpha:  0.9995908031440097

risk/gps:
  objective:  0.999001 -> 0.999001
  group risk: 2.6937669151682497 -> 2.6937669151682497

balanced/gnn:
  objective:  0.5 -> 0.5
  group risk: 9.276730045998226 -> 9.276730045998226
  note: optimizer reported ABNORMAL and returned baseline

balanced/gps:
  objective:  0.5 -> 0.5
  group risk: 2.6937669151682497 -> 2.6937669151682497

shed/gnn:
  objective:  0.000999 -> 0.000999
  group risk: 9.276730045998226 -> 9.276730045998226

shed/gps:
  objective:  0.000999 -> 0.000999
  group risk: 2.6937669151682497 -> 2.6937669151682497
```

The matching demand-weighted multistart set was also regenerated:

```powershell
python experiments/test/wildfire_initial_tests/run_multistart_optimization.py --tradeoff-sets --num-seed-points 11 --max-seeds 5
```

Latest demand-weighted multistart outputs:

```text
results/demand_weighted/multistart/risk/gnn/multistart_gnn_20260529_134608/
results/demand_weighted/multistart/risk/gps/multistart_gps_20260529_134657/
results/demand_weighted/multistart/balanced/gnn/multistart_gnn_20260529_134755/
results/demand_weighted/multistart/balanced/gps/multistart_gps_20260529_134825/
results/demand_weighted/multistart/shed/gnn/multistart_gnn_20260529_134854/
results/demand_weighted/multistart/shed/gps/multistart_gps_20260529_134901/
```

Demand-weighted multistart summary:

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

Interpretation: demand weighting makes the load penalty larger than the old
`1/N_bus` equal-bus normalization for the same complete selected-bus shed, but
the risk-preferred GPS and balanced GPS multistart cases still improve because
the risk reduction is large enough to offset about 3.35% demand-weighted
shedding. Shed-preferred cases still stay at baseline.

Later, `ac_opf_experiment.py` was added as a deliberately isolated hard
AC-OPF experiment. It follows the standard AC-OPF constraint categories from
MATPOWER/PowerModels-style formulations: AC P/Q nodal balance equalities,
branch apparent-flow thermal inequalities, generator P/Q bounds, voltage
magnitude bounds, reference-angle constraint, and selected alpha bounds. It is
not imported by the main GridFM workflow and can be removed by deleting the
module plus `results/ac_opf_connected_corridor/`.

The first run wrote:

```text
results/ac_opf_connected_corridor/ac_opf_20260528_223144/
```

Command:

```powershell
python experiments/test/wildfire_initial_tests/ac_opf_experiment.py --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml
```

Result: all three tradeoff solves failed to satisfy hard AC constraints. SLSQP
reported `Positive directional derivative for linesearch`; final max absolute
P/Q balance residuals stayed near `7767.5 MVA`, and final thermal margins were
about `-2884.6 MVA`. Treat these outputs as evidence that the local
`ScenarioData` approximation is not yet a reliable hard AC-OPF model, not as a
valid AC-OPF comparison result.

## Stage B.1/B.2 Multi-Group Wildfire Risk Sensitivity

On May 31, 2026, Stage B.1/B.2 was implemented as an extension of the fixed
topology first-pass workflow. The manual connected corridor remains available
through:

```yaml
wildfire:
  selection_method: manual_connected
```

The new automatic mode is:

```yaml
wildfire:
  selection_method: automatic_risk_components
```

Automatic ranking computes a baseline score for every candidate line:

```text
score_l = p_env * loading_l(base)^2 * I_l(base)
```

For Stage B.1/B.2, `p_env` is uniform before selection. The checked-in
automatic configs set `risk_score.candidate_p_env = 1.0`, so the ranking does
not reuse the old manual-corridor convention of `1.0` on selected lines and
`0.1` elsewhere. This avoids circular selection because selected lines do not
exist until after ranking.

Top-fraction selection uses:

```text
num_selected_lines = ceil(num_lines * requested_top_fraction)
realized_selected_fraction = num_selected_lines / num_lines
```

Selected lines are grouped by connected components into `G_1`, `G_2`, ... with
equal group weights. The automatic code allows a single connected component and
records whether the result collapsed to one group.

New configs:

```text
configs/automatic_multigroup_gps.yaml
configs/automatic_multigroup_gnn.yaml
```

New runner:

```powershell
python experiments/test/wildfire_initial_tests/run_multi_group_threshold_sensitivity.py --top-fractions 0.10 0.125 0.15 0.175 0.20
```

The runner defaults to grid-seeded multistart for `gps` and `gnn`, over
`risk`, `balanced`, and `shed`, with near-0/near-1 lambda cases:

```text
risk:     lambda_R = 0.999001, lambda_L = 0.000999
balanced: lambda_R = 0.5,      lambda_L = 0.5
shed:     lambda_R = 0.000999, lambda_L = 0.999001
```

New result root:

```text
results/multi_group/
```

Threshold folder names use:

```text
threshold_0p10
threshold_0p125
threshold_0p15
threshold_0p175
threshold_0p20
```

Each automatic run writes:

```text
automatic_line_risk_scores.csv
automatic_wildfire_groups.json
automatic_group_summary.csv
```

Multistart runs now also write the standard run artifacts needed for
comparison and visualization: baseline/final objective components,
baseline/risk-before-after CSVs, decision vectors, traces, normalizers,
`optimization_summary.json`, `analysis_summary.json`,
`figures/optimization_behavior.png`, and
`figures/ieee30_network_changes.png`.

The topology visualization now colors automatic high-risk lines by group and
adds group legend entries while preserving selected generator/load bus
highlighting. `visualization_summary.json` includes selection method,
requested/realized top fraction, selected line IDs, group IDs, group line and
bus IDs, collapsed-to-single-group status, and largest-group diagnostics.

Verified smoke command:

```powershell
python experiments/test/wildfire_initial_tests/run_multi_group_threshold_sensitivity.py --top-fractions 0.10 --models gps --tradeoff-cases risk --num-seed-points 3 --max-seeds 2
```

Latest smoke result:

```text
results/multi_group/threshold_0p10/risk/gps/multistart_gps_20260531_191421/
```

Smoke summary:

```text
requested_top_fraction: 0.10
num_lines: 110
num_selected_lines: 11
realized_selected_fraction: 0.10
num_groups: 5
largest_group_num_lines: 5
largest_group_fraction_of_selected_lines: 0.45454545454545453
collapsed_to_single_group: false
baseline_objective: 0.999001
best_seed_objective: 0.8983104240958543
best_objective: 0.898298444155839
best_grouped_risk: 2.5433063515171694
best_load_shedding: 0.03352434654418237
optimizer_success: true
```

The full 30-run default Stage B.2 sweep has not yet been run in full because
the verified smoke run took about 100 seconds for one reduced GPS/risk case
with `--num-seed-points 3 --max-seeds 2`; the default
`--num-seed-points 11 --max-seeds 5` sweep is materially heavier.

Focused verification:

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_multistart.py tests/test_wildfire_first_pass_multi_group_runner.py -q
```

Result:

```text
18 passed, 3 external deprecation warnings
```

Stage B.3/B.4 remain deferred. Distinct seeded grouping and manual
multi-region stress testing were intentionally not implemented and should only
be considered after manual review of Stage B.1/B.2 results under
`results/multi_group/`.

Later on May 31, 2026, GPS-only Stage B.2 results were generated for
20% and 30% thresholds using the default demand-weighted, automatic
multi-group, grid-seeded multistart methodology:

```powershell
python experiments/test/wildfire_initial_tests/run_multi_group_threshold_sensitivity.py --top-fractions 0.20 0.30 --models gps --tradeoff-cases risk balanced shed
```

The objective normalizers in these runs record:

```text
load_shedding_normalizer = 1.0
load_shedding_metric = demand_weighted_fraction
```

Output directories:

```text
results/multi_group/threshold_0p20/risk/gps/multistart_gps_20260531_192631/
results/multi_group/threshold_0p20/balanced/gps/multistart_gps_20260531_193233/
results/multi_group/threshold_0p20/shed/gps/multistart_gps_20260531_193552/
results/multi_group/threshold_0p30/risk/gps/multistart_gps_20260531_193631/
results/multi_group/threshold_0p30/balanced/gps/multistart_gps_20260531_194657/
results/multi_group/threshold_0p30/shed/gps/multistart_gps_20260531_195041/
```

Aggregate summary:

```text
results/multi_group/multi_group_threshold_sensitivity_summary.csv
results/multi_group/multi_group_threshold_sensitivity_summary.json
```

GPS 20% summary:

```text
num_selected_lines: 22 / 110
realized_selected_fraction: 0.20
num_groups: 6
largest_group_num_lines: 10
largest_group_fraction_of_selected_lines: 0.45454545454545453
collapsed_to_single_group: false
risk objective:     0.999001 -> 0.8985520880723752
balanced objective: 0.5      -> 0.4664756256271531
shed objective:     0.000999 -> 0.000999
```

GPS 30% summary:

```text
num_selected_lines: 33 / 110
realized_selected_fraction: 0.30
num_groups: 5
largest_group_num_lines: 24
largest_group_fraction_of_selected_lines: 0.7272727272727273
collapsed_to_single_group: false
risk objective:     0.999001 -> 0.8985455799965569
balanced objective: 0.5      -> 0.4664754236987482
shed objective:     0.000999 -> 0.000999
```

## Stage C PSPS Threshold Baseline

On May 31, 2026, Stage C was implemented as a deterministic PSPS-only
baseline/comparator. It does not optimize `z_l` or `y_n`, does not add
mixed-integer topology controls, and does not optimize continuous controls
after PSPS. Stage D optimized de-energization remains deferred.

Methodology:

```text
grouping_top_fraction = 0.30
psps_top_fraction = 0.10
lambda_R = 0.999001
lambda_L = 0.000999
evaluation_mode = psps_only
```

The grouping threshold builds the Stage B automatic multi-group scenario.
Only those candidate high-risk lines are eligible for Stage C PSPS
de-energization; all non-candidate lines keep `z_l = 1`.

Stage C uses a fixed demand-weighted line consequence score:

```text
I_l = max(0, S_D_base - S_D_outage_l) / S_D_base
S_D_base = sum_n P_D,n
```

This is computed once per line from the baseline one-line outage prediction.
Stage C does not use the old dynamic equal-bus `I_l(u)` consequence in risk
calculations.

For each environmental case, case-specific `p_env_l` is assigned before PSPS
ranking. The PSPS ranking score is:

```text
baseline_psps_risk_l = p_env_l * loading_l(base)^2 * I_l
```

The de-energization rule is:

```text
num_psps_lines = max(1, ceil(psps_top_fraction * num_candidate_lines))
realized_psps_fraction = num_psps_lines / num_candidate_lines
```

Then candidate lines are sorted by descending `baseline_psps_risk_l` and
ascending `line_id`; the top `num_psps_lines` get `z_l = 0`. A line with
`z_l = 0` contributes zero wildfire risk.

Implemented environmental cases:

```text
auto_env
largest_group_high
```

`largest_group_high` identifies the largest automatic connected group by line
count, then baseline group risk, then lower numeric group ID. It sets that
group to `p_env = 1.0`; other groups receive seeded random group-level
`p_env` values in `[0.6, 0.9]` with seed `30`.

New files:

```text
experiments/test/wildfire_initial_tests/stage_c_psps.py
experiments/test/wildfire_initial_tests/run_stage_c_psps_baseline.py
tests/test_wildfire_stage_c_psps.py
```

The GridFM runner now supports `predict_with_line_outages(...)` for one or
more removed scenario lines. For post-PSPS line loading, Stage C uses the
PSPS topology GridFM bus prediction on the original line indexing and
explicitly sets de-energized line loading to zero. This avoids a reduced-edge
loading shape mismatch in the existing overload evaluator while preserving
original line-ID auditability.

The planned long Stage C folder names were shortened because Windows/OneDrive
path length limits prevented writing required artifact filenames. Outputs are
still under `results/stage_c_psps/`:

```text
results/stage_c_psps/t0p30/p0p10/r/gps/auto/run_20260531_205156/
results/stage_c_psps/t0p30/p0p10/r/gps/lgh/run_20260531_205158/
```

Aggregate outputs:

```text
results/stage_c_psps/stage_c_psps_summary.csv
results/stage_c_psps/stage_c_psps_summary.json
```

Command run:

```powershell
python experiments/test/wildfire_initial_tests/run_stage_c_psps_baseline.py --grouping-top-fraction 0.30 --psps-top-fraction 0.10 --models gps --cases auto_env largest_group_high
```

GPS Stage C results:

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
  disconnected components:      1
  affected bus IDs:             [4, 5, 6, 7, 27]

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
  disconnected components:      1
  affected bus IDs:             [4, 5, 6, 7, 27]
```

Each run writes:

```text
fixed_line_consequence_scores.csv
psps_line_risk_scores.csv
psps_deenergization_decisions.csv
automatic_wildfire_groups.json
automatic_group_summary.csv
objective_trace.csv
optimization_summary.json
visualization_summary.json
figures/optimization_behavior.png
figures/ieee30_network_changes.png
```

`objective_trace.csv` has exactly two rows: baseline all-energized and
post-PSPS threshold topology. The network visualization now overlays PSPS
de-energized lines when the Stage C decision artifact is present.

Focused verification:

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_multistart.py tests/test_wildfire_stage_c_psps.py -q
```

Result:

```text
23 passed, 3 external deprecation warnings
```

Later on May 31, 2026, the Stage C network visualization was cleaned up so
PSPS de-energized lines are visually unambiguous. De-energized lines are now
drawn last with a red dashed stroke, red X markers, and `OFF <line_id>` edge
labels. The legend is placed below the lower-right side of the plot in two
columns so it no longer blocks the lower-left network area. The current Stage C
GPS figures were regenerated for:

```text
results/stage_c_psps/t0p30/p0p10/r/gps/auto/run_20260531_205156/
results/stage_c_psps/t0p30/p0p10/r/gps/lgh/run_20260531_205158/
```

The `results/stage_c_psps/` directory was also cleaned to retain only the
aggregate summary files and the two current near-lambda GPS runs with figures.
Older failed path-length attempts and pre-near-lambda runs were removed.

Verification:

```powershell
pytest tests/test_wildfire_stage_c_psps.py -q
```

Result:

```text
7 passed, 3 external deprecation warnings
```

Each `run_basic_case.py` execution creates a timestamped run directory:

```text
results/<run_name>_<YYYYMMDD_HHMMSS>/
```

For the current demand-weighted connected-corridor workflow, model runs are
organized as:

```text
results/demand_weighted/connected_corridor/<tradeoff_set>/gnn/<run_name>_<timestamp>/
results/demand_weighted/connected_corridor/<tradeoff_set>/gps/<run_name>_<timestamp>/
results/demand_weighted/connected_corridor/sweeps/<sweep_name>_<timestamp>/
```

Core machine-readable outputs stay at the run-directory root as JSON and CSV
files. Human-facing plots are grouped under:

```text
results/<run_name>_<timestamp>/figures/
```

Automatic figures from `run_basic_case.py`:

```text
figures/optimization_behavior.png
figures/ieee30_network_changes.png
visualization_summary.json
```

`optimization_behavior.png` shows objective, grouped wildfire risk, load
shedding over objective evaluations, and the configured objective weights
`lambda_R` and `lambda_L`. The weight panel is visual-only; realized weighted
normalized terms remain in `objective_trace.csv`. `ieee30_network_changes.png`
is generated from the actual IEEE-30 `edge_index` used by the scenario. The
layout is topology-accurate but not geographic: buses connected in the scenario
graph are connected in the figure, but coordinates come from a deterministic
spring layout. Bus labels are displayed as 1-based IEEE-style labels; high-risk
line labels are the internal scenario edge IDs used by the wildfire scenario
JSON and risk CSV files.

For stability sweeps, `run_stability_sweep.py` creates one timestamped sweep
directory and places the child optimization runs under:

```text
results/<sweep_name>_<timestamp>/runs/
```

This keeps repeated sweeps and their per-model/per-seed figures grouped
together instead of scattering child run folders at the top level.

## Commands To Reproduce

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py -q
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_objective_analysis.py -q
python experiments/test/wildfire_initial_tests/run_connected_corridor_tradeoffs.py --clear
python experiments/test/wildfire_initial_tests/run_connected_corridor_tradeoffs.py
python experiments/test/wildfire_initial_tests/objective_analysis.py --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml --line-id 23 --num-points 101
python experiments/test/wildfire_initial_tests/objective_analysis.py --mode decision --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml --num-points 21
python experiments/test/wildfire_initial_tests/objective_analysis.py --mode decision --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gnn.yaml --num-points 21
python experiments/test/wildfire_initial_tests/run_multistart_optimization.py --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml --num-seed-points 11 --max-seeds 5
python experiments/test/wildfire_initial_tests/run_multistart_optimization.py --tradeoff-sets --num-seed-points 11 --max-seeds 5
python experiments/test/wildfire_initial_tests/ac_opf_experiment.py --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml
python experiments/test/wildfire_initial_tests/run_basic_case.py --config experiments/test/wildfire_initial_tests/configs/basic_gps.yaml
python experiments/test/wildfire_initial_tests/run_basic_case.py --config experiments/test/wildfire_initial_tests/configs/basic_gnn.yaml
python experiments/test/wildfire_initial_tests/run_basic_case.py --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml
python experiments/test/wildfire_initial_tests/run_basic_case.py --config experiments/test/wildfire_initial_tests/configs/connected_corridor_gnn.yaml
python experiments/test/wildfire_initial_tests/run_stability_sweep.py --config experiments/test/wildfire_initial_tests/configs/stability_sweep.yaml
python experiments/test/wildfire_initial_tests/plot_optimization_behavior.py --run-dir experiments/test/wildfire_initial_tests/results/<run_name>
python experiments/test/wildfire_initial_tests/plot_network_changes.py --run-dir experiments/test/wildfire_initial_tests/results/<run_name>
```

## Important Implementation Notes

- Do not call GridFM through the CLI inside the objective loop.
- Use in-memory model loading through `pipeline_utils.py`.
- Use `NeuralSolverWrapper` for prediction.
- Counterfactual line-outage impact currently removes one scenario edge inside
  `GridFMRunner.predict_with_line_outage`; this is a surrogate consequence
  calculation, not an optimizer decision to de-energize the line.
- Keep generated objective components separate from scalar objective values.
- Invalid predictions should produce explicit penalties or validation failures, not silent crashes.
- Generated result directories are currently under `experiments/test/wildfire_initial_tests/results`.

## Next Session Launch Plan

When a new Codex session starts, read this section first and then inspect the
current code/results before editing. Preserve the current methodology:

- update code, configs, tests, generated outputs, `FIRST_PASS_SUMMARY.md`, and
  this `HISTORY.md` together when behavior changes
- run focused tests after implementation changes
- rerun the relevant model workflows when objective, decision, risk, or result
  semantics change
- record commands and outcomes here so this file remains a durable handoff

Current research transition: the simplified fixed-topology/fixed-energization
case is now polished enough for first-pass interpretation. The next major
methodology step is to evaluate behavior with de-energization included, while
keeping the existing simplified results as the baseline comparison.

Stage D update, May 31, 2026:

- Added `run_stage_d_deenergization.py` and Stage D helper logic for
  `evaluation_mode = limited_enumerated_z_only`.
- Stage D uses `grouping_top_fraction = 0.30`, computes the generated
  candidate count dynamically, and enumerates all `z_l` subsets with 0, 1, or
  2 de-energized candidate lines.
- The current GPS run generated 33 candidate lines and 562 evaluated subsets
  per environmental case.
- Subset evaluations are shared across the three Stage D lambda cases:

```text
risk_leaning:    lambda_R = 0.8, lambda_L = 0.2
balanced:        lambda_R = 0.5, lambda_L = 0.5
service_leaning: lambda_R = 0.2, lambda_L = 0.8
```

- Stage D uses the fixed demand-weighted `I_l` from Stage C:

```text
risk_l = z_l * p_env_l * loading_l^2 * I_l
```

- `R_group_baseline` is the all-energized baseline for the same model,
  environmental case, grouping threshold, fixed `I_l`, and candidate groups;
  it is not lambda-dependent.
- Stage C comparison is available for the GPS runs. The Stage C PSPS topology
  is fixed, while its scalar objective is re-scored under each Stage D lambda
  for apples-to-apples comparison.
- Generated outputs:

```text
results/stage_d_deenergization/t0p30/gps/auto/risk/run_20260531_212125/
results/stage_d_deenergization/t0p30/gps/auto/bal/run_20260531_212126/
results/stage_d_deenergization/t0p30/gps/auto/svc/run_20260531_212128/
results/stage_d_deenergization/t0p30/gps/lgh/risk/run_20260531_212136/
results/stage_d_deenergization/t0p30/gps/lgh/bal/run_20260531_212138/
results/stage_d_deenergization/t0p30/gps/lgh/svc/run_20260531_212139/
results/stage_d_deenergization/stage_d_deenergization_summary.csv
```

- GPS best subsets:

```text
auto_env/risk_leaning:          [23, 27]
auto_env/balanced:              [23, 27]
auto_env/service_leaning:       []
largest_group_high/risk_leaning:[23, 27]
largest_group_high/balanced:    [23, 27]
largest_group_high/service:     []
```

- `ieee30_network_changes.png` now recognizes
  `optimized_deenergization_decisions.csv` and labels those lines as
  Stage D optimized de-energized lines.
- Deferred: full mixed-integer topology optimization, relaxed continuous
  `z_l`, optimized `y_n`, AC feasibility enforcement, and continuous-control
  optimization after selecting `z_l`.

Latest Stage D verification:

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_multistart.py tests/test_wildfire_stage_c_psps.py tests/test_wildfire_stage_d_deenergization.py -q
python experiments/test/wildfire_initial_tests/run_stage_d_deenergization.py --grouping-top-fraction 0.30 --models gps --cases auto_env largest_group_high --evaluation-mode limited_enumerated_z_only --max-deenergized-lines 2
```

Test result: 30 passed, with 3 external deprecation warnings.

Immediate Stage B follow-up: run the full default multi-group threshold sweep
when compute time is available, then manually inspect
`results/multi_group/multi_group_threshold_sensitivity_summary.csv` and the
child automatic group artifacts before deciding whether Stage B.3/B.4 are
needed.

```powershell
python experiments/test/wildfire_initial_tests/run_multi_group_threshold_sensitivity.py --top-fractions 0.10 0.125 0.15 0.175 0.20
```

Priority 1: add a PSPS-style baseline with threshold de-energization.

The previous visual-clarity task for `optimization_behavior.png` was completed
on May 19, 2026: the plot now shows configured objective weights instead of
realized weighted term magnitudes.

Remaining methodology note: if future work again compares realized objective
terms across tradeoff sets, start from
`results/demand_weighted/connected_corridor/connected_corridor_tradeoff_summary.csv`
and the child `objective_trace.csv` files, and remember that the current
normalizers are `R_group / R_baseline` and the already normalized
demand-weighted `L_shed_weighted`.

- Implement a comparison baseline that de-energizes all lines above a wildfire
  risk threshold, representing a simple PSPS rule.
- Keep this separate from the current GridFM continuous optimization at first:
  it should be a baseline/comparator, not yet a mixed-integer optimizer.
- Decide how to represent de-energized lines in the current data path:
  likely a new result-generation module and reporting artifacts before
  introducing `z_l` into the optimizer.
- Compare at least:
  grouped wildfire risk, load shedding/service loss, affected buses/loads,
  disconnected components if any, and visualization of de-energized corridor
  lines.
- Add tests for threshold selection and reporting behavior before trusting the
  baseline results.

Priority 2 status: demand-weighted load-shedding cost has been implemented.

- Completed: the optimized load-shedding term now uses demand weights:

```text
w_n = P_D,n / sum_m P_D,m
L_shed_weighted = sum_n w_n * (1 - alpha_n)
```

- Completed: current generated runs use `load_shedding_normalizer = 1.0`
  because the demand weights sum to 1.
- Completed: tests were updated so a large-load bus contributes more to the
  objective than an equal fractional shed at a small-load bus.
- Completed: traces and multistart summaries now include diagnostic
  `equal_bus_load_shedding` and `unserved_demand_mw` where available.
- Remaining: if a future comparison needs the old equal-bus objective, add it
  as a named objective option rather than silently changing this metric back.

Deferred but still relevant:

1. Decide whether relaxed `[0, 1]` alpha should be the main setting or an
   emergency-style sensitivity, since `risk/gps` currently reaches
   `alpha = 0.0` on selected buses.
2. Run a reduced stability sweep and summarize variability.
3. Inspect sensitivity of the new counterfactual `I_l(u)` term to `alpha` and
   `Delta_Pg`, and decide whether to back it with a physical solver.
4. Add corridor-specific finite-difference control selection.
5. Add voltage and thermal violation diagnostics to CSV/JSON summaries.
6. Decide whether generated `results/` should be tracked or moved to ignored
   local output.
7. Later, revisit topology controls `y_n` and `z_l`, but do not add
   mixed-integer logic until the PSPS baseline is understood.
