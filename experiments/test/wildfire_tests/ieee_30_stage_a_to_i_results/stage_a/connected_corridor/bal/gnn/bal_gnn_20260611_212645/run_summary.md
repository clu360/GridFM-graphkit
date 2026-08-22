# Wildfire First Pass Summary

## What Was Implemented

A reduced fixed-topology wildfire predict-then-optimize experiment path was added under `experiments/test/wildfire_tests`.

## Reused Modules

- `experiments.test.wildfire_tests.gridfm_support.pipeline_utils` for scenario and checkpoint loading.
- `experiments.test.wildfire_tests.gridfm_support.neural_solver.NeuralSolverWrapper` for in-memory GridFM inference.
- `experiments.test.wildfire_tests.gridfm_support.scenario_data.ScenarioData` for the IEEE-30 scenario representation.
- `experiments.test.wildfire_tests.gridfm_support.overload_penalty` for branch loading reconstruction.

## New Modules Added

- `config.py`, `scenario.py`, `decision_vector.py`, `gridfm_runner.py`, `state_extraction.py`
- `wildfire_scenario.py`, `wildfire_risk.py`, `objective.py`, `optimization_problem.py`
- `validation.py`, `reporting.py`, `run_basic_case.py`, `run_stability_sweep.py`
- `plot_optimization_behavior.py`, `plot_network_changes.py`

## First-Pass Assumptions

The topology is fixed, all buses and lines are energized, Qg is fixed at baseline, and the optimizer only controls selected real-power generator redispatch and selected load-service fractions.

## Relation To Full Formulation

The full formulation includes Pg, Qg, alpha, bus energization, and line energization. This first pass keeps Qg, y, and z fixed and tests whether the continuous reduced controls can lower grouped wildfire exposure.

The optimized scalar objective is `lambda_R * (R_group / R_baseline) + lambda_L * L_shed_weighted`, where `L_shed_weighted = sum_n (Pd_n / sum_m Pd_m) * (1 - alpha_n)`. Generator movement is recorded as a diagnostic, not used as a cost term.

`R_group` uses `z_l * p_env_l * loading_l^2 * I_l(u)` summed over configured line groups. `z_l` is fixed at 1, and `I_l(u)` is the relative equal-weight served-load loss from a counterfactual one-line GridFM outage.

## GridFM Surrogate Use

GridFM is loaded once in memory and called through `NeuralSolverWrapper`; the CLI is not called inside optimization iterations.

## Grouped Wildfire Scenario

A synthetic high-risk corridor is represented as a `WildfireLineGroup`; risk is summed at line and group levels with counterfactual `impact` values written to the per-line CSVs.

## Basic Run

Latest run directory: `C:\Users\Caleb Lu\OneDrive\Documents\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\experiments\test\wildfire_tests\results\leq\stage_a\connected_corridor\bal\gnn\bal_gnn_20260611_212645`

## Stability Sweep

Implemented as `run_stability_sweep.py`; run results are only reported here after execution.

## Validation Checks

Baseline prediction validity, perturbation response, decision bounds, finite objective components, final prediction validity, alpha preservation, and risk/objective reduction are checked.

## Tests Added

Unit and smoke tests are intended under `tests/test_wildfire*.py`.

## Commands Run

- `python -m experiments.test.wildfire_tests.stage_a_first_pass.run_basic_case --config C:\Users\Caleb Lu\OneDrive\Documents\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\experiments\test\wildfire_tests\results\leq\generated_configs\stage_a\connected_corridor\bal_gnn.yaml`

## Final Results

- baseline prediction worked: True
- perturbation produced a nonzero response: True
- optimization reduced total objective: False
- optimization reduced grouped wildfire risk: False
- final mean alpha: 1.0
- final min alpha: 1.0
- max absolute Delta_Pg: 0.0
- optimizer success/failure status: False (ABNORMAL: )
- number of objective evaluations: 144
- GNN and GPS both ran if tested: not tested
- stability sweep summary if run: not run
- validation failures: ['objective_reduced', 'wildfire_risk_reduced', 'optimizer_success']
- optimization behavior plot: C:\Users\Caleb Lu\OneDrive\Documents\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\experiments\test\wildfire_tests\results\leq\stage_a\connected_corridor\bal\gnn\bal_gnn_20260611_212645\figures\optimization_behavior.png
- topology change plot: C:\Users\Caleb Lu\OneDrive\Documents\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\experiments\test\wildfire_tests\results\leq\stage_a\connected_corridor\bal\gnn\bal_gnn_20260611_212645\figures\ieee30_network_changes.png

## Commands to Reproduce

```powershell
python -m experiments.test.wildfire_tests.stage_a_first_pass.run_basic_case --config experiments/test/wildfire_tests/stage_a_first_pass/configs/basic_gps.yaml
python -m experiments.test.wildfire_tests.stage_a_first_pass.run_basic_case --config experiments/test/wildfire_tests/stage_a_first_pass/configs/basic_gnn.yaml
python -m experiments.test.wildfire_tests.stage_a_first_pass.run_stability_sweep --config experiments/test/wildfire_tests/stage_a_first_pass/configs/stability_sweep.yaml
python -m experiments.test.wildfire_tests.shared.plot_optimization_behavior --run-dir experiments/test/wildfire_tests/results/<run_name>
python -m experiments.test.wildfire_tests.shared.plot_network_changes --run-dir experiments/test/wildfire_tests/results/<run_name>
$files = Get-ChildItem tests -Filter 'test_wildfire*.py' | ForEach-Object { $_.FullName }; pytest $files -q
```

## Limitations And Next Steps

This is not a full wildfire-resilience-aware OPF. Next steps are to calibrate risk inputs, improve feasibility diagnostics, compare against a physical solver, and decide whether the reduced formulation should move toward topology controls.
