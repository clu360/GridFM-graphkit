# Improved Optimization

This package contains the improved wildfire-aware surrogate OPF workflow for the IEEE-30 experiment path.

## Purpose

The improved workflow keeps the surrogate-based OPF framing but replaces broad global controls with:

- targeted generator redispatch on selected PV buses
- targeted load shedding on selected PQ buses
- smooth wildfire-aware line-risk penalties
- Pareto and sensitivity studies built around the same reproducible pipeline

## File overview

- `config.py`: central experiment defaults
- `selection.py`: wildfire-sensitive line detection and targeted control screening
- `wildfire_metrics.py`: branch loading and wildfire penalty utilities
- `objective.py`: objective decomposition helpers
- `optimization_problem.py`: targeted SciPy optimization wrapper
- `validation.py`: safety and consistency checks
- `reporting.py`: CSV / JSON / markdown outputs
- `plots.py`: matplotlib figure generation
- `run_improved_optimization.py`: single-run entry point
- `run_pareto_sweep.py`: Pareto tradeoff sweep
- `run_objective_sensitivity.py`: weight / threshold / landscape diagnostics

## How to run

From the repo root:

```powershell
python experiments/test/improved_optimization/run_improved_optimization.py gnn
python experiments/test/improved_optimization/run_improved_optimization.py gps
python experiments/test/improved_optimization/run_pareto_sweep.py gnn
python experiments/test/improved_optimization/run_pareto_sweep.py gps
python experiments/test/improved_optimization/run_objective_sensitivity.py gnn
python experiments/test/improved_optimization/run_objective_sensitivity.py gps
```

## Outputs

Outputs are written under:

- `experiments/test/improved_optimization/results/improved_runs/`
- `experiments/test/improved_optimization/results/pareto/`
- `experiments/test/improved_optimization/results/sensitivity/`

Each run writes CSV summaries, a JSON run summary, and figures where applicable.

## Assumptions

- the improved workflow reuses the legacy homogeneous IEEE-30 data path
- GNN and GPS checkpoints are loaded through `experiments.test.pipeline_utils`
- branch loading is reconstructed from surrogate-predicted voltages using the existing simplified admittance model
