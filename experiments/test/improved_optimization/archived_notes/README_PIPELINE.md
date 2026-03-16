# Wildfire-Aware IEEE-30 Workflow

## Overview

`experiments/test` is the restored IEEE-30 research sandbox built on top of the
legacy homogeneous GridFM data path. It is not a standalone solver package.
It depends on:

- `tests/config/gridFMv0.1_dummy.yaml`
- `tests/data/case30_ieee`
- legacy homogeneous dataset/model compatibility files under `gridfm_graphkit`
- pretrained checkpoints in `examples/models`

The active workflow is:

1. Load the restored IEEE-30 test config.
2. Pull one graph from the batched test dataloader.
3. Extract a single scenario with `scenario_idx`.
4. Build a dispatch decision vector.
5. Run a pretrained GNN or GPS surrogate.
6. Convert predicted voltages into branch loading.
7. Optimize a wildfire-aware objective.

This remains a surrogate predict-then-optimize prototype, not an AC-OPF solver.

## Current Data and Scenario Path

The real-data scripts in this folder now use:

- config: `tests/config/gridFMv0.1_dummy.yaml`
- dataset: `tests/data/case30_ieee`
- datamodule: `gridfm_graphkit.datasets.powergrid_datamodule.LitGridDataModule`

On the current restored path, `scenario_idx=0` extracts:

- 30 buses
- 24 PQ buses
- 5 PV buses
- 1 REF bus

## Decision Variables

Two decision specifications are used:

- `PVDispatchDecisionSpec`
  - Phase 1 only
  - decision vector `u = u_Pg`
  - one active-power redispatch variable per controllable PV bus

- `ExtendedDispatchSpec`
  - extended wildfire experiment
  - decision vector `u = [u_Pg ; u_delta]`
  - `u_delta` is PQ-bus load shedding in MW

For the current extracted IEEE-30 graph:

- 5 PV redispatch variables
- 24 shedding variables
- 29 total variables in the extended case

## Objective

The active optimization objective in `optimization.py` is:

`J(u) = lambda_gen * sum((u_Pg - Pg_base)^2) + lambda_shed * sum(u_delta) + lambda_wf * wildfire_penalty(u)`

Defaults:

- `lambda_gen = 1.0`
- `lambda_shed = 50.0`
- `lambda_wf = 10.0`

Line loading enters the objective only through the wildfire penalty.
`overload_penalty.py` is still present for legacy postprocessing, but it is not
part of the current objective.

## Wildfire Penalty

`wildfire_penalty.py` reuses the branch-loading logic from
`overload_penalty.py` and applies:

- `branch_term_l = w_l * max(0, loading_l - eta_l)^2`
- `wildfire_penalty = sum(branch_term_l)`

Defaults:

- `w_l = 1.0`
- `eta_l = 0.7`

Returned diagnostics include:

- `branch_loading`
- `branch_risk_terms`
- `active_risk_mask`
- `n_active_risk_branches`
- `max_loading`

## File Roles

Core workflow files:

- `scenario_data.py`
  - canonical single-scenario representation
  - slices one graph from a PyG batch with `batch.ptr`
  - currently interprets the restored legacy 9-feature homogeneous node layout

- `pv_dispatch.py`
  - PV-only decision spec

- `load_shedding_spec.py`
  - PQ shedding spec in MW

- `extended_dispatch_spec.py`
  - combined PV redispatch + shedding spec

- `neural_solver.py`
  - wraps pretrained GNN or GPS checkpoints for surrogate inference

- `wildfire_penalty.py`
  - current line-risk objective module

- `optimization.py`
  - wildfire-only objective and `L-BFGS-B` wrapper

- `validation.py`
  - scenario, solver, and risk-evaluator checks

- `pipeline_utils.py`
  - shared setup helpers for config loading, datamodule setup, scenario
    extraction, and checkpoint resolution

Examples and checks:

- `example_optimization.py`
  - minimal PV-only IEEE-30 run

- `example_optimization_with_shedding.py`
  - extended dispatch run with three weight settings

- `test_pipeline.py`
  - synthetic smoke test

- `test_pipeline_ieee30.py`
  - real-data smoke test on the restored IEEE-30 path

- `test_gnn_vs_gps_shedding.py`
  - compares GNN and GPS on the extended dispatch baseline

- `ieee30_optimization_validation.ipynb`
  - presentation notebook for the same workflow

## Current Verified State

As of March 11, 2026, the restored workflow is consistent enough to run:

- `python experiments/test/test_pipeline.py`
- `python experiments/test/test_pipeline_ieee30.py`
- `python experiments/test/example_optimization.py`
- `python experiments/test/example_optimization_with_shedding.py`
- `python experiments/test/test_gnn_vs_gps_shedding.py`

The notebook source is aligned with the same restored path. In this shell
environment, automated `nbclient` execution is currently blocked by a local
Windows Jupyter runtime ACL issue rather than by notebook code drift.

## What The Current Results Mean

The framework is operational, but the surrogate remains the limiting factor.
Typical current behavior is:

- physical baseline wildfire cost is small or zero
- surrogate wildfire cost can still be very large
- validation may fail voltage reasonableness checks
- optimization often stops at the baseline with 0 iterations
- GPS is usually less stable than GNN on the extended decision space

So the main issue is no longer pipeline wiring. It is the quality and
compatibility of surrogate predictions under this restored workflow.

## Important Limitations

- this is a restored legacy IEEE-30 path, not the newer hetero dataset path
- branch ratings and admittance processing remain simplified in the risk model
- no explicit AC feasibility solve is inside the optimization loop
- wildfire weights are still uniform defaults
- exact pre-sync numbers are not guaranteed even though the same checkpoints are
  being loaded

## Recommended Next Steps

1. Compare current surrogate outputs against the known pre-sync notebook output
   cell by cell.
2. Add branch-specific wildfire weights if external exposure data is available.
3. Benchmark surrogate branch loading against a true PF calculation.
4. Revisit optimizer choice only after surrogate behavior is trusted again.
