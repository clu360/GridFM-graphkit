# Implementation Summary: Wildfire-Aware Dispatch Optimization

## Executive Summary

The `experiments/test` package has been refactored into a cleaner wildfire-aware optimization framework while keeping the existing decision variables unchanged.

Current design:

- decision vector remains `u = [u_Pg ; u_delta]`
- objective now uses:
  - generator deviation cost
  - linear load shedding cost
  - wildfire branch-risk cost
- thermal overload is no longer part of the optimization objective

This is now a clearer framework for wildfire-aware surrogate dispatch experiments on IEEE-30.

## What Was Implemented

### 1. Wildfire penalty module

Added:

- `experiments/test/wildfire_penalty.py`

This module:

- reuses the branch loading logic from `overload_penalty.py`
- computes wildfire risk with:
  - branch weights `w_l`
  - branch thresholds `eta_l`
- exposes debugging details:
  - `branch_loading`
  - `branch_risk_terms`
  - `active_risk_mask`

Default wildfire settings:

- `w_l = 1.0`
- `eta_l = 0.7`

### 2. Optimizer refactor

Updated:

- `experiments/test/optimization.py`

The current objective is:

- `J(u) = lambda_gen * sum((u_Pg - Pg_base)^2) + lambda_shed * sum(u_delta) + lambda_wf * wildfire_penalty(u)`

Default weights:

- `lambda_gen = 1.0`
- `lambda_shed = 50.0`
- `lambda_wf = 10.0`

The optimizer still keeps the same constraints:

- generator bounds
- shedding bounds

### 3. Framework cleanup

Added:

- `experiments/test/pipeline_utils.py`

This is the main structural cleanup.

It now centralizes:

- repo-root resolution
- config loading
- test datamodule setup
- batch loading
- single-scenario extraction
- checkpointed GNN loading
- checkpointed GPS loading

This removed repeated boilerplate across the real-data scripts.

### 4. Real-data scripts cleaned up

Updated:

- `example_optimization.py`
- `example_optimization_with_shedding.py`
- `test_pipeline_ieee30.py`
- `test_gnn_vs_gps_shedding.py`

These scripts now:

- use the shared setup helpers
- use the wildfire evaluator
- use the new optimizer weights
- report wildfire cost instead of overload penalty

### 5. Notebook updated

Updated:

- `ieee30_optimization_validation.ipynb`

The notebook now:

- matches the corrected single-scenario extraction path
- uses the wildfire-aware objective
- executes without errors

## Current Framework Structure

### Core data and decisions

- `scenario_data.py`
- `pv_dispatch.py`
- `load_shedding_spec.py`
- `extended_dispatch_spec.py`

### Surrogate and risk evaluation

- `neural_solver.py`
- `overload_penalty.py`
- `wildfire_penalty.py`

### Optimization and validation

- `optimization.py`
- `validation.py`

### Shared experiment setup

- `pipeline_utils.py`

### Examples and tests

- `example_optimization.py`
- `example_optimization_with_shedding.py`
- `test_pipeline.py`
- `test_pipeline_ieee30.py`
- `test_gnn_vs_gps_shedding.py`
- `ieee30_optimization_validation.ipynb`

## Current Verified Results

Verified locally on March 11, 2026:

- `python experiments/test/test_pipeline.py`
- `python experiments/test/test_pipeline_ieee30.py`
- `python experiments/test/test_gnn_vs_gps_shedding.py`
- `python experiments/test/example_optimization.py`
- `python experiments/test/example_optimization_with_shedding.py`
- notebook execution via `nbclient`

Observed behavior:

- the framework runs end to end
- the decision dimension for the extracted IEEE-30 graph remains 29
- the optimizer still often stays at the baseline with 0 iterations
- GNN remains much more stable than GPS under the wildfire objective

Representative current behavior:

- physical baseline wildfire cost can be zero while surrogate wildfire cost is nonzero
- this indicates that the remaining bottleneck is still surrogate fidelity, not pipeline wiring

## What Is Cleaner Now

The framework is cleaner in three main ways:

1. Shared setup is no longer duplicated across every script.
2. The optimization objective is now explicit and aligned with the wildfire use case.
3. The examples, tests, and notebook all use the same conceptual pipeline.

## Remaining Limitations

- the surrogate still fails some physical reasonableness checks
- the optimizer still frequently finds no improving step
- GPS remains unstable on the extended decision space
- branch wildfire weights are still uniform defaults
- no true PF solve is in the optimization loop

## Recommended Next Steps

1. Add real branch wildfire exposure weights.
2. Add case-specific thermal ratings and generator limits where available.
3. Benchmark surrogate wildfire cost against PF-derived branch flows.
4. Consider alternative optimizers if baseline sticking persists.
5. Add uncertainty-aware safeguards before interpreting optimized dispatches operationally.

## Bottom Line

`experiments/test` is now a clearer and more maintainable wildfire-aware research framework.

The software structure is in good shape. The main remaining issue is model fidelity: the surrogate predictions are still the limiting factor for meaningful optimization improvement.
