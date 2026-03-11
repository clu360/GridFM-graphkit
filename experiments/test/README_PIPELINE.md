# Wildfire-Aware Dispatch Pipeline for GridFM

## Overview

`experiments/test` now implements a wildfire-aware surrogate dispatch workflow for the IEEE-30 test case.

The pipeline:

1. Loads one graph from the batched IEEE-30 test loader.
2. Builds a decision vector over generator redispatch and load shedding.
3. Uses a pretrained GridFM model to predict bus states.
4. Computes branch loading from predicted voltages.
5. Optimizes a wildfire-aware objective over the existing dispatch variables.

This is still a surrogate optimization workflow, not a full AC-OPF solver.

## Decision Variables

The extended decision vector is unchanged:

`u = [u_Pg ; u_delta]`

where:

- `u_Pg` = active generation at controllable PV buses
- `u_delta` = load shedding at PQ buses, in MW

For the current extracted IEEE-30 graph:

- 5 PV generation variables
- 24 shedding variables
- 29 total decision variables

## Current Objective

The optimizer now uses:

`J(u) = lambda_gen * sum((u_Pg - Pg_base)^2) + lambda_shed * sum(u_delta) + lambda_wf * wildfire_penalty(u)`

Default weights:

- `lambda_gen = 1.0`
- `lambda_shed = 50.0`
- `lambda_wf = 10.0`

## Wildfire Penalty

Wildfire risk is computed from branch loading.

For each branch `l`:

- `loading_l = branch_loading_l`
- `branch_term_l = w_l * max(0, loading_l - eta_l)^2`

Total wildfire penalty:

- `wildfire_penalty = sum(branch_term_l)`

Default wildfire parameters:

- `w_l = 1.0`
- `eta_l = 0.7`

This means:

- no wildfire penalty below 70% loading
- quadratic growth above the threshold
- easy extension to branch-specific wildfire weights later

## Core Modules

### `scenario_data.py`

Canonical single-scenario representation.

Responsibilities:

- stores baseline bus variables and masks
- stores graph structure, edge parameters, positional encodings, and feature masks
- exposes provisional generator bounds
- slices one graph from a PyG batch using `batch.ptr`

### `pv_dispatch.py`

PV-only decision specification for Phase 1.

### `load_shedding_spec.py`

PQ-bus load shedding specification using MW shed.

### `extended_dispatch_spec.py`

Combined decision specification for:

- PV redispatch
- PQ shedding

### `neural_solver.py`

Unified GNN/GPS wrapper for:

- masked input construction
- inference
- denormalized predicted bus states

### `wildfire_penalty.py`

Wildfire-aware branch risk module.

Responsibilities:

- reuses branch loading logic from `overload_penalty.py`
- computes wildfire cost from thresholded branch loading
- returns debugging details including:
  - `branch_loading`
  - `branch_risk_terms`
  - `active_risk_mask`

### `optimization.py`

Wildfire-aware optimization wrapper around `scipy.optimize.minimize`.

Responsibilities:

- enforces generator and shedding bounds
- evaluates generator deviation cost
- evaluates shedding cost
- evaluates wildfire cost from surrogate predictions
- tracks optimization history

### `pipeline_utils.py`

Shared framework helpers for the real IEEE-30 workflow.

Responsibilities:

- load config and datamodule
- load the first test batch
- extract one scenario by `scenario_idx`
- load checkpointed GNN or GPS models

This is the main cleanup layer that removes repetitive setup code from the examples and tests.

### `validation.py`

Validation harness for:

- scenario structure
- decision specs
- solver behavior
- branch-risk evaluator behavior

## Entry Points

### `example_optimization.py`

Minimal PV-only wildfire-dispatch example.

### `example_optimization_with_shedding.py`

Extended wildfire-dispatch example with:

- PV-only case
- PV + shedding case
- lower shedding penalty case

### `test_pipeline.py`

Synthetic smoke test for the core module wiring.

### `test_pipeline_ieee30.py`

Real-data smoke test for one extracted IEEE-30 graph.

### `test_gnn_vs_gps_shedding.py`

Real-data comparison of GNN vs GPS under the wildfire objective.

### `ieee30_optimization_validation.ipynb`

Presentation notebook for the same workflow.

## Current Verified Behavior

As currently verified:

- single-scenario extraction works correctly
- all examples and tests under `experiments/test` run
- the notebook executes without errors
- the optimizer still tends to stop at the baseline with 0 iterations on the tested scenarios
- GNN is materially more stable than GPS under the extended wildfire objective

## Known Limitations

- optimization quality is still limited by surrogate fidelity
- voltage predictions can still fail reasonableness checks
- GPS remains unstable on the extended decision space
- branch risk is based on surrogate voltages and simplified branch postprocessing
- there is still no explicit AC feasibility enforcement in the objective loop

## Recommended Next Steps

1. Add branch-specific wildfire weights from external fire exposure data.
2. Replace provisional branch and generator metadata with case-specific values where available.
3. Compare surrogate wildfire cost against a true PF-based branch flow calculation.
4. Test derivative-free methods if `L-BFGS-B` continues to stop at the baseline.
5. Add uncertainty-aware decision logic if the surrogate remains noisy.
