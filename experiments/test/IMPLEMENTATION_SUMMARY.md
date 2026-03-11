# Implementation Summary

## Current Situation

`experiments/test` now reflects the restored IEEE-30 wildfire-dispatch workflow
that is currently runnable in this repo.

That means:

- the experiment is back on `case30_ieee`
- the loader path is the legacy homogeneous path
- the active objective is wildfire-only
- the extended decision vector is still `u = [u_Pg ; u_delta]`
- both pretrained checkpoints can be used again when present:
  - `GridFM_v0_1.pth` for GNN
  - `GridFM_v0_2.pth` for GPS

## What Has Been Restored

The synced repo had drifted away from the original IEEE-30 experiment path. The
current `experiments/test` layer assumes the restored legacy setup:

- config: `tests/config/gridFMv0.1_dummy.yaml`
- dataset: `tests/data/case30_ieee`
- datamodule: `LitGridDataModule`
- single-scenario extraction by `scenario_idx`

This restored the intended real-data experiment structure used by:

- `example_optimization.py`
- `example_optimization_with_shedding.py`
- `test_pipeline_ieee30.py`
- `test_gnn_vs_gps_shedding.py`
- `ieee30_optimization_validation.ipynb`

## Active Objective

The optimization objective in `optimization.py` is:

`J(u) = lambda_gen * sum((u_Pg - Pg_base)^2) + lambda_shed * sum(u_delta) + lambda_wf * wildfire_penalty(u)`

with defaults:

- `lambda_gen = 1.0`
- `lambda_shed = 50.0`
- `lambda_wf = 10.0`

Thermal overload is no longer a separate objective term.

## Decision Variables

The current decision-variable setup is:

- `u_Pg`
  - active-power redispatch at controllable PV buses

- `u_delta`
  - load shedding at PQ buses in MW

For the current extracted IEEE-30 graph:

- 5 PV variables
- 24 shedding variables
- 29 total extended variables

## Wildfire Risk Term

`wildfire_penalty.py` reuses branch loading from `overload_penalty.py` and
applies:

- `branch_term_l = w_l * max(0, loading_l - eta_l)^2`
- `wildfire_penalty = sum(branch_term_l)`

Current defaults:

- `w_l = 1.0`
- `eta_l = 0.7`

The objective details now expose:

- `total_cost`
- `generator_deviation_cost`
- `load_shedding_cost`
- `wildfire_cost`
- `branch_loading`
- `branch_risk_terms`
- `active_risk_mask`

## File Cleanup Status

The main files now represent the current situation as follows:

- `scenario_data.py`
  - extracts a single graph correctly from the batched IEEE-30 loader
  - matches the restored legacy homogeneous feature layout

- `load_shedding_spec.py`
  - documents shedding as MW, not normalized fraction

- `extended_dispatch_spec.py`
  - documents the current 29-variable IEEE-30 extended case correctly

- `pipeline_utils.py`
  - centralizes config loading, datamodule setup, scenario extraction, and
    checkpoint-path resolution

- `validation.py`
  - treats wildfire evaluation as the primary path
  - still tolerates the legacy overload evaluator for compatibility

- `__init__.py`
  - now describes `experiments/test` as the restored IEEE-30 wildfire sandbox
  - keeps overload utilities available only as legacy helpers

- `README_PIPELINE.md`
  - now describes the restored legacy IEEE-30 path accurately

## Current Verified Behavior

The current workflow is runnable end to end.

Verified scripts:

- `python experiments/test/test_pipeline.py`
- `python experiments/test/test_pipeline_ieee30.py`
- `python experiments/test/example_optimization.py`
- `python experiments/test/example_optimization_with_shedding.py`
- `python experiments/test/test_gnn_vs_gps_shedding.py`

Current notebook:

- `experiments/test/ieee30_optimization_validation.ipynb`

The notebook source has been updated to match the shared Python helpers,
including legacy checkpoint loading with `weights_only=False` for the restored
checkpoint files. Automated execution from this shell is currently blocked by a
local Windows Jupyter runtime permission issue, so the notebook should be
rerun from your normal VS Code/Jupyter session for fresh outputs.

## What Is Still Not Fixed

The framework is cleaner and consistent, but the optimization results are still
limited by surrogate behavior.

Current symptoms:

- surrogate voltage predictions can be physically unreasonable
- surrogate wildfire cost can be much larger than the physical baseline cost
- optimization often returns 0 iterations and 0 percent objective improvement
- GPS is usually less stable than GNN on the extended dispatch setup

So the remaining bottleneck is not the file structure in `experiments/test`.
It is the reliability of the restored surrogate inference path.

## Bottom Line

`experiments/test` now matches the current restored IEEE-30 wildfire-dispatch
workflow in this repo.

The code and summaries now consistently describe:

- the restored data/config path
- the current wildfire-only objective
- the current decision variables
- the current experimental limitation: surrogate fidelity rather than missing
  pipeline wiring
