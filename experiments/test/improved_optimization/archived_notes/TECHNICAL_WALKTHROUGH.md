# Technical Walkthrough: Wildfire-Aware IEEE-30 Surrogate Dispatch

## Purpose

This document is a step-by-step technical walkthrough of the current
`experiments/test` workflow. It is intended to explain:

- what components were used
- how the experiment is assembled
- what results were observed
- what conclusions can be defended from those results
- what the next technical steps should be

The current workflow is a restored IEEE-30 surrogate optimization prototype.
It is not a full AC-OPF implementation.

## High-Level Goal

The project goal in `experiments/test` is to evaluate whether pretrained GridFM
surrogate models can support a wildfire-aware dispatch optimization problem on
the IEEE-30 case.

The intended control actions are:

- generator redispatch at controllable PV buses
- load shedding at PQ buses

The intended optimization goal is:

- reduce wildfire-related branch loading risk
- while penalizing excessive generator movement
- and penalizing excessive load shedding

## Current Experimental Setting

The current restored workflow uses:

- configuration:
  - `tests/config/gridFMv0.1_dummy.yaml`
- dataset:
  - `tests/data/case30_ieee`
- loader path:
  - `gridfm_graphkit.datasets.powergrid_datamodule.LitGridDataModule`
- pretrained checkpoints:
  - `examples/models/GridFM_v0_1.pth`
  - `examples/models/GridFM_v0_2.pth`

This is the legacy homogeneous IEEE-30 path that existed before the sync drift.

## End-to-End Workflow

The workflow executed by the scripts and notebook is:

1. Load the IEEE-30 test configuration.
2. Build the datamodule and access the test dataloader.
3. Pull one batched test object.
4. Slice out one graph from the batch using `scenario_idx=0`.
5. Convert that graph into a canonical `ScenarioData` object.
6. Build decision variables for either:
   - PV redispatch only, or
   - PV redispatch plus load shedding
7. Load a pretrained GNN or GPS surrogate.
8. Inject the decision vector into the scenario features.
9. Use the surrogate to predict bus state variables.
10. Convert predicted voltages into branch loading.
11. Convert branch loading into wildfire risk.
12. Evaluate the objective function.
13. Run bounded optimization with `L-BFGS-B`.
14. Compare baseline and optimized objective values.

## Main Technical Components

### 1. Scenario extraction

File:

- `experiments/test/scenario_data.py`

Purpose:

- represent a single power-system scenario in a consistent in-memory structure

What it stores:

- baseline demand and generation
- baseline voltage magnitude and angle
- bus-type masks:
  - PQ
  - PV
  - REF
- graph topology
- branch parameters used for loading calculations
- positional encodings
- masking information
- provisional generator bounds

Important current behavior:

- `extract_scenario_from_batch(...)` now correctly slices one graph from the
  batch using `batch.ptr`
- this fixed the earlier accidental merged 60-bus issue

Current extracted single scenario:

- 30 buses
- 24 PQ buses
- 5 PV buses
- 1 REF bus

### 2. Decision-variable definitions

Files:

- `experiments/test/pv_dispatch.py`
- `experiments/test/load_shedding_spec.py`
- `experiments/test/extended_dispatch_spec.py`

Purpose:

- define what the optimizer is allowed to change

Phase 1 decision vector:

- `u = u_Pg`
- one active-power redispatch variable for each controllable PV bus

Extended decision vector:

- `u = [u_Pg ; u_delta]`

where:

- `u_Pg` = PV-bus active generation adjustments
- `u_delta` = load shedding at PQ buses in MW

Current IEEE-30 dimensions:

- 5 PV generation variables
- 24 shedding variables
- 29 total variables in the extended case

Constraints:

- `Pg_min <= u_Pg <= Pg_max`
- `0 <= u_delta <= Pd`

### 3. Shared experiment setup

File:

- `experiments/test/pipeline_utils.py`

Purpose:

- remove repeated setup code across scripts and tests

Responsibilities:

- resolve repo root
- load the IEEE-30 YAML config
- create the legacy homogeneous datamodule
- load the first test batch
- extract one scenario by `scenario_idx`
- resolve checkpoint paths
- load the pretrained GNN or GPS model

This file is now the main orchestration utility for the `experiments/test`
workflow.

### 4. Neural surrogate wrapper

File:

- `experiments/test/neural_solver.py`

Purpose:

- provide a common interface to both GNN and GPS models

Responsibilities:

- take the decision vector
- update the scenario input features accordingly
- apply masking
- run the model in evaluation mode
- return denormalized predictions for:
  - `Pd`
  - `Qd`
  - `Pg`
  - `Qg`
  - `Vm`
  - `Va`

This is the core surrogate layer that turns control actions into predicted
system states.

### 5. Branch-loading computation

File:

- `experiments/test/overload_penalty.py`

Purpose:

- compute branch loading from predicted bus voltages

Important note:

- this file still exists and still matters
- but only as a branch-loading postprocessing utility
- overload is no longer a separate objective term

Responsibilities:

- build simplified branch admittance matrices
- compute branch currents from predicted `Vm` and `Va`
- convert currents into normalized branch loading

This is the physical proxy layer underneath both the old overload metric and
the new wildfire metric.

### 6. Wildfire-risk evaluator

File:

- `experiments/test/wildfire_penalty.py`

Purpose:

- replace the overload-based objective term with a wildfire-aware loading term

Current wildfire penalty:

- `branch_term_l = w_l * max(0, loading_l - eta_l)^2`
- `wildfire_penalty = sum(branch_term_l)`

Default settings:

- `w_l = 1.0`
- `eta_l = 0.7`

Returned diagnostics include:

- `branch_loading`
- `branch_risk_terms`
- `active_risk_mask`
- `n_active_risk_branches`
- `max_loading`

Interpretation:

- no wildfire penalty below 70% loading
- risk grows quadratically above the threshold

### 7. Optimization problem

File:

- `experiments/test/optimization.py`

Purpose:

- define the current surrogate optimization problem

Current objective:

- `J(u) = lambda_gen * sum((u_Pg - Pg_base)^2) + lambda_shed * sum(u_delta) + lambda_wf * wildfire_penalty(u)`

Default weights:

- `lambda_gen = 1.0`
- `lambda_shed = 50.0`
- `lambda_wf = 10.0`

Method:

- bounded optimization via `scipy.optimize.minimize`
- solver method:
  - `L-BFGS-B`

The optimizer sees only the surrogate-based objective. There is no true AC
power flow solve inside the loop.

### 8. Validation harness

File:

- `experiments/test/validation.py`

Purpose:

- verify the pipeline before running optimization

Checks include:

- scenario structure
- decision-vector consistency
- solver prediction success
- whether predicted values are finite
- whether voltage magnitudes remain in a reasonable range
- whether the risk evaluator returns coherent outputs

This file helps distinguish:

- wiring bugs
- versus surrogate-quality problems

### 9. Scripts and notebook used to produce results

Files:

- `experiments/test/test_pipeline.py`
- `experiments/test/test_pipeline_ieee30.py`
- `experiments/test/example_optimization.py`
- `experiments/test/example_optimization_with_shedding.py`
- `experiments/test/test_gnn_vs_gps_shedding.py`
- `experiments/test/ieee30_optimization_validation.ipynb`

Roles:

- `test_pipeline.py`
  - synthetic smoke test of the module wiring

- `test_pipeline_ieee30.py`
  - real-data smoke test on the restored IEEE-30 path

- `example_optimization.py`
  - minimal PV-only optimization run with the wildfire objective

- `example_optimization_with_shedding.py`
  - extended dispatch run with shedding enabled

- `test_gnn_vs_gps_shedding.py`
  - side-by-side comparison of GNN and GPS on the extended decision space

- `ieee30_optimization_validation.ipynb`
  - presentation notebook for the same workflow

## Current Results

### What is working

The restored workflow is now technically coherent again:

- IEEE-30 config is back
- the single-scenario extraction is correct
- the current decision variables are implemented
- the wildfire-only objective is implemented
- GNN and GPS checkpoints can be loaded again
- the example/test scripts run end to end

### What is being observed

Typical observed behavior on the current restored IEEE-30 path:

- physical baseline wildfire cost is near zero
- surrogate wildfire cost is very large
- validation does not fully pass
- optimization stops immediately at the baseline
- objective improvement is often 0%

Representative current behavior:

- GNN baseline objective remains very large
- GPS baseline objective is usually much larger than GNN
- GPS is materially less stable than GNN on the extended dispatch space

### What this means

The pipeline itself is no longer the main issue.

The dominant issue is now surrogate behavior:

- the predicted voltages are often not physically reasonable
- those voltages produce exaggerated branch loading
- that drives an exaggerated wildfire objective
- the optimizer then sees a poor or unusable search landscape

## Primary Technical Limitations

### 1. Surrogate predictions are not reliable enough

This is the central limitation.

Evidence:

- voltage predictions fail reasonableness checks
- physical baseline risk and surrogate baseline risk disagree sharply
- GPS in particular produces extreme branch-loading values

Implication:

- the surrogate is not currently producing a trustworthy optimization signal

### 2. No meaningful descent direction is being exposed

Observed behavior:

- `L-BFGS-B` often stops with 0 iterations
- objective improvement remains 0%

Interpretation:

- even if the optimization wrapper is functioning correctly, the surrogate is
  not exposing a useful local direction for improving the objective

### 3. The wildfire metric is still a surrogate-derived proxy

Current wildfire cost depends on:

- predicted `Vm`
- predicted `Va`
- simplified branch loading reconstruction

There is no true PF solve in the loop.

Implication:

- a bad surrogate state can dominate the objective even if the true system
  would not behave that way

### 4. GPS is especially unstable

Observed behavior:

- GPS objectives are typically far larger than GNN objectives
- GPS branch loading values are more extreme

Implication:

- the extended dispatch setting appears to be particularly difficult for GPS in
  the current restored workflow

### 5. The restored workflow is compatible, but not guaranteed identical to the old environment

Important nuance:

- the same checkpoint files are being used
- but the full repo state had to be restored after sync drift

Implication:

- exact old notebook numbers are not guaranteed
- some residual compatibility risk still exists in the restored inference path

## Current Conclusion

The strongest defensible conclusion is:

- the wildfire-aware optimization framework is now implemented and runnable
- the current bottleneck is not the optimization wrapper
- the current bottleneck is surrogate fidelity and downstream suitability

More specifically:

- the pretrained surrogate models, as currently executed on the restored
  IEEE-30 path, are not providing a strong enough optimization signal to
  improve the wildfire objective

That means:

- the framework can be presented as a working prototype
- but not yet as a successful optimization result

## Recommended Next Steps

### Priority 1. Verify surrogate compatibility against the older known-good run

Best next action:

- compare current outputs against the earlier notebook outputs cell by cell
- identify where the restored path first diverges numerically

Why:

- this distinguishes compatibility drift from true model insufficiency

### Priority 2. Benchmark the surrogate against a true PF reference

Add a validation step that compares:

- surrogate-predicted branch loading
- PF-derived branch loading

Why:

- this will directly test whether the surrogate risk signal is trustworthy

### Priority 3. Improve the model for downstream use

If compatibility is confirmed and outputs are still poor, then the likely next
research step is:

- fine-tuning or retraining for this downstream optimization setting

Why:

- the current pretrained models were not trained specifically for wildfire-aware
  dispatch optimization

### Priority 4. Add stronger physics checks in the loop

Possible upgrades:

- feasibility filter
- PF re-evaluation step
- physics-informed postprocessing

Why:

- this can prevent clearly nonphysical surrogate states from dominating the
  objective

### Priority 5. Only then revisit optimizer choice

Current behavior does not yet justify changing the optimizer first.

Why:

- when the surrogate signal is poor, a different optimizer is unlikely to solve
  the core problem by itself

## Presentation Guidance

If you present this work now, the most accurate framing is:

- a restored IEEE-30 wildfire-aware surrogate optimization pipeline
- using pretrained GridFM GNN and GPS models
- with PV redispatch and optional load shedding
- where the implementation is working, but the optimization remains limited by
  surrogate fidelity

That framing is technically accurate and defensible.

## Bottom Line

You have successfully assembled and restored the full experimental pipeline:

- IEEE-30 data loading
- scenario extraction
- decision-variable construction
- pretrained surrogate inference
- wildfire-risk postprocessing
- bounded surrogate optimization
- GNN/GPS comparison

The current limitation is not missing code. It is that the surrogate does not
yet provide a sufficiently trustworthy signal for objective improvement in this
downstream wildfire-dispatch task.
