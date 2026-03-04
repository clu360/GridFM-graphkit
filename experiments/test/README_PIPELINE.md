# Predict-Then-Optimize Pipeline for GridFM

## Overview

This package implements a **predict-then-optimize pipeline** for power grid dispatch optimization using pretrained GridFM neural solvers (GNN_v0.1 or GPS_v0.2).

The pipeline enables **differentiable power grid optimization** by:
1. Using a pretrained neural surrogate model $\Phi_\theta$ to predict grid state
2. Formulating a constrained optimization problem in the decision variable space
3. Leveraging only generator bounds and surrogate predictions (no explicit AC-OPF constraints)

## Quick Start

### Basic Usage Example

```python
# 1. Load scenario
from experiments.test import *
scenario = extract_scenario_from_batch(batch, node_normalizer, edge_normalizer)

# 2. Define decision variables (PV-bus Pg)
decision_spec = PVDispatchDecisionSpec(scenario)

# 3. Load pretrained model and create solver wrapper
model_gnn = load_model(args)  # Load GNN_v0.1 or GPS_v0.2
solver = NeuralSolverWrapper(model_gnn, "gnn", scenario, decision_spec)

# 4. Create overload evaluator
overload_eval = OverloadPenaltyEvaluator(scenario)

# 5. Define and solve optimization problem
problem = DispatchOptimizationProblem(
    scenario, decision_spec, solver, overload_eval,
    alpha=1.0,      # Weight on baseline deviation
    lambda_=1.0,    # Weight on overload penalty
)
result = problem.optimize(method="L-BFGS-B", maxiter=100)

# 6. Analyze results
comparison = problem.compare_baseline_vs_optimized(result['u_opt'])
print(f"Objective improvement: {comparison['improvement']['objective_pct']:.2f}%")
```

## Module Reference

### 1. `scenario_data.py` – Canonical Scenario Representation

**Purpose**: Single in-memory representation of a power grid scenario with all fixed and variable components.

**Main Class**: `ScenarioData`

**Key Attributes**:
- Fixed scenario data:
  - `Pd_base, Qd_base` – demand (fixed)
  - `Pg_base, Qg_base, Vm_base, Va_base` – baseline state
  - `PQ_mask, PV_mask, REF_mask` – one-hot bus type indicators
  - `edge_index, G, B` – network topology and admittance
  - `pe` – positional encoding (pe_dim=20)
  - `mask` – node feature mask tensor

- Metadata:
  - `node_normalizer, edge_normalizer` – for denormalization
  - `Pg_min, Pg_max` – generator bounds per bus

**Functions**:
- `get_pv_buses()` – indices of PV buses
- `get_baseline_state()` – full bus state dict
- `to_device(device)` – move tensors to device
- `extract_scenario_from_batch()` – extract from PyTorch Geometric batch

### 2. `pv_dispatch.py` – PV-Bus Dispatch Decision Variables

**Purpose**: Map between optimization decision vector $u$ (Pg on PV buses) and full scenario representation.

**Main Class**: `PVDispatchDecisionSpec`

**Key Methods**:
- `u_to_Pg(u)` – expand decision vector to full Pg array
- `Pg_to_u(Pg)` – extract decision vector from full Pg
- `u_to_node_features(u)` – convert u to normalized node feature array
- `check_bounds(u)` – verify u satisfies bounds
- `get_distance_from_baseline(u)` – compute $\|u - u_{\text{base}}\|_2$

**Attributes**:
- `pv_bus_indices` – indices of PV buses
- `n_pv` – dimension of decision vector
- `u_base` – baseline Pg at PV buses
- `u_min, u_max` – bounds

### 3. `neural_solver.py` – Neural Solver Wrapper

**Purpose**: Unified interface to both GNN_v0.1 and GPS_v0.2 models with input preparation, inference, and denormalization.

**Main Class**: `NeuralSolverWrapper`

**Key Methods**:
- `predict_state(u)` – run inference on decision vector, return denormalized state
- `predict_normalized(u)` – return normalized predictions
- `predict_batch(u_batch)` – batch predictions
- `validate_baseline()` – compare predictions vs. baseline
- `prepare_input_tensors(u)` – internal: tensor preparation + masking

**Features**:
- ✓ Handles both GNN and GPS model types
- ✓ Applies PF masking during inference
- ✓ Normalizes inputs, denormalizes outputs
- ✓ pe_dim=20 for both models (no padding workaround)
- ✓ Single graph inference pipeline

### 4. `overload_penalty.py` – Line Loading & Overload Evaluation

**Purpose**: Compute branch loading, overload amounts, and total overload penalty from predicted bus state.

**Main Class**: `OverloadPenaltyEvaluator`

**Key Methods**:
- `compute_loading(Vm_pred, Va_pred)` – compute per-branch normalized loading
- `compute_overload_penalty(Vm_pred, Va_pred, penalty_type)` – total overload penalty
- `evaluate(Vm_pred, Va_pred)` – return overload metrics dict
- `evaluate_batch(Vm_pred_batch, Va_pred_batch)` – batch evaluation
- `evaluate_baseline()` – baseline overload

**Penalty Formulation**:

$$\rho_{\text{overload}}(\hat{v}) = \sum_{\ell} \max(0, \text{loading}_\ell - 1)^2$$

### 5. `optimization.py` – Dispatch Optimization Problem

**Purpose**: Define and solve the full optimization problem with objective, constraints, and optimization interface.

**Main Class**: `DispatchOptimizationProblem`

**Objective Function**:

$$J(u) = \alpha \|u - u_{\text{base}}\|_2^2 + \lambda \rho_{\text{overload}}(\Phi_\theta(u))$$

**Key Methods**:
- `objective(u, return_details)` – evaluate objective
- `optimize(method, maxiter, verbose)` – solve problem
- `compare_baseline_vs_optimized(u_opt)` – comparison metrics
- `cost_baseline_deviation(u)` – component 1
- `penalty_overload(u)` – component 2

**Optimization Interface**:
- Uses `scipy.optimize.minimize` with L-BFGS-B
- Respects generator bounds automatically
- Tracks history: cost, penalty, dispatch
- Numerical gradient computation

### 6. `validation.py` – Pipeline Validation Harness

**Purpose**: Systematic validation before optimization.

**Main Class**: `PipelineValidationHarness`

**Static Methods**:
- `validate_scenario_structure(scenario)` – check dimensions, consistency
- `validate_decision_spec(decision_spec)` – check decision variables
- `validate_solver(solver, decision_spec)` – test predictions, check dimensions
- `validate_overload_evaluator(overload_eval)` – test overload computation
- `full_validation(...)` – run all validations
- `print_validation_report(report)` – formatted console output

**Validation Checks**:
- ✓ Tensor shapes and types
- ✓ Bus type consistency (mutually exclusive)
- ✓ Bounds feasibility
- ✓ Model in eval mode
- ✓ Predictions are finite and reasonable
- ✓ Positional encoding dimension (pe_dim=20)
- ✓ Overload computation consistency

## Design Choices – Phase 1

| Choice | Value | Rationale |
|--------|-------|-----------|
| Mask policy | `pf` | Keeps solver near pretrained task distribution |
| Decision vector | `Pg` on PV buses | Avoids conflict with REF-bus PF masking |
| Cost term | Baseline deviation | $\|u - u_{\text{base}}\|_2^2$ |
| Risk term | Total overload | $\sum \max(0, \text{loading} - 1)^2$ |
| Constraints | Generator bounds | $P_{g,i}^{\min} \le u_i \le P_{g,i}^{\max}$ |
| Model compatibility | GNN v0.1 + GPS v0.2 | Both use true pe_dim=20 |

## Files in This Module

```
experiments/test/
├── __init__.py                          # Package exports
├── scenario_data.py                     # ScenarioData class and extraction
├── pv_dispatch.py                       # PVDispatchDecisionSpec class
├── neural_solver.py                     # NeuralSolverWrapper class
├── overload_penalty.py                  # OverloadPenaltyEvaluator class
├── optimization.py                      # DispatchOptimizationProblem class
├── validation.py                        # PipelineValidationHarness class
├── ieee30_optimization_validation.ipynb # Full demonstration notebook
└── README.md                            # This file
```

## Notebook: IEEE-30 Optimization Validation

The jupyter notebook `ieee30_optimization_validation.ipynb` provides a complete walkthrough:

1. Load IEEE-30 test data
2. Extract canonical scenario
3. Create decision spec (PV-bus Pg)
4. Load pretrained GNN model
5. Create neural solver wrapper
6. Create overload evaluator
7. Run full pipeline validation
8. Test solver predictions
9. Create optimization problem
10. Evaluate objective at baseline
11. Run optimization
12. Compare baseline vs. optimized
13. Plot optimization progress

**To run**:
```bash
jupyter notebook experiments/test/ieee30_optimization_validation.ipynb
```

## Known Limitations (Phase 1)

- ⚠ Only PV-bus `Pg` is optimized; REF-bus and other controls not yet included
- ⚠ Only generator bounds are enforced; full AC power flow feasibility not guaranteed
- ⚠ Surrogate predictions may become unreliable if $u$ moves far from baseline
- ⚠ Overload penalty depends on accurate branch postprocessing from predicted state
- ⚠ No explicit AC-OPF penalty for voltage or reactive power violations

## Future Extensions (Phases 2+)

### Phase 2: REF-Bus Participation
- Include REF-bus $V_m$ setpoint as decision variable
- Requires careful handling of slack bus in masking

### Phase 2: Reactive Power Control
- Add `Qg` at PV buses to decision vector
- Requires extending bounds and masking logic

### Phase 3: Advanced Constraints
- Explicit AC-OPF penalty terms
- Voltage magnitude constraints
- Reactive power limits
- Advanced ramp rate constraints

### Phase 3: Uncertainty Quantification
- Ensemble predictions from multiple scenarios
- Robustness to model prediction errors
- Distributionally robust optimization

## Generator Bounds Sourcing

**Current implementation**: Provisional bounds around baseline ($u_{\text{base}} \pm 20\%$)

**Future options**:
- Load from scenario metadata if available
- Parse from bus/generator parameter tables
- Use empirical percentiles from training data
- User-provided custom bounds

## Quick Reference: Class Instantiation

```python
# 1. Start with batch
scenario = extract_scenario_from_batch(batch, node_norm, edge_norm)

# 2. Define decision variables
decision_spec = PVDispatchDecisionSpec(scenario)

# 3. Create solver (GNN or GPS)
solver_gnn = NeuralSolverWrapper(model_gnn, "gnn", scenario, decision_spec)
solver_gps = NeuralSolverWrapper(model_gps, "gps", scenario, decision_spec)

# 4. Create overload evaluator
overload_eval = OverloadPenaltyEvaluator(scenario)

# 5. Create problem
problem = DispatchOptimizationProblem(
    scenario, decision_spec, solver_gnn, overload_eval,
    alpha=1.0, lambda_=1.0
)

# 6. Validate (optional but recommended)
from experiments.test import PipelineValidationHarness
report = PipelineValidationHarness.full_validation(
    scenario, decision_spec, solver_gnn, solver_gps, overload_eval
)
PipelineValidationHarness.print_validation_report(report)

# 7. Optimize
result = problem.optimize(method="L-BFGS-B", maxiter=100)
```

## Author & Version

Phase 1 Implementation: Predict-then-Optimize Pipeline for GridFM  
Base Model: GNN v0.1 (pe_dim=20, hidden_size=64)  
Tested on: IEEE-30 network  
Compatibility: GPS v0.2 (pe_dim=20, hidden_size=256)
