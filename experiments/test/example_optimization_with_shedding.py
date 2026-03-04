"""
Example: Predict-then-optimize with load shedding capability.

Demonstrates extended dispatch optimization including both:
  - PV bus active generation (Pg) adjustments
  - Load shedding at PQ buses (Pd reduction)
"""

import sys
from pathlib import Path
import numpy as np
import torch
import yaml

repo_root = Path('.').resolve()
sys.path.insert(0, str(repo_root))

from experiments.test import (
    extract_scenario_from_batch,
    ExtendedDispatchSpec,
    NeuralSolverWrapper,
    OverloadPenaltyEvaluator,
    DispatchOptimizationProblem,
)
from gridfm_graphkit.io.param_handler import NestedNamespace, load_model
from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule

print("=" * 80)
print("PREDICT-THEN-OPTIMIZE WITH LOAD SHEDDING - IEEE-30 EXAMPLE")
print("=" * 80)

# ============================================================================
# Load IEEE-30 Test Data
# ============================================================================
print("\n[1] Loading IEEE-30 test data...")
config_path = repo_root / "tests" / "config" / "gridFMv0.1_dummy.yaml"
with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)
args = NestedNamespace(**config_dict)

data_dir = repo_root / "tests" / "data"
datamodule = LitGridDataModule(args, data_dir=str(data_dir))
datamodule.setup(stage="test")

test_loader = datamodule.test_dataloader()[0]
batch = next(iter(test_loader))
print(f"✓ Loaded batch: {batch.num_nodes} nodes, {batch.num_edges} edges")

# ============================================================================
# Extract Scenario and Initialize Components
# ============================================================================
print("\n[2] Setting up scenario and decision specs...")
node_normalizer = datamodule.node_normalizers[0]
edge_normalizer = datamodule.edge_normalizers[0]

scenario = extract_scenario_from_batch(
    batch,
    node_normalizer,
    edge_normalizer,
    scenario_id="IEEE-30-with-shedding",
)

# Create EXTENDED dispatch spec with load shedding capability
decision_spec = ExtendedDispatchSpec(
    scenario,
    max_shed_fraction=1.0,  # Allow up to 100% shedding at each bus
)

print(f"✓ Decision variables:")
print(f"  - PV buses: {decision_spec.n_pv}")
print(f"  - PQ buses (can shed): {decision_spec.n_pq}")
print(f"  - Total decision dim: {decision_spec.n_total}")
print(f"  - Max total shedding: {decision_spec.get_summary()['shed_max_shed_total_MW']:.1f} MW")

# ============================================================================
# Load Neural Model and Create Solver
# ============================================================================
print("\n[3] Loading neural model...")
device = "cpu"
model = load_model(args)

solver = NeuralSolverWrapper(
    model,
    model_type="gnn",
    scenario=scenario,
    decision_spec=decision_spec,
    device=device,
)
print(f"✓ Loaded {solver.model_type.upper()} model")

# ============================================================================
# Create Overload Evaluator
# ============================================================================
print("\n[4] Computing baseline overload...")
overload_eval = OverloadPenaltyEvaluator(scenario)
baseline_eval = overload_eval.evaluate_baseline()

print(f"✓ Baseline metrics:")
print(f"  - Total penalty: {baseline_eval['total_penalty']:.2f}")
print(f"  - Overloaded lines: {baseline_eval['n_overloaded_lines']}")
print(f"  - Max line loading: {baseline_eval['max_loading']:.2f} pu")
print(f"  - Mean line loading: {baseline_eval['mean_loading']:.4f} pu")

# ============================================================================
# SCENARIO 1: PV Generation Only (Original Phase 1)
# ============================================================================
print("\n[5] SCENARIO 1: PV Generation Adjustment Only (α=1, β=0, λ=1)")
print("-" * 80)

problem_pv_only = DispatchOptimizationProblem(
    scenario=scenario,
    decision_spec=decision_spec,
    solver=solver,
    overload_eval=overload_eval,
    alpha=1.0,
    lambda_=1.0,
    beta=0.0,  # No shedding cost (shedding not encouraged)
)

result_pv_only = problem_pv_only.optimize(method="L-BFGS-B", maxiter=50, verbose=False)

print(f"Optimization completed:")
print(f"  - Success: {result_pv_only['success']}")
print(f"  - Iterations: {result_pv_only['n_iter']}")
print(f"  - Final objective: {result_pv_only['obj_opt']:.2f}")
print(f"  - Final penalty: {result_pv_only['penalty_opt']:.2f}")
print(f"  - Final shedding cost: {result_pv_only['details']['cost_shedding']:.4f}")

# Extract results
u_opt_pv = result_pv_only['u_opt']
Pg_pv_opt, delta_opt_pv = decision_spec.split_decision_vector(u_opt_pv)
print(f"\nOptimized dispatch (PV only):")
print(f"  - PV generation change: {np.linalg.norm(Pg_pv_opt - decision_spec.pv_spec.u_base):.6f} MW (L2)")
print(f"  - Load shed: {np.sum(delta_opt_pv):.6f} MW (total)")
print(f"  - Improvement vs baseline: {result_pv_only['details']['penalty_overload'] - baseline_eval['total_penalty']:.2f}")

# ============================================================================
# SCENARIO 2: With Load Shedding (New Phase 2 Extension)
# ============================================================================
print("\n[6] SCENARIO 2: PV + Load Shedding (α=1, β=10, λ=1)")
print("-" * 80)
print("β=10 means shedding 1 MW costs same as changing generation by √10 ≈ 3.16 MW")

problem_with_shedding = DispatchOptimizationProblem(
    scenario=scenario,
    decision_spec=decision_spec,
    solver=solver,
    overload_eval=overload_eval,
    alpha=1.0,
    lambda_=1.0,
    beta=10.0,  # Penalize shedding (but allow it if needed)
)

result_with_shedding = problem_with_shedding.optimize(
    method="L-BFGS-B", 
    maxiter=100, 
    verbose=False,
)

print(f"Optimization completed:")
print(f"  - Success: {result_with_shedding['success']}")
print(f"  - Iterations: {result_with_shedding['n_iter']}")
print(f"  - Final objective: {result_with_shedding['obj_opt']:.2f}")
print(f"  - Final penalty: {result_with_shedding['penalty_opt']:.2f}")
print(f"  - Final shedding cost: {result_with_shedding['details']['cost_shedding']:.4f}")

# Extract results
u_opt_shedding = result_with_shedding['u_opt']
Pg_pv_opt_shed, delta_opt_shed = decision_spec.split_decision_vector(u_opt_shedding)
print(f"\nOptimized dispatch (with shedding):")
print(f"  - PV generation change: {np.linalg.norm(Pg_pv_opt_shed - decision_spec.pv_spec.u_base):.6f} MW (L2)")
print(f"  - Load shed: {np.sum(delta_opt_shed):.6f} MW (total)")
print(f"  - Shed as % of PQ demand: {100 * decision_spec.shed_spec.get_shed_fraction(delta_opt_shed):.4f}%")
print(f"  - Improvement vs baseline: {result_with_shedding['details']['penalty_overload'] - baseline_eval['total_penalty']:.2f}")

# ============================================================================
# SCENARIO 3: Aggressive Shedding (New Phase 2 Extension)
# ============================================================================
print("\n[7] SCENARIO 3: Aggressive Shedding (α=1, β=1, λ=1)")
print("-" * 80)
print("β=1 means shedding is equally weighted as generation/penalty")

problem_aggressive_shed = DispatchOptimizationProblem(
    scenario=scenario,
    decision_spec=decision_spec,
    solver=solver,
    overload_eval=overload_eval,
    alpha=1.0,
    lambda_=1.0,
    beta=1.0,  # More aggressive shedding is accepted
)

result_aggressive = problem_aggressive_shed.optimize(
    method="L-BFGS-B",
    maxiter=100,
    verbose=False,
)

print(f"Optimization completed:")
print(f"  - Success: {result_aggressive['success']}")
print(f"  - Iterations: {result_aggressive['n_iter']}")
print(f"  - Final objective: {result_aggressive['obj_opt']:.2f}")
print(f"  - Final penalty: {result_aggressive['penalty_opt']:.2f}")
print(f"  - Final shedding cost: {result_aggressive['details']['cost_shedding']:.4f}")

# Extract results
u_opt_aggressive = result_aggressive['u_opt']
Pg_pv_opt_agg, delta_opt_agg = decision_spec.split_decision_vector(u_opt_aggressive)
print(f"\nOptimized dispatch (aggressive shedding):")
print(f"  - PV generation change: {np.linalg.norm(Pg_pv_opt_agg - decision_spec.pv_spec.u_base):.6f} MW (L2)")
print(f"  - Load shed: {np.sum(delta_opt_agg):.6f} MW (total)")
print(f"  - Shed as % of PQ demand: {100 * decision_spec.shed_spec.get_shed_fraction(delta_opt_agg):.4f}%")
print(f"  - Improvement vs baseline: {result_aggressive['details']['penalty_overload'] - baseline_eval['total_penalty']:.2f}")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n[8] COMPARISON OF SCENARIOS")
print("=" * 80)

comparison_data = [
    ("Baseline", 
     baseline_eval['total_penalty'],
     0.0,
     0.0,
     0.0,
     baseline_eval['n_overloaded_lines'],
     baseline_eval['max_loading']),
    
    ("PV only (β=0)",
     result_pv_only['details']['penalty_overload'],
     result_pv_only['details']['cost_deviation'],
     result_pv_only['details']['cost_shedding'],
     np.sum(delta_opt_pv),
     result_pv_only['details']['n_overloaded_lines'],
     result_pv_only['details']['max_loading']),
    
    ("With shedding (β=10)",
     result_with_shedding['details']['penalty_overload'],
     result_with_shedding['details']['cost_deviation'],
     result_with_shedding['details']['cost_shedding'],
     np.sum(delta_opt_shed),
     result_with_shedding['details']['n_overloaded_lines'],
     result_with_shedding['details']['max_loading']),
    
    ("Aggressive (β=1)",
     result_aggressive['details']['penalty_overload'],
     result_aggressive['details']['cost_deviation'],
     result_aggressive['details']['cost_shedding'],
     np.sum(delta_opt_agg),
     result_aggressive['details']['n_overloaded_lines'],
     result_aggressive['details']['max_loading']),
]

print(f"{'Scenario':<20} {'Penalty':<12} {'Dev Cost':<12} {'Shed Cost':<12} {'Shed (MW)':<12} {'Overload':<12} {'Max Load':<12}")
print("-" * 80)

for scenario_name, penalty, dev, shed_cost, shed_mw, n_over, max_load in comparison_data:
    print(f"{scenario_name:<20} {penalty:>10.2f}   {dev:>10.4f}   {shed_cost:>10.4f}   {shed_mw:>10.2f}   {n_over:>10d}   {max_load:>10.2f}")

print("=" * 80)

# ============================================================================
# Single Bus Shedding Breakdown
# ============================================================================
print("\n[9] LOAD SHEDDING BY BUS (Aggressive Scenario)")
print("-" * 80)

pq_buses = decision_spec.shed_spec.pq_bus_indices
shed_by_bus = np.zeros(len(pq_buses))

for i, pq_idx in enumerate(pq_buses):
    if i < len(delta_opt_agg):
        shed_by_bus[i] = delta_opt_agg[i]

shed_sorted_idx = np.argsort(-shed_by_bus)  # Sort descending

print(f"Top 10 buses with load shedding:")
print(f"{'Rank':<5} {'Bus IDX':<10} {'Shed (MW)':<15} {'% of Pd':<15}")
print("-" * 45)

baseline_pd = scenario.Pd_base[pq_buses]
for rank, i in enumerate(shed_sorted_idx[:10]):
    pq_idx = pq_buses[i]
    shed_mw = shed_by_bus[i]
    shed_pct = 100 * shed_mw / baseline_pd[i] if baseline_pd[i] > 0 else 0
    print(f"{rank+1:<5} {pq_idx:<10} {shed_mw:>13.4f}   {shed_pct:>13.2f}%")

print("\n" + "=" * 80)
print("✅ LOAD SHEDDING OPTIMIZATION COMPLETE")
print("=" * 80)
