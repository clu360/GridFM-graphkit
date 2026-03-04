"""
Evaluate load shedding optimization with GNN vs GPS models.
Compares performance on IEEE-30 test case.
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

print('=' * 80)
print('LOAD SHEDDING EVALUATION - GNN vs GPS MODELS')
print('=' * 80)

# Load data
config_path = repo_root / 'tests' / 'config' / 'gridFMv0.1_dummy.yaml'
with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)
args = NestedNamespace(**config_dict)

data_dir = repo_root / 'tests' / 'data'
datamodule = LitGridDataModule(args, data_dir=str(data_dir))
datamodule.setup(stage='test')

test_loader = datamodule.test_dataloader()[0]
batch = next(iter(test_loader))

node_normalizer = datamodule.node_normalizers[0]
edge_normalizer = datamodule.edge_normalizers[0]

# Extract scenario
scenario = extract_scenario_from_batch(
    batch, node_normalizer, edge_normalizer, scenario_id='IEEE-30-test'
)

# Create decision spec with shedding
decision_spec = ExtendedDispatchSpec(scenario, max_shed_fraction=1.0)

# Load both models
device = 'cpu'
model_gnn = load_model(args)  # GNN_v0.1
print('✓ Loaded GNN model')

# For GPS_v0.2, need GPS config
config_gps_path = repo_root / 'tests' / 'config' / 'gridFMv0.2_dummy.yaml'
if config_gps_path.exists():
    with open(config_gps_path, 'r') as f:
        config_gps_dict = yaml.safe_load(f)
    args_gps = NestedNamespace(**config_gps_dict)
    model_gps = load_model(args_gps)
    print('✓ Loaded GPS model')
    has_gps = True
else:
    print('⚠ GPS config not found, checking for GPS model checkpoint...')
    gps_model_path = repo_root / 'examples' / 'models' / 'GridFM_v0_2.pth'
    if gps_model_path.exists():
        # Load with default args but GPS architecture
        args.model.model_type = 'gps'
        args.model.pe_dim = 20
        model_gps = load_model(args)
        print('✓ Loaded GPS model from checkpoint')
        has_gps = True
    else:
        print('✗ GPS model not found at expected paths')
        has_gps = False

# Create solvers
solver_gnn = NeuralSolverWrapper(model_gnn, 'gnn', scenario, decision_spec, device=device)
print('✓ Created GNN solver')

if has_gps:
    solver_gps = NeuralSolverWrapper(model_gps, 'gps', scenario, decision_spec, device=device)
    print('✓ Created GPS solver')

# Create overload evaluator
overload_eval = OverloadPenaltyEvaluator(scenario)

print()
print('Baseline metrics:')
baseline_eval = overload_eval.evaluate_baseline()
print(f'  Penalty: {baseline_eval["total_penalty"]:.2f}')
print(f'  Overloaded lines: {baseline_eval["n_overloaded_lines"]}')
print(f'  Max loading: {baseline_eval["max_loading"]:.2f} pu')

print()
print('-' * 80)
print('GNN_v0.1 WITH LOAD SHEDDING')
print('-' * 80)

problem_gnn = DispatchOptimizationProblem(
    scenario, decision_spec, solver_gnn, overload_eval,
    alpha=1.0, lambda_=1.0, beta=10.0
)

result_gnn = problem_gnn.optimize(method='L-BFGS-B', maxiter=100, verbose=False)

print(f'Optimization converged: {result_gnn["success"]}')
print(f'Iterations: {result_gnn["n_iter"]}')
print(f'Message: {result_gnn["message"]}')
print()
print('Objective:')
print(f'  Baseline: 222455.84')
print(f'  Optimized: {result_gnn["obj_opt"]:.2f}')
print(f'  Improvement: {222455.84 - result_gnn["obj_opt"]:.2f}')
print()
print('Components:')
print(f'  Cost (Pg change): {result_gnn["cost_opt"]:.2f}')
print(f'  Penalty (overloads): {result_gnn["penalty_opt"]:.2f}')
print()
u_opt_gnn = result_gnn['u_opt']
n_pv = decision_spec.pv_spec.n_pv
Pg_base = decision_spec.pv_spec.u_base
Pg_change = np.sum(np.abs(u_opt_gnn[:n_pv] - Pg_base))
shed_total = np.sum(u_opt_gnn[n_pv:])
print('Dispatch changes:')
print(f'  Total Pg adjustment: {Pg_change:.4f} MW')
print(f'  Total load shed: {shed_total:.4f} MW')

if has_gps:
    print()
    print('-' * 80)
    print('GPS_v0.2 WITH LOAD SHEDDING')
    print('-' * 80)
    
    problem_gps = DispatchOptimizationProblem(
        scenario, decision_spec, solver_gps, overload_eval,
        alpha=1.0, lambda_=1.0, beta=10.0
    )
    
    result_gps = problem_gps.optimize(method='L-BFGS-B', maxiter=100, verbose=False)
    
    print(f'Optimization converged: {result_gps["success"]}')
    print(f'Iterations: {result_gps["n_iter"]}')
    print(f'Message: {result_gps["message"]}')
    print()
    print('Objective:')
    print(f'  Baseline: 222455.84')
    print(f'  Optimized: {result_gps["obj_opt"]:.2f}')
    print(f'  Improvement: {222455.84 - result_gps["obj_opt"]:.2f}')
    print()
    print('Components:')
    print(f'  Cost (Pg change): {result_gps["cost_opt"]:.2f}')
    print(f'  Penalty (overloads): {result_gps["penalty_opt"]:.2f}')
    print()
    u_opt_gps = result_gps['u_opt']
    Pg_change_gps = np.sum(np.abs(u_opt_gps[:n_pv] - Pg_base))
    shed_total_gps = np.sum(u_opt_gps[n_pv:])
    print('Dispatch changes:')
    print(f'  Total Pg adjustment: {Pg_change_gps:.4f} MW')
    print(f'  Total load shed: {shed_total_gps:.4f} MW')
    
    print()
    print('=' * 80)
    print('COMPARISON: GNN vs GPS')
    print('=' * 80)
    print(f'GNN objective:  {result_gnn["obj_opt"]:.2f}')
    print(f'GPS objective:  {result_gps["obj_opt"]:.2f}')
    diff = result_gnn['obj_opt'] - result_gps['obj_opt']
    if abs(diff) < 0.01:
        print(f'Difference: {abs(diff):.2f} (similar performance)')
    elif diff > 0:
        print(f'Difference: {diff:.2f} (GPS is better)')
    else:
        print(f'Difference: {abs(diff):.2f} (GNN is better)')
    
    print()
    print(f'GNN Pg changes:     {Pg_change:.4f} MW')
    print(f'GPS Pg changes:     {Pg_change_gps:.4f} MW')
    print(f'GNN load shed:      {shed_total:.4f} MW')
    print(f'GPS load shed:      {shed_total_gps:.4f} MW')

print()
print('=' * 80)
