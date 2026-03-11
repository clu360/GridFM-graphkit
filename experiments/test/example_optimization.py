"""
Simple example: Dispatch optimization with GridFM neural solver.

This script demonstrates the minimal code needed to set up and run
a predict-then-optimize dispatch optimization experiment.

Usage:
    python example_optimization.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from experiments.test import (
    PVDispatchDecisionSpec,
    NeuralSolverWrapper,
    WildfirePenaltyEvaluator,
    DispatchOptimizationProblem,
    PipelineValidationHarness,
    load_gnn_model,
    load_single_test_scenario,
)

STANDARD_BRANCH_RATING_MVA = 100.0


def main():
    """Run full optimization pipeline."""
    
    print("=" * 70)
    print("GridFM Predict-Then-Optimize Pipeline Example")
    print("=" * 70)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")
    
    # 1. Load configuration and data
    print("[1/8] Loading test data...")
    context = load_single_test_scenario(scenario_idx=0, scenario_id="IEEE-30")
    repo_root = context.repo_root
    args = context.args
    scenario = context.scenario
    print("[OK] Loaded IEEE-30 test batch\n")
    
    # 2. Extract scenario
    print("[2/8] Extracting scenario...")
    print(f"[OK] Scenario with {scenario.num_buses} buses, "
          f"{np.sum(scenario.PV_mask)} PV buses\n")
    
    # 3. Define decision variables
    print("[3/8] Creating decision spec...")
    decision_spec = PVDispatchDecisionSpec(scenario)
    print(f"[OK] Decision vector dimension: {decision_spec.n_pv}\n")
    
    # 4. Load and wrap model
    print("[4/8] Loading pretrained model...")
    model = load_gnn_model(args, repo_root=repo_root, device=device)
    print("[OK] Loaded weights from GridFM_v0_1.pth\n")
    
    # 5. Create solver wrapper
    print("[5/8] Creating solver wrapper...")
    solver = NeuralSolverWrapper(
        model, "gnn", scenario, decision_spec, device=device
    )
    print("[OK] Solver ready for inference\n")
    
    # 6. Create wildfire evaluator
    print("[6/8] Creating wildfire evaluator...")
    wildfire_eval = WildfirePenaltyEvaluator(
        scenario,
        standard_rate_a_mva=STANDARD_BRANCH_RATING_MVA,
    )
    baseline_risk = wildfire_eval.evaluate_baseline()
    print(
        f"[OK] Baseline wildfire cost: {baseline_risk['wildfire_cost']:.6f}, "
        f"active risk branches {baseline_risk['n_active_risk_branches']}, "
        f"max loading {baseline_risk['max_loading']:.3f}\n"
    )
    
    # 7. Quick validation
    print("[7/8] Running pipeline validation...")
    validation_report = PipelineValidationHarness.full_validation(
        scenario, decision_spec, solver, None, wildfire_eval
    )
    if validation_report['all_passed']:
        print("[OK] All validation checks passed\n")
    else:
        print("[WARN] Some validation checks failed (see details above)\n")
    
    # 8. Optimize
    print("[8/8] Running optimization...")
    problem = DispatchOptimizationProblem(
        scenario,
        decision_spec,
        solver,
        wildfire_eval,
        lambda_gen=1.0,
        lambda_shed=50.0,
        lambda_wf=10.0,
    )
    
    result = problem.optimize(method="L-BFGS-B", maxiter=50, verbose=False)
    
    # Results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    
    u_opt = result['u_opt']
    comparison = problem.compare_baseline_vs_optimized(u_opt)
    
    print(f"\nObjective:")
    print(f"  Baseline:  {comparison['baseline']['objective']:.6f}")
    print(f"  Optimized: {comparison['optimized']['objective']:.6f}")
    print(f"  Improvement: {comparison['improvement']['objective_pct']:.2f}%")
    
    print(f"\nGenerator deviation cost:")
    print(f"  Baseline:  {comparison['baseline']['generator_deviation_cost']:.6f}")
    print(f"  Optimized: {comparison['optimized']['generator_deviation_cost']:.6f}")
    
    print(f"\nWildfire cost:")
    print(f"  Baseline:  {comparison['baseline']['wildfire_cost']:.6f}")
    print(f"  Optimized: {comparison['optimized']['wildfire_cost']:.6f}")
    print(f"  Reduction: {comparison['improvement']['wildfire_pct']:.2f}%")
    
    print(f"\nActive wildfire-risk branches:")
    print(f"  Baseline:  {comparison['baseline']['n_active_risk_branches']}")
    print(f"  Optimized: {comparison['optimized']['n_active_risk_branches']}")
    
    print(f"\nMax line loading:")
    print(f"  Baseline:  {comparison['baseline']['max_loading']:.4f}")
    print(f"  Optimized: {comparison['optimized']['max_loading']:.4f}")
    
    print(f"\nOptimization:")
    print(f"  Iterations: {result['n_iter']}")
    print(f"  Success: {result['success']}")
    
    print("\n" + "=" * 70)
    print("[OK] Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
