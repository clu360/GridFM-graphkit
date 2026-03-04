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
import yaml

# Add repo to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from experiments.test import (
    ScenarioData,
    extract_scenario_from_batch,
    PVDispatchDecisionSpec,
    NeuralSolverWrapper,
    OverloadPenaltyEvaluator,
    DispatchOptimizationProblem,
    PipelineValidationHarness,
)

from gridfm_graphkit.io.param_handler import NestedNamespace, load_model
from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule


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
    config_path = repo_root / "tests" / "config" / "gridFMv0.1_dummy.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    args = NestedNamespace(**config_dict)
    
    data_dir = repo_root / "tests" / "data"  # Use tests/data, not data/
    datamodule = LitGridDataModule(args, data_dir=str(data_dir))
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()[0]
    batch = next(iter(test_loader))
    print("✓ Loaded IEEE-30 test batch\n")
    
    # 2. Extract scenario
    print("[2/8] Extracting scenario...")
    scenario = extract_scenario_from_batch(
        batch,
        datamodule.node_normalizers[0],
        datamodule.edge_normalizers[0],
        scenario_id="IEEE-30",
    )
    print(f"✓ Scenario with {scenario.num_buses} buses, "
          f"{np.sum(scenario.PV_mask)} PV buses\n")
    
    # 3. Define decision variables
    print("[3/8] Creating decision spec...")
    decision_spec = PVDispatchDecisionSpec(scenario)
    print(f"✓ Decision vector dimension: {decision_spec.n_pv}\n")
    
    # 4. Load and wrap model
    print("[4/8] Loading pretrained model...")
    model = load_model(args)
    model_path = repo_root / "examples" / "models" / "GridFM_v0_1.pth"
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"✓ Loaded weights from {model_path.name}\n")
    else:
        print(f"⚠ Model file not found, using untrained model\n")
    
    # 5. Create solver wrapper
    print("[5/8] Creating solver wrapper...")
    solver = NeuralSolverWrapper(
        model, "gnn", scenario, decision_spec, device=device
    )
    print("✓ Solver ready for inference\n")
    
    # 6. Create overload evaluator
    print("[6/8] Creating overload evaluator...")
    overload_eval = OverloadPenaltyEvaluator(scenario)
    baseline_overload = overload_eval.evaluate_baseline()
    print(f"✓ Baseline: {baseline_overload['n_overloaded_lines']} overloaded lines, "
          f"max loading {baseline_overload['max_loading']:.3f}\n")
    
    # 7. Quick validation
    print("[7/8] Running pipeline validation...")
    validation_report = PipelineValidationHarness.full_validation(
        scenario, decision_spec, solver, None, overload_eval
    )
    if validation_report['all_passed']:
        print("✓ All validation checks passed\n")
    else:
        print("⚠ Some validation checks failed (see details above)\n")
    
    # 8. Optimize
    print("[8/8] Running optimization...")
    problem = DispatchOptimizationProblem(
        scenario, decision_spec, solver, overload_eval,
        alpha=1.0, lambda_=1.0
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
    
    print(f"\nCost (baseline deviation):")
    print(f"  Baseline:  {comparison['baseline']['cost']:.6f}")
    print(f"  Optimized: {comparison['optimized']['cost']:.6f}")
    
    print(f"\nPenalty (overload):")
    print(f"  Baseline:  {comparison['baseline']['penalty']:.6f}")
    print(f"  Optimized: {comparison['optimized']['penalty']:.6f}")
    print(f"  Reduction: {comparison['improvement']['penalty_pct']:.2f}%")
    
    print(f"\nOverloaded lines:")
    print(f"  Baseline:  {comparison['baseline']['n_overloaded']}")
    print(f"  Optimized: {comparison['optimized']['n_overloaded']}")
    
    print(f"\nMax line loading:")
    print(f"  Baseline:  {comparison['baseline']['max_loading']:.4f}")
    print(f"  Optimized: {comparison['optimized']['max_loading']:.4f}")
    
    print(f"\nOptimization:")
    print(f"  Iterations: {result['n_iter']}")
    print(f"  Success: {result['success']}")
    
    print("\n" + "=" * 70)
    print("✓ Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
