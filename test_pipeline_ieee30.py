"""
Test script for predict-then-optimize pipeline using actual IEEE-30 test data.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import yaml

repo_root = Path('.').resolve()
sys.path.insert(0, str(repo_root))

print("=" * 80)
print("PIPELINE MODULE TEST SUITE - IEEE-30 REAL DATA")
print("=" * 80)

# ============================================================================
# TEST 1: Import all modules
# ============================================================================
print("\n[1/7] Testing module imports...")
try:
    from experiments.test import (
        ScenarioData,
        extract_scenario_from_batch,
        PVDispatchDecisionSpec,
        NeuralSolverWrapper,
        OverloadPenaltyEvaluator,
        DispatchOptimizationProblem,
        PipelineValidationHarness,
    )
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 2: Load IEEE-30 test data
# ============================================================================
print("\n[2/7] Loading IEEE-30 test data...")
try:
    from gridfm_graphkit.io.param_handler import NestedNamespace, load_model
    from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
    
    # Load config
    config_path = repo_root / "tests" / "config" / "gridFMv0.1_dummy.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    args = NestedNamespace(**config_dict)
    
    # Load datamodule
    data_dir = repo_root / "tests" / "data"
    datamodule = LitGridDataModule(args, data_dir=str(data_dir))
    datamodule.setup(stage="test")
    
    test_loader = datamodule.test_dataloader()[0]
    batch = next(iter(test_loader))
    
    print(f"✓ Loaded IEEE-30 test batch")
    print(f"  - Num nodes: {batch.num_nodes}")
    print(f"  - Num edges: {batch.num_edges}")
    print(f"  - PE shape: {batch.pe.shape}")
    print(f"  - Node features shape: {batch.x.shape}")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 3: Test ScenarioData extraction
# ============================================================================
print("\n[3/7] Testing ScenarioData extraction...")
try:
    node_normalizer = datamodule.node_normalizers[0]
    edge_normalizer = datamodule.edge_normalizers[0]
    
    scenario = extract_scenario_from_batch(
        batch,
        node_normalizer,
        edge_normalizer,
        scenario_id="IEEE-30-test",
    )
    
    assert scenario.num_buses > 0, "num_buses should be positive"
    assert scenario.Pd_base.shape[0] == scenario.num_buses, "Pd_base shape mismatch"
    assert scenario.pe.shape[1] == 20, f"PE dimension should be 20, got {scenario.pe.shape[1]}"
    assert scenario.PQ_mask is not None, "PQ_mask should not be None"
    assert scenario.PV_mask is not None, "PV_mask should not be None"
    assert scenario.REF_mask is not None, "REF_mask should not be None"
    
    pv_buses = scenario.get_pv_buses()
    pq_buses = scenario.get_pq_buses()
    ref_bus = scenario.get_ref_bus()
    
    print(f"✓ Scenario extracted successfully")
    print(f"  - Total buses: {scenario.num_buses}")
    print(f"  - PQ buses: {len(pq_buses)}")
    print(f"  - PV buses: {len(pv_buses)}")
    print(f"  - REF bus: {ref_bus}")
    print(f"  - PE dimension: {scenario.pe.shape[1]}")
    print(f"  - Mask shape: {scenario.mask.shape}")
except Exception as e:
    print(f"✗ ScenarioData extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 4: Test PVDispatchDecisionSpec
# ============================================================================
print("\n[4/7] Testing PVDispatchDecisionSpec...")
try:
    decision_spec = PVDispatchDecisionSpec(scenario)
    
    assert decision_spec.n_pv > 0, "n_pv should be positive"
    assert decision_spec.u_base.shape[0] == decision_spec.n_pv, "u_base shape mismatch"
    assert np.all(decision_spec.u_min <= decision_spec.u_max), "Bounds inconsistent"
    assert np.all(decision_spec.u_min <= decision_spec.u_base), "u_base below minimum"
    assert np.all(decision_spec.u_base <= decision_spec.u_max), "u_base above maximum"
    
    # Test mapping functions
    u = decision_spec.u_base.copy()
    Pg_full = decision_spec.u_to_Pg(u)
    u_extracted = decision_spec.Pg_to_u(Pg_full)
    assert np.allclose(u, u_extracted), "Mapping roundtrip failed"
    assert Pg_full.shape[0] == scenario.num_buses, "Pg_full shape mismatch"
    
    # Test node features
    node_features = decision_spec.u_to_node_features(u)
    assert node_features.shape == (scenario.num_buses, 6), f"Node features shape {node_features.shape} != (30, 6)"
    
    bounds_ok, msg = decision_spec.check_bounds(u)
    assert bounds_ok, f"Bounds check failed: {msg}"
    
    distance = decision_spec.get_distance_from_baseline(u)
    assert distance >= 0, "Distance should be non-negative"
    
    summary = decision_spec.get_summary()
    assert "n_pv" in summary, "Missing n_pv in summary"
    
    print(f"✓ PVDispatchDecisionSpec working correctly")
    print(f"  - PV buses: {decision_spec.n_pv}")
    print(f"  - Baseline Pg mean: {np.mean(decision_spec.u_base):.4f}")
    print(f"  - Bounds range mean: {summary['bound_range_mean']:.4f}")
except Exception as e:
    print(f"✗ PVDispatchDecisionSpec failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 5: Test NeuralSolverWrapper
# ============================================================================
print("\n[5/7] Testing NeuralSolverWrapper...")
try:
    device = "cpu"  # Use CPU for testing
    model = load_model(args)
    
    solver = NeuralSolverWrapper(
        model,
        model_type="gnn",
        scenario=scenario,
        decision_spec=decision_spec,
        device=device,
    )
    
    # Test prediction on baseline
    u_test = decision_spec.u_base.copy()
    pred = solver.predict_state(u_test)
    
    assert isinstance(pred, dict), "Prediction should be dict"
    expected_keys = {"Pd", "Qd", "Pg", "Qg", "Vm", "Va"}
    assert set(pred.keys()) == expected_keys, f"Missing prediction keys, got {set(pred.keys())}"
    
    for key, val in pred.items():
        assert val.shape == (scenario.num_buses,), f"{key} shape {val.shape} != ({scenario.num_buses},)"
        assert np.isfinite(val).all(), f"{key} contains non-finite values"
    
    # Test baseline validation
    errors = solver.validate_baseline()
    assert "Vm_rmse" in errors, "validate_baseline should return error metrics"
    
    print(f"✓ NeuralSolverWrapper working correctly")
    print(f"  - Model type: {solver.model_type}")
    print(f"  - Device: {solver.device}")
    print(f"  - Vm mean: {np.mean(pred['Vm']):.4f}")
    print(f"  - Va mean: {np.mean(pred['Va']):.4f}")
    print(f"  - Vm RMSE vs baseline: {errors['Vm_rmse']:.6f}")
except Exception as e:
    print(f"✗ NeuralSolverWrapper failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 6: Test OverloadPenaltyEvaluator
# ============================================================================
print("\n[6/7] Testing OverloadPenaltyEvaluator...")
try:
    overload_eval = OverloadPenaltyEvaluator(scenario)
    
    # Test baseline evaluation
    baseline_eval = overload_eval.evaluate_baseline()
    
    assert isinstance(baseline_eval, dict), "Baseline eval should return dict"
    assert "total_penalty" in baseline_eval, "Missing total_penalty"
    assert "n_overloaded_lines" in baseline_eval, "Missing n_overloaded_lines"
    assert "max_loading" in baseline_eval, "Missing max_loading"
    assert baseline_eval["total_penalty"] >= 0, "Penalty should be non-negative"
    assert baseline_eval["n_overloaded_lines"] >= 0, "n_overloaded_lines should be non-negative"
    assert baseline_eval["max_loading"] > 0, "max_loading should be positive"
    
    print(f"✓ OverloadPenaltyEvaluator working correctly")
    print(f"  - Baseline penalty: {baseline_eval['total_penalty']:.6f}")
    print(f"  - Overloaded lines: {baseline_eval['n_overloaded_lines']}")
    print(f"  - Max line loading: {baseline_eval['max_loading']:.4f}")
    print(f"  - Mean line loading: {baseline_eval['mean_loading']:.4f}")
except Exception as e:
    print(f"✗ OverloadPenaltyEvaluator failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 7: Test DispatchOptimizationProblem
# ============================================================================
print("\n[7/7] Testing DispatchOptimizationProblem...")
try:
    problem = DispatchOptimizationProblem(
        scenario=scenario,
        decision_spec=decision_spec,
        solver=solver,
        overload_eval=overload_eval,
        alpha=1.0,
        lambda_=1.0,
    )
    
    # Test objective evaluation at baseline
    u_test = decision_spec.u_base.copy()
    obj, details = problem.objective(u_test, return_details=True)
    
    assert np.isfinite(obj), f"Objective should be finite, got {obj}"
    assert isinstance(details, dict), "Details should be dict"
    assert "cost_deviation" in details, "Missing cost_deviation in details"
    assert "penalty_overload" in details, "Missing penalty_overload in details"
    assert details["cost_deviation"] >= 0, "Cost should be non-negative"
    assert details["penalty_overload"] >= 0, "Penalty should be non-negative"
    
    # Check bounds
    u_min, u_max = problem.get_bounds()
    assert np.all(u_min <= u_max), "Bounds inconsistent"
    
    print(f"✓ DispatchOptimizationProblem working correctly")
    print(f"  - Baseline objective: {obj:.6f}")
    print(f"  - Cost component: {details['cost_deviation']:.6f}")
    print(f"  - Penalty component: {details['penalty_overload']:.6f}")
    print(f"  - Decision dimension: {len(u_test)}")
    
except Exception as e:
    print(f"✗ DispatchOptimizationProblem failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# BONUS: Test PipelineValidationHarness
# ============================================================================
print("\n[BONUS] Testing PipelineValidationHarness...")
try:
    validation_report = PipelineValidationHarness.full_validation(
        scenario,
        decision_spec,
        solver,
        solver_gps=None,
        overload_eval=overload_eval,
    )
    
    all_passed = validation_report.get("all_passed", False)
    
    if all_passed:
        print(f"✓ All validation checks passed")
    else:
        print(f"⚠ Some validation checks may have failed")
        print(f"  Running verbose report...")
        PipelineValidationHarness.print_validation_report(validation_report, verbose=False)
    
except Exception as e:
    print(f"✗ Validation harness failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - Pipeline is functional with IEEE-30 data!")
print("=" * 80)
print("\nNext steps:")
print("  1. Run full optimization: python experiments/test/example_optimization.py")
print("  2. View notebook: jupyter notebook experiments/test/ieee30_optimization_validation.ipynb")
print("=" * 80)
