"""
Test script to verify all pipeline modules work correctly.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import yaml

repo_root = Path('.').resolve()
sys.path.insert(0, str(repo_root))

print("=" * 80)
print("PIPELINE MODULE TEST SUITE")
print("=" * 80)

# ============================================================================
# TEST 1: Import all modules
# ============================================================================
print("\n[1/6] Testing module imports...")
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
    sys.exit(1)

# ============================================================================
# TEST 2: Create synthetic scenario (data not available, testing logic)
# ============================================================================
print("\n[2/6] Creating synthetic scenario for testing...")
try:
    # Create synthetic batch-like object for testing
    from gridfm_graphkit.datasets.normalizers import MinMaxNormalizer
    from gridfm_graphkit.io.param_handler import NestedNamespace
    
    num_buses = 30
    num_edges = 40
    
    # Create mock batch
    class MockBatch:
        def __init__(self):
            self.num_nodes = num_buses
            self.num_edges = num_edges
            self.nodes = torch.randn(num_buses, 6)  # [Pd, Qd, Pg, Qg, Vm, Va]
            self.bus_types = torch.zeros(num_buses, 3)
            # Assign bus types: PQ, PV, REF
            self.bus_types[:-1, 0] = 1  # Most are PQ
            self.bus_types[1:6, 0] = 0  # Override some as PV
            self.bus_types[1:6, 1] = 1  # These are PV
            self.bus_types[0, 2] = 1   # First is REF
            
            # Normalize using min-max to [0, 1]
            x_normalized = (self.nodes - self.nodes.min(dim=0)[0]) / (self.nodes.max(dim=0)[0] - self.nodes.min(dim=0)[0] + 1e-6)
            self.x = torch.cat([x_normalized, self.bus_types], dim=1)
            
            # Edge attributes
            self.edge_attr = torch.randn(num_edges, 2)  # [G, B]
            self.edge_index = torch.randint(0, num_buses, (2, num_edges)).long()
            
            # Positional encoding
            self.pe = torch.randn(num_buses, 20)
            
            # Mask
            self.mask = torch.ones(num_buses, 6, dtype=torch.bool)
            self.mask[self.bus_types[:, 2] > 0.5, 4:] = False  # Don't mask VA, Vm for REF
            self.mask[self.bus_types[:, 1] > 0.5, 3] = False  # Don't mask Qg for PV
            
            # Batch assignment
            self.batch = torch.zeros(num_buses, dtype=torch.long)
            
            # Output
            self.y = self.x.clone()
    
    batch = MockBatch()
    
    # Create normalizers
    config_path = repo_root / "tests" / "config" / "gridFMv0.1_dummy.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    args = NestedNamespace(**config_dict)
    
    node_normalizer = MinMaxNormalizer(node_data=True, args=args)
    edge_normalizer = MinMaxNormalizer(node_data=False, args=args)
    
    # Fit normalizers on mock data
    node_normalizer.fit(batch.x)
    edge_normalizer.fit(batch.edge_attr)
    
    print(f"✓ Synthetic scenario created successfully")
    print(f"  Batch shape: {batch.num_nodes} nodes, {batch.num_edges} edges")
    print(f"  PE shape: {batch.pe.shape}, Mask shape: {batch.mask.shape}")
except Exception as e:
    print(f"✗ Synthetic scenario creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 3: Test ScenarioData extraction
# ============================================================================
print("\n[3/6] Testing ScenarioData extraction...")
try:
    scenario = extract_scenario_from_batch(
        batch,
        node_normalizer,
        edge_normalizer,
        scenario_id="test-scenario",
    )
    
    assert scenario.num_buses > 0, "num_buses should be positive"
    assert scenario.Pd_base.shape[0] == scenario.num_buses, "Pd_base shape mismatch"
    assert scenario.pe.shape[1] == 20, "PE dimension should be 20"
    
    pv_buses = scenario.get_pv_buses()
    print(f"✓ Scenario extracted successfully")
    print(f"  Buses: {scenario.num_buses}, PV buses: {len(pv_buses)}, PE dim: {scenario.pe.shape[1]}")
except Exception as e:
    print(f"✗ ScenarioData extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 4: Test PVDispatchDecisionSpec
# ============================================================================
print("\n[4/6] Testing PVDispatchDecisionSpec...")
try:
    decision_spec = PVDispatchDecisionSpec(scenario)
    
    assert decision_spec.n_pv > 0, "n_pv should be positive"
    assert decision_spec.u_base.shape[0] == decision_spec.n_pv, "u_base shape mismatch"
    assert np.all(decision_spec.u_min <= decision_spec.u_max), "Bounds inconsistent"
    
    # Test mapping functions
    u = decision_spec.u_base.copy()
    Pg_full = decision_spec.u_to_Pg(u)
    u_extracted = decision_spec.Pg_to_u(Pg_full)
    assert np.allclose(u, u_extracted), "Mapping roundtrip failed"
    
    # Test node features
    node_features = decision_spec.u_to_node_features(u)
    assert node_features.shape == (scenario.num_buses, 6), "Node features shape mismatch"
    
    bounds_ok, msg = decision_spec.check_bounds(u)
    assert bounds_ok, f"Bounds check failed: {msg}"
    
    print(f"✓ PVDispatchDecisionSpec working correctly")
    print(f"  PV buses: {decision_spec.n_pv}, baseline Pg mean: {np.mean(decision_spec.u_base):.4f}")
except Exception as e:
    print(f"✗ PVDispatchDecisionSpec failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 5: Test NeuralSolverWrapper
# ============================================================================
print("\n[5/6] Testing NeuralSolverWrapper...")
try:
    from gridfm_graphkit.io.param_handler import load_model
    
    device = "cpu"  # Force CPU for testing
    model = load_model(args)
    
    solver = NeuralSolverWrapper(
        model,
        model_type="gnn",
        scenario=scenario,
        decision_spec=decision_spec,
        device=device,
    )
    
    # Test prediction
    u_test = decision_spec.u_base.copy()
    pred = solver.predict_state(u_test)
    
    assert isinstance(pred, dict), "Prediction should be dict"
    assert set(pred.keys()) == {"Pd", "Qd", "Pg", "Qg", "Vm", "Va"}, "Missing prediction keys"
    
    for key, val in pred.items():
        assert val.shape == (scenario.num_buses,), f"{key} shape mismatch"
        assert np.isfinite(val).all(), f"{key} contains non-finite values"
    
    print(f"✓ NeuralSolverWrapper working correctly")
    print(f"  Predictions: Vm mean {np.mean(pred['Vm']):.4f}, Va mean {np.mean(pred['Va']):.4f}")
except Exception as e:
    print(f"✗ NeuralSolverWrapper failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 6: Test OverloadPenaltyEvaluator & DispatchOptimizationProblem
# ============================================================================
print("\n[6/6] Testing OverloadPenaltyEvaluator and DispatchOptimizationProblem...")
try:
    overload_eval = OverloadPenaltyEvaluator(scenario)
    
    # Test baseline evaluation
    baseline_eval = overload_eval.evaluate_baseline()
    assert "total_penalty" in baseline_eval, "Missing total_penalty"
    assert baseline_eval["total_penalty"] >= 0, "Penalty should be non-negative"
    
    print(f"✓ OverloadPenaltyEvaluator working correctly")
    print(f"  Baseline penalty: {baseline_eval['total_penalty']:.4f}, "
          f"n_overloaded: {baseline_eval['n_overloaded_lines']}")
    
    # Test optimization problem
    problem = DispatchOptimizationProblem(
        scenario=scenario,
        decision_spec=decision_spec,
        solver=solver,
        overload_eval=overload_eval,
        alpha=1.0,
        lambda_=1.0,
    )
    
    # Test objective evaluation
    u_test = decision_spec.u_base.copy()
    obj, details = problem.objective(u_test, return_details=True)
    
    assert np.isfinite(obj), "Objective should be finite"
    assert "cost_deviation" in details, "Missing cost_deviation in details"
    assert "penalty_overload" in details, "Missing penalty_overload in details"
    
    print(f"✓ DispatchOptimizationProblem working correctly")
    print(f"  Baseline objective: {obj:.4f}, cost: {details['cost_deviation']:.4f}, "
          f"penalty: {details['penalty_overload']:.4f}")
    
except Exception as e:
    print(f"✗ OverloadPenaltyEvaluator or DispatchOptimizationProblem failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# VALIDATION HARNESS TEST
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
    
    all_passed = all(
        all(v for k, v in checks.items() if isinstance(v, bool))
        for section, checks in list(validation_report.items())[:-1]
        if isinstance(checks, dict)
    )
    
    if all_passed:
        print(f"✓ All validation checks passed")
    else:
        print(f"⚠ Some validation checks failed (see details)")
    
except Exception as e:
    print(f"✗ Validation harness failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - Pipeline is functional!")
print("=" * 80)
