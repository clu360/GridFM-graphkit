"""
Validation utilities for the predict-then-optimize pipeline.

Provides a validation harness to verify scenario integrity, solver behavior,
and branch-loading risk evaluation before optimization.
"""

import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from .scenario_data import ScenarioData
from .pv_dispatch import PVDispatchDecisionSpec
from .neural_solver import NeuralSolverWrapper
from .wildfire_penalty import WildfirePenaltyEvaluator
from .overload_penalty import OverloadPenaltyEvaluator


class PipelineValidationHarness:
    """
    Validation harness for the predict-then-optimize pipeline.
    
    Performs systematic checks before optimization to ensure:
    1. Models load correctly (GNN and GPS)
    2. Positional encodings are correct (pe_dim=20 for both)
    3. Solver produces reasonable predictions
    4. Branch-risk computation is consistent
    5. Pipeline components work together
    """
    
    @staticmethod
    def _is_bool_like(value) -> bool:
        return isinstance(value, (bool, np.bool_))

    @staticmethod
    def validate_scenario_structure(scenario: ScenarioData) -> Dict[str, bool]:
        """
        Validate that scenario has complete and consistent structure.
        
        Parameters
        ----------
        scenario : ScenarioData
            Scenario to validate
        
        Returns
        -------
        dict
            Validation results
        """
        checks = {}
        
        # Check dimensions
        checks["num_buses_consistent"] = len(scenario.bus_indices) == scenario.num_buses
        checks["Pd_shape"] = scenario.Pd_base.shape == (scenario.num_buses,)
        checks["Qd_shape"] = scenario.Qd_base.shape == (scenario.num_buses,)
        checks["Pg_shape"] = scenario.Pg_base.shape == (scenario.num_buses,)
        checks["Qg_shape"] = scenario.Qg_base.shape == (scenario.num_buses,)
        checks["Vm_shape"] = scenario.Vm_base.shape == (scenario.num_buses,)
        checks["Va_shape"] = scenario.Va_base.shape == (scenario.num_buses,)
        
        # Check bus types
        checks["PQ_shape"] = scenario.PQ_mask.shape == (scenario.num_buses,)
        checks["PV_shape"] = scenario.PV_mask.shape == (scenario.num_buses,)
        checks["REF_shape"] = scenario.REF_mask.shape == (scenario.num_buses,)
        
        # Check that bus types are mutually exclusive
        type_sum = (
            scenario.PQ_mask.astype(int)
            + scenario.PV_mask.astype(int)
            + scenario.REF_mask.astype(int)
        )
        checks["bus_types_mutually_exclusive"] = np.all(type_sum == 1)
        
        # Check at least one REF bus
        checks["has_ref_bus"] = np.sum(scenario.REF_mask) >= 1
        
        # Check edge data
        num_edges = scenario.edge_index.shape[1]
        checks["G_shape"] = scenario.G.shape == (num_edges,)
        checks["B_shape"] = scenario.B.shape == (num_edges,)
        
        # Check positional encoding
        checks["pe_pe_dim"] = scenario.pe.shape == (scenario.num_buses, 20)
        
        # Check mask
        checks["mask_shape"] = scenario.mask.shape == (scenario.num_buses, 6)
        
        # Check bounds consistency
        checks["Pg_bounds_tight"] = np.all(scenario.Pg_min <= scenario.Pg_base)
        checks["Pg_bounds_tight_2"] = np.all(scenario.Pg_base <= scenario.Pg_max)
        
        return checks
    
    @staticmethod
    def validate_decision_spec(
        decision_spec: PVDispatchDecisionSpec,
    ) -> Dict[str, bool]:
        """
        Validate decision variable specification.
        
        Parameters
        ----------
        decision_spec : PVDispatchDecisionSpec
            Decision spec to validate
        
        Returns
        -------
        dict
            Validation results
        """
        checks = {}
        
        # Check PV buses identified
        checks["n_pv_positive"] = decision_spec.n_pv > 0
        
        # Check baseline
        checks["u_base_shape"] = decision_spec.u_base.shape == (decision_spec.n_pv,)
        
        # Check bounds
        checks["u_min_shape"] = decision_spec.u_min.shape == (decision_spec.n_pv,)
        checks["u_max_shape"] = decision_spec.u_max.shape == (decision_spec.n_pv,)
        checks["bounds_consistent"] = np.all(decision_spec.u_min <= decision_spec.u_max)
        
        # Check baseline within bounds
        checks["u_base_within_bounds"] = np.all(
            (decision_spec.u_min <= decision_spec.u_base)
            & (decision_spec.u_base <= decision_spec.u_max)
        )
        
        return checks
    
    @staticmethod
    def validate_solver(
        solver: NeuralSolverWrapper,
        decision_spec: PVDispatchDecisionSpec,
    ) -> Dict[str, any]:
        """
        Validate neural solver wrapper.
        
        Parameters
        ----------
        solver : NeuralSolverWrapper
            Solver wrapper to validate
        decision_spec : PVDispatchDecisionSpec
            Decision spec for baseline dispatch
        
        Returns
        -------
        dict
            Validation results
        """
        checks = {}
        
        # Check model type
        checks["model_type_valid"] = solver.model_type in ["gnn", "gps"]
        
        # Check model is in eval mode
        checks["model_in_eval"] = not solver.model.training
        
        # Try prediction on baseline
        try:
            u_base = decision_spec.u_base
            pred = solver.predict_state(u_base)
            checks["prediction_successful"] = True
            checks["prediction_keys"] = set(pred.keys()) == {"Pd", "Qd", "Pg", "Qg", "Vm", "Va"}
            
            # Check prediction shapes
            num_buses = solver.scenario.num_buses
            for key in pred.keys():
                checks[f"pred_{key}_shape"] = pred[key].shape == (num_buses,)
            
            # Check predictions are finite
            all_finite = all(np.isfinite(pred[key]).all() for key in pred.keys())
            checks["predictions_finite"] = all_finite
            
            # Check voltage magnitudes are reasonable (0.8 to 1.2 pu typical)
            vm_reasonable = np.all((pred["Vm"] > 0.5) & (pred["Vm"] < 1.5))
            checks["Vm_reasonable_range"] = vm_reasonable
            
        except Exception as e:
            checks["prediction_successful"] = False
            checks["prediction_error"] = str(e)
        
        # Validate baseline comparison
        try:
            errors = solver.validate_baseline()
            checks["baseline_validation_successful"] = True
            checks["baseline_errors"] = errors
        except Exception as e:
            checks["baseline_validation_successful"] = False
            checks["baseline_error_msg"] = str(e)
        
        return checks
    
    @staticmethod
    def validate_risk_evaluator(
        risk_eval,
    ) -> Dict[str, any]:
        """
        Validate a branch-risk evaluator.
        
        Parameters
        ----------
        risk_eval : OverloadPenaltyEvaluator or WildfirePenaltyEvaluator
            Risk evaluator to validate.
        
        Returns
        -------
        dict
            Validation results
        """
        checks = {}
        
        # Check admittance matrices exist
        checks["Yf_exists"] = risk_eval.scenario.Yf is not None
        checks["Yt_exists"] = risk_eval.scenario.Yt is not None
        
        # Try baseline evaluation
        try:
            baseline_eval = risk_eval.evaluate_baseline()
            checks["baseline_eval_successful"] = True
            
            # Check all keys present
            expected_keys = {
                "total_penalty",
                "n_overloaded_lines",
                "max_loading",
                "mean_loading",
            }
            checks["evaluation_keys_complete"] = expected_keys.issubset(
                set(baseline_eval.keys())
            )
            
            # Check values are reasonable
            checks["penalty_nonnegative"] = baseline_eval["total_penalty"] >= 0
            checks["n_overloaded_reasonable"] = baseline_eval["n_overloaded_lines"] >= 0
            checks["max_loading_positive"] = baseline_eval["max_loading"] > 0
            
        except Exception as e:
            checks["baseline_eval_successful"] = False
            checks["baseline_eval_error"] = str(e)
        
        return checks
    
    @staticmethod
    def full_validation(
        scenario: ScenarioData,
        decision_spec: PVDispatchDecisionSpec,
        solver_gnn: NeuralSolverWrapper,
        solver_gps: Optional[NeuralSolverWrapper] = None,
        overload_eval: Optional[OverloadPenaltyEvaluator] = None,
        wildfire_eval: Optional[WildfirePenaltyEvaluator] = None,
    ) -> Dict:
        """
        Run full validation harness on all pipeline components.
        
        Parameters
        ----------
        scenario : ScenarioData
            Scenario to validate
        decision_spec : PVDispatchDecisionSpec
            Decision spec to validate
        solver_gnn : NeuralSolverWrapper
            GNN solver to validate
        solver_gps : NeuralSolverWrapper, optional
            GPS solver to validate
        overload_eval : OverloadPenaltyEvaluator, optional
            Legacy alias for the branch-risk evaluator.
        wildfire_eval : WildfirePenaltyEvaluator, optional
            Current wildfire evaluator to validate.
        
        Returns
        -------
        dict
            Complete validation report
        """
        report = {}
        
        # Validate scenario
        report["scenario"] = PipelineValidationHarness.validate_scenario_structure(scenario)
        
        # Validate decision spec
        report["decision_spec"] = PipelineValidationHarness.validate_decision_spec(
            decision_spec
        )
        
        # Validate solvers
        report["solver_gnn"] = PipelineValidationHarness.validate_solver(solver_gnn, decision_spec)
        
        if solver_gps is not None:
            report["solver_gps"] = PipelineValidationHarness.validate_solver(
                solver_gps,
                decision_spec,
            )
        
        risk_eval = wildfire_eval if wildfire_eval is not None else overload_eval
        if risk_eval is not None:
            report["risk_evaluator"] = PipelineValidationHarness.validate_risk_evaluator(
                risk_eval
            )
        
        # Summary
        all_passes = all(
            all(v for k, v in checks.items() if PipelineValidationHarness._is_bool_like(v))
            for section, checks in report.items()
            if isinstance(checks, dict)
        )
        
        report["all_passed"] = all_passes
        
        return report
    
    @staticmethod
    def print_validation_report(report: Dict, verbose: bool = True):
        """
        Print formatted validation report.
        
        Parameters
        ----------
        report : dict
            Validation report from full_validation
        verbose : bool, optional
            Print detailed results (default: True)
        """
        print("=" * 80)
        print("PIPELINE VALIDATION REPORT")
        print("=" * 80)
        
        all_passed = report.pop("all_passed", False)
        
        for section, checks in report.items():
            if not isinstance(checks, dict):
                continue
            
            print(f"\n[{section.upper()}]")
            
            bool_checks = {
                k: bool(v)
                for k, v in checks.items()
                if PipelineValidationHarness._is_bool_like(v)
            }
            n_pass = sum(bool_checks.values())
            n_total = len(bool_checks)
            
            status = "PASS" if n_pass == n_total else "FAIL"
            print(f"  Status: {status} ({n_pass}/{n_total})")
            
            if verbose:
                for check_name, result in bool_checks.items():
                    symbol = "[OK]" if result else "[FAIL]"
                    print(f"    {symbol} {check_name}")
            
            # Print non-bool details
            for key, val in checks.items():
                if not PipelineValidationHarness._is_bool_like(val) and verbose:
                    print(f"    {key}: {val}")
        
        print("\n" + "=" * 80)
        overall_status = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
        print(f"Overall: {overall_status}")
        print("=" * 80)

    validate_overload_evaluator = validate_risk_evaluator
