"""
Dispatch optimization problem definition.

Defines the full optimization problem including objective, constraints,
and optimization interfaces.
"""

import numpy as np
import scipy.optimize as opt
from typing import Dict, Tuple
from .scenario_data import ScenarioData
from .neural_solver import NeuralSolverWrapper
from .wildfire_penalty import WildfirePenaltyEvaluator


class DispatchOptimizationProblem:
    """
    Dispatch redispatch optimization problem using pretrained neural solver.
    
    Minimizes:
        J(u) = lambda_gen * ||u_pg - u_pg,base||_2^2
             + lambda_shed * sum(u_delta)
             + lambda_wf * wildfire_penalty(Phi_theta(u))
    
    Subject to:
        u_min <= u <= u_max
        Phi_theta(u) is the neural solver prediction
    
    Attributes
    ----------
    scenario : ScenarioData
        Canonical scenario data
    decision_spec : PVDispatchDecisionSpec
        Decision variable spec for PV-bus Pg
    solver : NeuralSolverWrapper
        Neural solver wrapper
    wildfire_eval : WildfirePenaltyEvaluator
        Wildfire penalty evaluator
    """
    
    def __init__(
        self,
        scenario: ScenarioData,
        decision_spec,  # PVDispatchDecisionSpec or ExtendedDispatchSpec
        solver: NeuralSolverWrapper,
        wildfire_eval: WildfirePenaltyEvaluator,
        lambda_gen: float = 1.0,
        lambda_shed: float = 50.0,
        lambda_wf: float = 10.0,
    ):
        """
        Initialize optimization problem.
        
        Parameters
        ----------
        scenario : ScenarioData
            Canonical scenario data
        decision_spec : PVDispatchDecisionSpec or ExtendedDispatchSpec
            Decision variable specification
        solver : NeuralSolverWrapper
            Neural solver wrapper (GNN or GPS)
        wildfire_eval : WildfirePenaltyEvaluator
            Wildfire penalty evaluator
        lambda_gen : float, optional
            Weight for generator deviation cost
        lambda_shed : float, optional
            Weight for load shedding cost
        lambda_wf : float, optional
            Weight for wildfire penalty
        """
        self.scenario = scenario
        self.decision_spec = decision_spec
        self.solver = solver
        self.wildfire_eval = wildfire_eval
        self.lambda_gen = lambda_gen
        self.lambda_shed = lambda_shed
        self.lambda_wf = lambda_wf
        
        # Detect if extended dispatch spec (with shedding)
        self.has_shedding = hasattr(decision_spec, 'shed_spec')
        
        # Track optimization history
        self.history = {
            "u": [],
            "generator_deviation": [],
            "load_shedding": [],
            "wildfire": [],
            "objective": [],
        }
    
    def cost_generator_deviation(self, u: np.ndarray) -> float:
        """
        Compute PV generator deviation cost.
        
        Parameters
        ----------
        u : np.ndarray
            Decision vector, shape (n_pv,) or (n_pv + n_pq,)
        
        Returns
        -------
        float
            Generator deviation cost
        """
        if self.has_shedding:
            u_pg, _ = self.decision_spec.split_decision_vector(u)
            u_pg_base = self.decision_spec.pv_spec.u_base
            return float(np.sum((u_pg - u_pg_base) ** 2))

        u_base = self.decision_spec.u_base
        return float(np.sum((u - u_base) ** 2))
    
    def cost_load_shedding(self, u: np.ndarray) -> float:
        """
        Compute load shedding cost: sum(delta).
        
        Only applicable if decision spec has shedding (ExtendedDispatchSpec).
        
        Parameters
        ----------
        u : np.ndarray
            Extended decision vector, shape (n_pv + n_pq,)
        
        Returns
        -------
        float
            Shedding cost (0 if no shedding in spec)
        """
        if not self.has_shedding:
            return 0.0
        
        # Extract shedding part
        _, delta = self.decision_spec.split_decision_vector(u)
        return float(np.sum(delta))
    
    def penalty_wildfire(self, u: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute wildfire penalty from neural solver prediction.
        
        Parameters
        ----------
        u : np.ndarray
            Decision vector, shape (n_pv,)
        
        Returns
        -------
        tuple
            (wildfire_cost: float, details: dict)
        """
        pred = self.solver.predict_state(u)

        wildfire_cost, details = self.wildfire_eval.compute_wildfire_penalty(
            pred["Vm"],
            pred["Va"],
            return_details=True,
        )

        return wildfire_cost, details
    
    def objective(self, u: np.ndarray, return_details: bool = False) -> float:
        """
        Evaluate objective function.
        
        Parameters
        ----------
        u : np.ndarray
            Decision vector, shape (n_pv,) or (n_pv + n_pq,)
        return_details : bool, optional
            If True, return (objective, details) instead of just objective
        
        Returns
        -------
        float or tuple
            Objective value, or (objective, details) dict
        """
        # Check bounds
        if not self.decision_spec.check_bounds(u)[0]:
            # Return large penalty for infeasible
            return 1e10

        cost_gen = self.cost_generator_deviation(u)
        cost_shed = self.cost_load_shedding(u)
        wildfire_cost, details = self.penalty_wildfire(u)

        obj = (
            self.lambda_gen * cost_gen
            + self.lambda_shed * cost_shed
            + self.lambda_wf * wildfire_cost
        )
        
        if return_details:
            details_full = {
                "objective": obj,
                "total_cost": obj,
                "generator_deviation_cost": cost_gen,
                "load_shedding_cost": cost_shed,
                "wildfire_cost": wildfire_cost,
                # Backward-compatible aliases for existing scripts.
                "cost_deviation": cost_gen,
                "cost_shedding": cost_shed,
            }
            details_full.update(details)
            return obj, details_full
        
        return obj
    
    def objective_scipy(self, u: np.ndarray) -> float:
        """
        Objective for scipy optimizer (constraints already handled).
        
        Parameters
        ----------
        u : np.ndarray
            Decision vector, shape (n_pv,)
        
        Returns
        -------
        float
            Objective value
        """
        return self.objective(u, return_details=False)
    
    def gradient_numerical(
        self,
        u: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """
        Compute numerical gradient of objective.
        
        Parameters
        ----------
        u : np.ndarray
            Decision vector, shape (n_pv,)
        eps : float, optional
            Finite difference step size (default: 1e-5)
        
        Returns
        -------
        np.ndarray
            Gradient, shape (n_pv,)
        """
        grad = np.zeros(len(u))
        obj0 = self.objective(u)
        
        for i in range(len(u)):
            u_plus = u.copy()
            u_plus[i] += eps
            obj_plus = self.objective(u_plus)
            grad[i] = (obj_plus - obj0) / eps
        
        return grad
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get optimization bounds for decision variables.
        
        Returns
        -------
        tuple
            (u_min, u_max)
        """
        return self.decision_spec.u_min, self.decision_spec.u_max
    
    def optimize(
        self,
        method: str = "L-BFGS-B",
        maxiter: int = 100,
        verbose: bool = False,
    ) -> Dict:
        """
        Solve optimization problem.
        
        Parameters
        ----------
        method : str, optional
            Optimization method (default: "L-BFGS-B")
        maxiter : int, optional
            Maximum iterations (default: 100)
        verbose : bool, optional
            Print progress (default: False)
        
        Returns
        -------
        dict
            Optimization result with keys: success, u_opt, obj_opt, n_iter, etc.
        """
        self.reset_history()

        # Initial point: baseline
        u0 = self.decision_spec.u_base.copy()
        
        # Bounds
        bounds = list(zip(self.decision_spec.u_min, self.decision_spec.u_max))
        
        # Define callback to track history
        def callback(u_iter):
            obj, details = self.objective(u_iter, return_details=True)
            self.history["u"].append(u_iter.copy())
            self.history["generator_deviation"].append(details["generator_deviation_cost"])
            self.history["load_shedding"].append(details["load_shedding_cost"])
            self.history["wildfire"].append(details["wildfire_cost"])
            self.history["objective"].append(obj)
        
        # Solve
        result = opt.minimize(
            self.objective_scipy,
            u0,
            method=method,
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "disp": verbose,
            },
            callback=callback,
        )
        
        u_opt = result.x
        cost_gen_opt = self.cost_generator_deviation(u_opt)
        cost_shed_opt = self.cost_load_shedding(u_opt)
        wildfire_opt, wildfire_details_opt = self.penalty_wildfire(u_opt)
        obj_opt = (
            self.lambda_gen * cost_gen_opt
            + self.lambda_shed * cost_shed_opt
            + self.lambda_wf * wildfire_opt
        )
        details_opt = {
            "objective": obj_opt,
            "total_cost": obj_opt,
            "generator_deviation_cost": cost_gen_opt,
            "load_shedding_cost": cost_shed_opt,
            "wildfire_cost": wildfire_opt,
            "cost_deviation": cost_gen_opt,
            "cost_shedding": cost_shed_opt,
        }
        details_opt.update(wildfire_details_opt)
        
        return {
            "success": result.success,
            "u_opt": u_opt,
            "obj_opt": obj_opt,
            "cost_opt": cost_gen_opt,
            "shedding_cost_opt": cost_shed_opt,
            "wildfire_opt": wildfire_opt,
            "n_iter": result.nit,
            "message": result.message,
            "history": self.history,
            "details": details_opt,
        }
    
    def compare_baseline_vs_optimized(
        self,
        u_opt: np.ndarray,
    ) -> Dict:
        """
        Compare baseline and optimized dispatch.
        
        Parameters
        ----------
        u_opt : np.ndarray
            Optimized decision vector, shape (n_pv,)
        
        Returns
        -------
        dict
            Comparison metrics
        """
        u_base = self.decision_spec.u_base
        
        # Baseline metrics
        cost_base = self.cost_generator_deviation(u_base)
        cost_shed_base = self.cost_load_shedding(u_base)
        wildfire_base, wildfire_details_base = self.penalty_wildfire(u_base)
        obj_base = (
            self.lambda_gen * cost_base
            + self.lambda_shed * cost_shed_base
            + self.lambda_wf * wildfire_base
        )
        details_base = {
            "objective": obj_base,
            "total_cost": obj_base,
            "generator_deviation_cost": cost_base,
            "load_shedding_cost": cost_shed_base,
            "wildfire_cost": wildfire_base,
            "cost_deviation": cost_base,
            "cost_shedding": cost_shed_base,
        }
        details_base.update(wildfire_details_base)
        pred_base = self.solver.predict_state(u_base)
        
        cost_opt = self.cost_generator_deviation(u_opt)
        cost_shed_opt = self.cost_load_shedding(u_opt)
        wildfire_opt, wildfire_details_opt = self.penalty_wildfire(u_opt)
        obj_opt = (
            self.lambda_gen * cost_opt
            + self.lambda_shed * cost_shed_opt
            + self.lambda_wf * wildfire_opt
        )
        details_opt = {
            "objective": obj_opt,
            "total_cost": obj_opt,
            "generator_deviation_cost": cost_opt,
            "load_shedding_cost": cost_shed_opt,
            "wildfire_cost": wildfire_opt,
            "cost_deviation": cost_opt,
            "cost_shedding": cost_shed_opt,
        }
        details_opt.update(wildfire_details_opt)
        pred_opt = self.solver.predict_state(u_opt)
        
        obj_improvement = float(obj_base - obj_opt)
        wildfire_improvement = details_base["wildfire_cost"] - details_opt["wildfire_cost"]
        
        return {
            "baseline": {
                "u": u_base,
                "objective": obj_base,
                "generator_deviation_cost": details_base["generator_deviation_cost"],
                "load_shedding_cost": details_base["load_shedding_cost"],
                "wildfire_cost": details_base["wildfire_cost"],
                "cost": details_base["generator_deviation_cost"],
                "penalty": details_base["wildfire_cost"],
                "n_active_risk_branches": details_base["n_active_risk_branches"],
                "max_loading": details_base["max_loading"],
            },
            "optimized": {
                "u": u_opt,
                "objective": obj_opt,
                "generator_deviation_cost": details_opt["generator_deviation_cost"],
                "load_shedding_cost": details_opt["load_shedding_cost"],
                "wildfire_cost": details_opt["wildfire_cost"],
                "cost": details_opt["generator_deviation_cost"],
                "penalty": details_opt["wildfire_cost"],
                "n_active_risk_branches": details_opt["n_active_risk_branches"],
                "max_loading": details_opt["max_loading"],
            },
            "improvement": {
                "objective": obj_improvement,
                "objective_pct": 100 * obj_improvement / abs(obj_base) if obj_base != 0 else 0.0,
                "wildfire": wildfire_improvement,
                "wildfire_pct": 100 * wildfire_improvement / abs(details_base["wildfire_cost"])
                if details_base["wildfire_cost"] != 0
                else 0.0,
                # Backward-compatible aliases.
                "penalty": wildfire_improvement,
                "penalty_pct": 100 * wildfire_improvement / abs(details_base["wildfire_cost"])
                if details_base["wildfire_cost"] != 0
                else 0.0,
            },
            "predictions": {
                "baseline": pred_base,
                "optimized": pred_opt,
            },
        }
    
    def reset_history(self):
        """Reset optimization history."""
        self.history = {
            "u": [],
            "generator_deviation": [],
            "load_shedding": [],
            "wildfire": [],
            "objective": [],
        }
