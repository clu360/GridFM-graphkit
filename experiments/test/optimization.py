"""
Dispatch optimization problem definition.

Defines the full optimization problem including objective, constraints,
and optimization interfaces.
"""

import numpy as np
import scipy.optimize as opt
from typing import Dict, Callable, Optional, Tuple
from .scenario_data import ScenarioData
from .pv_dispatch import PVDispatchDecisionSpec
from .neural_solver import NeuralSolverWrapper
from .overload_penalty import OverloadPenaltyEvaluator


class DispatchOptimizationProblem:
    """
    Dispatch redispatch optimization problem using pretrained neural solver.
    
    Minimizes:
        J(u) = alpha * ||u - u_base||_2^2 + lambda * rho_overload(Phi_theta(u))
    
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
    overload_eval : OverloadPenaltyEvaluator
        Overload evaluator
    alpha : float
        Cost weight for baseline deviation
    lambda_ : float
        Cost weight for overload penalty
    """
    
    def __init__(
        self,
        scenario: ScenarioData,
        decision_spec,  # PVDispatchDecisionSpec or ExtendedDispatchSpec
        solver: NeuralSolverWrapper,
        overload_eval: OverloadPenaltyEvaluator,
        alpha: float = 1.0,
        lambda_: float = 1.0,
        beta: float = 0.0,  # Shedding cost weight (0 if no shedding)
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
        overload_eval : OverloadPenaltyEvaluator
            Overload penalty evaluator
        alpha : float, optional
            Weight for baseline deviation cost (default: 1.0)
        lambda_ : float, optional
            Weight for overload penalty (default: 1.0)
        beta : float, optional
            Weight for load shedding cost (default: 0.0 = no shedding)
        """
        self.scenario = scenario
        self.decision_spec = decision_spec
        self.solver = solver
        self.overload_eval = overload_eval
        self.alpha = alpha
        self.lambda_ = lambda_
        self.beta = beta
        
        # Detect if extended dispatch spec (with shedding)
        self.has_shedding = hasattr(decision_spec, 'shed_spec')
        
        # Track optimization history
        self.history = {
            "u": [],
            "cost": [],
            "deviation": [],
            "penalty": [],
            "shedding": [],
        }
    
    def cost_baseline_deviation(self, u: np.ndarray) -> float:
        """
        Compute baseline deviation cost: ||u - u_base||_2^2.
        
        Parameters
        ----------
        u : np.ndarray
            Decision vector, shape (n_pv,) or (n_pv + n_pq,)
        
        Returns
        -------
        float
            Cost
        """
        u_base = self.decision_spec.u_base
        return float(np.sum((u - u_base) ** 2))
    
    def cost_load_shedding(self, u: np.ndarray) -> float:
        """
        Compute load shedding cost: ||delta||_2^2.
        
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
        return float(np.sum(delta ** 2))
    
    def penalty_overload(self, u: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute overload penalty from neural solver prediction.
        
        Parameters
        ----------
        u : np.ndarray
            Decision vector, shape (n_pv,)
        
        Returns
        -------
        tuple
            (penalty: float, details: dict)
        """
        # Get predicted state
        pred = self.solver.predict_state(u)
        
        # Compute overload penalty
        penalty, details = self.overload_eval.compute_overload_penalty(
            pred["Vm"],
            pred["Va"],
        )
        
        return penalty, details
    
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
        
        # Baseline deviation cost
        cost_dev = self.cost_baseline_deviation(u)
        
        # Load shedding cost
        cost_shed = self.cost_load_shedding(u)
        
        # Overload penalty
        penalty, details = self.penalty_overload(u)
        
        # Total objective
        obj = self.alpha * cost_dev + self.beta * cost_shed + self.lambda_ * penalty
        
        if return_details:
            details_full = {
                "objective": obj,
                "cost_deviation": cost_dev,
                "cost_shedding": cost_shed,
                "penalty_overload": penalty,
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
        # Initial point: baseline
        u0 = self.decision_spec.u_base.copy()
        
        # Bounds
        bounds = list(zip(self.decision_spec.u_min, self.decision_spec.u_max))
        
        # Define callback to track history
        def callback(u_iter):
            obj, details = self.objective(u_iter, return_details=True)
            self.history["u"].append(u_iter.copy())
            self.history["cost"].append(details["cost_deviation"])
            self.history["penalty"].append(details["penalty_overload"])
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
        
        # Compute final metrics
        u_opt = result.x
        obj_opt, details_opt = self.objective(u_opt, return_details=True)
        
        return {
            "success": result.success,
            "u_opt": u_opt,
            "obj_opt": obj_opt,
            "cost_opt": details_opt["cost_deviation"],
            "penalty_opt": details_opt["penalty_overload"],
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
        obj_base, details_base = self.objective(u_base, return_details=True)
        pred_base = self.solver.predict_state(u_base)
        
        # Optimized metrics
        obj_opt, details_opt = self.objective(u_opt, return_details=True)
        pred_opt = self.solver.predict_state(u_opt)
        
        # Improvements
        obj_improvement = float(obj_base - obj_opt)
        penalty_improvement = details_base["penalty_overload"] - details_opt["penalty_overload"]
        
        return {
            "baseline": {
                "u": u_base,
                "objective": obj_base,
                "cost": details_base["cost_deviation"],
                "penalty": details_base["penalty_overload"],
                "n_overloaded": details_base["n_overloaded_lines"],
                "max_loading": details_base["max_loading"],
            },
            "optimized": {
                "u": u_opt,
                "objective": obj_opt,
                "cost": details_opt["cost_deviation"],
                "penalty": details_opt["penalty_overload"],
                "n_overloaded": details_opt["n_overloaded_lines"],
                "max_loading": details_opt["max_loading"],
            },
            "improvement": {
                "objective": obj_improvement,
                "objective_pct": 100 * obj_improvement / abs(obj_base),
                "penalty": penalty_improvement,
                "penalty_pct": 100 * penalty_improvement / abs(details_base["penalty_overload"]),
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
            "cost": [],
            "penalty": [],
            "objective": [],
        }
