"""
Targeted optimization problem for Improved Optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import scipy.optimize as opt

from experiments.test.neural_solver import NeuralSolverWrapper
from experiments.test.scenario_data import ScenarioData
from .config import ExperimentConfig
from .objective import compute_objective_breakdown
from .selection import TargetedDispatchDecisionSpec
from .wildfire_metrics import compute_line_loading_ratios, compute_line_weights, compute_wildfire_penalty


@dataclass
class OptimizationRecord:
    u: List[np.ndarray] = field(default_factory=list)
    objective: List[float] = field(default_factory=list)
    generator_deviation_cost: List[float] = field(default_factory=list)
    load_shedding_cost: List[float] = field(default_factory=list)
    wildfire_cost: List[float] = field(default_factory=list)
    total_shed_mw: List[float] = field(default_factory=list)
    max_risky_line_loading: List[float] = field(default_factory=list)
    decision_norm: List[float] = field(default_factory=list)


class ImprovedDispatchOptimizationProblem:
    def __init__(
        self,
        scenario: ScenarioData,
        decision_spec: TargetedDispatchDecisionSpec,
        solver: NeuralSolverWrapper,
        config: ExperimentConfig,
    ):
        self.scenario = scenario
        self.decision_spec = decision_spec
        self.solver = solver
        self.config = config
        self.history = OptimizationRecord()

        baseline_pred = self.solver.predict_state(self.decision_spec.u_base)
        baseline_loading = compute_line_loading_ratios(
            self.scenario,
            baseline_pred["Vm"],
            baseline_pred["Va"],
            standard_rate_a_mva=self.config.wildfire.standard_rate_a_mva,
        )
        self.line_weights = compute_line_weights(baseline_loading, self.config.wildfire)
        risky_mask = baseline_loading >= (self.config.wildfire.wildfire_threshold - self.config.wildfire.threshold_buffer)
        if int(np.sum(risky_mask)) < self.config.wildfire.top_n_risky_lines:
            top_idx = np.argsort(baseline_loading)[::-1][: self.config.wildfire.top_n_risky_lines]
            risky_mask[top_idx] = True
        self.risky_mask = risky_mask

    def decode_decision(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.decision_spec.split_decision_vector(u)

    def evaluate_state(self, u: np.ndarray) -> Dict[str, np.ndarray]:
        return self.solver.predict_state(u)

    def compute_objective_breakdown(self, u: np.ndarray) -> Dict:
        feasible, message = self.decision_spec.check_bounds(u)
        if not feasible:
            return {"objective": 1e10, "feasible": False, "message": message}

        delta_pg, shed = self.decode_decision(u)
        pred = self.evaluate_state(u)
        loading = compute_line_loading_ratios(
            self.scenario,
            pred["Vm"],
            pred["Va"],
            standard_rate_a_mva=self.config.wildfire.standard_rate_a_mva,
        )
        wildfire = compute_wildfire_penalty(
            loading,
            self.line_weights,
            self.config.wildfire.wildfire_threshold,
            self.config.wildfire,
            active_mask=self.risky_mask,
        )
        breakdown = compute_objective_breakdown(
            delta_pg,
            shed,
            wildfire["wildfire_cost"],
            self.config.objective.lambda_g,
            self.config.objective.lambda_s,
            self.config.objective.lambda_w,
            generator_weights=self.decision_spec.generator_weights,
            shed_weights=self.decision_spec.shed_weights,
        )
        breakdown.update(
            {
                "predictions": pred,
                "loading": loading,
                "feasible": True,
                "message": message,
                "max_risky_line_loading": float(np.max(loading[self.risky_mask])) if np.any(self.risky_mask) else float(np.max(loading)),
                "n_risky_lines": int(np.sum(self.risky_mask)),
            }
        )
        breakdown.update(wildfire)
        return breakdown

    def objective(self, u: np.ndarray) -> float:
        return float(self.compute_objective_breakdown(u)["objective"])

    def _record(self, u: np.ndarray, breakdown: Dict) -> None:
        self.history.u.append(np.asarray(u, dtype=float).copy())
        self.history.objective.append(float(breakdown["objective"]))
        self.history.generator_deviation_cost.append(float(breakdown["generator_deviation_cost"]))
        self.history.load_shedding_cost.append(float(breakdown["load_shedding_cost"]))
        self.history.wildfire_cost.append(float(breakdown["wildfire_cost"]))
        self.history.total_shed_mw.append(float(breakdown["total_shed_mw"]))
        self.history.max_risky_line_loading.append(float(breakdown["max_risky_line_loading"]))
        self.history.decision_norm.append(float(breakdown["decision_norm"]))

    def optimize(self) -> Dict:
        self.history = OptimizationRecord()
        u0 = self.decision_spec.u_base.copy()
        bounds = list(zip(self.decision_spec.u_min, self.decision_spec.u_max))

        def callback(u_iter: np.ndarray) -> None:
            self._record(u_iter, self.compute_objective_breakdown(u_iter))

        result = opt.minimize(
            self.objective,
            u0,
            method=self.config.optimization.method,
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": self.config.optimization.maxiter,
                "ftol": self.config.optimization.ftol,
                "gtol": self.config.optimization.gtol,
                "eps": self.config.optimization.eps,
                "disp": self.config.optimization.disp,
            },
        )

        final_breakdown = self.compute_objective_breakdown(result.x)
        if not self.history.objective:
            self._record(result.x, final_breakdown)

        return {
            "success": bool(result.success),
            "message": result.message,
            "n_iter": int(result.nit),
            "u_opt": result.x,
            "history": self.history,
            "raw_result": result,
            "final": final_breakdown,
            "baseline": self.compute_objective_breakdown(u0),
        }
