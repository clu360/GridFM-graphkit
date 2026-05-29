from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import scipy.optimize as opt

from .config import FirstPassConfig
from .decision_vector import FirstPassDecisionVector
from .objective import compute_first_pass_objective_components
from .state_extraction import extract_state_quantities
from .wildfire_risk import compute_counterfactual_line_impacts
from .wildfire_scenario import WildfireScenario


@dataclass
class ObjectiveTrace:
    rows: List[Dict] = field(default_factory=list)
    failure_count: int = 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class FirstPassOptimizationProblem:
    def __init__(
        self,
        scenario,
        decision_vector: FirstPassDecisionVector,
        runner,
        wildfire: WildfireScenario,
        config: FirstPassConfig,
    ):
        self.scenario = scenario
        self.decision_vector = decision_vector
        self.runner = runner
        self.wildfire = wildfire
        self.config = config
        self.trace = ObjectiveTrace()

    def evaluate(self, u: np.ndarray, record: bool = False) -> Dict:
        feasible, message = self.decision_vector.check_bounds(u)
        if not feasible:
            if self.config.objective.strict:
                raise ValueError(message)
            return {
                "objective_total": float(self.config.objective.invalid_prediction_penalty),
                "wildfire_group_risk": float("nan"),
                "load_shedding": float("nan"),
                "normalized_wildfire_group_risk": float("nan"),
                "normalized_load_shedding": float("nan"),
                "risk_objective_term": float("nan"),
                "load_shedding_objective_term": float("nan"),
                "generator_movement": float("nan"),
                "max_loading_ratio": float("nan"),
                "max_voltage": float("nan"),
                "min_voltage": float("nan"),
                "num_nan": 1,
                "num_inf": 0,
                "message": message,
            }
        try:
            prediction = self.runner.predict(u)
            state = extract_state_quantities(
                self.scenario,
                prediction,
                standard_rate_a_mva=self.config.wildfire.standard_rate_a_mva,
            )
            if state["prediction_has_nan"] or state["prediction_has_inf"]:
                raise FloatingPointError("Invalid prediction contains NaN or inf.")
            line_impact = compute_counterfactual_line_impacts(
                u,
                self.scenario,
                self.runner,
                self.wildfire,
                prediction,
            )
            _, components = compute_first_pass_objective_components(
                u,
                self.decision_vector,
                state,
                self.wildfire,
                self.config.objective.lambda_R,
                self.config.objective.lambda_L,
                normalize_terms=self.config.objective.normalize_terms,
                risk_normalizer=self.config.objective.risk_normalizer,
                load_shedding_normalizer=self.config.objective.load_shedding_normalizer,
                line_impact=line_impact,
            )
            components["message"] = "ok"
        except Exception as exc:
            if self.config.objective.strict:
                raise
            self.trace.failure_count += 1
            components = {
                "objective_total": float(self.config.objective.invalid_prediction_penalty),
                "wildfire_group_risk": float("nan"),
                "load_shedding": float("nan"),
                "normalized_wildfire_group_risk": float("nan"),
                "normalized_load_shedding": float("nan"),
                "risk_objective_term": float("nan"),
                "load_shedding_objective_term": float("nan"),
                "generator_movement": float("nan"),
                "max_loading_ratio": float("nan"),
                "max_voltage": float("nan"),
                "min_voltage": float("nan"),
                "num_nan": 1,
                "num_inf": 0,
                "message": f"invalid_prediction: {exc}",
            }
        if record:
            delta_pg, alpha = self.decision_vector.split_decision_vector(u)
            self.trace.rows.append(
                {
                    "eval_idx": len(self.trace.rows),
                    "objective_total": float(components["objective_total"]),
                    "wildfire_group_risk": float(components["wildfire_group_risk"]),
                    "load_shedding": float(components["load_shedding"]),
                    "equal_bus_load_shedding": float(components.get("equal_bus_load_shedding", np.nan)),
                    "unserved_demand_mw": float(components.get("unserved_demand_mw", np.nan)),
                    "normalized_wildfire_group_risk": float(components["normalized_wildfire_group_risk"]),
                    "normalized_load_shedding": float(components["normalized_load_shedding"]),
                    "risk_objective_term": float(components["risk_objective_term"]),
                    "load_shedding_objective_term": float(components["load_shedding_objective_term"]),
                    "generator_movement": float(components["generator_movement"]),
                    "max_loading_ratio": float(components["max_loading_ratio"]),
                    "mean_alpha": float(np.mean(alpha)) if len(alpha) else 1.0,
                    "max_abs_delta_pg": float(np.max(np.abs(delta_pg))) if len(delta_pg) else 0.0,
                    "message": components.get("message", ""),
                }
            )
        return components

    def objective(self, u: np.ndarray) -> float:
        return float(self.evaluate(u, record=True)["objective_total"])

    def _minimize_from(self, u0: np.ndarray) -> Dict:
        self.trace = ObjectiveTrace()
        u0 = np.asarray(u0, dtype=float)
        bounds = list(zip(self.decision_vector.u_min, self.decision_vector.u_max))
        baseline = self.evaluate(self.decision_vector.u_base, record=False)
        result = opt.minimize(
            self.objective,
            u0,
            method=self.config.optimizer.method,
            bounds=bounds,
            options={
                "maxiter": self.config.optimizer.maxiter,
                "ftol": self.config.optimizer.ftol,
                "gtol": self.config.optimizer.gtol,
                "eps": self.config.optimizer.eps,
                "disp": self.config.optimizer.disp,
            },
        )
        final = self.evaluate(result.x, record=False)
        return {
            "success": bool(result.success),
            "message": str(result.message),
            "n_iter": int(result.nit),
            "num_objective_evals": int(result.nfev),
            "u_initial": u0,
            "u_final": result.x,
            "baseline": baseline,
            "final": final,
            "trace": self.trace,
            "raw_result": result,
        }

    def optimize(self, u0: np.ndarray | None = None) -> Dict:
        u0 = self.decision_vector.u_base.copy() if u0 is None else np.asarray(u0, dtype=float)
        return self._minimize_from(u0)

    def optimize_multistart(self, starts: Sequence[np.ndarray]) -> Dict:
        start_results = []
        best_result = None
        for start_idx, start in enumerate(starts):
            start = np.asarray(start, dtype=float)
            start_components = self.evaluate(start, record=False)
            result = self._minimize_from(start)
            result["start_idx"] = int(start_idx)
            result["start_objective"] = float(start_components["objective_total"])
            result["start_components"] = start_components
            start_results.append(result)
            if best_result is None or result["final"]["objective_total"] < best_result["final"]["objective_total"]:
                best_result = result
        if best_result is None:
            raise ValueError("optimize_multistart requires at least one start.")
        return {
            **best_result,
            "multistart_results": start_results,
            "num_starts": int(len(start_results)),
            "best_start_idx": int(best_result["start_idx"]),
        }
