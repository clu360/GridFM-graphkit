"""
Wildfire-aware line loading penalty for surrogate dispatch optimization.

This module reuses the branch loading computation from the overload evaluator,
then applies a wildfire activation threshold and branch-specific risk weights.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np

from .overload_penalty import OverloadPenaltyEvaluator
from .scenario_data import ScenarioData


def compute_wildfire_penalty(
    pred_vm: np.ndarray,
    pred_va: np.ndarray,
    branch_data,
    risk_weights: Optional[np.ndarray] = None,
    risk_thresholds: Optional[Union[float, np.ndarray]] = None,
    return_details: bool = False,
):
    """
    Compute wildfire penalty from predicted voltages.

    Parameters
    ----------
    pred_vm : np.ndarray
        Predicted bus voltage magnitudes.
    pred_va : np.ndarray
        Predicted bus voltage angles.
    branch_data : WildfirePenaltyEvaluator or ScenarioData
        Source of branch loading data.
    risk_weights : np.ndarray, optional
        Per-branch wildfire weights. Defaults to all ones.
    risk_thresholds : float or np.ndarray, optional
        Per-branch wildfire activation thresholds. Defaults to 0.7.
    return_details : bool, optional
        If True, also return diagnostics.
    """
    if isinstance(branch_data, WildfirePenaltyEvaluator):
        evaluator = branch_data
    elif isinstance(branch_data, ScenarioData):
        evaluator = WildfirePenaltyEvaluator(branch_data)
    else:
        raise TypeError(
            "branch_data must be a WildfirePenaltyEvaluator or ScenarioData instance."
        )

    return evaluator.compute_wildfire_penalty(
        pred_vm,
        pred_va,
        risk_weights=risk_weights,
        risk_thresholds=risk_thresholds,
        return_details=return_details,
    )


class WildfirePenaltyEvaluator(OverloadPenaltyEvaluator):
    """
    Evaluate wildfire-aware line risk from predicted bus state.

    Uses the same branch loading calculation as OverloadPenaltyEvaluator, but
    replaces the thermal overload threshold of 1.0 with configurable wildfire
    activation thresholds eta_l and branch risk weights w_l.
    """

    def __init__(
        self,
        scenario: ScenarioData,
        sn_mva: float = 100.0,
        standard_rate_a_mva: float = 100.0,
        risk_weights: Optional[np.ndarray] = None,
        risk_thresholds: Optional[Union[float, np.ndarray]] = 0.7,
    ):
        super().__init__(
            scenario=scenario,
            sn_mva=sn_mva,
            standard_rate_a_mva=standard_rate_a_mva,
        )
        n_branches = int(self.scenario.edge_index.shape[1])
        self.risk_weights = self._coerce_branch_vector(risk_weights, n_branches, 1.0)
        self.risk_thresholds = self._coerce_branch_vector(risk_thresholds, n_branches, 0.7)

    @staticmethod
    def _coerce_branch_vector(
        values: Optional[Union[float, np.ndarray]],
        n_branches: int,
        default: float,
    ) -> np.ndarray:
        if values is None:
            return np.full(n_branches, default, dtype=float)

        if np.isscalar(values):
            return np.full(n_branches, float(values), dtype=float)

        arr = np.asarray(values, dtype=float)
        if arr.shape != (n_branches,):
            raise ValueError(
                f"Expected branch vector of shape ({n_branches},), got {arr.shape}."
            )
        return arr

    def compute_wildfire_penalty(
        self,
        pred_vm: np.ndarray,
        pred_va: np.ndarray,
        risk_weights: Optional[np.ndarray] = None,
        risk_thresholds: Optional[Union[float, np.ndarray]] = None,
        return_details: bool = False,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Compute wildfire penalty from branch loading.
        """
        loading = self.compute_loading(pred_vm, pred_va)
        n_branches = loading.shape[0]

        weights = self._coerce_branch_vector(risk_weights, n_branches, 1.0)
        thresholds = self._coerce_branch_vector(risk_thresholds, n_branches, 0.7)

        active_risk_mask = loading > thresholds
        exceedance = np.maximum(0.0, loading - thresholds)
        branch_risk_terms = weights * (exceedance ** 2)
        wildfire_cost = float(np.sum(branch_risk_terms))

        details = {
            "branch_loading": loading,
            "branch_risk_terms": branch_risk_terms,
            "active_risk_mask": active_risk_mask,
            "risk_weights": weights,
            "risk_thresholds": thresholds,
            "n_active_risk_branches": int(np.sum(active_risk_mask)),
            "n_overloaded_lines": int(np.sum(active_risk_mask)),
            "max_loading": float(np.max(loading)),
            "mean_loading": float(np.mean(loading)),
            "max_branch_risk": float(np.max(branch_risk_terms)),
            "wildfire_cost": wildfire_cost,
        }

        if return_details:
            return wildfire_cost, details
        return wildfire_cost, details

    def evaluate(
        self,
        pred_vm: np.ndarray,
        pred_va: np.ndarray,
    ) -> Dict[str, float]:
        wildfire_cost, details = self.compute_wildfire_penalty(
            pred_vm,
            pred_va,
            return_details=True,
        )
        return {
            "wildfire_cost": wildfire_cost,
            "total_penalty": wildfire_cost,
            "n_active_risk_branches": details["n_active_risk_branches"],
            "n_overloaded_lines": details["n_active_risk_branches"],
            "max_loading": details["max_loading"],
            "mean_loading": details["mean_loading"],
            "max_branch_risk": details["max_branch_risk"],
        }

    def evaluate_baseline(self) -> Dict[str, float]:
        baseline = self.scenario.get_baseline_state()
        return self.evaluate(baseline["Vm"], baseline["Va"])
