from __future__ import annotations

from typing import Dict

import numpy as np


def validate_baseline_prediction(state: Dict) -> Dict:
    return {
        "prediction_has_nan": bool(state["prediction_has_nan"]),
        "prediction_has_inf": bool(state["prediction_has_inf"]),
        "state_extraction_passed": bool(state["state_extraction_passed"]),
        "passed": bool(
            not state["prediction_has_nan"]
            and not state["prediction_has_inf"]
            and state["state_extraction_passed"]
        ),
    }


def validate_optimization_result(
    baseline_components: Dict,
    final_components: Dict,
    decision_vector,
    u_final: np.ndarray,
    optimizer_success: bool,
    optimizer_message: str,
) -> Dict:
    bounds_ok, bounds_message = decision_vector.check_bounds(u_final)
    _, alpha = decision_vector.split_decision_vector(u_final)
    full_alpha = decision_vector.full_alpha(u_final)
    selected_mean_alpha = float(np.mean(alpha)) if len(alpha) else 1.0
    selected_min_alpha = float(np.min(alpha)) if len(alpha) else 1.0
    mean_alpha = float(np.mean(full_alpha))
    min_alpha = float(np.min(full_alpha))
    delta_pg, _ = decision_vector.split_decision_vector(u_final)
    return {
        "objective_reduced": bool(final_components["objective_total"] < baseline_components["objective_total"]),
        "wildfire_risk_reduced": bool(final_components["wildfire_group_risk"] < baseline_components["wildfire_group_risk"]),
        "mean_alpha": mean_alpha,
        "min_alpha": min_alpha,
        "selected_mean_alpha": selected_mean_alpha,
        "selected_min_alpha": selected_min_alpha,
        "mean_alpha_ok": bool(mean_alpha >= 0.95),
        "bounds_ok": bool(bounds_ok),
        "bounds_message": bounds_message,
        "final_prediction_no_nan": bool(final_components["num_nan"] == 0 and final_components["num_inf"] == 0),
        "optimizer_success": bool(optimizer_success),
        "optimizer_message": str(optimizer_message),
        "max_abs_delta_pg": float(np.max(np.abs(delta_pg))) if len(delta_pg) else 0.0,
    }
