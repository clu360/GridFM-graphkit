from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .decision_vector import FirstPassDecisionVector
from .wildfire_risk import compute_grouped_wildfire_risk
from .wildfire_scenario import WildfireScenario


def compute_first_pass_objective_components(
    u: np.ndarray,
    decision_vector: FirstPassDecisionVector,
    state: Dict,
    wildfire: WildfireScenario,
    lambda_R: float,
    lambda_L: float,
    normalize_terms: bool = True,
    risk_normalizer: float = 1.0,
    load_shedding_normalizer: float = 1.0,
    line_impact: np.ndarray | None = None,
) -> Tuple[float, Dict]:
    risk_total, line_df, group_df = compute_grouped_wildfire_risk(
        np.asarray(state["loading_ratio"], dtype=float),
        wildfire,
        impact=line_impact,
    )
    load_shedding = decision_vector.load_shedding(u)
    equal_bus_load_shedding = decision_vector.equal_bus_load_shedding(u)
    unserved_demand_mw = decision_vector.unserved_demand(u)
    gen_move = decision_vector.normalized_generator_movement(u)
    risk_scale = max(float(risk_normalizer), 1e-12) if normalize_terms else 1.0
    shedding_scale = max(float(load_shedding_normalizer), 1e-12) if normalize_terms else 1.0
    normalized_risk = risk_total / risk_scale
    normalized_load_shedding = load_shedding / shedding_scale
    risk_objective_term = lambda_R * normalized_risk
    load_shedding_objective_term = lambda_L * normalized_load_shedding
    objective = risk_objective_term + load_shedding_objective_term
    components = {
        "objective_total": float(objective),
        "wildfire_group_risk": float(risk_total),
        "load_shedding": float(load_shedding),
        "load_shedding_metric": "demand_weighted_fraction",
        "equal_bus_load_shedding": float(equal_bus_load_shedding),
        "unserved_demand_mw": float(unserved_demand_mw),
        "normalized_wildfire_group_risk": float(normalized_risk),
        "normalized_load_shedding": float(normalized_load_shedding),
        "risk_objective_term": float(risk_objective_term),
        "load_shedding_objective_term": float(load_shedding_objective_term),
        "risk_normalizer": float(risk_scale),
        "load_shedding_normalizer": float(shedding_scale),
        "objective_terms_normalized": bool(normalize_terms),
        "generator_movement": float(gen_move),
        "generator_movement_objective_weight": 0.0,
        "max_loading_ratio": float(state["max_loading_ratio"]),
        "max_voltage": float(state["max_voltage"]),
        "min_voltage": float(state["min_voltage"]),
        "num_nan": int(state["num_nan"]),
        "num_inf": int(state["num_inf"]),
        "risk_by_line": line_df,
        "risk_by_group": group_df,
    }
    return float(objective), components
