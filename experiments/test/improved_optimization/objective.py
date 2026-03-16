"""
Objective decomposition utilities for Improved Optimization.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_generator_deviation_cost(
    delta_pg: np.ndarray,
    generator_weights: Optional[np.ndarray] = None,
) -> float:
    if generator_weights is None:
        generator_weights = np.ones_like(delta_pg)
    return float(np.sum(generator_weights * (delta_pg ** 2)))


def compute_load_shedding_cost(
    shed: np.ndarray,
    shed_weights: Optional[np.ndarray] = None,
) -> float:
    if shed_weights is None:
        shed_weights = np.ones_like(shed)
    return float(np.sum(shed_weights * shed))


def compute_total_objective(
    generator_cost: float,
    shedding_cost: float,
    wildfire_cost: float,
    lambda_g: float,
    lambda_s: float,
    lambda_w: float,
) -> float:
    return float(lambda_g * generator_cost + lambda_s * shedding_cost + lambda_w * wildfire_cost)


def compute_objective_breakdown(
    delta_pg: np.ndarray,
    shed: np.ndarray,
    wildfire_cost: float,
    lambda_g: float,
    lambda_s: float,
    lambda_w: float,
    generator_weights: Optional[np.ndarray] = None,
    shed_weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    generator_cost = compute_generator_deviation_cost(delta_pg, generator_weights)
    shedding_cost = compute_load_shedding_cost(shed, shed_weights)
    total = compute_total_objective(generator_cost, shedding_cost, wildfire_cost, lambda_g, lambda_s, lambda_w)
    return {
        "objective": total,
        "generator_deviation_cost": generator_cost,
        "load_shedding_cost": shedding_cost,
        "wildfire_cost": float(wildfire_cost),
        "total_shed_mw": float(np.sum(shed)),
        "decision_norm": float(np.linalg.norm(np.hstack([delta_pg, shed]))),
    }
