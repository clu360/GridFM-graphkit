"""
Wildfire-risk and branch-loading utilities for Improved Optimization.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from experiments.test.overload_penalty import OverloadPenaltyEvaluator
from experiments.test.scenario_data import ScenarioData
from .config import WildfireConfig


def reconstruct_branch_flows(
    scenario: ScenarioData,
    vm: np.ndarray,
    va: np.ndarray,
    standard_rate_a_mva: float = 100.0,
) -> Dict[str, np.ndarray]:
    evaluator = OverloadPenaltyEvaluator(
        scenario,
        sn_mva=scenario.sn_mva,
        standard_rate_a_mva=standard_rate_a_mva,
    )
    if_pu, it_pu = evaluator.compute_branch_currents_pu(vm, va)
    loading = evaluator.compute_loading(vm, va)
    return {
        "from_current_pu": if_pu,
        "to_current_pu": it_pu,
        "loading": loading,
    }


def compute_line_loading_ratios(
    scenario: ScenarioData,
    vm: np.ndarray,
    va: np.ndarray,
    standard_rate_a_mva: float = 100.0,
) -> np.ndarray:
    return reconstruct_branch_flows(scenario, vm, va, standard_rate_a_mva)["loading"]


def compute_line_weights(loading: np.ndarray, config: WildfireConfig) -> np.ndarray:
    if config.risk_weight_mode == "uniform":
        return np.ones_like(loading)
    if config.risk_weight_mode == "loading":
        return np.maximum(1.0, loading)
    raise ValueError(f"Unknown risk weight mode: {config.risk_weight_mode}")


def softplus_penalty(values: np.ndarray, alpha: float, threshold: np.ndarray) -> np.ndarray:
    shifted = alpha * (values - threshold)
    stable = np.log1p(np.exp(-np.abs(shifted))) + np.maximum(shifted, 0.0)
    return stable ** 2


def quadratic_threshold_penalty(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, values - threshold) ** 2


def compute_wildfire_penalty(
    loading: np.ndarray,
    line_weights: np.ndarray,
    threshold: np.ndarray,
    config: WildfireConfig,
    active_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    if active_mask is None:
        active_mask = np.ones_like(loading, dtype=bool)

    threshold_vec = np.broadcast_to(np.asarray(threshold, dtype=float), loading.shape)
    if config.use_softplus:
        penalty_terms = softplus_penalty(loading, config.softplus_alpha, threshold_vec)
    else:
        penalty_terms = quadratic_threshold_penalty(loading, threshold_vec)

    penalty_terms = penalty_terms * line_weights * active_mask.astype(float)
    return {
        "wildfire_cost": float(np.sum(penalty_terms)),
        "branch_risk_terms": penalty_terms,
        "active_mask": active_mask,
        "line_weights": line_weights,
        "thresholds": threshold_vec,
    }


def summarize_top_risky_lines(
    loading: np.ndarray,
    risk_terms: np.ndarray,
    line_weights: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "line_idx": np.arange(len(loading)),
            "loading": loading,
            "risk_term": risk_terms,
            "weight": line_weights,
        }
    )
    return df.sort_values(["risk_term", "loading"], ascending=False).head(top_n).reset_index(drop=True)
