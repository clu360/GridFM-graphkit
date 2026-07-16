from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .wildfire_scenario import WildfireScenario


def estimate_equal_weight_service(prediction: Dict[str, np.ndarray], scenario) -> float:
    """Estimate served load as an equal-bus alpha sum from predicted Pd."""
    pd_base = np.asarray(scenario.Pd_base, dtype=float)
    pd_pred = np.asarray(prediction["Pd"], dtype=float)
    alpha = np.ones_like(pd_base, dtype=float)
    load_mask = pd_base > 1e-12
    alpha[load_mask] = np.clip(pd_pred[load_mask] / pd_base[load_mask], 0.0, 1.0)
    return float(np.sum(alpha))


def compute_counterfactual_line_impacts(
    u: np.ndarray,
    scenario,
    runner,
    wildfire: WildfireScenario,
    current_prediction: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute I_l(u) as relative equal-weight served-load loss under line outage.

    This keeps line energization fixed in the optimizer. The line is removed only
    for a counterfactual consequence calculation:

        I_l(u) = max(0, S_current(u) - S_outage_l(u)) / max(S_current(u), eps)
    """
    if hasattr(scenario.edge_index, "shape"):
        num_lines = int(scenario.edge_index.shape[1])
    else:
        line_ids = [int(line_id) for group in wildfire.line_groups for line_id in group.line_ids]
        num_lines = max(line_ids, default=-1) + 1
    impacts = wildfire.impact_vector(num_lines)
    current_service = estimate_equal_weight_service(current_prediction, scenario)
    service_scale = max(current_service, 1e-12)
    line_ids = sorted({int(line_id) for group in wildfire.line_groups for line_id in group.line_ids})
    for line_id in line_ids:
        outage_prediction = runner.predict_with_line_outage(u, line_id)
        outage_service = estimate_equal_weight_service(outage_prediction, scenario)
        impacts[line_id] = max(0.0, current_service - outage_service) / service_scale
    return impacts


def compute_line_risk(
    loading_ratio: np.ndarray,
    wildfire: WildfireScenario,
    impact: np.ndarray | None = None,
) -> np.ndarray:
    loading = np.asarray(loading_ratio, dtype=float)
    hazard = wildfire.hazard_vector(len(loading))
    impact_vector = wildfire.impact_vector(len(loading)) if impact is None else np.asarray(impact, dtype=float)
    if impact_vector.shape != loading.shape:
        raise ValueError(f"impact has shape {impact_vector.shape}; expected {loading.shape}.")
    return hazard * np.square(loading) * impact_vector


def compute_operational_wildfire_exposure(
    loading_ratio: np.ndarray,
    p_env_by_line: Dict[int, float],
    z_by_line: Dict[int, int],
    candidate_line_ids,
) -> Tuple[float, Dict[int, float]]:
    """
    Compute loading-dependent wildfire exposure over the selected candidate scope.

    This revised true evaluation term intentionally excludes the historical
    impact/consequence score. Single-line service consequence belongs in proxy
    candidate generation, not in this GridFM-evaluated exposure index.
    """
    loading = np.asarray(loading_ratio, dtype=float)
    exposure_by_line: Dict[int, float] = {}
    total = 0.0
    for line_id in candidate_line_ids:
        line_id = int(line_id)
        if line_id < 0 or line_id >= len(loading):
            raise ValueError(f"line_id={line_id} is invalid for loading vector of length {len(loading)}.")
        z_l = int(z_by_line.get(line_id, 1))
        exposure = float(z_l * float(p_env_by_line.get(line_id, 0.0)) * loading[line_id] ** 2)
        exposure_by_line[line_id] = exposure
        total += exposure
    return float(total), exposure_by_line


def compute_grouped_wildfire_risk(
    loading_ratio: np.ndarray,
    wildfire: WildfireScenario,
    impact: np.ndarray | None = None,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    line_risk = compute_line_risk(loading_ratio, wildfire, impact=impact)
    impact_vector = wildfire.impact_vector(len(line_risk)) if impact is None else np.asarray(impact, dtype=float)
    line_rows = []
    group_rows = []
    for line_id, risk in enumerate(line_risk):
        line_rows.append(
            {
                "line_id": int(line_id),
                "loading_ratio": float(loading_ratio[line_id]),
                "hazard": float(wildfire.hazard_vector(len(line_risk))[line_id]),
                "impact": float(impact_vector[line_id]),
                "risk": float(risk),
            }
        )
    total = 0.0
    for group in wildfire.line_groups:
        group_risk = float(np.sum(line_risk[group.line_ids]))
        weighted = float(group.group_weight * group_risk)
        total += weighted
        group_rows.append(
            {
                "group_name": group.name,
                "line_ids": ",".join(str(i) for i in group.line_ids),
                "group_weight": float(group.group_weight),
                "raw_group_risk": group_risk,
                "weighted_group_risk": weighted,
            }
        )
    return float(total), pd.DataFrame(line_rows), pd.DataFrame(group_rows)


def risk_summary_dict(total: float, line_df: pd.DataFrame, group_df: pd.DataFrame) -> Dict:
    return {
        "total_grouped_wildfire_risk": float(total),
        "num_lines": int(len(line_df)),
        "num_groups": int(len(group_df)),
        "max_line_risk": float(line_df["risk"].max()) if len(line_df) else 0.0,
        "max_group_risk": float(group_df["weighted_group_risk"].max()) if len(group_df) else 0.0,
    }
