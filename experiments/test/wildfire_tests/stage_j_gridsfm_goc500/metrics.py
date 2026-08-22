"""Metric and objective helpers for Stage J."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from .schemas import PacWeights, StageJObjective, UnitCompatibilityReport


def require_positive_ratea(ratea_by_line: Mapping[int, float], line_ids: Iterable[int], *, epsilon: float = 1e-9) -> None:
    """Raise if any risk branch lacks a finite positive thermal rating."""

    for line_id in line_ids:
        value = float(ratea_by_line[int(line_id)])
        if not np.isfinite(value) or value <= epsilon:
            raise ValueError(f"rateA for line {line_id} must be finite and positive, got {value}")


def compute_ac_loading_two_ended(
    p_from: Mapping[int, float],
    q_from: Mapping[int, float],
    p_to: Mapping[int, float],
    q_to: Mapping[int, float],
    ratea_by_line: Mapping[int, float],
    line_ids: Iterable[int],
) -> dict[int, float]:
    """Compute AC/GridSFM branch loading using the larger apparent flow at either end."""

    ids = [int(line_id) for line_id in line_ids]
    require_positive_ratea(ratea_by_line, ids)
    loading: dict[int, float] = {}
    for line_id in ids:
        s_from = float(np.hypot(float(p_from[line_id]), float(q_from[line_id])))
        s_to = float(np.hypot(float(p_to[line_id]), float(q_to[line_id])))
        loading[line_id] = max(s_from, s_to) / float(ratea_by_line[line_id])
    return loading


def compute_r_base(
    p_env_by_line: Mapping[int, float],
    baseline_loading_by_line: Mapping[int, float],
    risk_line_ids: Iterable[int],
    *,
    epsilon: float = 1e-6,
) -> float:
    """Compute and validate the shared Stage J risk denominator."""

    total = 0.0
    for line_id in risk_line_ids:
        lid = int(line_id)
        p_env = float(p_env_by_line[lid])
        loading = float(baseline_loading_by_line[lid])
        if not np.isfinite(p_env) or not np.isfinite(loading):
            raise ValueError(f"nonfinite R_base input for line {lid}")
        total += p_env * loading * loading
    if total <= epsilon:
        raise ValueError(f"R_base must exceed epsilon={epsilon}, got {total}")
    return float(total)


def compute_j_trade(lambda_r: float, r_norm: float, l_shed_total: float) -> float:
    """Compute the weighted wildfire/service tradeoff objective."""

    if lambda_r < 0.0 or lambda_r > 1.0:
        raise ValueError(f"lambda_r must be in [0, 1], got {lambda_r}")
    return float(lambda_r * r_norm + (1.0 - lambda_r) * l_shed_total)


def compute_pac_total(
    pac_operational: float,
    pac_ac: float,
    pac_model: float,
    weights: PacWeights,
) -> float:
    """Compute PAC_total with frozen calibrated weights."""

    weights.validate()
    components = {
        "PAC_operational": pac_operational,
        "PAC_AC": pac_ac,
        "PAC_model": pac_model,
    }
    for name, value in components.items():
        if value < 0 or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    return float(weights.w_op * pac_operational + weights.w_ac * pac_ac + weights.w_model * pac_model)


def compute_j_total_sfm(
    *,
    lambda_r: float,
    r_norm: float,
    l_shed_total: float,
    pac_operational: float,
    pac_ac: float,
    pac_model: float,
    weights: PacWeights,
) -> StageJObjective:
    """Compute the GridSFM candidate-selection merit and all saved components."""

    j_trade = compute_j_trade(lambda_r, r_norm, l_shed_total)
    pac_total = compute_pac_total(pac_operational, pac_ac, pac_model, weights)
    return StageJObjective(
        r_norm=float(r_norm),
        l_shed_total=float(l_shed_total),
        j_trade=j_trade,
        pac_operational=float(pac_operational),
        pac_ac=float(pac_ac),
        pac_model=float(pac_model),
        pac_total=pac_total,
        j_total=float(j_trade + weights.rho_phys * pac_total),
    )


def require_unit_compatibility(report: UnitCompatibilityReport) -> None:
    """Hard unit gate before using flow/rating ratios in wildfire metrics."""

    report.require_compatible()
