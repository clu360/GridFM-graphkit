"""Common Stage K risk, service, PAC, and reference discrepancies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math


def compute_r_base(
    line_ids: Iterable[int], p_env: Mapping[int, float], baseline_loading: Mapping[int, float]
) -> float:
    value = sum(float(p_env[int(i)]) * float(baseline_loading[int(i)]) ** 2 for i in line_ids)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"R_base must be positive and finite, got {value}")
    return float(value)


def compute_r_norm(
    line_ids: Iterable[int],
    offline_line_ids: Iterable[int],
    p_env: Mapping[int, float],
    loading: Mapping[int, float],
    r_base: float,
) -> float:
    offline = {int(value) for value in offline_line_ids}
    raw = sum(
        float(p_env[int(i)]) * float(loading[int(i)]) ** 2
        for i in line_ids
        if int(i) not in offline
    )
    if r_base <= 0.0 or not math.isfinite(r_base):
        raise ValueError("R_base must be positive and finite")
    return float(raw / r_base)


def compute_l_shed(pd_by_load: Mapping[int, float], alpha_effective: Mapping[int, float]) -> float:
    total = sum(max(0.0, float(value)) for value in pd_by_load.values())
    if total <= 0.0:
        raise ValueError("total requested active demand must be positive")
    served = sum(max(0.0, float(pd_by_load[i])) * float(alpha_effective[i]) for i in pd_by_load)
    return float(max(0.0, min(1.0, 1.0 - served / total)))


def compute_j_trade(lambda_r: float, r_norm: float, l_shed: float) -> float:
    value = float(lambda_r)
    if not 0.0 <= value <= 1.0:
        raise ValueError("lambda_r must be in [0, 1]")
    return float(value * r_norm + (1.0 - value) * l_shed)


def compute_pac_total(pac_operational: float, pac_ac: float, pac_model: float = 0.0) -> float:
    return float(pac_operational + pac_ac)


def compute_j_gridsfm(j_trade: float, pac_total: float, rho_phys: float = 2.0) -> float:
    return float(j_trade + rho_phys * pac_total)


def reference_deltas(
    *, native_r_norm: float, reference_a_r_norm: float, native_j_trade: float,
    reference_a_j_trade: float, selected_service: float, reference_b_max_service: float,
) -> dict[str, float]:
    return {
        "delta_r_a": float(native_r_norm - reference_a_r_norm),
        "delta_j_a": float(native_j_trade - reference_a_j_trade),
        "delta_s_b": float(reference_b_max_service - selected_service),
    }
