from __future__ import annotations

from typing import Dict

import numpy as np

from experiments.test.wildfire_tests.gridfm_support.overload_penalty import OverloadPenaltyEvaluator
from experiments.test.wildfire_tests.gridfm_support.scenario_data import ScenarioData


def compute_line_loading_ratios(
    scenario: ScenarioData,
    vm: np.ndarray,
    va: np.ndarray,
    standard_rate_a_mva: float = 100.0,
) -> np.ndarray:
    evaluator = OverloadPenaltyEvaluator(
        scenario,
        sn_mva=scenario.sn_mva,
        standard_rate_a_mva=standard_rate_a_mva,
    )
    return evaluator.compute_loading(vm, va)


def extract_state_quantities(
    scenario: ScenarioData,
    prediction: Dict[str, np.ndarray],
    standard_rate_a_mva: float = 100.0,
) -> Dict[str, np.ndarray | float | bool | int]:
    vm = np.asarray(prediction["Vm"], dtype=float)
    va = np.asarray(prediction["Va"], dtype=float)
    loading = compute_line_loading_ratios(
        scenario,
        vm,
        va,
        standard_rate_a_mva=standard_rate_a_mva,
    )
    all_arrays = [np.asarray(v, dtype=float) for v in prediction.values()] + [loading]
    num_nan = int(sum(np.isnan(arr).sum() for arr in all_arrays))
    num_inf = int(sum(np.isinf(arr).sum() for arr in all_arrays))
    rate_a = getattr(scenario, "rate_a", None)
    if rate_a is None:
        apparent_flow_proxy = loading * float(standard_rate_a_mva)
    else:
        rate = np.asarray(rate_a, dtype=float)
        apparent_flow_proxy = np.zeros_like(loading, dtype=float)
        valid = np.isfinite(rate) & (rate > 0.0)
        apparent_flow_proxy[valid] = loading[valid] * rate[valid]
    is_self_loop = getattr(scenario, "is_self_loop", np.zeros(len(loading), dtype=bool))
    mapping_status = getattr(scenario, "branch_mapping_status", np.asarray(["unknown"] * len(loading), dtype=object))
    impossible_voltage = bool(np.nanmin(vm) < 0.0 or np.nanmax(vm) > 2.0)
    return {
        "Vm": vm,
        "Va": va,
        "loading_ratio": loading,
        "apparent_flow_proxy": apparent_flow_proxy,
        "prediction_has_nan": bool(num_nan > 0),
        "prediction_has_inf": bool(num_inf > 0),
        "num_nan": num_nan,
        "num_inf": num_inf,
        "max_voltage": float(np.nanmax(vm)),
        "min_voltage": float(np.nanmin(vm)),
        "max_loading_ratio": float(np.nanmax(loading)),
        "num_lines": int(len(loading)),
        "num_self_loop_edges": int(np.sum(np.asarray(is_self_loop, dtype=bool))),
        "num_mapped_physical_edges": int(np.sum(np.asarray(mapping_status, dtype=object) == "mapped")),
        "impossible_voltage": impossible_voltage,
        "state_extraction_passed": bool(num_nan == 0 and num_inf == 0 and len(loading) > 0),
    }


def compare_states(baseline: Dict, perturbed: Dict) -> Dict[str, float | bool]:
    delta_v = np.asarray(perturbed["Vm"]) - np.asarray(baseline["Vm"])
    delta_a = np.asarray(perturbed["Va"]) - np.asarray(baseline["Va"])
    delta_l = np.asarray(perturbed["loading_ratio"]) - np.asarray(baseline["loading_ratio"])
    return {
        "norm_delta_voltage": float(np.linalg.norm(delta_v)),
        "norm_delta_angle": float(np.linalg.norm(delta_a)),
        "norm_delta_branch_loading": float(np.linalg.norm(delta_l)),
        "max_abs_delta_loading": float(np.max(np.abs(delta_l))) if len(delta_l) else 0.0,
        "nonzero_response": bool(
            np.linalg.norm(delta_v) > 1e-12
            or np.linalg.norm(delta_a) > 1e-12
            or np.linalg.norm(delta_l) > 1e-12
        ),
    }
