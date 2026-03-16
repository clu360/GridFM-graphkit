"""
Validation helpers for the Improved Optimization workflow.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .selection import TargetedDispatchDecisionSpec


def validate_selected_buses(spec: TargetedDispatchDecisionSpec) -> Dict[str, bool]:
    scenario = spec.scenario
    return {
        "generators_are_pv": bool(np.all(scenario.PV_mask[spec.selected_generator_buses])) if spec.n_generators else True,
        "loads_are_pq": bool(np.all(scenario.PQ_mask[spec.selected_load_buses])) if spec.n_loads else True,
    }


def validate_bounds(spec: TargetedDispatchDecisionSpec) -> Dict[str, bool]:
    return {
        "generator_bounds_valid": bool(np.all(spec.delta_pg_min <= spec.delta_pg_max)) if spec.n_generators else True,
        "load_bounds_valid": bool(np.all(spec.shed_min <= spec.shed_max)) if spec.n_loads else True,
        "baseline_in_bounds": bool(np.all(spec.u_base >= spec.u_min) and np.all(spec.u_base <= spec.u_max)),
    }


def validate_objective_value(value: float) -> Dict[str, bool]:
    return {"objective_is_finite": bool(np.isfinite(value))}


def validate_prediction_dict(pred: Dict[str, np.ndarray], num_buses: int) -> Dict[str, bool]:
    checks = {}
    for key in ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]:
        checks[f"{key}_shape_ok"] = pred[key].shape == (num_buses,)
        checks[f"{key}_finite"] = bool(np.all(np.isfinite(pred[key])))
    return checks


def validate_run(spec: TargetedDispatchDecisionSpec, pred: Dict[str, np.ndarray], objective_value: float) -> Dict[str, bool]:
    report = {}
    report.update(validate_selected_buses(spec))
    report.update(validate_bounds(spec))
    report.update(validate_prediction_dict(pred, spec.scenario.num_buses))
    report.update(validate_objective_value(objective_value))
    report["all_passed"] = all(report.values())
    return report
