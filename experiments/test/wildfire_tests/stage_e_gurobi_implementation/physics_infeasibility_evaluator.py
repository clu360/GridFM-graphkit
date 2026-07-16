from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from experiments.test.wildfire_tests.shared.state_extraction import extract_state_quantities
from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
from experiments.test.wildfire_tests.gridfm_support.branch_metadata import expand_to_physical_line_ids
from experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps import (
    demand_weighted_load_shed_from_prediction,
    scenario_with_line_outages,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    normalize_true_exposure,
)


DEFAULT_PHYSICS_WEIGHTS = {
    "voltage_limits": 1.0,
    "thermal_limits": 1.0,
    "generator_limits": 1.0,
    "island_source_feasibility": 1.0,
    "p_balance": 0.0,
    "q_balance": 0.0,
    "branch_flow_consistency": 0.0,
}


@dataclass
class PhysicsEvaluationResult:
    row: Dict
    trace: List[Dict] = field(default_factory=list)


def _edge_array(scenario) -> np.ndarray:
    return scenario.edge_index.cpu().numpy() if hasattr(scenario.edge_index, "cpu") else np.asarray(scenario.edge_index)


def _as_csv_ints(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in sorted(int(item) for item in values))


def _active_line_ids(num_lines: int, removed_line_ids: Iterable[int], scenario=None) -> List[int]:
    removed_values = expand_to_physical_line_ids(scenario, removed_line_ids) if scenario is not None else removed_line_ids
    removed = {int(line_id) for line_id in removed_values}
    return [line_id for line_id in range(int(num_lines)) if line_id not in removed]


def source_less_island_buses(scenario, removed_line_ids: Iterable[int]) -> List[int]:
    edge_array = _edge_array(scenario)
    graph = nx.Graph()
    graph.add_nodes_from(range(int(scenario.num_buses)))
    removed = {int(line_id) for line_id in expand_to_physical_line_ids(scenario, removed_line_ids)}
    for line_id, (src, dst) in enumerate(edge_array.T):
        if int(line_id) in removed:
            continue
        graph.add_edge(int(src), int(dst))

    pg_base = np.asarray(scenario.Pg_base, dtype=float)
    source_buses = set(int(bus) for bus in np.where(pg_base > 1e-9)[0])
    if hasattr(scenario, "get_pv_buses"):
        source_buses.update(int(bus) for bus in scenario.get_pv_buses())
    if hasattr(scenario, "get_ref_bus"):
        ref_bus = scenario.get_ref_bus()
        if ref_bus is not None:
            source_buses.add(int(ref_bus))

    source_less: set[int] = set()
    for component in nx.connected_components(graph):
        component_set = {int(bus) for bus in component}
        if not component_set.intersection(source_buses):
            source_less.update(component_set)
    return sorted(source_less)


def recourse_bounds_with_island_limits(decision_vector, source_less_buses: Iterable[int]) -> tuple[np.ndarray, np.ndarray, int]:
    lower = np.asarray(decision_vector.u_min, dtype=float).copy()
    upper = np.asarray(decision_vector.u_max, dtype=float).copy()
    source_less = {int(bus) for bus in source_less_buses}
    offset = int(decision_vector.n_generators)
    forced = 0
    for idx, bus in enumerate(np.asarray(decision_vector.selected_load_buses, dtype=int)):
        if int(bus) in source_less:
            decision_idx = offset + idx
            lower[decision_idx] = 0.0
            upper[decision_idx] = 0.0
            forced += 1
    return lower, upper, forced


def risk_only_stage_c_scores(
    candidate_line_ids: Iterable[int],
    p_env_by_line: Dict[int, float],
    baseline_loading: np.ndarray,
) -> Dict[int, float]:
    loading = np.asarray(baseline_loading, dtype=float)
    return {
        int(line_id): float(p_env_by_line.get(int(line_id), 0.0)) * float(loading[int(line_id)]) ** 2
        for line_id in candidate_line_ids
    }


def _full_loading_vector(num_lines: int, keep_mask: List[int], active_loading: np.ndarray, removed_line_ids: Iterable[int]) -> np.ndarray:
    full = np.zeros(int(num_lines), dtype=float)
    active_loading = np.asarray(active_loading, dtype=float)
    for active_idx, original_line_id in enumerate(keep_mask):
        if active_idx < len(active_loading):
            full[int(original_line_id)] = float(active_loading[active_idx])
    for line_id in removed_line_ids:
        full[int(line_id)] = 0.0
    return full


def _predict_topology_state(model_context: dict, u: np.ndarray, removed_line_ids: List[int]) -> tuple[Dict, Dict, List[int]]:
    config = model_context["config"]
    scenario = model_context["scenario"]
    runner = model_context["runner"]
    num_lines = int(_edge_array(scenario).shape[1])
    if removed_line_ids:
        with scenario_with_line_outages(scenario, removed_line_ids) as keep_mask:
            original_yf = getattr(scenario, "Yf", None)
            original_yt = getattr(scenario, "Yt", None)
            try:
                scenario.Yf = None
                scenario.Yt = None
                prediction = runner.predict(u)
                active_state = extract_state_quantities(
                    scenario,
                    prediction,
                    standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
                )
            finally:
                scenario.Yf = original_yf
                scenario.Yt = original_yt
        loading = _full_loading_vector(num_lines, keep_mask, active_state["loading_ratio"], removed_line_ids)
        state = dict(active_state)
        state["loading_ratio"] = loading
        state["apparent_flow_proxy"] = loading * float(config.wildfire.standard_rate_a_mva)
        state["num_lines"] = num_lines
        state["max_loading_ratio"] = float(np.nanmax(loading)) if len(loading) else 0.0
        return prediction, state, [int(line_id) for line_id in keep_mask]

    prediction = runner.predict(u)
    state = extract_state_quantities(
        scenario,
        prediction,
        standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
    )
    return prediction, state, list(range(num_lines))


def _generator_limit_penalty(decision_vector, u: np.ndarray) -> float:
    delta_pg, _alpha = decision_vector.split_decision_vector(u)
    if not len(delta_pg):
        return 0.0
    buses = np.asarray(decision_vector.selected_generator_buses, dtype=int)
    pg = np.asarray(decision_vector.scenario.Pg_base, dtype=float)[buses] + np.asarray(delta_pg, dtype=float)
    lower = np.asarray(decision_vector.scenario.Pg_min, dtype=float)[buses]
    upper = np.asarray(decision_vector.scenario.Pg_max, dtype=float)[buses]
    scale = np.maximum(np.abs(upper - lower), 1.0)
    low = np.maximum(0.0, lower - pg) / scale
    high = np.maximum(0.0, pg - upper) / scale
    return float(np.mean(low**2 + high**2))


def _physics_components(
    model_context: dict,
    u: np.ndarray,
    state: Dict,
    active_line_ids: Iterable[int],
    source_less_buses: Iterable[int],
) -> Dict[str, float]:
    decision_vector = model_context["decision_vector"]
    scenario = model_context["scenario"]
    vm = np.asarray(state.get("Vm", []), dtype=float)
    if len(vm):
        voltage = float(np.mean(np.maximum(0.0, 0.95 - vm) ** 2 + np.maximum(0.0, vm - 1.05) ** 2))
    else:
        voltage = 0.0

    loading = np.asarray(state.get("loading_ratio", []), dtype=float)
    active = [int(line_id) for line_id in active_line_ids if int(line_id) < len(loading)]
    thermal = float(np.mean(np.maximum(0.0, loading[active] - 1.0) ** 2)) if active else 0.0

    full_alpha = decision_vector.full_alpha(u)
    demand = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    total_demand = max(float(np.sum(demand)), 1e-12)
    island_buses = [int(bus) for bus in source_less_buses]
    if island_buses:
        island = float(np.sum(demand[island_buses] * full_alpha[island_buses] ** 2) / total_demand)
    else:
        island = 0.0

    return {
        "voltage_limits": voltage,
        "thermal_limits": thermal,
        "generator_limits": _generator_limit_penalty(decision_vector, u),
        "island_source_feasibility": island,
        "p_balance": 0.0,
        "q_balance": 0.0,
        "branch_flow_consistency": 0.0,
    }


def weighted_pac_total(components: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    weights = DEFAULT_PHYSICS_WEIGHTS if weights is None else weights
    return float(sum(float(weights.get(name, 0.0)) * float(value) for name, value in components.items()))


def evaluate_topology_with_physics_recourse(
    model_context: dict,
    candidate_line_ids: List[int],
    p_env_by_line: Dict[int, float],
    baseline_R_raw: float,
    shutoff_line_ids: Iterable[int],
    lambda_R: float,
    lambda_L: float,
    rho_phys: float,
    stage: str,
    case_name: str,
    lambda_case: str,
    eval_id: int,
    proposal_method: str,
    model_name: str = "gnn",
    topology_budget: int | None = None,
    stage_e_iteration: int | None = None,
    stage_c_proposal_score: float | None = None,
    proxy_fields: Dict | None = None,
    physics_weights: Dict[str, float] | None = None,
    recourse_maxiter: int | None = None,
    optimize_recourse: bool = True,
) -> PhysicsEvaluationResult:
    config = model_context["config"]
    scenario = model_context["scenario"]
    decision_vector = model_context["decision_vector"]
    num_lines = int(_edge_array(scenario).shape[1])
    removed = sorted({int(line_id) for line_id in shutoff_line_ids})
    active = _active_line_ids(num_lines, removed, scenario=scenario)
    z_by_line = {line_id: int(line_id not in set(removed)) for line_id in range(num_lines)}
    source_less = source_less_island_buses(scenario, removed)
    lower, upper, forced_alpha_count = recourse_bounds_with_island_limits(decision_vector, source_less)
    u0 = np.minimum(np.maximum(np.asarray(decision_vector.u_base, dtype=float), lower), upper)
    bounds = list(zip(lower.tolist(), upper.tolist()))
    physics_weights = DEFAULT_PHYSICS_WEIGHTS if physics_weights is None else physics_weights
    optimizer_maxiter = int(config.optimizer.maxiter if recourse_maxiter is None else recourse_maxiter)
    trace: List[Dict] = []
    gridfm_calls = 0
    started = time.time()

    def objective(u_vec: np.ndarray) -> float:
        nonlocal gridfm_calls
        gridfm_calls += 1
        try:
            _prediction, state, _keep = _predict_topology_state(model_context, np.asarray(u_vec, dtype=float), removed)
            components = _physics_components(model_context, np.asarray(u_vec, dtype=float), state, active, source_less)
            pac = weighted_pac_total(components, physics_weights)
            raw, _by_line = compute_operational_wildfire_exposure(
                np.asarray(state["loading_ratio"], dtype=float),
                p_env_by_line,
                z_by_line,
                candidate_line_ids,
            )
            r_norm = normalize_true_exposure(raw, baseline_R_raw)
            l_shed = demand_weighted_load_shed_from_prediction(_prediction, scenario)
            value = float(lambda_R) * r_norm + float(lambda_L) * l_shed + float(rho_phys) * pac
            trace.append(
                {
                    "eval_id": int(eval_id),
                    "call_idx": int(gridfm_calls),
                    "stage": stage,
                    "case_name": case_name,
                    "lambda_case": lambda_case,
                    "rho_phys": float(rho_phys),
                    "R_norm": float(r_norm),
                    "L_shed": float(l_shed),
                    "PAC_total": float(pac),
                    "J_true": float(value),
                    "max_loading_ratio": float(state.get("max_loading_ratio", np.nan)),
                }
            )
            return value if math.isfinite(value) else float(config.objective.invalid_prediction_penalty)
        except Exception:
            return float(config.objective.invalid_prediction_penalty)

    status = "ok"
    scipy_success = False
    scipy_message = ""
    u_best = u0.copy()
    if optimize_recourse:
        try:
            result = minimize(
                objective,
                u0,
                method=config.optimizer.method,
                bounds=bounds,
                options={
                    "maxiter": optimizer_maxiter,
                    "ftol": float(config.optimizer.ftol),
                    "gtol": float(config.optimizer.gtol),
                    "eps": float(config.optimizer.eps),
                    "disp": bool(config.optimizer.disp),
                },
            )
            scipy_success = bool(result.success)
            scipy_message = str(result.message)
            u_best = np.asarray(result.x, dtype=float)
        except Exception as exc:
            status = "failed"
            scipy_message = str(exc)
    else:
        scipy_success = True
        scipy_message = "fixed_control_no_continuous_recourse"

    final_error = ""
    try:
        prediction, state, _keep = _predict_topology_state(model_context, u_best, removed)
        gridfm_calls += 1
        components = _physics_components(model_context, u_best, state, active, source_less)
        pac = weighted_pac_total(components, physics_weights)
        true_R_raw, exposure_by_line = compute_operational_wildfire_exposure(
            np.asarray(state["loading_ratio"], dtype=float),
            p_env_by_line,
            z_by_line,
            candidate_line_ids,
        )
        r_norm = normalize_true_exposure(true_R_raw, baseline_R_raw)
        l_shed = demand_weighted_load_shed_from_prediction(prediction, scenario)
        j_no_phys = float(lambda_R) * r_norm + float(lambda_L) * l_shed
        j_true = j_no_phys + float(rho_phys) * pac
        max_loading = float(state.get("max_loading_ratio", np.nan))
        vm = np.asarray(state.get("Vm", []), dtype=float)
        prediction_has_nan = any(
            np.any(~np.isfinite(np.asarray(value, dtype=float)))
            for value in prediction.values()
            if isinstance(value, (list, tuple, np.ndarray))
        )
    except Exception as exc:
        status = "failed"
        final_error = str(exc)
        components = {key: np.nan for key in DEFAULT_PHYSICS_WEIGHTS}
        pac = np.nan
        true_R_raw = np.nan
        exposure_by_line = {}
        r_norm = np.nan
        l_shed = np.nan
        j_no_phys = np.nan
        j_true = np.nan
        max_loading = np.nan
        vm = np.asarray([], dtype=float)
        prediction_has_nan = True

    delta_pg, alpha = decision_vector.split_decision_vector(u_best)
    proxy_fields = proxy_fields or {}
    row = {
        "stage": stage,
        "case_name": case_name,
        "model": model_name,
        "candidate_set": "t0p30",
        "lambda_case": lambda_case,
        "lambda_R": float(lambda_R),
        "lambda_L": float(lambda_L),
        "rho_phys": float(rho_phys),
        "eval_id": int(eval_id),
        "proposal_method": proposal_method,
        "topology_budget": np.nan if topology_budget is None else int(topology_budget),
        "stage_e_iteration": np.nan if stage_e_iteration is None else int(stage_e_iteration),
        "num_candidate_lines": int(len(candidate_line_ids)),
        "num_shutoff_lines": int(len(removed)),
        "shutoff_line_ids": _as_csv_ints(removed),
        "removed_line_ids": _as_csv_ints(removed),
        "active_line_ids": _as_csv_ints(active),
        "source_less_bus_ids": _as_csv_ints(source_less),
        "source_less_selected_alpha_forced_zero": int(forced_alpha_count),
        "R_raw": float(true_R_raw) if np.isfinite(true_R_raw) else np.nan,
        "R_base": float(baseline_R_raw),
        "R_norm": float(r_norm) if np.isfinite(r_norm) else np.nan,
        "L_shed": float(l_shed) if np.isfinite(l_shed) else np.nan,
        "L_shed_definition": "gridfm_predicted_demand_weighted_service_loss",
        "PAC_total": float(pac) if np.isfinite(pac) else np.nan,
        "J_no_phys": float(j_no_phys) if np.isfinite(j_no_phys) else np.nan,
        "J_true": float(j_true) if np.isfinite(j_true) else np.nan,
        "risk_contribution": float(lambda_R) * float(r_norm) if np.isfinite(r_norm) else np.nan,
        "load_contribution": float(lambda_L) * float(l_shed) if np.isfinite(l_shed) else np.nan,
        "physics_contribution": float(rho_phys) * float(pac) if np.isfinite(pac) else np.nan,
        "PAC_voltage_limits": float(components["voltage_limits"]) if np.isfinite(components["voltage_limits"]) else np.nan,
        "PAC_thermal_limits": float(components["thermal_limits"]) if np.isfinite(components["thermal_limits"]) else np.nan,
        "PAC_generator_limits": float(components["generator_limits"]) if np.isfinite(components["generator_limits"]) else np.nan,
        "PAC_island_source_feasibility": float(components["island_source_feasibility"]) if np.isfinite(components["island_source_feasibility"]) else np.nan,
        "diag_p_balance": float(components["p_balance"]) if np.isfinite(components["p_balance"]) else np.nan,
        "diag_q_balance": float(components["q_balance"]) if np.isfinite(components["q_balance"]) else np.nan,
        "diag_branch_flow_consistency": float(components["branch_flow_consistency"]) if np.isfinite(components["branch_flow_consistency"]) else np.nan,
        "scipy_success": bool(scipy_success),
        "scipy_message": scipy_message,
        "recourse_maxiter": int(optimizer_maxiter),
        "continuous_recourse_optimized": bool(optimize_recourse),
        "gridfm_status": status,
        "gridfm_error": final_error,
        "gridfm_calls": int(gridfm_calls),
        "prediction_has_nan": bool(prediction_has_nan),
        "max_loading_ratio": max_loading,
        "min_voltage": float(np.nanmin(vm)) if len(vm) else np.nan,
        "max_voltage": float(np.nanmax(vm)) if len(vm) else np.nan,
        "mean_alpha": float(np.mean(alpha)) if len(alpha) else 1.0,
        "min_alpha": float(np.min(alpha)) if len(alpha) else 1.0,
        "max_abs_delta_pg": float(np.max(np.abs(delta_pg))) if len(delta_pg) else 0.0,
        "runtime_seconds": float(time.time() - started),
        "stage_c_proposal_score_r_TH": np.nan if stage_c_proposal_score is None else float(stage_c_proposal_score),
        "stage_c_final_metric_uses_revised_true_exposure": True,
        "stage_c_final_metric_uses_impact": False,
        "stage_e_proxy_R_hat": proxy_fields.get("proxy_R_hat", np.nan),
        "stage_e_proxy_L_hat": proxy_fields.get("proxy_L_hat", np.nan),
        "stage_e_proxy_objective": proxy_fields.get("proxy_objective", np.nan),
        "stage_e_proxy_R_denominator": proxy_fields.get("proxy_R_denominator", np.nan),
        "stage_e_gurobi_objective": proxy_fields.get("gurobi_objective", np.nan),
        "stage_e_gurobi_status": proxy_fields.get("gurobi_status", np.nan),
        "true_exposure_by_line": ";".join(f"{int(k)}:{float(v):.12g}" for k, v in sorted(exposure_by_line.items())),
    }
    return PhysicsEvaluationResult(row=row, trace=trace)
