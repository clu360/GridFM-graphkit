from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.gridfm_support.branch_metadata import (
    canonicalize_line_ids,
    expand_to_physical_line_ids,
    physical_line_ids,
)
from experiments.test.wildfire_tests.shared.decision_vector import PgQgAlphaDecisionVector
from experiments.test.wildfire_tests.shared.gridfm_runner import GridFMRunner
from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import git_metadata, make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_tests.shared.state_extraction import extract_state_quantities
from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
from experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps import scenario_with_line_outages
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import (
    MODEL_CONFIGS,
    _build_model_context,
)
from experiments.test.wildfire_tests.stage_d_deenergization.stage_d_deenergization import (
    enumerate_deenergization_subsets,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.gurobi_master import (
    solve_gurobi_master_next_candidate,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.physics_infeasibility_evaluator import (
    source_less_island_buses,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    DEFAULT_PROXY_TYPE,
    deenergized_from_y,
    normalize_true_exposure,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_decision_quality import (
    FIXED_T0P30_CANDIDATE_LINE_IDS,
    _consequence_by_line,
    _csv_ints,
    _edge_array,
    _parse_line_ids,
    _scenario_baseline_exposure,
    _validate_candidate_line_ids,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_physics_decision_quality import (
    STAGE_D,
    STAGE_E_K2,
    STAGE_E_UNCONSTRAINED,
    _canonicalize_decision_quality_scenario,
    _expected_outcome,
    _lambda_case,
    _nondominated_mask,
    _pareto_tables,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import (
    DecisionQualityScenario,
    get_scenarios,
)
from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_scenario_baseline_physics_sensitivity import (
    BASELINE_LOADING_SOURCE,
    EXCLUDED_MARGIN_TARGET_LINE_IDS,
    P_ENV_MODE_TARGET_MARGIN,
    POST_TOPOLOGY_EVALUATION_SOURCE,
    build_scenario_baseline_loading_ranking,
    p_env_for_target_margin_scenario,
    remove_margin_excluded_targets,
    scenario_baseline_loading_vector,
)


RESULT_ROOT = RESULTS_ROOT / "leq" / "stage_g" / "rev_cont"
DEFAULT_LAMBDAS = [0.0, 0.2, 0.5, 0.8, 1.0]
DEFAULT_RHO_VALUES = [0.0, 2.0]
DEFAULT_STAGE_D_LIMIT = None
DEFAULT_STAGE_E_BUDGET = 50
DEFAULT_CALL_BUDGET = 100
STAGES = [STAGE_D, STAGE_E_K2, STAGE_E_UNCONSTRAINED]
STAGE_LABELS = {
    STAGE_D: "Stage D exhaustive",
    STAGE_E_K2: "Stage E k2",
    STAGE_E_UNCONSTRAINED: "Stage E unconstrained",
}
FEATURE_NAMES = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]
CONTROLLED_FEATURES = {"Pd", "Qd", "Pg", "Qg"}
POST_TOPOLOGY_CONTINUOUS_SOURCE = "gridfm_inference_with_intervention_mask_and_clamp"
L_SHED_SOURCE = "hybrid_commanded_and_gridfm_effective_alpha_with_source_less_island_correction"
LOAD_SHED_MODE = "hybrid"
PAC_GROUP_WEIGHTS = {
    "operational": 1.0,
    "ac": 1.0,
    "model_consistency": 1.0,
}
CHECKPOINT_TABLES = {
    "results": "continuous_recourse_results.csv",
    "traces": "continuous_objective_call_trace.csv",
    "load": "load_shedding_provenance.csv",
    "risk": "wildfire_risk_provenance.csv",
    "controlled": "controlled_state_consistency.csv",
    "mask": "masking_clamping_audit.csv",
}
CONTROL_CLAMP_ATOL = 1e-5


class CallBudgetReached(RuntimeError):
    pass


def _as_float_list(values: Sequence[float]) -> List[float]:
    return [float(value) for value in values]


def _normalize_stages(stages: Sequence[str] | None) -> List[str]:
    selected = STAGES if stages is None else list(stages)
    unknown = sorted(set(selected) - set(STAGES))
    if unknown:
        raise ValueError(f"Unsupported stage(s): {unknown}. Valid stages: {STAGES}")
    deduped: List[str] = []
    for stage in selected:
        if stage not in deduped:
            deduped.append(stage)
    if not deduped:
        raise ValueError("At least one stage must be selected.")
    return deduped


def _line_key(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in sorted({int(item) for item in values}))


def _parse_line_key(value) -> List[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    return sorted({int(float(part)) for part in text.split(",") if part.strip()})


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _append_checkpoint(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    header = not checkpoint_path.exists()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(checkpoint_path, mode="a", header=header, index=False)


def _savefig(fig, path: Path, **kwargs) -> None:
    output_path = Path(path).resolve()
    _mkdir_long(output_path.parent)
    try:
        fig.savefig(_long_path(output_path), **kwargs)
    except FileNotFoundError:
        _mkdir_long(output_path.parent)
        fig.savefig(_long_path(output_path), **kwargs)


def _long_path(path: Path) -> str:
    path = Path(path)
    if os.name == "nt":
        absolute = str(path.absolute())
        return absolute if absolute.startswith("\\\\?\\") else "\\\\?\\" + absolute
    return str(path)


def _mkdir_long(path: Path) -> None:
    path = Path(path)
    if os.name == "nt":
        Path(_long_path(path)).mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def _write_progress(
    progress_path: Path,
    *,
    completed: int,
    total: int,
    started: float,
    last_result_id: int | None,
    status: str = "running",
) -> None:
    write_json(
        progress_path,
        {
            "status": status,
            "completed_topology_rho_optimizations": int(completed),
            "total_topology_rho_optimizations": int(total),
            "remaining_topology_rho_optimizations": int(max(total - completed, 0)),
            "completion_fraction": float(completed / total) if total else 1.0,
            "last_completed_result_id": None if last_result_id is None else int(last_result_id),
            "elapsed_seconds": float(time.perf_counter() - started),
            "updated_unix_seconds": float(time.time()),
        },
    )


def _read_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _run_signature(
    *,
    model_type: str,
    scenario_ids: List[str] | None,
    lambda_values: Sequence[float],
    proxy_lambda_values: Sequence[float] | None,
    rho_values: Sequence[float],
    stages: Sequence[str],
    stage_d_limit: int | None,
    stage_e_budget: int,
    call_budget: int,
    delta_qg_bound_mvar: float,
) -> dict:
    return {
        "model": str(model_type),
        "scenario_ids": None if scenario_ids is None else [str(item) for item in scenario_ids],
        "lambda_values": _as_float_list(lambda_values),
        "proxy_lambda_values": None if proxy_lambda_values is None else _as_float_list(proxy_lambda_values),
        "rho_phys_values": _as_float_list(rho_values),
        "stages": list(stages),
        "stage_d_limit": None if stage_d_limit is None else int(stage_d_limit),
        "stage_e_budget": int(stage_e_budget),
        "call_budget": int(call_budget),
        "delta_qg_bound_mvar": float(delta_qg_bound_mvar),
    }


def _final_artifacts_complete(run_dir: Path) -> bool:
    tables_dir = run_dir / "tables"
    required_tables = [
        "methodology_fidelity_checks.csv",
        "continuous_recourse_results.csv",
        "best_by_rho_scenario_lambda_stage.csv",
        "expected_vs_selected_by_rho.csv",
        "runtime_and_solver_diagnostics.csv",
        "fixed_vs_continuous_comparison.csv",
        "controlled_state_consistency.csv",
        "masking_clamping_audit.csv",
        "load_shedding_provenance.csv",
        "wildfire_risk_provenance.csv",
        "continuous_objective_call_trace.csv",
    ]
    if not all((tables_dir / name).exists() for name in required_tables):
        return False
    if not (run_dir / "inputs" / "metadata.json").exists():
        return False
    return any((run_dir / "plots").rglob("*.png"))


def _latest_incomplete_run(output_root: Path, expected_signature: dict) -> Path | None:
    if not output_root.exists():
        return None
    candidates = sorted(
        (path for path in output_root.glob("run_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        progress_path = run_dir / "inputs" / "progress.json"
        if not progress_path.exists():
            continue
        signature = _read_optional_json(run_dir / "inputs" / "run_signature.json")
        if signature != expected_signature:
            continue
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                progress = json.load(f)
        except Exception:
            continue
        completed = int(progress.get("completed_topology_rho_optimizations", 0))
        total = int(progress.get("total_topology_rho_optimizations", 0))
        status = str(progress.get("status", "running"))
        if total and completed >= total and status == "complete" and not _final_artifacts_complete(run_dir):
            return run_dir
        if total and completed < total and status != "complete":
            return run_dir
    return None


def _make_continuous_context(model_type: str, delta_qg_bound_mvar: float) -> dict:
    context = _build_model_context(model_type, grouping_top_fraction=0.30)
    old_decision = context["decision_vector"]
    scenario = context["scenario"]
    config = context["config"]
    decision = PgQgAlphaDecisionVector(
        scenario,
        old_decision.selected_generator_buses,
        old_decision.selected_load_buses,
        delta_pg_bound_mw=old_decision.delta_pg_bound_mw,
        delta_qg_bound_mvar=delta_qg_bound_mvar,
        alpha_min=old_decision.alpha_min,
        alpha_max=old_decision.alpha_max,
    )
    model = context["runner"].solver.model
    runner = GridFMRunner(model, model_type, scenario, decision, device=config.model.device)
    context = dict(context)
    context["decision_vector"] = decision
    context["runner"] = runner
    context["continuous_decision_vector_type"] = "[Delta_Pg, Delta_Qg, alpha]"
    return context


def _full_loading_vector(num_lines: int, keep_mask: List[int], active_loading: np.ndarray, removed_line_ids: Iterable[int]) -> np.ndarray:
    full = np.zeros(int(num_lines), dtype=float)
    for active_idx, original_line_id in enumerate(keep_mask):
        if active_idx < len(active_loading):
            full[int(original_line_id)] = float(active_loading[active_idx])
    for line_id in removed_line_ids:
        if 0 <= int(line_id) < len(full):
            full[int(line_id)] = 0.0
    return full


def _active_line_ids(scenario, removed_line_ids: Iterable[int]) -> List[int]:
    num_lines = int(_edge_array(scenario).shape[1])
    removed = set(expand_to_physical_line_ids(scenario, removed_line_ids))
    return [line_id for line_id in range(num_lines) if int(line_id) not in removed]


def _clamp_prediction(decision_vector: PgQgAlphaDecisionVector, u: np.ndarray, prediction: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    delta_pg, delta_qg, alpha = decision_vector.split_decision_vector(u)
    combined = {key: np.asarray(value, dtype=float).copy() for key, value in prediction.items()}
    if decision_vector.n_generators:
        buses = np.asarray(decision_vector.selected_generator_buses, dtype=int)
        combined["Pg"][buses] = np.asarray(decision_vector.scenario.Pg_base, dtype=float)[buses] + delta_pg
        combined["Qg"][buses] = np.asarray(decision_vector.scenario.Qg_base, dtype=float)[buses] + delta_qg
    if decision_vector.n_loads:
        buses = np.asarray(decision_vector.selected_load_buses, dtype=int)
        combined["Pd"][buses] = np.asarray(decision_vector.scenario.Pd_base, dtype=float)[buses] * alpha
        combined["Qd"][buses] = np.asarray(decision_vector.scenario.Qd_base, dtype=float)[buses] * alpha
    return combined


def _attach_ac_topology_state(scenario, state: Dict) -> Dict:
    enriched = dict(state)
    yf = getattr(scenario, "Yf", None)
    yt = getattr(scenario, "Yt", None)
    edge_index = getattr(scenario, "edge_index", None)
    if yf is not None and yt is not None and edge_index is not None:
        enriched["_ac_Yf"] = yf.copy() if hasattr(yf, "copy") else yf
        enriched["_ac_Yt"] = yt.copy() if hasattr(yt, "copy") else yt
        if hasattr(edge_index, "detach"):
            enriched["_ac_edge_index"] = edge_index.detach().cpu().numpy().copy()
        else:
            enriched["_ac_edge_index"] = np.asarray(edge_index, dtype=int).copy()
        enriched["_ac_balance_available"] = True
    else:
        enriched["_ac_balance_available"] = False
    return enriched


def _predict_combined_state(context: dict, u: np.ndarray, removed_line_ids: List[int]) -> tuple[Dict, Dict, Dict, List[int]]:
    scenario = context["scenario"]
    runner = context["runner"]
    decision = context["decision_vector"]
    config = context["config"]
    num_lines = int(_edge_array(scenario).shape[1])
    if removed_line_ids:
        with scenario_with_line_outages(scenario, removed_line_ids) as keep_mask:
            original_yf = getattr(scenario, "Yf", None)
            original_yt = getattr(scenario, "Yt", None)
            try:
                scenario.Yf = None
                scenario.Yt = None
                raw_prediction = runner.predict(u)
                combined_prediction = _clamp_prediction(decision, u, raw_prediction)
                active_state = extract_state_quantities(
                    scenario,
                    combined_prediction,
                    standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
                )
                active_state = _attach_ac_topology_state(scenario, active_state)
            finally:
                scenario.Yf = original_yf
                scenario.Yt = original_yt
        loading = _full_loading_vector(num_lines, keep_mask, active_state["loading_ratio"], removed_line_ids)
        state = dict(active_state)
        state["loading_ratio"] = loading
        state["num_lines"] = num_lines
        state["max_loading_ratio"] = float(np.nanmax(loading)) if len(loading) else 0.0
        return raw_prediction, combined_prediction, state, [int(line_id) for line_id in keep_mask]

    raw_prediction = runner.predict(u)
    combined_prediction = _clamp_prediction(decision, u, raw_prediction)
    state = extract_state_quantities(
        scenario,
        combined_prediction,
        standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
    )
    state = _attach_ac_topology_state(scenario, state)
    return raw_prediction, combined_prediction, state, list(range(num_lines))


def _normalized_mse(observed: np.ndarray, target: np.ndarray, scale: np.ndarray | float) -> float:
    observed = np.asarray(observed, dtype=float)
    target = np.asarray(target, dtype=float)
    scale_arr = np.asarray(scale, dtype=float) + np.zeros_like(observed, dtype=float)
    denom = np.maximum(np.abs(scale_arr), 1.0)
    finite = np.isfinite(observed) & np.isfinite(target) & np.isfinite(denom) & (denom > 0.0)
    if not np.any(finite):
        return 0.0
    return float(np.mean(((observed[finite] - target[finite]) / denom[finite]) ** 2))


def _normalized_bound_penalty(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    scale = np.maximum(np.abs(upper - lower), 1.0)
    finite = np.isfinite(values) & np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
    if not np.any(finite):
        return 0.0
    low = np.maximum(0.0, lower[finite] - values[finite]) / scale[finite]
    high = np.maximum(0.0, values[finite] - upper[finite]) / scale[finite]
    return float(np.mean(low**2 + high**2))


def _generator_bus_mask(scenario) -> np.ndarray:
    pg_base = np.asarray(getattr(scenario, "Pg_base", []), dtype=float)
    pg_min = np.asarray(getattr(scenario, "Pg_min", np.zeros_like(pg_base)), dtype=float)
    pg_max = np.asarray(getattr(scenario, "Pg_max", np.zeros_like(pg_base)), dtype=float)
    pv = np.asarray(getattr(scenario, "PV_mask", np.zeros_like(pg_base, dtype=bool)), dtype=bool)
    ref = np.asarray(getattr(scenario, "REF_mask", np.zeros_like(pg_base, dtype=bool)), dtype=bool)
    bounds = np.isfinite(pg_min) & np.isfinite(pg_max) & (pg_max > pg_min)
    active_base = np.abs(pg_base) > 1e-9
    return bounds | active_base | pv | ref


def _generator_limit_penalty_for_prediction(scenario, prediction: Dict[str, np.ndarray]) -> float:
    mask = _generator_bus_mask(scenario)
    if not np.any(mask):
        return 0.0
    pg = np.asarray(prediction.get("Pg", np.zeros(int(getattr(scenario, "num_buses", 0)))), dtype=float)
    lower = np.asarray(getattr(scenario, "Pg_min", np.zeros_like(pg)), dtype=float)
    upper = np.asarray(getattr(scenario, "Pg_max", np.zeros_like(pg)), dtype=float)
    return _normalized_bound_penalty(pg[mask], lower[mask], upper[mask])


def _ac_balance_components(context: dict, state: Dict, prediction: Dict[str, np.ndarray]) -> Dict[str, float | bool]:
    scenario = context["scenario"]
    if not bool(state.get("_ac_balance_available", False)):
        return {"p_balance": np.nan, "q_balance": np.nan, "p_balance_available": False, "q_balance_available": False}
    try:
        yf = state["_ac_Yf"]
        yt = state["_ac_Yt"]
        edge_index = np.asarray(state["_ac_edge_index"], dtype=int)
        vm = np.asarray(prediction["Vm"], dtype=float)
        va = np.asarray(prediction["Va"], dtype=float)
        if len(va) and np.nanmax(np.abs(va)) > (2.0 * np.pi + 1e-9):
            va = np.deg2rad(va)
        v = vm * np.exp(1j * va)
        if_complex = yf @ v
        it_complex = yt @ v
        bus_current = np.zeros(int(scenario.num_buses), dtype=complex)
        f = edge_index[0, :].astype(int)
        t = edge_index[1, :].astype(int)
        np.add.at(bus_current, f, if_complex)
        np.add.at(bus_current, t, it_complex)
        s_network = v * np.conj(bus_current) * float(getattr(scenario, "sn_mva", 100.0))
        p_injection = np.asarray(prediction["Pg"], dtype=float) - np.asarray(prediction["Pd"], dtype=float)
        q_injection = np.asarray(prediction["Qg"], dtype=float) - np.asarray(prediction["Qd"], dtype=float)
        p_resid = p_injection - np.real(s_network)
        q_resid = q_injection - np.imag(s_network)
        scale = max(float(getattr(scenario, "sn_mva", 100.0)), 1.0)
        return {
            "p_balance": float(np.mean((p_resid / scale) ** 2)),
            "q_balance": float(np.mean((q_resid / scale) ** 2)),
            "p_balance_available": True,
            "q_balance_available": True,
        }
    except Exception:
        return {"p_balance": np.nan, "q_balance": np.nan, "p_balance_available": False, "q_balance_available": False}


def _command_consistency_components(
    context: dict,
    u: np.ndarray,
    raw_prediction: Dict[str, np.ndarray],
) -> Dict[str, float]:
    decision = context["decision_vector"]
    scenario = context["scenario"]
    commanded = decision.u_to_node_features(u)
    penalties = {"load": 0.0, "Pg": 0.0, "Qg": 0.0}
    if decision.n_loads:
        buses = np.asarray(decision.selected_load_buses, dtype=int)
        pd_penalty = _normalized_mse(np.asarray(raw_prediction["Pd"])[buses], commanded[buses, 0], np.asarray(scenario.Pd_base)[buses])
        qd_penalty = _normalized_mse(np.asarray(raw_prediction["Qd"])[buses], commanded[buses, 1], np.asarray(scenario.Qd_base)[buses])
        penalties["load"] = float(np.mean([pd_penalty, qd_penalty]))
    if decision.n_generators:
        buses = np.asarray(decision.selected_generator_buses, dtype=int)
        penalties["Pg"] = _normalized_mse(np.asarray(raw_prediction["Pg"])[buses], commanded[buses, 2], np.asarray(scenario.Pg_base)[buses])
        penalties["Qg"] = _normalized_mse(np.asarray(raw_prediction["Qg"])[buses], commanded[buses, 3], np.asarray(scenario.Qg_base)[buses])
    return penalties


def _physics_components(
    context: dict,
    u: np.ndarray,
    state: Dict,
    active_line_ids: Iterable[int],
    source_less_buses: Iterable[int],
    raw_prediction: Dict[str, np.ndarray],
    combined_prediction: Dict[str, np.ndarray],
) -> Dict[str, float | bool]:
    decision = context["decision_vector"]
    scenario = context["scenario"]
    vm = np.asarray(state.get("Vm", []), dtype=float)
    if len(vm):
        voltage = float(np.mean(np.maximum(0.0, 0.95 - vm) ** 2 + np.maximum(0.0, vm - 1.05) ** 2))
    else:
        voltage = 0.0
    loading = np.asarray(state.get("loading_ratio", []), dtype=float)
    active = [int(line_id) for line_id in active_line_ids if int(line_id) < len(loading)]
    thermal = float(np.mean(np.maximum(0.0, loading[active] - 1.0) ** 2)) if active else 0.0
    full_alpha = decision.full_alpha(u)
    demand = np.maximum(np.asarray(decision.scenario.Pd_base, dtype=float), 0.0)
    total_demand = max(float(np.sum(demand)), 1e-12)
    island_buses = [int(bus) for bus in source_less_buses]
    island = float(np.sum(demand[island_buses] * full_alpha[island_buses] ** 2) / total_demand) if island_buses else 0.0
    ac = _ac_balance_components(context, state, combined_prediction)
    cmd = _command_consistency_components(context, u, raw_prediction)
    generator_eval = _generator_limit_penalty_for_prediction(scenario, combined_prediction)
    generator_raw = _generator_limit_penalty_for_prediction(scenario, raw_prediction)
    p_balance = float(ac["p_balance"]) if bool(ac["p_balance_available"]) else np.nan
    q_balance = float(ac["q_balance"]) if bool(ac["q_balance_available"]) else np.nan
    pac_operational = float(voltage + thermal + generator_eval + island)
    pac_ac = float((p_balance if np.isfinite(p_balance) else 0.0) + (q_balance if np.isfinite(q_balance) else 0.0))
    pac_model = float(cmd["load"] + cmd["Pg"] + cmd["Qg"] + generator_raw)
    pac_total = float(
        PAC_GROUP_WEIGHTS["operational"] * pac_operational
        + PAC_GROUP_WEIGHTS["ac"] * pac_ac
        + PAC_GROUP_WEIGHTS["model_consistency"] * pac_model
    )
    return {
        "voltage_limits": voltage,
        "thermal_limits": thermal,
        "generator_limits": generator_eval,
        "generator_limits_eval": generator_eval,
        "generator_limits_raw": generator_raw,
        "island_source_feasibility": island,
        "p_balance": p_balance,
        "q_balance": q_balance,
        "p_balance_available": bool(ac["p_balance_available"]),
        "q_balance_available": bool(ac["q_balance_available"]),
        "branch_flow_consistency": np.nan,
        "branch_flow_consistency_available": False,
        "branch_flow_consistency_weight": 0.0,
        "cmd_load": float(cmd["load"]),
        "cmd_Pg": float(cmd["Pg"]),
        "cmd_Qg": float(cmd["Qg"]),
        "PAC_operational": pac_operational,
        "PAC_AC": pac_ac,
        "PAC_model_consistency": pac_model,
        "PAC_total": pac_total,
        "pac_operational_weight": float(PAC_GROUP_WEIGHTS["operational"]),
        "pac_ac_weight": float(PAC_GROUP_WEIGHTS["ac"]),
        "pac_model_consistency_weight": float(PAC_GROUP_WEIGHTS["model_consistency"]),
        "Qg_bounds_available": False,
    }


def _load_shedding_provenance(
    context: dict,
    u: np.ndarray,
    source_less_buses: Iterable[int],
    result_id: int,
    raw_prediction: Dict[str, np.ndarray] | None = None,
    load_shed_mode: str = LOAD_SHED_MODE,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    decision = context["decision_vector"]
    scenario = context["scenario"]
    demand = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    total = max(float(np.sum(demand)), 1e-12)
    alpha_commanded = decision.full_alpha(u)
    raw_pd = np.asarray(raw_prediction.get("Pd", demand), dtype=float) if raw_prediction is not None else demand.copy()
    selected = {int(bus) for bus in decision.selected_load_buses}
    source_less = {int(bus) for bus in source_less_buses}
    rows = []
    totals = {
        "L_shed_cmd": 0.0,
        "L_shed_gridfm_raw": 0.0,
        "L_shed_gridfm_effective": 0.0,
        "L_shed_hybrid": 0.0,
    }
    for bus in range(int(scenario.num_buses)):
        is_source_less = bus in source_less
        if demand[bus] > 1e-12:
            alpha_gridfm_raw = float(raw_pd[bus] / demand[bus])
        else:
            alpha_gridfm_raw = 1.0
        alpha_gridfm_effective = float(np.clip(alpha_gridfm_raw, 0.0, 1.0))
        alpha_cmd_effective = 0.0 if is_source_less else float(alpha_commanded[bus])
        alpha_gridfm_island_effective = 0.0 if is_source_less else alpha_gridfm_effective
        if is_source_less:
            alpha_hybrid = 0.0
            source = "source_less_forced_unserved"
        elif bus in selected:
            alpha_hybrid = float(alpha_commanded[bus])
            source = "selected_commanded_alpha"
        else:
            alpha_hybrid = alpha_gridfm_effective
            source = "nonselected_gridfm_effective_alpha"
        load_cmd = float(demand[bus] * (1.0 - alpha_cmd_effective))
        load_gridfm_raw = float(demand[bus] * (1.0 - alpha_gridfm_raw))
        load_gridfm_effective = float(demand[bus] * (1.0 - alpha_gridfm_island_effective))
        load_hybrid = float(demand[bus] * (1.0 - alpha_hybrid))
        weighted_cmd = float(load_cmd / total)
        weighted_gridfm_raw = float(load_gridfm_raw / total)
        weighted_gridfm_effective = float(load_gridfm_effective / total)
        weighted_hybrid = float(load_hybrid / total)
        totals["L_shed_cmd"] += weighted_cmd
        totals["L_shed_gridfm_raw"] += weighted_gridfm_raw
        totals["L_shed_gridfm_effective"] += weighted_gridfm_effective
        totals["L_shed_hybrid"] += weighted_hybrid
        active_alpha = alpha_hybrid if load_shed_mode == "hybrid" else alpha_cmd_effective
        active_load = load_hybrid if load_shed_mode == "hybrid" else load_cmd
        active_weighted = weighted_hybrid if load_shed_mode == "hybrid" else weighted_cmd
        rows.append(
            {
                "result_id": int(result_id),
                "bus_id": int(bus),
                "bus_role": "selected_load" if bus in selected else "nonselected",
                "alpha_source": source,
                "alpha_source_hybrid": source,
                "is_source_less_islanded": bool(is_source_less),
                "alpha_commanded": float(alpha_commanded[bus]),
                "alpha_gridfm_raw": float(alpha_gridfm_raw),
                "alpha_gridfm_effective": float(alpha_gridfm_effective),
                "alpha_effective_cmd": float(alpha_cmd_effective),
                "alpha_effective_gridfm": float(alpha_gridfm_island_effective),
                "alpha_effective_hybrid": float(alpha_hybrid),
                "alpha_effective": float(active_alpha),
                "alpha_value": float(active_alpha),
                "source_less_island_correction_applied": bool(is_source_less),
                "Pd_base": float(demand[bus]),
                "Pd_gridfm_raw": float(raw_pd[bus]),
                "load_shed_cmd_mw": load_cmd,
                "load_shed_gridfm_raw_mw": load_gridfm_raw,
                "load_shed_gridfm_effective_mw": load_gridfm_effective,
                "load_shed_hybrid_mw": load_hybrid,
                "load_shed_cmd_weighted": weighted_cmd,
                "load_shed_gridfm_raw_weighted": weighted_gridfm_raw,
                "load_shed_gridfm_effective_weighted": weighted_gridfm_effective,
                "load_shed_hybrid_weighted": weighted_hybrid,
                "load_shed_mw": active_load,
                "load_shed_weighted": active_weighted,
                "gridfm_predicted_pd_used_for_l_shed": bool(load_shed_mode == "hybrid" and bus not in selected and not is_source_less),
                "load_shed_mode": load_shed_mode,
                "included_in_objective": True,
            }
        )
    totals["L_shed"] = float(totals["L_shed_hybrid"] if load_shed_mode == "hybrid" else totals["L_shed_cmd"])
    return pd.DataFrame(rows), {key: float(value) for key, value in totals.items()}


def _wildfire_risk_provenance(
    context: dict,
    loading: np.ndarray,
    p_env_by_line: Dict[int, float],
    z_by_line: Dict[int, int],
    candidate_line_ids: Sequence[int],
    result_id: int,
) -> tuple[pd.DataFrame, float]:
    scenario = context["scenario"]
    edge_array = _edge_array(scenario)
    loading = np.asarray(loading, dtype=float)
    rows = []
    total = 0.0
    for line_id in [int(item) for item in candidate_line_ids]:
        exposure = float(int(z_by_line.get(line_id, 1)) * float(p_env_by_line.get(line_id, 0.0)) * loading[line_id] ** 2)
        total += exposure
        rows.append(
            {
                "result_id": int(result_id),
                "canonical_line_id": line_id,
                "from_bus": int(edge_array[0, line_id]),
                "to_bus": int(edge_array[1, line_id]),
                "z_line": int(z_by_line.get(line_id, 1)),
                "p_env": float(p_env_by_line.get(line_id, 0.0)),
                "loading_ratio": float(loading[line_id]),
                "loading_source": "reconstructed_from_combined_state",
                "endpoint_feature_sources": "controlled_clamped_for_selected_Pg_Qg_Pd_Qd;gridfm_predicted_for_Vm_Va;baseline_for_unselected_controls",
                "impact_used_in_true_risk": False,
                "risk_raw_contribution": exposure,
                "included_in_objective": True,
            }
        )
    return pd.DataFrame(rows), float(total)


def _controlled_state_audit(
    context: dict,
    u: np.ndarray,
    raw_prediction: Dict[str, np.ndarray],
    combined_prediction: Dict[str, np.ndarray],
    result_id: int,
) -> pd.DataFrame:
    decision = context["decision_vector"]
    scenario = context["scenario"]
    node_features = decision.u_to_node_features(u)
    feature_index = {"Pd": 0, "Qd": 1, "Pg": 2, "Qg": 3, "Vm": 4, "Va": 5}
    rows = []
    for bus in [int(item) for item in decision.selected_generator_buses]:
        for feature in ["Pg", "Qg"]:
            commanded = float(node_features[bus, feature_index[feature]])
            gridfm = float(np.asarray(raw_prediction[feature], dtype=float)[bus])
            eval_value = float(np.asarray(combined_prediction[feature], dtype=float)[bus])
            rows.append(
                {
                    "result_id": int(result_id),
                    "bus_id": bus,
                    "feature": feature,
                    "source_role": "controlled_clamped",
                    "is_decision_feature": True,
                    "baseline_value": float(getattr(scenario, f"{feature}_base")[bus]),
                    "commanded_value": commanded,
                    "gridfm_visible_input_value": commanded,
                    "raw_gridfm_predicted_value": gridfm,
                    "objective_evaluation_value": eval_value,
                    "evaluation_value_after_clamp": eval_value,
                    "abs_command_vs_eval_error": abs(commanded - eval_value),
                    "abs_command_vs_raw_gridfm_error": abs(commanded - gridfm),
                }
            )
    for bus in [int(item) for item in decision.selected_load_buses]:
        for feature in ["Pd", "Qd"]:
            commanded = float(node_features[bus, feature_index[feature]])
            gridfm = float(np.asarray(raw_prediction[feature], dtype=float)[bus])
            eval_value = float(np.asarray(combined_prediction[feature], dtype=float)[bus])
            rows.append(
                {
                    "result_id": int(result_id),
                    "bus_id": bus,
                    "feature": feature,
                    "source_role": "controlled_clamped",
                    "is_decision_feature": True,
                    "baseline_value": float(getattr(scenario, f"{feature}_base")[bus]),
                    "commanded_value": commanded,
                    "gridfm_visible_input_value": commanded,
                    "raw_gridfm_predicted_value": gridfm,
                    "objective_evaluation_value": eval_value,
                    "evaluation_value_after_clamp": eval_value,
                    "abs_command_vs_eval_error": abs(commanded - eval_value),
                    "abs_command_vs_raw_gridfm_error": abs(commanded - gridfm),
                }
            )
    return pd.DataFrame(rows)


def _masking_audit(context: dict, result_id: int) -> pd.DataFrame:
    solver = context["runner"].solver
    original = np.asarray(getattr(solver, "last_original_mask", np.zeros((0, 0))), dtype=bool)
    effective = np.asarray(getattr(solver, "last_effective_mask", np.zeros_like(original)), dtype=bool)
    controlled = np.asarray(getattr(solver, "last_controlled_feature_mask", np.zeros_like(original)), dtype=bool)
    rows = []
    for bus in range(original.shape[0]):
        for idx in range(original.shape[1]):
            feature = FEATURE_NAMES[idx]
            rows.append(
                {
                    "result_id": int(result_id),
                    "bus_id": int(bus),
                    "feature": feature,
                    "original_masked": bool(original[bus, idx]),
                    "effective_masked": bool(effective[bus, idx]),
                    "is_controlled_feature": bool(controlled[bus, idx]),
                    "controlled_visible_to_gridfm": bool((not effective[bus, idx]) if controlled[bus, idx] else True),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_control(
    context: dict,
    candidate_line_ids: List[int],
    p_env_by_line: Dict[int, float],
    baseline_r: float,
    shutoff_line_ids: List[int],
    u: np.ndarray,
    lambda_r: float,
    rho_phys: float,
    result_id: int,
) -> Dict:
    scenario = context["scenario"]
    removed = sorted({int(line_id) for line_id in shutoff_line_ids})
    active = _active_line_ids(scenario, removed)
    z_by_line = {line_id: int(line_id not in set(expand_to_physical_line_ids(scenario, removed))) for line_id in range(int(_edge_array(scenario).shape[1]))}
    source_less = source_less_island_buses(scenario, removed)
    raw_prediction, combined_prediction, state, _keep = _predict_combined_state(context, np.asarray(u, dtype=float), removed)
    components = _physics_components(context, np.asarray(u, dtype=float), state, active, source_less, raw_prediction, combined_prediction)
    pac = float(components["PAC_total"])
    risk_raw, exposure_by_line = compute_operational_wildfire_exposure(
        np.asarray(state["loading_ratio"], dtype=float),
        p_env_by_line,
        z_by_line,
        candidate_line_ids,
    )
    r_norm = normalize_true_exposure(risk_raw, baseline_r)
    load_prov, load_metrics = _load_shedding_provenance(
        context,
        np.asarray(u, dtype=float),
        source_less,
        result_id,
        raw_prediction=raw_prediction,
        load_shed_mode=LOAD_SHED_MODE,
    )
    l_shed = float(load_metrics["L_shed"])
    risk_prov, risk_from_prov = _wildfire_risk_provenance(
        context,
        np.asarray(state["loading_ratio"], dtype=float),
        p_env_by_line,
        z_by_line,
        candidate_line_ids,
        result_id,
    )
    lambda_l = 1.0 - float(lambda_r)
    j_no_phys = float(lambda_r) * float(r_norm) + lambda_l * float(l_shed)
    j_true = j_no_phys + float(rho_phys) * float(pac)
    delta_pg, delta_qg, alpha = context["decision_vector"].split_decision_vector(np.asarray(u, dtype=float))
    return {
        "raw_prediction": raw_prediction,
        "combined_prediction": combined_prediction,
        "state": state,
        "components": components,
        "source_less": source_less,
        "R_raw": float(risk_raw),
        "R_raw_from_provenance": float(risk_from_prov),
        "R_norm": float(r_norm),
        "L_shed": float(l_shed),
        "L_shed_cmd": float(load_metrics["L_shed_cmd"]),
        "L_shed_gridfm_raw": float(load_metrics["L_shed_gridfm_raw"]),
        "L_shed_gridfm_effective": float(load_metrics["L_shed_gridfm_effective"]),
        "L_shed_hybrid": float(load_metrics["L_shed_hybrid"]),
        "load_shed_mode": LOAD_SHED_MODE,
        "PAC_total": float(pac),
        "PAC_operational": float(components["PAC_operational"]),
        "PAC_AC": float(components["PAC_AC"]),
        "PAC_model_consistency": float(components["PAC_model_consistency"]),
        "pac_operational_weight": float(components["pac_operational_weight"]),
        "pac_ac_weight": float(components["pac_ac_weight"]),
        "pac_model_consistency_weight": float(components["pac_model_consistency_weight"]),
        "J_no_phys": float(j_no_phys),
        "J_true": float(j_true),
        "risk_contribution": float(lambda_r) * float(r_norm),
        "load_contribution": lambda_l * float(l_shed),
        "physics_contribution": float(rho_phys) * float(pac),
        "max_loading_ratio": float(state.get("max_loading_ratio", np.nan)),
        "min_voltage": float(state.get("min_voltage", np.nan)),
        "max_voltage": float(state.get("max_voltage", np.nan)),
        "load_provenance": load_prov,
        "risk_provenance": risk_prov,
        "controlled_audit": _controlled_state_audit(context, np.asarray(u, dtype=float), raw_prediction, combined_prediction, result_id),
        "mask_audit": _masking_audit(context, result_id),
        "exposure_by_line": exposure_by_line,
        "delta_pg": delta_pg,
        "delta_qg": delta_qg,
        "alpha": alpha,
    }


def _recourse_bounds_with_islands(context: dict, source_less: Iterable[int]) -> tuple[np.ndarray, np.ndarray, int]:
    decision = context["decision_vector"]
    lower = np.asarray(decision.u_min, dtype=float).copy()
    upper = np.asarray(decision.u_max, dtype=float).copy()
    forced = 0
    source_less_set = {int(bus) for bus in source_less}
    for idx, bus in enumerate(np.asarray(decision.selected_load_buses, dtype=int)):
        if int(bus) in source_less_set:
            decision_idx = decision.alpha_offset + idx
            lower[decision_idx] = 0.0
            upper[decision_idx] = 0.0
            forced += 1
    return lower, upper, forced


def _optimize_topology(
    context: dict,
    candidate_line_ids: List[int],
    p_env_by_line: Dict[int, float],
    baseline_r: float,
    topology: List[int],
    lambda_r: float,
    rho_phys: float,
    result_id: int,
    call_budget: int,
) -> tuple[Dict, List[Dict], Dict]:
    decision = context["decision_vector"]
    source_less = source_less_island_buses(context["scenario"], topology)
    lower, upper, forced_alpha_count = _recourse_bounds_with_islands(context, source_less)
    u0 = np.minimum(np.maximum(np.asarray(decision.u_base, dtype=float), lower), upper)
    bounds = list(zip(lower.tolist(), upper.tolist()))
    best = None
    traces = []
    calls = 0
    invalid_calls = 0
    started = time.perf_counter()

    def objective(u_raw: np.ndarray) -> float:
        nonlocal best, calls, invalid_calls
        if calls >= int(call_budget):
            raise CallBudgetReached(f"GridFM objective-call budget {call_budget} reached.")
        calls += 1
        u = np.minimum(np.maximum(np.asarray(u_raw, dtype=float), lower), upper)
        call_started = time.perf_counter()
        try:
            components = _evaluate_control(
                context,
                candidate_line_ids,
                p_env_by_line,
                baseline_r,
                topology,
                u,
                lambda_r,
                rho_phys,
                result_id,
            )
            value = float(components["J_true"])
            valid = bool(np.isfinite(value))
            error = ""
        except Exception as exc:
            components = {}
            value = float(context["config"].objective.invalid_prediction_penalty)
            valid = False
            error = str(exc)
            invalid_calls += 1
        improved = bool(valid and (best is None or value < float(best["J_true"]) - 1e-15))
        if improved:
            best = {**components, "u": u.copy(), "call_idx": int(calls)}
        traces.append(
            {
                "result_id": int(result_id),
                "call_idx": int(calls),
                "lambda_R": float(lambda_r),
                "lambda_L": float(1.0 - lambda_r),
                "rho_phys": float(rho_phys),
                "topology_key": _line_key(topology),
                "J_true": float(value),
                "R_norm": components.get("R_norm", np.nan),
                "L_shed": components.get("L_shed", np.nan),
                "L_shed_cmd": components.get("L_shed_cmd", np.nan),
                "L_shed_gridfm_raw": components.get("L_shed_gridfm_raw", np.nan),
                "L_shed_gridfm_effective": components.get("L_shed_gridfm_effective", np.nan),
                "L_shed_hybrid": components.get("L_shed_hybrid", np.nan),
                "PAC_total": components.get("PAC_total", np.nan),
                "PAC_operational": components.get("PAC_operational", np.nan),
                "PAC_AC": components.get("PAC_AC", np.nan),
                "PAC_model_consistency": components.get("PAC_model_consistency", np.nan),
                "valid": bool(valid),
                "is_best_within_topology": bool(improved),
                "call_runtime_seconds": float(time.perf_counter() - call_started),
                "error": error,
            }
        )
        return value

    scipy_success = False
    scipy_message = ""
    termination_reason = ""
    try:
        result = minimize(
            objective,
            u0,
            method=context["config"].optimizer.method,
            bounds=bounds,
            options={
                "maxiter": 1000,
                "ftol": float(context["config"].optimizer.ftol),
                "gtol": float(context["config"].optimizer.gtol),
                "eps": float(context["config"].optimizer.eps),
                "disp": False,
            },
        )
        scipy_success = bool(result.success)
        scipy_message = str(result.message)
        termination_reason = "scipy_converged" if result.success else "scipy_terminated"
    except CallBudgetReached as exc:
        scipy_message = str(exc)
        termination_reason = "call_budget_reached"
    except Exception as exc:
        scipy_message = str(exc)
        termination_reason = "optimizer_exception"
    if best is None:
        raise RuntimeError(f"No valid continuous recourse call for topology={_line_key(topology)}.")

    u_best = np.asarray(best["u"], dtype=float)
    delta_pg, delta_qg, alpha = decision.split_decision_vector(u_best)
    row = {
        "result_id": int(result_id),
        "J_true": float(best["J_true"]),
        "J_no_phys": float(best["J_no_phys"]),
        "R_raw": float(best["R_raw"]),
        "R_norm": float(best["R_norm"]),
        "L_shed": float(best["L_shed"]),
        "L_shed_cmd": float(best["L_shed_cmd"]),
        "L_shed_gridfm_raw": float(best["L_shed_gridfm_raw"]),
        "L_shed_gridfm_effective": float(best["L_shed_gridfm_effective"]),
        "L_shed_hybrid": float(best["L_shed_hybrid"]),
        "load_shed_mode": str(best["load_shed_mode"]),
        "L_shed_source": L_SHED_SOURCE,
        "PAC_total": float(best["PAC_total"]),
        "PAC_operational": float(best["PAC_operational"]),
        "PAC_AC": float(best["PAC_AC"]),
        "PAC_model_consistency": float(best["PAC_model_consistency"]),
        "pac_operational_weight": float(best["pac_operational_weight"]),
        "pac_ac_weight": float(best["pac_ac_weight"]),
        "pac_model_consistency_weight": float(best["pac_model_consistency_weight"]),
        "risk_contribution": float(best["risk_contribution"]),
        "load_contribution": float(best["load_contribution"]),
        "physics_contribution": float(best["physics_contribution"]),
        "PAC_voltage_limits": float(best["components"]["voltage_limits"]),
        "PAC_thermal_limits": float(best["components"]["thermal_limits"]),
        "PAC_generator_limits": float(best["components"]["generator_limits"]),
        "PAC_generator_limits_eval": float(best["components"]["generator_limits_eval"]),
        "PAC_generator_limits_raw": float(best["components"]["generator_limits_raw"]),
        "PAC_island_source_feasibility": float(best["components"]["island_source_feasibility"]),
        "PAC_p_balance": float(best["components"]["p_balance"]) if np.isfinite(best["components"]["p_balance"]) else np.nan,
        "PAC_q_balance": float(best["components"]["q_balance"]) if np.isfinite(best["components"]["q_balance"]) else np.nan,
        "PAC_p_balance_available": bool(best["components"]["p_balance_available"]),
        "PAC_q_balance_available": bool(best["components"]["q_balance_available"]),
        "PAC_branch_flow_consistency": np.nan,
        "PAC_branch_flow_consistency_available": False,
        "PAC_branch_flow_consistency_weight": 0.0,
        "PAC_cmd_load": float(best["components"]["cmd_load"]),
        "PAC_cmd_Pg": float(best["components"]["cmd_Pg"]),
        "PAC_cmd_Qg": float(best["components"]["cmd_Qg"]),
        "Qg_bounds_available": bool(best["components"]["Qg_bounds_available"]),
        "min_voltage": float(best["min_voltage"]),
        "max_voltage": float(best["max_voltage"]),
        "max_loading_ratio": float(best["max_loading_ratio"]),
        "gridfm_calls": int(calls),
        "invalid_calls": int(invalid_calls),
        "call_budget": int(call_budget),
        "budget_exhausted": bool(calls >= int(call_budget)),
        "termination_reason": termination_reason,
        "scipy_success": bool(scipy_success),
        "scipy_message": scipy_message,
        "runtime_seconds": float(time.perf_counter() - started),
        "best_call_idx": int(best["call_idx"]),
        "source_less_bus_ids": _line_key(source_less),
        "source_less_selected_alpha_forced_zero": int(forced_alpha_count),
        "u_best_json": json.dumps(u_best.tolist()),
        "delta_pg_json": json.dumps(delta_pg.tolist()),
        "delta_qg_json": json.dumps(delta_qg.tolist()),
        "alpha_selected_json": json.dumps(alpha.tolist()),
        "max_abs_delta_pg": float(np.max(np.abs(delta_pg))) if len(delta_pg) else 0.0,
        "max_abs_delta_qg": float(np.max(np.abs(delta_qg))) if len(delta_qg) else 0.0,
        "mean_alpha": float(np.mean(alpha)) if len(alpha) else 1.0,
        "min_alpha": float(np.min(alpha)) if len(alpha) else 1.0,
        "num_variables_at_lower_bound": int(np.sum(np.isclose(u_best, lower, atol=1e-8))),
        "num_variables_at_upper_bound": int(np.sum(np.isclose(u_best, upper, atol=1e-8))),
    }
    audit_frames = {
        "load": best["load_provenance"],
        "risk": best["risk_provenance"],
        "controlled": best["controlled_audit"],
        "mask": best["mask_audit"],
    }
    return row, traces, audit_frames


def _topology_candidates_for_context(
    context: dict,
    scenario_ids: List[str] | None,
    lambdas: Sequence[float],
    proxy_lambdas: Sequence[float] | None,
    stages: Sequence[str] | None,
    stage_d_limit: int | None,
    stage_e_budget: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[DecisionQualityScenario], dict]:
    scenario = context["scenario"]
    selected_stages = _normalize_stages(stages)
    candidate_line_ids = canonicalize_line_ids(scenario, FIXED_T0P30_CANDIDATE_LINE_IDS)
    _validate_candidate_line_ids(candidate_line_ids, int(_edge_array(scenario).shape[1]))
    physical_ids = physical_line_ids(scenario)
    baseline_loading = scenario_baseline_loading_vector(context)
    c_by_line = _consequence_by_line(context["consequence_df"])
    canonical_scenarios = [
        remove_margin_excluded_targets(_canonicalize_decision_quality_scenario(scenario_def, scenario))
        for scenario_def in get_scenarios(scenario_ids)
    ]
    rows = []
    p_env_frames = []
    for scenario_def in canonical_scenarios:
        p_env, p_env_frame = p_env_for_target_margin_scenario(
            scenario_def,
            int(_edge_array(scenario).shape[1]),
            physical_ids,
            baseline_loading,
        )
        p_env_frames.append(p_env_frame)
        baseline_r, _ = _scenario_baseline_exposure(baseline_loading, p_env, physical_ids)
        lambda_pairs = [(float(v), float(v)) for v in lambdas]
        if proxy_lambdas is not None:
            lambda_pairs = [(float(inner), float(proxy)) for proxy in proxy_lambdas for inner in lambdas]
        for lambda_r, lambda_r_proxy in lambda_pairs:
            lambda_l = 1.0 - float(lambda_r)
            lambda_l_proxy = 1.0 - float(lambda_r_proxy)
            if STAGE_D in selected_stages:
                subsets = enumerate_deenergization_subsets(candidate_line_ids, max_deenergized_lines=2)
                if stage_d_limit is not None:
                    subsets = subsets[: int(stage_d_limit)]
                for index, subset in enumerate(subsets):
                    rows.append(
                        {
                            "scenario_id": scenario_def.scenario_id,
                            "scenario_name": scenario_def.scenario_name,
                            "stage": STAGE_D,
                            "stage_label": STAGE_LABELS[STAGE_D],
                            "lambda_R": float(lambda_r),
                            "lambda_L": lambda_l,
                            "lambda_case": _lambda_case(lambda_r),
                            "lambda_R_proxy": float(lambda_r_proxy),
                            "lambda_L_proxy": lambda_l_proxy,
                            "lambda_proxy_case": _lambda_case(lambda_r_proxy),
                            "topology_iteration": int(index + 1),
                            "proposal_method": "stage_d_k2_exhaustive_limited",
                            "shutoff_line_ids": _line_key(subset),
                            "baseline_R": float(baseline_r),
                            "p_env_json": json.dumps({str(int(k)): float(v) for k, v in p_env.items()}),
                            "proxy_objective": np.nan,
                            "topology_proxy_excludes_pac": True,
                        }
                    )
            stage_e_specs = [
                (STAGE_E_K2, 2),
                (STAGE_E_UNCONSTRAINED, None),
            ]
            for stage, max_lines in [(stage, max_lines) for stage, max_lines in stage_e_specs if stage in selected_stages]:
                evaluated_y = []
                for iteration in range(int(stage_e_budget)):
                    proposal = solve_gurobi_master_next_candidate(
                        candidate_line_ids,
                        p_env,
                        baseline_loading,
                        c_by_line,
                        float(lambda_r_proxy),
                        lambda_l_proxy,
                        max_deenergized_lines=max_lines,
                        evaluated_y_vectors=evaluated_y,
                        proxy_type=DEFAULT_PROXY_TYPE,
                    )
                    y_by_line = {int(key): int(value) for key, value in proposal["y_by_line"].items()}
                    evaluated_y.append(y_by_line)
                    rows.append(
                        {
                            "scenario_id": scenario_def.scenario_id,
                            "scenario_name": scenario_def.scenario_name,
                            "stage": stage,
                            "stage_label": STAGE_LABELS[stage],
                            "lambda_R": float(lambda_r),
                            "lambda_L": lambda_l,
                            "lambda_case": _lambda_case(lambda_r),
                            "lambda_R_proxy": float(lambda_r_proxy),
                            "lambda_L_proxy": lambda_l_proxy,
                            "lambda_proxy_case": _lambda_case(lambda_r_proxy),
                            "topology_iteration": int(iteration + 1),
                            "proposal_method": "stage_e_gurobi_proxy_limited",
                            "shutoff_line_ids": _line_key(deenergized_from_y(y_by_line)),
                            "baseline_R": float(baseline_r),
                            "p_env_json": json.dumps({str(int(k)): float(v) for k, v in p_env.items()}),
                            "proxy_objective": proposal.get("proxy_objective", np.nan),
                            "stage_e_proxy_R_hat": proposal.get("proxy_R_hat", np.nan),
                            "stage_e_proxy_L_hat": proposal.get("proxy_L_hat", np.nan),
                            "stage_e_gurobi_status": proposal.get("gurobi_status", np.nan),
                            "topology_proxy_excludes_pac": True,
                        }
                    )
            print(f"[topology pool] {scenario_def.scenario_id} lambda_R={lambda_r:g} proxy={lambda_r_proxy:g}", flush=True)
    p_env_table = pd.concat(p_env_frames, ignore_index=True, sort=False)
    ranking = build_scenario_baseline_loading_ranking(context, baseline_loading, p_env_frames)
    metadata = {
        "candidate_line_ids": candidate_line_ids,
        "physical_line_ids": physical_ids,
        "baseline_loading": baseline_loading,
        "stages": selected_stages,
        "proxy_lambda_values": None if proxy_lambdas is None else _as_float_list(proxy_lambdas),
    }
    return pd.DataFrame(rows).reset_index(drop=True), p_env_table, ranking, canonical_scenarios, metadata


def _run_continuous_pool(
    context: dict,
    topology_pool: pd.DataFrame,
    candidate_line_ids: List[int],
    rho_values: Sequence[float],
    call_budget: int,
    checkpoint_dir: Path | None = None,
    progress_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    rho_values = [float(rho) for rho in rho_values]
    total = int(len(topology_pool) * len(rho_values))
    checkpoint_paths = {
        name: checkpoint_dir / filename for name, filename in CHECKPOINT_TABLES.items()
    } if checkpoint_dir is not None else {}
    existing_results = _read_optional_csv(checkpoint_paths["results"]) if checkpoint_paths else pd.DataFrame()
    completed_ids = (
        set(existing_results["result_id"].astype(int).tolist())
        if "result_id" in existing_results
        else set()
    )
    if progress_path is not None:
        _write_progress(
            progress_path,
            completed=len(completed_ids),
            total=total,
            started=started,
            last_result_id=max(completed_ids) if completed_ids else None,
        )

    for source_index, source in enumerate(topology_pool.itertuples(index=False)):
        p_env = {int(k): float(v) for k, v in json.loads(source.p_env_json).items()}
        topology = _parse_line_key(source.shutoff_line_ids)
        for rho_index, rho in enumerate(rho_values):
            result_id = int(source_index * len(rho_values) + rho_index)
            if result_id in completed_ids:
                continue
            row, traces, audits = _optimize_topology(
                context,
                candidate_line_ids,
                p_env,
                float(source.baseline_R),
                topology,
                float(source.lambda_R),
                float(rho),
                result_id,
                call_budget,
            )
            result_row = {
                **source._asdict(),
                **row,
                "rho_phys": float(rho),
                "model_type": "gnn",
                "candidate_set": "t0p30",
                "p_env_mode": P_ENV_MODE_TARGET_MARGIN,
                "baseline_loading_source": BASELINE_LOADING_SOURCE,
                "R_base_s_source": BASELINE_LOADING_SOURCE,
                "proxy_loading_source": BASELINE_LOADING_SOURCE,
                "post_topology_evaluation_source": POST_TOPOLOGY_CONTINUOUS_SOURCE,
                "continuous_recourse_optimized": True,
                "topology_proxy_excludes_pac": True,
                "stage_c_final_metric_uses_revised_true_exposure": True,
                "stage_c_final_metric_uses_impact": False,
                "line_id_key": source.shutoff_line_ids,
                "num_shutoff_lines": len(topology),
            }
            trace_frame = pd.DataFrame(
                {
                    **trace,
                    "scenario_id": source.scenario_id,
                    "stage": source.stage,
                    "topology_iteration": int(source.topology_iteration),
                }
                for trace in traces
            )
            if checkpoint_paths:
                _append_checkpoint(checkpoint_paths["traces"], trace_frame)
                _append_checkpoint(checkpoint_paths["load"], audits["load"])
                _append_checkpoint(checkpoint_paths["risk"], audits["risk"])
                _append_checkpoint(checkpoint_paths["controlled"], audits["controlled"])
                _append_checkpoint(checkpoint_paths["mask"], audits["mask"])
                _append_checkpoint(checkpoint_paths["results"], pd.DataFrame([result_row]))
            completed_ids.add(result_id)
            completed = len(completed_ids)
            if progress_path is not None:
                _write_progress(
                    progress_path,
                    completed=completed,
                    total=total,
                    started=started,
                    last_result_id=result_id,
                )
            if completed % 50 == 0 or completed == total:
                print(f"[continuous] completed {completed}/{total} topology/rho optimizations", flush=True)

    if not checkpoint_paths:
        raise ValueError("checkpoint_dir is required so continuous runs can be resumed safely.")

    results = _read_optional_csv(checkpoint_paths["results"]).drop_duplicates("result_id", keep="last")
    valid_ids = set(results["result_id"].astype(int).tolist()) if "result_id" in results else set()

    def load_checkpoint(name: str) -> pd.DataFrame:
        frame = _read_optional_csv(checkpoint_paths[name])
        if frame.empty or "result_id" not in frame:
            return frame
        frame = frame[frame["result_id"].astype(int).isin(valid_ids)].copy()
        return frame.drop_duplicates().reset_index(drop=True)

    traces = load_checkpoint("traces")
    load_prov = load_checkpoint("load")
    risk_prov = load_checkpoint("risk")
    controlled = load_checkpoint("controlled")
    mask = load_checkpoint("mask")
    if progress_path is not None:
        _write_progress(
            progress_path,
            completed=len(valid_ids),
            total=total,
            started=started,
            last_result_id=max(valid_ids) if valid_ids else None,
            status="complete" if len(valid_ids) == total else "running",
        )
    return (
        results.sort_values("result_id", kind="mergesort").reset_index(drop=True),
        traces,
        load_prov,
        risk_prov,
        controlled,
        mask,
    )


def _best_by_rho_scenario_lambda_stage(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[np.isfinite(results["J_true"].astype(float))].copy()
    keys = ["model_type", "scenario_id", "scenario_name", "lambda_R", "lambda_L", "rho_phys", "stage"]
    for optional in ["lambda_R_proxy", "lambda_L_proxy", "lambda_proxy_case"]:
        if optional in ok.columns:
            keys.append(optional)
    return (
        ok.sort_values(keys + ["J_true", "L_shed", "num_shutoff_lines", "line_id_key"], kind="mergesort")
        .groupby(keys, as_index=False, dropna=False)
        .first()
        .reset_index(drop=True)
    )


def _expected_vs_observed_all_lambdas(best: pd.DataFrame, scenarios: List[DecisionQualityScenario]) -> pd.DataFrame:
    rows = []
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    for _, row in best.iterrows():
        scenario = by_id[str(row["scenario_id"])]
        expected = set(int(line_id) for line_id in scenario.expected_target_set)
        observed = set(_parse_line_key(row.get("shutoff_line_ids", "")))
        observed_targets = observed.intersection(expected)
        observed_non_targets = observed.difference(expected)
        missed_targets = expected.difference(observed)
        target_recall = float(len(observed_targets) / max(len(expected), 1))
        target_precision = float(len(observed_targets) / len(observed)) if observed else 0.0
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.scenario_name,
                "stage": row["stage"],
                "stage_label": row["stage_label"],
                "lambda_R": float(row["lambda_R"]),
                "lambda_L": float(row["lambda_L"]),
                "rho_phys": float(row["rho_phys"]),
                "expected_target_line_ids": _line_key(expected),
                "expected_possible_behavior": _expected_outcome(scenario, float(row["lambda_R"])),
                "observed_shutoff_line_ids": row.get("shutoff_line_ids", ""),
                "observed_target_subset": _line_key(observed_targets),
                "observed_non_target_lines": _line_key(observed_non_targets),
                "expected_targets_not_selected": _line_key(missed_targets),
                "num_expected_target_lines": int(len(expected)),
                "num_observed_shutoff_lines": int(len(observed)),
                "num_observed_target_lines": int(len(observed_targets)),
                "num_observed_non_target_lines": int(len(observed_non_targets)),
                "target_recall": target_recall,
                "target_precision": target_precision,
                "target_overlap_fraction": target_recall,
                "creates_source_less_island": bool(row.get("creates_source_less_island", False)),
                "R_norm": float(row["R_norm"]),
                "L_shed": float(row["L_shed"]),
                "PAC_total": float(row["PAC_total"]),
                "J_true": float(row["J_true"]),
            }
        )
    return pd.DataFrame(rows)


def _runtime_summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["scenario_id", "lambda_R", "rho_phys", "stage", "stage_label"], as_index=False)
        .agg(
            num_topologies=("result_id", "count"),
            total_runtime_seconds=("runtime_seconds", "sum"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            total_gridfm_calls=("gridfm_calls", "sum"),
            mean_gridfm_calls=("gridfm_calls", "mean"),
            invalid_calls=("invalid_calls", "sum"),
            budget_exhaustion_rate=("budget_exhausted", "mean"),
            scipy_success_rate=("scipy_success", "mean"),
        )
        .sort_values(["scenario_id", "rho_phys", "lambda_R", "stage"])
    )


def _fixed_vs_continuous(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results[results["topology_iteration"].astype(int).eq(1)].copy()
    # This table is named for the comparison slot; the current reduced study does
    # not re-run a separate fixed-control evaluator, so baseline rows identify
    # first-iteration topology values for quick sanity checking.
    return baseline[
        [
            "scenario_id",
            "lambda_R",
            "rho_phys",
            "stage",
            "shutoff_line_ids",
            "R_norm",
            "L_shed",
            "PAC_total",
            "J_true",
        ]
    ].copy()


def _methodology_checks(
    results: pd.DataFrame,
    p_env_table: pd.DataFrame,
    ranking: pd.DataFrame,
    load_prov: pd.DataFrame,
    risk_prov: pd.DataFrame,
    controlled: pd.DataFrame,
    mask: pd.DataFrame,
    scenario,
    candidate_line_ids: Sequence[int],
) -> pd.DataFrame:
    rows = []

    def add(name: str, passed: bool, details: str, severity: str = "hard") -> None:
        rows.append({"check_name": name, "passed": bool(passed), "severity": severity, "details": details})

    add("baseline_loading_source", results["baseline_loading_source"].astype(str).eq(BASELINE_LOADING_SOURCE).all(), BASELINE_LOADING_SOURCE)
    add("proxy_loading_source", results["proxy_loading_source"].astype(str).eq(BASELINE_LOADING_SOURCE).all(), BASELINE_LOADING_SOURCE)
    add("R_base_s_source", results["R_base_s_source"].astype(str).eq(BASELINE_LOADING_SOURCE).all(), BASELINE_LOADING_SOURCE)
    add("post_topology_evaluation_source", results["post_topology_evaluation_source"].astype(str).eq(POST_TOPOLOGY_CONTINUOUS_SOURCE).all(), POST_TOPOLOGY_CONTINUOUS_SOURCE)
    add("continuous_recourse_optimized", results["continuous_recourse_optimized"].astype(bool).all(), "Continuous recourse must be enabled.")
    add("topology_proxy_excludes_pac", results["topology_proxy_excludes_pac"].astype(bool).all(), "Topology proxy must exclude PAC.")
    add("L_shed_source_effective_alpha", results["L_shed_source"].astype(str).eq(L_SHED_SOURCE).all(), L_SHED_SOURCE)
    physical = set(int(item) for item in physical_line_ids(scenario))
    add("candidate_lines_are_canonical_physical", set(int(item) for item in candidate_line_ids).issubset(physical), "Candidate IDs must be canonical physical branches.")
    add("line32_excluded_from_targets", not p_env_table[p_env_table["is_target"].astype(bool)]["line_id"].astype(int).eq(32).any(), "Line 32 must not be a target.")
    add("target_p_env_high", p_env_table[p_env_table["is_target"].astype(bool)]["p_env"].astype(float).eq(1.0).all(), "Targets must have p_env=1.")
    non_targets = p_env_table[~p_env_table["is_target"].astype(bool)]
    add("non_target_margin_scores", (non_targets["line_score_after_calibration"].astype(float) <= non_targets["non_target_cap_score"].astype(float) + 1e-10).all(), "Non-target scores must stay below the margin cap.")
    add("controlled_features_visible", mask[mask["is_controlled_feature"].astype(bool)]["controlled_visible_to_gridfm"].astype(bool).all(), "Controlled features must not be masked.")
    clamp_err = float(controlled["abs_command_vs_eval_error"].astype(float).max()) if not controlled.empty else np.inf
    add("controlled_values_clamped", clamp_err <= CONTROL_CLAMP_ATOL, f"max_abs_error={clamp_err:.3g}; atol={CONTROL_CLAMP_ATOL:g}")
    load_sum = load_prov.groupby("result_id")["load_shed_weighted"].sum()
    load_obs = results.set_index("result_id")["L_shed"].astype(float)
    load_err = float((load_sum - load_obs.loc[load_sum.index]).abs().max()) if len(load_sum) else np.inf
    add("load_provenance_sums_to_L_shed", load_err <= 1e-9, f"max_abs_error={load_err:.3g}")
    for prov_col, result_col, check_name in [
        ("load_shed_cmd_weighted", "L_shed_cmd", "load_shed_cmd_sums_to_L_shed_cmd"),
        ("load_shed_gridfm_raw_weighted", "L_shed_gridfm_raw", "load_shed_gridfm_raw_sums_to_L_shed_gridfm_raw"),
        ("load_shed_gridfm_effective_weighted", "L_shed_gridfm_effective", "load_shed_gridfm_effective_sums_to_L_shed_gridfm_effective"),
        ("load_shed_hybrid_weighted", "L_shed_hybrid", "load_shed_hybrid_sums_to_L_shed_hybrid"),
    ]:
        if prov_col in load_prov.columns and result_col in results.columns:
            local_sum = load_prov.groupby("result_id")[prov_col].sum()
            local_obs = results.set_index("result_id")[result_col].astype(float)
            local_err = float((local_sum - local_obs.loc[local_sum.index]).abs().max()) if len(local_sum) else np.inf
            add(check_name, local_err <= 1e-9, f"max_abs_error={local_err:.3g}")
    if "source_less_island_correction_applied" in load_prov.columns and "alpha_effective" in load_prov.columns:
        corrected = load_prov[load_prov["source_less_island_correction_applied"].astype(bool)]
        corrected_ok = corrected["alpha_effective"].astype(float).eq(0.0).all() if not corrected.empty else True
        add("source_less_island_loads_counted_unserved", corrected_ok, "All source-less islanded buses must have alpha_effective=0.")
    if {"bus_role", "is_source_less_islanded", "alpha_effective_hybrid", "alpha_gridfm_effective"}.issubset(load_prov.columns):
        nonselected_connected = load_prov[
            load_prov["bus_role"].astype(str).eq("nonselected")
            & ~load_prov["is_source_less_islanded"].astype(bool)
            & (load_prov["Pd_base"].astype(float) > 1e-12)
        ]
        err = (
            float((nonselected_connected["alpha_effective_hybrid"].astype(float) - nonselected_connected["alpha_gridfm_effective"].astype(float)).abs().max())
            if not nonselected_connected.empty
            else 0.0
        )
        add("nonselected_connected_load_uses_gridfm_alpha_in_hybrid", err <= 1e-9, f"max_abs_error={err:.3g}")
    if {"bus_role", "is_source_less_islanded", "alpha_effective_hybrid", "alpha_commanded"}.issubset(load_prov.columns):
        selected_connected = load_prov[
            load_prov["bus_role"].astype(str).eq("selected_load")
            & ~load_prov["is_source_less_islanded"].astype(bool)
        ]
        err = (
            float((selected_connected["alpha_effective_hybrid"].astype(float) - selected_connected["alpha_commanded"].astype(float)).abs().max())
            if not selected_connected.empty
            else 0.0
        )
        add("selected_connected_load_uses_commanded_alpha_in_hybrid", err <= 1e-9, f"max_abs_error={err:.3g}")
    risk_sum = risk_prov.groupby("result_id")["risk_raw_contribution"].sum()
    risk_obs = results.set_index("result_id")["R_raw"].astype(float)
    risk_err = float((risk_sum - risk_obs.loc[risk_sum.index]).abs().max()) if len(risk_sum) else np.inf
    add("risk_provenance_sums_to_R_raw", risk_err <= 1e-9, f"max_abs_error={risk_err:.3g}")
    j_no_phys = results["lambda_R"].astype(float) * results["R_norm"].astype(float) + (1.0 - results["lambda_R"].astype(float)) * results["L_shed"].astype(float)
    j_err = float((j_no_phys - results["J_no_phys"].astype(float)).abs().max()) if len(results) else np.inf
    add("J_no_phys_formula", j_err <= 1e-9, f"max_abs_error={j_err:.3g}")
    j_true = results["J_no_phys"].astype(float) + results["rho_phys"].astype(float) * results["PAC_total"].astype(float)
    jt_err = float((j_true - results["J_true"].astype(float)).abs().max()) if len(results) else np.inf
    add("J_true_formula", jt_err <= 1e-9, f"max_abs_error={jt_err:.3g}")
    if {"PAC_operational", "PAC_AC", "PAC_model_consistency", "PAC_total", "pac_operational_weight", "pac_ac_weight", "pac_model_consistency_weight"}.issubset(results.columns):
        pac_recalc = (
            results["pac_operational_weight"].astype(float) * results["PAC_operational"].astype(float)
            + results["pac_ac_weight"].astype(float) * results["PAC_AC"].astype(float)
            + results["pac_model_consistency_weight"].astype(float) * results["PAC_model_consistency"].astype(float)
        )
        pac_err = float((pac_recalc - results["PAC_total"].astype(float)).abs().max()) if len(results) else np.inf
        add("pac_total_equals_operational_plus_ac_plus_model", pac_err <= 1e-9, f"max_abs_error={pac_err:.3g}")
    if {"PAC_branch_flow_consistency_available", "PAC_branch_flow_consistency_weight"}.issubset(results.columns):
        unavailable = ~results["PAC_branch_flow_consistency_available"].astype(bool)
        weight_zero = results["PAC_branch_flow_consistency_weight"].astype(float).eq(0.0)
        add("branch_flow_consistency_unavailable_zero_weight", bool((unavailable & weight_zero).all()), "GridFM has no independent branch-flow channel.")
    add("true_risk_excludes_impact", risk_prov["impact_used_in_true_risk"].astype(str).eq("False").all() if not risk_prov.empty else False, "Impact must not be used in true risk.")
    self_loops = np.asarray(getattr(scenario, "is_self_loop", np.zeros(int(_edge_array(scenario).shape[1]))), dtype=bool)
    add("self_loops_excluded", not any(bool(self_loops[int(line_id)]) for line_id in candidate_line_ids), "Candidate set must exclude self-loops.")
    return pd.DataFrame(rows)


def _plot_per_rho(results: pd.DataFrame, best: pd.DataFrame, expected: pd.DataFrame, plots_dir: Path) -> None:
    for (rho, scenario_id), group in results.groupby(["rho_phys", "scenario_id"], sort=True):
        out = plots_dir / "per_rho" / f"rho{rho:g}" / str(scenario_id)
        scored = group.copy()
        scored["status"] = "ok"
        local_best = best[
            np.isclose(best["rho_phys"].astype(float), float(rho)) & best["scenario_id"].eq(scenario_id)
        ].copy()
        local_expected = expected[np.isclose(expected["rho_phys"].astype(float), float(rho)) & expected["scenario_id"].eq(scenario_id)]
        pareto_points, frontier = _pareto_tables(scored)
        _plot_scenario_outputs_dynamic(
            str(scenario_id),
            scored,
            local_best,
            local_expected,
            pareto_points,
            frontier,
            out,
            float(rho),
        )


def _selected_contains_line(value: str, line_id: int) -> bool:
    return int(line_id) in set(_parse_line_key(value))


def _actual_lambda_values(frame: pd.DataFrame) -> List[float]:
    return sorted(float(value) for value in frame["lambda_R"].dropna().unique())


def _actual_stage_values(frame: pd.DataFrame) -> List[str]:
    present = set(frame["stage"].dropna().astype(str))
    ordered = [stage for stage in STAGES if stage in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _wrapped_line_ids(value, chunk_size: int = 6) -> str:
    values = _parse_line_ids(value)
    if not values:
        return ""
    chunks = [values[index : index + int(chunk_size)] for index in range(0, len(values), int(chunk_size))]
    return "\n".join(",".join(str(line_id) for line_id in chunk) for chunk in chunks)


def _plot_scenario_outputs_dynamic(
    scenario_id: str,
    scored: pd.DataFrame,
    best: pd.DataFrame,
    expected_comparison: pd.DataFrame,
    pareto_points: pd.DataFrame,
    frontier: pd.DataFrame,
    output_dir: Path,
    rho_phys: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {STAGE_D: "#4C78A8", STAGE_E_K2: "#F58518", STAGE_E_UNCONSTRAINED: "#54A24B"}
    lambda_values = _actual_lambda_values(scored[scored["scenario_id"].eq(scenario_id)])
    stage_values = _actual_stage_values(scored[scored["scenario_id"].eq(scenario_id)])
    if not lambda_values or not stage_values:
        return

    fig, axes = plt.subplots(1, len(lambda_values), figsize=(5.8 * len(lambda_values), 5.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, lambda_r in zip(axes, lambda_values):
        local = scored[
            scored["scenario_id"].eq(scenario_id) & np.isclose(scored["lambda_R"].astype(float), lambda_r)
        ].copy()
        topology_lines = []
        max_iterations = max(int(local["topology_iteration"].max()), 1) if not local.empty else 1
        for stage in stage_values:
            group = local[local["stage"].eq(stage)].sort_values("topology_iteration", kind="mergesort")
            if group.empty:
                continue
            best_so_far = group["J_true"].astype(float).cummin()
            color = colors.get(stage, "#6B6B6B")
            ax.step(
                group["topology_iteration"],
                best_so_far,
                where="post",
                color=color,
                linewidth=2,
                label=STAGE_LABELS.get(stage, stage),
            )
            final_row = group.loc[group["J_true"].idxmin()]
            ax.scatter(
                [int(group["topology_iteration"].iloc[-1])],
                [float(best_so_far.iloc[-1])],
                color="#D62728",
                s=38,
                zorder=5,
            )
            topology_lines.append(f"{STAGE_LABELS.get(stage, stage)}: [{final_row['shutoff_line_ids'] or 'none'}]")
        if topology_lines:
            ax.text(
                0.98,
                0.78,
                "\n".join(topology_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "#C8C8C8", "alpha": 0.88, "pad": 4},
            )
        ax.set_xlim(left=0, right=max_iterations * 1.03)
        ax.set_title(f"lambda_R={lambda_r:g}, lambda_L={1.0 - lambda_r:g}")
        ax.set_xlabel("Topology iteration")
        ax.set_ylabel("Best topology objective so far, J_true")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"{scenario_id}: Lambda-Zoom Decision Quality, rho_phys={rho_phys:g}\n"
        f"J = lambda_R R_norm + lambda_L L_shed + {rho_phys:g} PAC_total"
    )
    _savefig(fig, output_dir / "traditional_lambda_objective_convergence.png", dpi=180)
    _savefig(fig, output_dir / "lambda_zoom_objective_convergence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(stage_values), figsize=(5.8 * len(stage_values), 5.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, stage in zip(axes, stage_values):
        points = pareto_points[
            pareto_points["scenario_id"].eq(scenario_id) & pareto_points["stage"].eq(stage)
        ]
        front = frontier[frontier["scenario_id"].eq(scenario_id) & frontier["stage"].eq(stage)]
        ax.scatter(points["L_shed"], points["R_norm"], s=20, color="#B8B8B8", alpha=0.35, label="Evaluated")
        if not front.empty:
            ordered = front.sort_values(["L_shed", "R_norm"], kind="mergesort")
            ax.scatter(ordered["L_shed"], ordered["R_norm"], s=48, color="#D62728", label="Nondominated")
            ax.plot(ordered["L_shed"], ordered["R_norm"], color="#D62728", linewidth=1.4)
        ax.set_title(STAGE_LABELS.get(stage, stage))
        ax.set_xlabel("Effective load shed, L_shed")
        ax.set_ylabel("Wildfire exposure, R_norm")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"{scenario_id}: Pareto Frontier for Lambda Zoom\n"
        f"lambda_R={min(lambda_values):g}..{max(lambda_values):g}; stages={', '.join(stage_values)}"
    )
    _savefig(fig, output_dir / "pareto_frontier_scatter.png", dpi=180)
    _savefig(fig, output_dir / "lambda_zoom_pareto_frontier_scatter.png", dpi=180)
    plt.close(fig)

    comparison = expected_comparison[expected_comparison["scenario_id"].eq(scenario_id)].copy()
    if comparison.empty:
        return
    comparison_lambdas = _actual_lambda_values(comparison)
    comparison_stages = _actual_stage_values(comparison)
    expected_targets = str(comparison["expected_target_line_ids"].iloc[0])
    fig, axes = plt.subplots(
        len(comparison_stages),
        len(comparison_lambdas),
        figsize=(4.2 * len(comparison_lambdas), 2.8 * len(comparison_stages)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, stage in enumerate(comparison_stages):
        for column_index, lambda_r in enumerate(comparison_lambdas):
            ax = axes[row_index, column_index]
            row = comparison[
                comparison["stage"].eq(stage) & np.isclose(comparison["lambda_R"].astype(float), lambda_r)
            ]
            ax.set_xticks([])
            ax.set_yticks([])
            if row.empty:
                ax.set_facecolor("#F2F2F2")
                ax.text(0.5, 0.5, "Not evaluated", ha="center", va="center", transform=ax.transAxes)
                continue
            item = row.iloc[0]
            recall = float(item.get("target_recall", item.get("target_overlap_fraction", 0.0)))
            precision = float(item.get("target_precision", 0.0))
            creates_island = bool(item["creates_source_less_island"])
            if creates_island:
                facecolor = "#F8C9C9"
            elif recall > 0.0:
                facecolor = "#D8EFD3"
            elif str(item["observed_shutoff_line_ids"]).strip() in {"", "nan"}:
                facecolor = "#E8E8E8"
            else:
                facecolor = "#F6E3B4"
            ax.set_facecolor(facecolor)
            observed = _wrapped_line_ids(item["observed_shutoff_line_ids"])
            target_hits = _wrapped_line_ids(item["observed_target_subset"])
            non_targets = _wrapped_line_ids(item["observed_non_target_lines"])
            text = (
                f"Selected:\n[{observed}]\n"
                f"Target hits: [{target_hits}]\n"
                f"Other:\n[{non_targets}]\n"
                f"Recall={recall:.2f}, precision={precision:.2f}\n"
                f"PAC={float(item['PAC_total']):.3g}\n"
                f"Island={'yes' if creates_island else 'no'}"
            )
            fontsize = 6.2 if len(_parse_line_ids(item["observed_shutoff_line_ids"])) > 10 else 8
            ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize, transform=ax.transAxes)
            if row_index == 0:
                ax.set_title(f"lambda_R={lambda_r:g}", fontsize=10)
            if column_index == 0:
                ax.set_ylabel(STAGE_LABELS.get(stage, stage), fontsize=9)
    fig.suptitle(
        f"{scenario_id}: Expected Versus Selected Shutoff Lines, rho_phys={rho_phys:g}\n"
        f"Expected target set: [{expected_targets}]"
    )
    _savefig(fig, output_dir / "expected_vs_selected_shutoff_lines.png", dpi=180)
    _savefig(fig, output_dir / "lambda_zoom_expected_vs_selected_shutoff_lines.png", dpi=180)
    plt.close(fig)


def _plot_lambda_zoom_and_summary(
    best: pd.DataFrame,
    expected: pd.DataFrame,
    mask: pd.DataFrame,
    plots_dir: Path,
    rho_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    physics_rows = []
    cost_rows = []
    frequency_rows = []
    cmap = plt.cm.viridis

    group_cols = ["scenario_id", "stage"]
    has_proxy_axis = "lambda_R_proxy" in best.columns and best["lambda_R_proxy"].nunique(dropna=True) > 1
    if has_proxy_axis:
        group_cols.append("lambda_R_proxy")

    for group_key, group in best.groupby(group_cols, sort=True):
        if has_proxy_axis:
            scenario_id, stage, proxy_lambda = group_key
        else:
            scenario_id, stage = group_key
            proxy_lambda = None
        group = group.sort_values("lambda_R", kind="mergesort")
        lambda_values = _actual_lambda_values(group)
        out = plots_dir / "lambda_zoom" / f"rho{rho_value:g}" / str(scenario_id) / str(stage)
        title_suffix = ""
        if proxy_lambda is not None:
            out = out / f"proxy_lambda_{float(proxy_lambda):g}"
            title_suffix = f", proxy={float(proxy_lambda):g}"
        out.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        scatter = ax.scatter(
            group["L_shed"],
            group["R_norm"],
            c=group["lambda_R"].astype(float),
            cmap=cmap,
            s=64,
            edgecolor="#333333",
            linewidth=0.6,
        )
        ax.plot(group["L_shed"], group["R_norm"], color="#4C78A8", linewidth=1.4, alpha=0.75)
        for _, row in group.iterrows():
            ax.annotate(f"{float(row['lambda_R']):g}", (row["L_shed"], row["R_norm"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Lambda-Zoom Selected Path, rho={rho_value:g}{title_suffix}")
        ax.set_xlabel("Effective load shed, L_shed")
        ax.set_ylabel("Wildfire exposure, R_norm")
        ax.grid(True, alpha=0.25)
        fig.colorbar(scatter, ax=ax, label="lambda_R")
        _savefig(fig, out / "risk_load_selected_path_by_lambda.png", dpi=180)
        plt.close(fig)

        unique = group.drop_duplicates(["line_id_key", "R_norm", "L_shed"]).copy()
        unique["is_nondominated_selected"] = _nondominated_mask(unique)
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        ax.scatter(unique["L_shed"], unique["R_norm"], color="#C8C8C8", alpha=0.45, s=38, label="Selected")
        nd = unique[unique["is_nondominated_selected"]]
        if not nd.empty:
            scatter = ax.scatter(
                nd["L_shed"],
                nd["R_norm"],
                c=nd["lambda_R"].astype(float),
                cmap=cmap,
                s=70,
                edgecolor="#333333",
                linewidth=0.6,
                label="Nondominated selected",
            )
            fig.colorbar(scatter, ax=ax, label="lambda_R")
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Nondominated Selected Points{title_suffix}")
        ax.set_xlabel("Effective load shed, L_shed")
        ax.set_ylabel("Wildfire exposure, R_norm")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        _savefig(fig, out / "nondominated_selected_points_by_lambda.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
        ax.plot(group["lambda_R"], group["PAC_total"], marker="o", linewidth=2, color="#4C78A8")
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Physics Penalty Along Lambda Zoom{title_suffix}")
        ax.set_xlabel("lambda_R")
        ax.set_ylabel("PAC_total")
        ax.grid(True, alpha=0.25)
        _savefig(fig, out / "physics_penalty_by_lambda.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
        ax.plot(group["lambda_R"], group["J_no_phys"], marker="o", linewidth=2, color="#F58518")
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Nonphysics Tradeoff Along Lambda Zoom{title_suffix}")
        ax.set_xlabel("lambda_R")
        ax.set_ylabel("J_no_phys = lambda_R R_norm + lambda_L L_shed")
        ax.grid(True, alpha=0.25)
        _savefig(fig, out / "nonphysics_tradeoff_by_lambda.png", dpi=180)
        plt.close(fig)

        line23_values = []
        for _, row in group.iterrows():
            selected = _selected_contains_line(str(row["shutoff_line_ids"]), 23)
            line23_values.append(float(selected))
            physics_rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage": stage,
                    "lambda_R": float(row["lambda_R"]),
                    "lambda_R_proxy": float(row.get("lambda_R_proxy", row["lambda_R"])),
                    "rho_phys": float(row["rho_phys"]),
                    "PAC_total": float(row["PAC_total"]),
                    "PAC_voltage_limits": float(row["PAC_voltage_limits"]),
                    "PAC_thermal_limits": float(row["PAC_thermal_limits"]),
                    "PAC_generator_limits": float(row["PAC_generator_limits"]),
                    "PAC_island_source_feasibility": float(row["PAC_island_source_feasibility"]),
                    "max_abs_delta_qg": float(row.get("max_abs_delta_qg", np.nan)),
                    "L_shed": float(row["L_shed"]),
                    "shutoff_line_ids": row["shutoff_line_ids"],
                }
            )
            cost_rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage": stage,
                    "lambda_R": float(row["lambda_R"]),
                    "lambda_R_proxy": float(row.get("lambda_R_proxy", row["lambda_R"])),
                    "rho_phys": float(row["rho_phys"]),
                    "nonphysics_tradeoff_T": float(row["J_no_phys"]),
                    "cost_of_feasibility_vs_rho0": np.nan,
                    "PAC_total": float(row["PAC_total"]),
                    "L_shed": float(row["L_shed"]),
                    "shutoff_line_ids": row["shutoff_line_ids"],
                }
            )
            frequency_rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage": stage,
                    "rho_phys": float(row["rho_phys"]),
                    "lambda_R": float(row["lambda_R"]),
                    "lambda_R_proxy": float(row.get("lambda_R_proxy", row["lambda_R"])),
                    "line23_selection_frequency": float(selected),
                    "num_selected_rows": 1,
                }
            )

        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        ax.plot(lambda_values, line23_values, marker="o", linewidth=2, color="#4C78A8")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Line 23 Selection Along Lambda Zoom{title_suffix}")
        ax.set_xlabel("lambda_R")
        ax.set_ylabel("Line 23 selected")
        ax.grid(True, alpha=0.25)
        _savefig(fig, out / "line23_selection_by_lambda.png", dpi=180)
        plt.close(fig)

        for y, filename, title in [
            ("L_shed", "effective_load_shed_by_lambda.png", "Effective Load Shed"),
            ("max_abs_delta_qg", "pq_control_clamping_consistency_by_lambda.png", "PQ Control Movement"),
            ("gridfm_calls", "recourse_solver_runtime_and_calls_by_lambda.png", "Recourse GridFM Calls"),
        ]:
            fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
            ax.plot(group["lambda_R"], group[y], marker="o", linewidth=2)
            ax.set_xlabel("lambda_R")
            ax.set_ylabel(y)
            ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: {title}{title_suffix}")
            ax.grid(True, alpha=0.25)
            _savefig(fig, out / filename, dpi=180)
            plt.close(fig)

    summary_dir = plots_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    physics = pd.DataFrame(physics_rows)
    cost = pd.DataFrame(cost_rows)
    line23 = pd.DataFrame(frequency_rows)
    tradeoff = (
        best.groupby(["scenario_id", "lambda_R"], as_index=False)
        .agg(
            avg_PAC_total=("PAC_total", "mean"),
            avg_R_norm=("R_norm", "mean"),
            avg_L_shed=("L_shed", "mean"),
            avg_J_no_phys=("J_no_phys", "mean"),
            avg_J_true=("J_true", "mean"),
        )
        .sort_values(["scenario_id", "lambda_R"])
    )
    target = expected.groupby(["scenario_id", "lambda_R"], as_index=False).agg(
        avg_target_recall=("target_recall", "mean"),
        avg_target_precision=("target_precision", "mean"),
        avg_target_overlap=("target_overlap_fraction", "mean"),
    )
    line23_summary = line23.groupby(["scenario_id", "lambda_R"], as_index=False).agg(
        line23_selection_frequency=("line23_selection_frequency", "mean")
    )
    mask_summary = mask.groupby("is_controlled_feature", as_index=False).agg(visible_rate=("controlled_visible_to_gridfm", "mean"))

    def lambda_plot(frame: pd.DataFrame, y: str, title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
        for scenario_id, local in frame.groupby("scenario_id", sort=True):
            ax.plot(local["lambda_R"], local[y], marker="o", linewidth=2, label=scenario_id)
        ax.set_title(title)
        ax.set_xlabel("lambda_R")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(summary_dir / filename, dpi=180)
        plt.close(fig)

    lambda_plot(tradeoff, "avg_PAC_total", "Average PAC by Lambda_R, rho=2", "Average PAC_total", "lambda_zoom_avg_pac_by_lambda.png")
    lambda_plot(tradeoff, "avg_R_norm", "Average Wildfire Exposure by Lambda_R, rho=2", "Average R_norm", "lambda_zoom_avg_r_norm_by_lambda.png")
    lambda_plot(tradeoff, "avg_L_shed", "Average Effective Load Shed by Lambda_R, rho=2", "Average L_shed", "lambda_zoom_avg_l_shed_by_lambda.png")
    lambda_plot(target, "avg_target_recall", "Target Recall by Lambda_R, rho=2", "Average target recall", "lambda_zoom_target_recall_by_lambda.png")
    lambda_plot(target, "avg_target_precision", "Target Precision by Lambda_R, rho=2", "Average target precision", "lambda_zoom_target_precision_by_lambda.png")
    lambda_plot(target, "avg_target_overlap", "Target Recall by Lambda_R, rho=2", "Average target recall", "lambda_zoom_target_overlap_by_lambda.png")
    lambda_plot(line23_summary, "line23_selection_frequency", "Line 23 Frequency by Lambda_R, rho=2", "Selection frequency", "lambda_zoom_line23_by_lambda.png")

    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    ax.bar(mask_summary["is_controlled_feature"].astype(str), mask_summary["visible_rate"].astype(float), color="#4C78A8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Visible rate")
    ax.set_title("Masking/Clamping Audit Summary")
    fig.savefig(summary_dir / "masking_clamping_audit_summary.png", dpi=180)
    plt.close(fig)
    return physics, cost, line23, tradeoff


def _plot_cross_and_summary(
    results: pd.DataFrame,
    best: pd.DataFrame,
    expected: pd.DataFrame,
    mask: pd.DataFrame,
    plots_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    physics_rows = []
    cost_rows = []
    frequency_rows = []
    rho_values = sorted(float(value) for value in best["rho_phys"].dropna().unique())
    if len(rho_values) == 1:
        return _plot_lambda_zoom_and_summary(best, expected, mask, plots_dir, rho_values[0])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(rho_values), 1)))
    color_by_rho = {rho: colors[index] for index, rho in enumerate(rho_values)}

    for (scenario_id, stage), group in best.groupby(["scenario_id", "stage"], sort=True):
        out = plots_dir / "cross_rho" / str(scenario_id) / str(stage)
        out.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        for rho in rho_values:
            local = group[np.isclose(group["rho_phys"].astype(float), rho)].sort_values("lambda_R")
            if local.empty:
                continue
            ax.plot(local["R_norm"], local["L_shed"], marker="o", linewidth=1.5, markersize=3.5, color=color_by_rho[rho], label=f"rho={rho:g}")
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Risk/Load Selected Path")
        ax.set_xlabel("R_norm")
        ax.set_ylabel("L_shed")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        _savefig(fig, out / "risk_load_pareto_overlay_all_lambdas.png", dpi=180)
        plt.close(fig)

        unique = group.drop_duplicates(["rho_phys", "line_id_key", "R_norm", "L_shed"]).copy()
        unique["is_nondominated_selected"] = _nondominated_mask(unique)
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        ax.scatter(unique["R_norm"], unique["L_shed"], color="#C8C8C8", alpha=0.35, s=28, label="Selected")
        nd = unique[unique["is_nondominated_selected"]]
        for rho in rho_values:
            local = nd[np.isclose(nd["rho_phys"].astype(float), rho)]
            if local.empty:
                continue
            ax.scatter(local["R_norm"], local["L_shed"], color=color_by_rho[rho], s=55, label=f"rho={rho:g}")
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Nondominated Selected Points")
        ax.set_xlabel("R_norm")
        ax.set_ylabel("L_shed")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        _savefig(fig, out / "nondominated_points_colored_by_rho_all_lambdas.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
        lambda_values = sorted(float(value) for value in group["lambda_R"].dropna().unique())
        for lambda_r in lambda_values:
            local = group[np.isclose(group["lambda_R"].astype(float), lambda_r)].sort_values("rho_phys")
            if local.empty:
                continue
            ax.plot(local["rho_phys"], local["PAC_total"], marker="o", linewidth=2, label=f"lambda_R={lambda_r:g}")
            for _, row in local.iterrows():
                physics_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "stage": stage,
                        "lambda_R": float(row["lambda_R"]),
                        "rho_phys": float(row["rho_phys"]),
                        "PAC_total": float(row["PAC_total"]),
                        "PAC_voltage_limits": float(row["PAC_voltage_limits"]),
                        "PAC_thermal_limits": float(row["PAC_thermal_limits"]),
                        "PAC_generator_limits": float(row["PAC_generator_limits"]),
                        "PAC_island_source_feasibility": float(row["PAC_island_source_feasibility"]),
                        "max_abs_delta_qg": float(row.get("max_abs_delta_qg", np.nan)),
                        "L_shed": float(row["L_shed"]),
                        "shutoff_line_ids": row["shutoff_line_ids"],
                    }
                )
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Physics Sensitivity")
        ax.set_xlabel("rho_phys")
        ax.set_ylabel("PAC_total")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        _savefig(fig, out / "physics_feasibility_sensitivity_traditional_lambdas.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
        for lambda_r in lambda_values:
            local = group[np.isclose(group["lambda_R"].astype(float), lambda_r)].sort_values("rho_phys")
            if local.empty:
                continue
            rho0 = local[np.isclose(local["rho_phys"].astype(float), 0.0)]
            baseline = float(rho0["J_no_phys"].iloc[0]) if not rho0.empty else float(local["J_no_phys"].iloc[0])
            delta = local["J_no_phys"].astype(float) - baseline
            ax.plot(local["rho_phys"], delta, marker="o", linewidth=2, label=f"lambda_R={lambda_r:g}")
            for (_, row), value in zip(local.iterrows(), delta):
                cost_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "stage": stage,
                        "lambda_R": float(row["lambda_R"]),
                        "rho_phys": float(row["rho_phys"]),
                        "nonphysics_tradeoff_T": float(row["J_no_phys"]),
                        "cost_of_feasibility_vs_rho0": float(value),
                        "PAC_total": float(row["PAC_total"]),
                        "L_shed": float(row["L_shed"]),
                        "shutoff_line_ids": row["shutoff_line_ids"],
                    }
                )
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Cost of Feasibility")
        ax.set_xlabel("rho_phys")
        ax.set_ylabel("T(rho) - T(rho=0)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        _savefig(fig, out / "cost_of_feasibility_traditional_lambdas.png", dpi=180)
        plt.close(fig)

        freq_values = []
        for rho in rho_values:
            local = group[np.isclose(group["rho_phys"].astype(float), rho)]
            frequency = float(local["shutoff_line_ids"].fillna("").map(lambda value: _selected_contains_line(value, 23)).mean()) if not local.empty else np.nan
            freq_values.append(frequency)
            frequency_rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage": stage,
                    "rho_phys": float(rho),
                    "line23_selection_frequency": frequency,
                    "num_selected_rows": int(len(local)),
                }
            )
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        ax.plot(rho_values, freq_values, marker="o", linewidth=2, color="#4C78A8")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Line 23 Frequency")
        ax.set_xlabel("rho_phys")
        ax.set_ylabel("Fraction of lambda sweep selections")
        ax.grid(True, alpha=0.25)
        _savefig(fig, out / "line23_frequency_by_rho.png", dpi=180)
        plt.close(fig)

        for y, filename, title in [
            ("L_shed", "effective_load_shed.png", "Effective Load Shed"),
            ("max_abs_delta_qg", "pq_control_clamping_consistency.png", "PQ Control Movement"),
            ("gridfm_calls", "recourse_solver_runtime_and_calls.png", "Recourse GridFM Calls"),
        ]:
            fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
            for lambda_r, local in group.groupby("lambda_R", sort=True):
                ax.plot(local["rho_phys"], local[y], marker="o", label=f"lambda_R={lambda_r:g}")
            ax.set_xlabel("rho_phys")
            ax.set_ylabel(y)
            ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: {title}")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            _savefig(fig, out / filename, dpi=180)
            plt.close(fig)

    summary_dir = plots_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    physics = pd.DataFrame(physics_rows)
    cost = pd.DataFrame(cost_rows)
    line23 = pd.DataFrame(frequency_rows)
    tradeoff = (
        best.groupby(["scenario_id", "rho_phys"], as_index=False)
        .agg(
            avg_PAC_total=("PAC_total", "mean"),
            avg_R_norm=("R_norm", "mean"),
            avg_L_shed=("L_shed", "mean"),
            avg_J_no_phys=("J_no_phys", "mean"),
            avg_J_true=("J_true", "mean"),
        )
        .sort_values(["scenario_id", "rho_phys"])
    )
    mask_summary = mask.groupby("is_controlled_feature", as_index=False).agg(visible_rate=("controlled_visible_to_gridfm", "mean"))

    def line_plot(frame: pd.DataFrame, y: str, title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
        for scenario_id, local in frame.groupby("scenario_id", sort=True):
            ax.plot(local["rho_phys"], local[y], marker="o", linewidth=2, label=scenario_id)
        ax.set_title(title)
        ax.set_xlabel("rho_phys")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(summary_dir / filename, dpi=180)
        plt.close(fig)

    line_plot(tradeoff, "avg_PAC_total", "Average PAC by Rho", "Average PAC_total", "avg_pac_by_rho.png")
    target = expected.groupby(["scenario_id", "rho_phys"], as_index=False).agg(
        avg_target_recall=("target_recall", "mean"),
        avg_target_precision=("target_precision", "mean"),
        avg_target_overlap=("target_overlap_fraction", "mean"),
    )
    line_plot(target, "avg_target_recall", "Target Recall by Rho", "Average target recall", "target_recall_by_rho.png")
    line_plot(target, "avg_target_precision", "Target Precision by Rho", "Average target precision", "target_precision_by_rho.png")
    line_plot(target, "avg_target_overlap", "Target Recall by Rho", "Average target recall", "target_overlap_by_rho.png")
    cost_summary = cost.groupby(["scenario_id", "rho_phys"], as_index=False).agg(
        avg_cost_of_feasibility=("cost_of_feasibility_vs_rho0", "mean")
    )
    line_plot(cost_summary, "avg_cost_of_feasibility", "Average Cost of Feasibility by Rho", "Average T(rho)-T(0)", "avg_cost_of_feasibility_by_rho.png")
    line23_summary = line23.groupby(["scenario_id", "rho_phys"], as_index=False).agg(
        line23_selection_frequency=("line23_selection_frequency", "mean")
    )
    line_plot(line23_summary, "line23_selection_frequency", "Line 23 Frequency by Rho", "Selection frequency", "line23_frequency_by_rho_all_scenarios.png")
    line_plot(tradeoff, "avg_L_shed", "Average Effective Load Shed by Rho", "Average L_shed", "avg_alpha_clamped_l_shed_by_rho.png")

    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    ax.bar(mask_summary["is_controlled_feature"].astype(str), mask_summary["visible_rate"].astype(float), color="#4C78A8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Visible rate")
    ax.set_title("Masking/Clamping Audit Summary")
    fig.savefig(summary_dir / "masking_clamping_audit_summary.png", dpi=180)
    plt.close(fig)
    return physics, cost, line23, tradeoff


def run_revised_continuous_implementation(
    models: List[str] | None = None,
    scenario_ids: List[str] | None = None,
    lambda_values: Sequence[float] = DEFAULT_LAMBDAS,
    proxy_lambda_values: Sequence[float] | None = None,
    rho_values: Sequence[float] = DEFAULT_RHO_VALUES,
    stages: Sequence[str] | None = STAGES,
    stage_d_limit: int | None = DEFAULT_STAGE_D_LIMIT,
    stage_e_budget: int = DEFAULT_STAGE_E_BUDGET,
    call_budget: int = DEFAULT_CALL_BUDGET,
    delta_qg_bound_mvar: float = 5.0,
    output_root: Path = RESULT_ROOT,
) -> Path:
    started = time.perf_counter()
    models = ["gnn"] if models is None else list(models)
    if len(models) != 1:
        raise ValueError("The revised continuous implementation currently supports one model per run.")
    model_type = models[0]
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model_type={model_type}.")
    selected_stages = _normalize_stages(stages)
    output_root = Path(output_root)
    run_signature = _run_signature(
        model_type=model_type,
        scenario_ids=scenario_ids,
        lambda_values=lambda_values,
        proxy_lambda_values=proxy_lambda_values,
        rho_values=rho_values,
        stages=selected_stages,
        stage_d_limit=stage_d_limit,
        stage_e_budget=int(stage_e_budget),
        call_budget=int(call_budget),
        delta_qg_bound_mvar=float(delta_qg_bound_mvar),
    )
    run_dir = _latest_incomplete_run(output_root, run_signature)
    if run_dir is None:
        run_dir = make_run_dir(output_root, "run")
        resume_mode = False
    else:
        print(f"[resume] continuing incomplete run at {run_dir}", flush=True)
        resume_mode = True
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    inputs_dir = run_dir / "inputs"
    checkpoint_dir = tables_dir / "checkpoints"
    progress_path = inputs_dir / "progress.json"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    write_json(inputs_dir / "run_signature.json", run_signature)
    context = _make_continuous_context(model_type, delta_qg_bound_mvar)
    write_json(inputs_dir / "selected_decision_buses.json", {
        "selected_generator_buses": [int(v) for v in context["decision_vector"].selected_generator_buses],
        "selected_load_buses": [int(v) for v in context["decision_vector"].selected_load_buses],
    })
    topology_pool, p_env_table, ranking, scenarios, metadata = _topology_candidates_for_context(
        context,
        scenario_ids,
        _as_float_list(lambda_values),
        None if proxy_lambda_values is None else _as_float_list(proxy_lambda_values),
        selected_stages,
        stage_d_limit,
        int(stage_e_budget),
    )
    write_dataframe(tables_dir / "topology_proposal_pool.csv", topology_pool)
    write_dataframe(tables_dir / "p_env_by_scenario.csv", p_env_table)
    write_dataframe(tables_dir / "scenario_baseline_loading_ranking.csv", ranking)
    results, traces, load_prov, risk_prov, controlled, mask = _run_continuous_pool(
        context,
        topology_pool,
        metadata["candidate_line_ids"],
        _as_float_list(rho_values),
        int(call_budget),
        checkpoint_dir=checkpoint_dir,
        progress_path=progress_path,
    )
    best = _best_by_rho_scenario_lambda_stage(results)
    expected = _expected_vs_observed_all_lambdas(best, scenarios)
    runtime = _runtime_summary(results)
    fixed_vs = _fixed_vs_continuous(results)
    checks = _methodology_checks(
        results,
        p_env_table,
        ranking,
        load_prov,
        risk_prov,
        controlled,
        mask,
        context["scenario"],
        metadata["candidate_line_ids"],
    )
    hard_failures = checks[checks["severity"].eq("hard") & ~checks["passed"].astype(bool)]
    if not hard_failures.empty:
        write_dataframe(tables_dir / "methodology_fidelity_checks.csv", checks)
        raise RuntimeError(f"Hard methodology checks failed: {hard_failures['check_name'].tolist()}")

    _plot_per_rho(results, best, expected, plots_dir)
    physics, cost, line23, tradeoff = _plot_cross_and_summary(results, best, expected, mask, plots_dir)

    write_dataframe(tables_dir / "methodology_fidelity_checks.csv", checks)
    write_dataframe(tables_dir / "masking_clamping_audit.csv", mask)
    write_dataframe(tables_dir / "controlled_state_consistency.csv", controlled)
    write_dataframe(tables_dir / "load_shedding_provenance.csv", load_prov)
    write_dataframe(tables_dir / "wildfire_risk_provenance.csv", risk_prov)
    write_dataframe(tables_dir / "continuous_objective_call_trace.csv", traces)
    write_dataframe(tables_dir / "continuous_recourse_results.csv", results)
    write_dataframe(tables_dir / "best_by_rho_scenario_lambda_stage.csv", best)
    write_dataframe(tables_dir / "expected_vs_selected_by_rho.csv", expected)
    write_dataframe(tables_dir / "runtime_and_solver_diagnostics.csv", runtime)
    write_dataframe(tables_dir / "fixed_vs_continuous_comparison.csv", fixed_vs)
    write_dataframe(tables_dir / "physics_sensitivity_by_rho.csv", physics)
    write_dataframe(tables_dir / "cost_of_feasibility_by_rho.csv", cost)
    write_dataframe(tables_dir / "line23_frequency_by_rho.csv", line23)
    write_dataframe(tables_dir / "rho_tradeoff_summary.csv", tradeoff)
    write_dataframe(tables_dir / "topology_proposal_pool.csv", topology_pool)
    write_json(
        inputs_dir / "metadata.json",
        {
            **git_metadata(),
            "study": "physics_infeasibility_revised_continuous_implementation",
            "model": model_type,
            "scenario_ids": scenario_ids or [scenario.scenario_id for scenario in scenarios],
            "lambda_values": _as_float_list(lambda_values),
            "proxy_lambda_values": None if proxy_lambda_values is None else _as_float_list(proxy_lambda_values),
            "rho_phys_values": _as_float_list(rho_values),
            "stages": selected_stages,
            "stage_d_limit": None if stage_d_limit is None else int(stage_d_limit),
            "stage_e_budget_per_stage_lambda_scenario": int(stage_e_budget),
            "call_budget_per_topology": int(call_budget),
            "continuous_decision_vector": "[Delta_Pg, Delta_Qg, alpha]",
            "delta_qg_bound_mvar": float(delta_qg_bound_mvar),
            "p_env_mode": P_ENV_MODE_TARGET_MARGIN,
            "excluded_target_line_ids": list(EXCLUDED_MARGIN_TARGET_LINE_IDS),
            "baseline_loading_source": BASELINE_LOADING_SOURCE,
            "post_topology_evaluation_source": POST_TOPOLOGY_CONTINUOUS_SOURCE,
            "L_shed_source": L_SHED_SOURCE,
            "load_shed_mode": LOAD_SHED_MODE,
            "load_shed_metrics": ["L_shed_cmd", "L_shed_gridfm_raw", "L_shed_gridfm_effective", "L_shed_hybrid"],
            "pac_group_weights": PAC_GROUP_WEIGHTS,
            "branch_flow_consistency_available": False,
            "true_risk_formula": "sum_l z_l * p_env_l * loading_l^2; impact_l excluded",
            "topology_proxy_excludes_pac": True,
            "num_topology_proposals": int(len(topology_pool)),
            "num_continuous_results": int(len(results)),
            "resumed_existing_run": bool(resume_mode),
            "progress_checkpoint": str(progress_path),
            "checkpoint_tables_dir": str(checkpoint_dir),
            "runtime_seconds": float(time.perf_counter() - started),
        },
    )
    _write_progress(
        progress_path,
        completed=int(len(results)),
        total=int(len(topology_pool) * len(_as_float_list(rho_values))),
        started=started,
        last_result_id=int(results["result_id"].astype(int).max()) if len(results) else None,
        status="complete",
    )
    return run_dir


def regenerate_revised_continuous_plots(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    results = pd.read_csv(tables_dir / "continuous_recourse_results.csv")
    best = pd.read_csv(tables_dir / "best_by_rho_scenario_lambda_stage.csv")
    expected = pd.read_csv(tables_dir / "expected_vs_selected_by_rho.csv")
    mask = pd.read_csv(tables_dir / "masking_clamping_audit.csv")
    if plots_dir.exists():
        resolved = plots_dir.resolve()
        if resolved.parent != run_dir.resolve():
            raise ValueError(f"Refusing to clear unexpected plots directory: {resolved}")
        shutil.rmtree(resolved)

    _plot_per_rho(results, best, expected, plots_dir)
    physics, cost, line23, tradeoff = _plot_cross_and_summary(results, best, expected, mask, plots_dir)
    write_dataframe(tables_dir / "physics_sensitivity_by_rho.csv", physics)
    write_dataframe(tables_dir / "cost_of_feasibility_by_rho.csv", cost)
    write_dataframe(tables_dir / "line23_frequency_by_rho.csv", line23)
    write_dataframe(tables_dir / "rho_tradeoff_summary.csv", tradeoff)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage G revised continuous recourse implementation.")
    parser.add_argument("--models", nargs="+", default=["gnn"])
    parser.add_argument("--scenario-ids", nargs="+", default=None)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--proxy-lambda-values", nargs="+", type=float, default=None)
    parser.add_argument("--rho-phys", nargs="+", type=float, default=DEFAULT_RHO_VALUES)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--stage-d-limit", type=int, default=DEFAULT_STAGE_D_LIMIT)
    parser.add_argument("--stage-e-budget", type=int, default=DEFAULT_STAGE_E_BUDGET)
    parser.add_argument("--call-budget", type=int, default=DEFAULT_CALL_BUDGET)
    parser.add_argument("--delta-qg-bound-mvar", type=float, default=5.0)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--regenerate-plots-run-dir",
        type=Path,
        default=None,
        help="Regenerate revised-continuous plots and rho summary tables from an existing completed run directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.regenerate_plots_run_dir is not None:
        run_dir = regenerate_revised_continuous_plots(args.regenerate_plots_run_dir)
        print(f"Regenerated Stage G revised continuous plots for {run_dir}")
        return
    run_dir = run_revised_continuous_implementation(
        models=args.models,
        scenario_ids=args.scenario_ids,
        lambda_values=args.lambda_values,
        proxy_lambda_values=args.proxy_lambda_values,
        rho_values=args.rho_phys,
        stages=args.stages,
        stage_d_limit=args.stage_d_limit,
        stage_e_budget=args.stage_e_budget,
        call_budget=args.call_budget,
        delta_qg_bound_mvar=args.delta_qg_bound_mvar,
        output_root=args.output_root,
    )
    print(f"Wrote Stage G revised continuous implementation to {run_dir}")


if __name__ == "__main__":
    main()
