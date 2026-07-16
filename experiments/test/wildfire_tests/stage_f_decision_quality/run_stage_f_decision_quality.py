from __future__ import annotations

import argparse
import itertools
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.config import write_config_copy
from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import git_metadata, make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
from experiments.test.wildfire_tests.gridfm_support.branch_metadata import (
    canonicalize_line_ids,
    expand_to_physical_line_ids,
    physical_line_ids,
)
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import (
    MODEL_CONFIGS,
    _build_model_context,
    compact_fraction_label,
)
from experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps import (
    affected_buses,
    demand_weighted_load_shed_from_prediction,
    disconnected_component_count,
    predict_psps_state,
)
from experiments.test.wildfire_tests.stage_d_deenergization.stage_d_deenergization import (
    disconnected_buses_from_reference,
    enumerate_deenergization_subsets,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.gurobi_master import (
    solve_gurobi_master_next_candidate,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_gurobi_gridfm import (
    _candidate_group_maps,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    DEFAULT_PROXY_TYPE,
    STAGE_E_LAMBDA_CASES,
    candidate_y_from_deenergized,
    compute_proxy_metrics,
    deenergized_from_y,
    normalize_true_exposure,
    vector_string,
    z_from_y,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import (
    DecisionQualityScenario,
    get_scenarios,
    p_env_for_scenario,
)


RESULT_ROOT = (
    RESULTS_ROOT
    / "leq"
    / "stage_f"
    / "decision_quality_analysis"
    / "without_physics_infeasibility"
)
STAGE_D_NAME = "stage_d_k2_exhaustive"
STAGE_E_NAME = "stage_e_k2_gurobi"
RISK_SCOPE = "all_lines"
FIXED_T0P30_CANDIDATE_LINE_IDS = (
    2,
    5,
    6,
    8,
    10,
    12,
    16,
    18,
    19,
    20,
    22,
    23,
    25,
    26,
    27,
    32,
    33,
    35,
    36,
    37,
    38,
    40,
    47,
    50,
    51,
    72,
    74,
    77,
    79,
    88,
    91,
    97,
    101,
)


def _edge_array(scenario) -> np.ndarray:
    return scenario.edge_index.cpu().numpy() if hasattr(scenario.edge_index, "cpu") else np.asarray(scenario.edge_index)


def _csv_ints(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in sorted(int(item) for item in values))


def _parse_line_ids(value) -> List[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(float(item)) for item in text.replace("[", "").replace("]", "").split(",") if item.strip()]


def _line_id_key(values: Iterable[int]) -> str:
    return _csv_ints(values)


def _candidate_line_ids(wildfire=None) -> List[int]:
    return list(FIXED_T0P30_CANDIDATE_LINE_IDS)


def _risk_line_ids(scenario) -> List[int]:
    return physical_line_ids(scenario)


def _validate_candidate_line_ids(candidate_line_ids: Iterable[int], num_lines: int) -> None:
    invalid = [int(line_id) for line_id in candidate_line_ids if int(line_id) < 0 or int(line_id) >= int(num_lines)]
    if invalid:
        raise ValueError(f"Stage F fixed candidate line IDs outside 0..{int(num_lines)-1}: {invalid}")


def _consequence_by_line(consequence_df: pd.DataFrame) -> Dict[int, float]:
    source = "c_l" if "c_l" in consequence_df.columns else "I_l"
    return {int(row["line_id"]): float(row[source]) for _, row in consequence_df.iterrows()}


def _active_line_ids(num_lines: int, removed_line_ids: Iterable[int], scenario=None) -> List[int]:
    removed_values = expand_to_physical_line_ids(scenario, removed_line_ids) if scenario is not None else removed_line_ids
    removed = {int(line_id) for line_id in removed_values}
    return [line_id for line_id in range(int(num_lines)) if line_id not in removed]


def _source_buses(scenario) -> set[int]:
    pg = np.asarray(scenario.Pg_base, dtype=float)
    sources = set(int(bus) for bus in np.where(pg > 1e-9)[0])
    if hasattr(scenario, "get_pv_buses"):
        sources.update(int(bus) for bus in scenario.get_pv_buses())
    if hasattr(scenario, "get_ref_bus"):
        ref_bus = scenario.get_ref_bus()
        if ref_bus is not None:
            sources.add(int(ref_bus))
    return sources


def _graph_after_removal(scenario, removed_line_ids: Iterable[int]) -> nx.Graph:
    edge_array = _edge_array(scenario)
    removed = {int(line_id) for line_id in expand_to_physical_line_ids(scenario, removed_line_ids)}
    graph = nx.Graph()
    graph.add_nodes_from(range(int(scenario.num_buses)))
    for line_id, (src, dst) in enumerate(edge_array.T):
        if int(line_id) in removed:
            continue
        graph.add_edge(int(src), int(dst))
    return graph


def _source_less_diagnostics(scenario, removed_line_ids: Iterable[int]) -> Dict:
    graph = _graph_after_removal(scenario, removed_line_ids)
    demand = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    sources = _source_buses(scenario)
    source_less_buses: list[int] = []
    source_less_load = 0.0
    for component in nx.connected_components(graph):
        buses = sorted(int(bus) for bus in component)
        if not set(buses).intersection(sources):
            load = float(demand[buses].sum())
            if load > 1e-12:
                source_less_buses.extend(buses)
                source_less_load += load
    return {
        "source_less_bus_ids": _csv_ints(source_less_buses),
        "source_less_load_mw": float(source_less_load),
        "creates_source_less_island": bool(source_less_load > 1e-12),
    }


def _scenario_baseline_exposure(
    baseline_loading: np.ndarray,
    p_env_by_line: Dict[int, float],
    risk_line_ids: List[int],
) -> tuple[float, Dict[int, float]]:
    z_all_on = {line_id: 1 for line_id in range(len(baseline_loading))}
    return compute_operational_wildfire_exposure(baseline_loading, p_env_by_line, z_all_on, risk_line_ids)


def _evaluate_topology(
    model_context: dict,
    scenario_def: DecisionQualityScenario,
    candidate_line_ids: List[int],
    risk_line_ids: List[int],
    p_env_by_line: Dict[int, float],
    baseline_R: float,
    shutoff_line_ids: Iterable[int],
    eval_id: int,
    stage: str,
    lambda_case: str | None = None,
    lambda_R: float | None = None,
    lambda_L: float | None = None,
    proposal_iteration: int | None = None,
    proxy_fields: Dict | None = None,
) -> Dict:
    config = model_context["config"]
    scenario = model_context["scenario"]
    runner = model_context["runner"]
    decision_vector = model_context["decision_vector"]
    baseline_prediction = model_context["baseline_prediction"]
    baseline_state = model_context["baseline_state"]
    num_lines = int(_edge_array(scenario).shape[1])
    deenergized = sorted({int(line_id) for line_id in shutoff_line_ids})
    y_by_line = candidate_y_from_deenergized(candidate_line_ids, deenergized)
    z_by_line = z_from_y(y_by_line, num_lines)
    status = "ok"
    error = ""
    try:
        if deenergized:
            prediction, state = predict_psps_state(
                scenario,
                runner,
                decision_vector.u_base,
                deenergized,
                standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
            )
        else:
            prediction = baseline_prediction
            state = baseline_state
        loading = np.asarray(state["loading_ratio"], dtype=float)
        true_R_raw, exposure_by_line = compute_operational_wildfire_exposure(
            loading,
            p_env_by_line,
            z_by_line,
            risk_line_ids,
        )
        true_R_norm = normalize_true_exposure(true_R_raw, baseline_R)
        true_L_shed = demand_weighted_load_shed_from_prediction(prediction, scenario) if deenergized else 0.0
        true_objective = np.nan if lambda_R is None or lambda_L is None else float(lambda_R) * true_R_norm + float(lambda_L) * true_L_shed
        max_loading = float(state.get("max_loading_ratio", np.nan))
        min_voltage = float(state.get("min_voltage", np.nan))
        max_voltage = float(state.get("max_voltage", np.nan))
    except Exception as exc:
        status = "failed"
        error = str(exc)
        true_R_raw = np.nan
        true_R_norm = np.nan
        true_L_shed = np.nan
        true_objective = np.nan
        exposure_by_line = {}
        max_loading = np.nan
        min_voltage = np.nan
        max_voltage = np.nan

    affected = affected_buses(scenario.edge_index, deenergized)
    disconnected = disconnected_buses_from_reference(scenario.edge_index, scenario.num_buses, deenergized)
    source_less = _source_less_diagnostics(scenario, deenergized)
    proxy_fields = proxy_fields or {}
    return {
        "stage": stage,
        "scenario_id": scenario_def.scenario_id,
        "scenario_name": scenario_def.scenario_name,
        "lambda_case": "" if lambda_case is None else lambda_case,
        "lambda_R": np.nan if lambda_R is None else float(lambda_R),
        "lambda_L": np.nan if lambda_L is None else float(lambda_L),
        "eval_id": int(eval_id),
        "proposal_iteration": np.nan if proposal_iteration is None else int(proposal_iteration),
        "candidate_set": "t0p30",
        "risk_scope": RISK_SCOPE,
        "num_candidate_lines": int(len(candidate_line_ids)),
        "num_risk_lines": int(len(risk_line_ids)),
        "num_shutoff_lines": int(len(deenergized)),
        "shutoff_line_ids": _csv_ints(deenergized),
        "line_id_key": _line_id_key(deenergized),
        "active_line_ids": _csv_ints(_active_line_ids(num_lines, deenergized, scenario=scenario)),
        "y_vector": vector_string(candidate_line_ids, y_by_line),
        "z_vector": vector_string(candidate_line_ids, {line_id: 1 - int(y_by_line.get(line_id, 0)) for line_id in candidate_line_ids}),
        "R_wf": float(true_R_raw) if np.isfinite(true_R_raw) else np.nan,
        "R_base_s": float(baseline_R),
        "R_norm": float(true_R_norm) if np.isfinite(true_R_norm) else np.nan,
        "L_shed": float(true_L_shed) if np.isfinite(true_L_shed) else np.nan,
        "J_true": float(true_objective) if np.isfinite(true_objective) else np.nan,
        "risk_contribution": float(lambda_R) * float(true_R_norm) if lambda_R is not None and np.isfinite(true_R_norm) else np.nan,
        "load_contribution": float(lambda_L) * float(true_L_shed) if lambda_L is not None and np.isfinite(true_L_shed) else np.nan,
        "proxy_R_hat": proxy_fields.get("proxy_R_hat", np.nan),
        "proxy_L_hat": proxy_fields.get("proxy_L_hat", np.nan),
        "proxy_objective": proxy_fields.get("proxy_objective", np.nan),
        "proxy_R_denominator": proxy_fields.get("proxy_R_denominator", np.nan),
        "gurobi_objective": proxy_fields.get("gurobi_objective", np.nan),
        "gurobi_status": proxy_fields.get("gurobi_status", np.nan),
        "status": status,
        "error": error,
        "affected_bus_ids": _csv_ints(affected),
        "disconnected_bus_ids": _csv_ints(disconnected),
        "num_disconnected_components": int(disconnected_component_count(scenario.edge_index, scenario.num_buses, deenergized)),
        "max_loading_ratio": max_loading,
        "min_voltage": min_voltage,
        "max_voltage": max_voltage,
        "true_exposure_by_line": ";".join(f"{int(k)}:{float(v):.12g}" for k, v in sorted(exposure_by_line.items())),
        **source_less,
    }


def _select_best_by_lambda(stage_rows: pd.DataFrame, lambda_cases: Dict[str, tuple[float, float]]) -> pd.DataFrame:
    rows = []
    ok = stage_rows[stage_rows["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    for lambda_case, (lambda_R, lambda_L) in lambda_cases.items():
        frame = ok.copy()
        frame["lambda_case"] = lambda_case
        frame["lambda_R"] = float(lambda_R)
        frame["lambda_L"] = float(lambda_L)
        frame["J_true"] = float(lambda_R) * frame["R_norm"].astype(float) + float(lambda_L) * frame["L_shed"].astype(float)
        frame["risk_contribution"] = float(lambda_R) * frame["R_norm"].astype(float)
        frame["load_contribution"] = float(lambda_L) * frame["L_shed"].astype(float)
        best = frame.sort_values(["J_true", "L_shed", "num_shutoff_lines", "line_id_key"], kind="mergesort").iloc[0]
        rows.append(best.to_dict())
    return pd.DataFrame(rows)


def _run_stage_d(
    model_context: dict,
    scenario_def: DecisionQualityScenario,
    candidate_line_ids: List[int],
    risk_line_ids: List[int],
    p_env_by_line: Dict[int, float],
    baseline_R: float,
    max_deenergized_lines: int,
) -> pd.DataFrame:
    rows = []
    for eval_id, subset in enumerate(enumerate_deenergization_subsets(candidate_line_ids, max_deenergized_lines=max_deenergized_lines)):
        rows.append(
            _evaluate_topology(
                model_context,
                scenario_def,
                candidate_line_ids,
                risk_line_ids,
                p_env_by_line,
                baseline_R,
                subset,
                eval_id=eval_id,
                stage=STAGE_D_NAME,
            )
        )
    return pd.DataFrame(rows)


def _run_stage_e(
    model_context: dict,
    scenario_def: DecisionQualityScenario,
    candidate_line_ids: List[int],
    risk_line_ids: List[int],
    p_env_by_line: Dict[int, float],
    baseline_R: float,
    baseline_loading: np.ndarray,
    c_by_line: Dict[int, float],
    lambda_case: str,
    lambda_R: float,
    lambda_L: float,
    max_deenergized_lines: int,
    evaluation_budget: int,
) -> pd.DataFrame:
    rows = []
    evaluated_y: list[dict[int, int]] = []
    for iteration in range(int(evaluation_budget)):
        try:
            proposal = solve_gurobi_master_next_candidate(
                candidate_line_ids,
                p_env_by_line,
                baseline_loading,
                c_by_line,
                lambda_R,
                lambda_L,
                max_deenergized_lines=max_deenergized_lines,
                evaluated_y_vectors=evaluated_y,
                proxy_type=DEFAULT_PROXY_TYPE,
            )
        except Exception as exc:
            rows.append(
                {
                    "stage": STAGE_E_NAME,
                    "scenario_id": scenario_def.scenario_id,
                    "scenario_name": scenario_def.scenario_name,
                    "lambda_case": lambda_case,
                    "lambda_R": float(lambda_R),
                    "lambda_L": float(lambda_L),
                    "eval_id": int(iteration),
                    "proposal_iteration": int(iteration),
                    "candidate_set": "t0p30",
                    "risk_scope": RISK_SCOPE,
                    "status": "proposal_failed",
                    "error": str(exc),
                }
            )
            break
        y_by_line = {int(k): int(v) for k, v in proposal["y_by_line"].items()}
        evaluated_y.append(y_by_line)
        deenergized = deenergized_from_y(y_by_line)
        rows.append(
            _evaluate_topology(
                model_context,
                scenario_def,
                candidate_line_ids,
                risk_line_ids,
                p_env_by_line,
                baseline_R,
                deenergized,
                eval_id=iteration,
                stage=STAGE_E_NAME,
                lambda_case=lambda_case,
                lambda_R=lambda_R,
                lambda_L=lambda_L,
                proposal_iteration=iteration,
                proxy_fields=proposal,
            )
        )
    return pd.DataFrame(rows)


def _best_stage_e(stage_e_rows: pd.DataFrame) -> pd.DataFrame:
    ok = stage_e_rows[stage_e_rows["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = ["model_type", "scenario_id", "scenario_name", "lambda_case", "lambda_R", "lambda_L"]
    idx = ok.groupby(group_cols, dropna=False)["J_true"].idxmin()
    return ok.loc[idx].sort_values(group_cols).reset_index(drop=True)


def _nondominated_mask(frame: pd.DataFrame, columns: Iterable[str] = ("R_norm", "L_shed")) -> np.ndarray:
    values = frame[list(columns)].to_numpy(dtype=float)
    finite = np.all(np.isfinite(values), axis=1)
    mask = np.zeros(len(frame), dtype=bool)
    for idx, point in enumerate(values):
        if not finite[idx]:
            continue
        dominated = False
        for other_idx, other in enumerate(values):
            if idx == other_idx or not finite[other_idx]:
                continue
            if np.all(other <= point + 1e-12) and np.any(other < point - 1e-12):
                dominated = True
                break
        mask[idx] = not dominated
    return mask


def _pareto_tables(stage_d_rows: pd.DataFrame, stage_e_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    if not stage_d_rows.empty:
        frames.append(stage_d_rows[stage_d_rows["status"].eq("ok")].copy())
    if not stage_e_rows.empty:
        frames.append(stage_e_rows[stage_e_rows["status"].eq("ok")].copy())
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    pareto = pd.concat(frames, ignore_index=True, sort=False)
    pieces = []
    for _key, group in pareto.groupby(["scenario_id", "stage"], dropna=False):
        local = group.copy()
        local["is_nondominated_within_stage"] = _nondominated_mask(local, ("R_norm", "L_shed"))
        pieces.append(local)
    pareto = pd.concat(pieces, ignore_index=True, sort=False)
    combined = []
    for _key, group in pareto.groupby(["scenario_id"], dropna=False):
        local = group.copy()
        local["is_nondominated_within_scenario"] = _nondominated_mask(local, ("R_norm", "L_shed"))
        combined.append(local)
    pareto = pd.concat(combined, ignore_index=True, sort=False)
    nondominated = pareto[pareto["is_nondominated_within_stage"] | pareto["is_nondominated_within_scenario"]].copy()
    return pareto.reset_index(drop=True), nondominated.reset_index(drop=True)


def _hypothesis_comparison(
    scenarios: List[DecisionQualityScenario],
    stage_d_best: pd.DataFrame,
    stage_e_best: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    selected = []
    if not stage_d_best.empty:
        selected.append(stage_d_best)
    if not stage_e_best.empty:
        selected.append(stage_e_best)
    if not selected:
        return pd.DataFrame()
    combined = pd.concat(selected, ignore_index=True, sort=False)
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    for _, row in combined.iterrows():
        scenario = scenario_by_id[str(row["scenario_id"])]
        targets = set(int(line_id) for line_id in scenario.expected_target_set)
        chosen = set(_parse_line_ids(row.get("shutoff_line_ids", "")))
        overlap = len(chosen.intersection(targets)) / max(len(targets), 1)
        hamming = len(chosen.symmetric_difference(targets))
        supports = bool(overlap > 0.0) if row["lambda_case"] != "load_priority" else bool(len(chosen) <= 2)
        rows.append(
            {
                "stage": row["stage"],
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.scenario_name,
                "lambda_case": row["lambda_case"],
                "lambda_R": float(row["lambda_R"]),
                "lambda_L": float(row["lambda_L"]),
                "selected_line_ids": row.get("shutoff_line_ids", ""),
                "hypothesis_target_line_ids": _csv_ints(targets),
                "overlap_score": float(overlap),
                "hamming_distance": int(hamming),
                "J_true": float(row["J_true"]),
                "R_norm": float(row["R_norm"]),
                "L_shed": float(row["L_shed"]),
                "num_shutoff_lines": int(row["num_shutoff_lines"]),
                "hypothesis_expected_behavior": scenario.hypothesis_expected_behavior,
                "observed_selection_summary": f"Selected [{row.get('shutoff_line_ids', '')}] with R_norm={float(row['R_norm']):.6g}, L_shed={float(row['L_shed']):.6g}, J={float(row['J_true']):.6g}.",
                "supports_hypothesis": supports,
                "interpretation_note": scenario.interpretation_focus,
                "manual_review_flag": bool(scenario.scenario_id in {"S3", "S4", "S5"} or overlap == 0.0),
            }
        )
    return pd.DataFrame(rows)


def _gap_table(stage_d_best: pd.DataFrame, stage_e_best: pd.DataFrame) -> pd.DataFrame:
    if stage_d_best.empty or stage_e_best.empty:
        return pd.DataFrame()
    keys = ["model_type", "scenario_id", "scenario_name", "lambda_case", "lambda_R", "lambda_L"]
    merged = stage_e_best.merge(stage_d_best, on=keys, suffixes=("_stage_e", "_stage_d"))
    rows = []
    for _, row in merged.iterrows():
        gap = float(row["J_true_stage_e"]) - float(row["J_true_stage_d"])
        rows.append(
            {
                **{key: row[key] for key in keys},
                "stage_e_selected_line_ids": row["shutoff_line_ids_stage_e"],
                "stage_d_selected_line_ids": row["shutoff_line_ids_stage_d"],
                "stage_e_J_true": float(row["J_true_stage_e"]),
                "stage_d_J_true": float(row["J_true_stage_d"]),
                "gap_J": float(gap),
                "relative_gap_J": float(gap / max(abs(float(row["J_true_stage_d"])), 1e-12)),
                "stage_e_R_norm": float(row["R_norm_stage_e"]),
                "stage_d_R_norm": float(row["R_norm_stage_d"]),
                "stage_e_L_shed": float(row["L_shed_stage_e"]),
                "stage_d_L_shed": float(row["L_shed_stage_d"]),
            }
        )
    return pd.DataFrame(rows)


def _candidate_line_diagnostics(
    model_context: dict,
    scenarios: List[DecisionQualityScenario],
    candidate_line_ids: List[int],
) -> pd.DataFrame:
    scenario = model_context["scenario"]
    wildfire = model_context["wildfire"]
    consequence_df = model_context["consequence_df"]
    baseline_loading = np.asarray(model_context["baseline_state"]["loading_ratio"], dtype=float)
    c_by_line = _consequence_by_line(consequence_df)
    line_to_group = _candidate_group_maps(wildfire)
    candidate_set = set(int(line_id) for line_id in candidate_line_ids)
    edge_array = _edge_array(scenario)
    rows = []
    for scenario_def in scenarios:
        p_env = p_env_for_scenario(scenario_def, int(edge_array.shape[1]))
        for line_id in range(int(edge_array.shape[1])):
            diag = _source_less_diagnostics(scenario, [line_id])
            rows.append(
                {
                    "scenario_id": scenario_def.scenario_id,
                    "scenario_name": scenario_def.scenario_name,
                    "line_id": int(line_id),
                    "from_bus": int(edge_array[0, line_id]),
                    "to_bus": int(edge_array[1, line_id]),
                    "in_t0p30_candidate_set": bool(line_id in candidate_set),
                    "connected_group_id": line_to_group.get(int(line_id), ""),
                    "p_env": float(p_env[line_id]),
                    "baseline_loading": float(baseline_loading[line_id]),
                    "consequence_proxy_c_l": float(c_by_line.get(int(line_id), np.nan)),
                    "single_removal_source_less_island": bool(diag["creates_source_less_island"]),
                    "single_removal_source_less_load_mw": float(diag["source_less_load_mw"]),
                    "single_removal_source_less_bus_ids": diag["source_less_bus_ids"],
                }
            )
    return pd.DataFrame(rows)


def _base_candidate_groups(model_context: dict, candidate_line_ids: List[int]) -> pd.DataFrame:
    scenario = model_context["scenario"]
    baseline_loading = np.asarray(model_context["baseline_state"]["loading_ratio"], dtype=float)
    edge_array = _edge_array(scenario)
    candidate_set = set(int(line_id) for line_id in candidate_line_ids)
    graph = nx.Graph()
    graph.add_nodes_from(range(int(scenario.num_buses)))
    for line_id in candidate_set:
        graph.add_edge(int(edge_array[0, line_id]), int(edge_array[1, line_id]))
    rows = []
    for index, bus_component in enumerate(nx.connected_components(graph)):
        bus_set = set(int(bus) for bus in bus_component)
        line_ids = sorted(
            line_id
            for line_id in candidate_set
            if int(edge_array[0, line_id]) in bus_set and int(edge_array[1, line_id]) in bus_set
        )
        if not line_ids:
            continue
        bus_ids = sorted(bus_set)
        rows.append(
            {
                "group_id": f"G_{index + 1}",
                "line_ids": _csv_ints(line_ids),
                "bus_ids": _csv_ints(bus_ids),
                "num_lines": int(len(line_ids)),
                "baseline_group_risk": float(np.sum(np.square(baseline_loading[line_ids]))) if line_ids else 0.0,
                "group_weight": 1.0,
                "largest_group_flag": False,
            }
        )
    if rows:
        largest = max(range(len(rows)), key=lambda i: rows[i]["num_lines"])
        rows[largest]["largest_group_flag"] = True
    return pd.DataFrame(rows)


def _plot_outputs(stage_d_rows: pd.DataFrame, stage_d_best: pd.DataFrame, stage_e_rows: pd.DataFrame, stage_e_best: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_scenarios = sorted(set(stage_d_rows.get("scenario_id", pd.Series(dtype=str)).dropna().astype(str)))
    for scenario_id in all_scenarios:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        d = stage_d_rows[(stage_d_rows["scenario_id"].eq(scenario_id)) & (stage_d_rows["status"].eq("ok"))]
        if not d.empty:
            ax.scatter(d["R_norm"], d["L_shed"], s=18, alpha=0.25, color="#8A8A8A", label="Stage D all K<=2")
        db = stage_d_best[stage_d_best["scenario_id"].eq(scenario_id)] if not stage_d_best.empty else pd.DataFrame()
        if not db.empty:
            ax.scatter(db["R_norm"], db["L_shed"], s=70, marker="o", label="Stage D lambda optima")
        e = stage_e_rows[(stage_e_rows["scenario_id"].eq(scenario_id)) & (stage_e_rows["status"].eq("ok"))] if not stage_e_rows.empty else pd.DataFrame()
        if not e.empty:
            ax.scatter(e["R_norm"], e["L_shed"], s=28, alpha=0.45, marker="^", label="Stage E evaluated")
        eb = stage_e_best[stage_e_best["scenario_id"].eq(scenario_id)] if not stage_e_best.empty else pd.DataFrame()
        if not eb.empty:
            ax.scatter(eb["R_norm"], eb["L_shed"], s=90, marker="*", label="Stage E best")
        ax.set_xlabel("R_norm")
        ax.set_ylabel("L_shed")
        ax.set_title(f"Stage F diagnostic Pareto plot: {scenario_id}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"{scenario_id}_r_norm_vs_l_shed.png", dpi=180)
        plt.close(fig)

    selected = []
    if not stage_d_best.empty:
        selected.append(stage_d_best.assign(summary_stage="Stage D"))
    if not stage_e_best.empty:
        selected.append(stage_e_best.assign(summary_stage="Stage E"))
    if selected:
        frame = pd.concat(selected, ignore_index=True, sort=False)
        frame["label"] = frame["scenario_id"] + "|" + frame["summary_stage"] + "|" + frame["lambda_case"]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(frame["label"], frame["num_shutoff_lines"].astype(float), color="#4C78A8")
        ax.set_ylabel("Number of shutoff lines")
        ax.set_title("Stage F selected topology size by scenario/lambda")
        ax.tick_params(axis="x", labelrotation=75)
        fig.tight_layout()
        fig.savefig(output_dir / "selected_topology_size_summary.png", dpi=180)
        plt.close(fig)


def run_stage_f_decision_quality(
    models: List[str] | None = None,
    scenario_ids: List[str] | None = None,
    lambda_cases: List[str] | None = None,
    grouping_top_fraction: float = 0.30,
    max_deenergized_lines: int = 2,
    stage_e_budget: int = 100,
    clear: bool = False,
) -> Path:
    models = ["gnn"] if models is None else models
    scenario_defs = get_scenarios(scenario_ids)
    lambda_cases = ["risk_priority", "balanced", "load_priority"] if lambda_cases is None else lambda_cases
    lambda_map = {name: STAGE_E_LAMBDA_CASES[name] for name in lambda_cases}
    root = RESULT_ROOT
    if clear and root.exists():
        resolved = root.resolve()
        expected_parent = (RESULTS_ROOT / "leq" / "stage_f").resolve()
        if resolved.parent != expected_parent:
            raise ValueError(f"Refusing to delete unexpected Stage F path: {resolved}")
        shutil.rmtree(resolved)
    run_dir = make_run_dir(root, "run")
    plots_dir = run_dir / "plots"

    all_stage_d_rows = []
    all_stage_d_best = []
    all_stage_e_rows = []
    all_stage_e_best = []
    model_metadata = []
    started = time.perf_counter()
    for model_type in models:
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model_type={model_type}; supported={sorted(MODEL_CONFIGS)}")
        model_context = _build_model_context(model_type, grouping_top_fraction)
        config = model_context["config"]
        write_config_copy(config, run_dir / f"config_{model_type}.yaml")
        wildfire = model_context["wildfire"]
        scenario = model_context["scenario"]
        candidate_line_ids = canonicalize_line_ids(scenario, _candidate_line_ids(wildfire))
        num_lines = int(_edge_array(scenario).shape[1])
        _validate_candidate_line_ids(candidate_line_ids, num_lines)
        risk_line_ids = _risk_line_ids(scenario)
        baseline_loading = np.asarray(model_context["baseline_state"]["loading_ratio"], dtype=float)
        c_by_line = _consequence_by_line(model_context["consequence_df"])
        base_groups = _base_candidate_groups(model_context, candidate_line_ids)
        line_diag = _candidate_line_diagnostics(model_context, scenario_defs, candidate_line_ids)
        write_dataframe(run_dir / f"base_candidate_groups_{model_type}.csv", base_groups)
        write_dataframe(run_dir / f"candidate_line_diagnostics_{model_type}.csv", line_diag)
        if len(models) == 1:
            write_dataframe(run_dir / "base_candidate_groups.csv", base_groups)
            write_dataframe(run_dir / "candidate_line_diagnostics.csv", line_diag)

        for scenario_def in scenario_defs:
            p_env = p_env_for_scenario(scenario_def, int(_edge_array(scenario).shape[1]))
            baseline_R, _baseline_by_line = _scenario_baseline_exposure(baseline_loading, p_env, risk_line_ids)
            if baseline_R <= 1e-12:
                raise ValueError(f"Scenario {scenario_def.scenario_id} has zero baseline wildfire exposure.")
            stage_d_rows = _run_stage_d(
                model_context,
                scenario_def,
                candidate_line_ids,
                risk_line_ids,
                p_env,
                baseline_R,
                max_deenergized_lines,
            )
            stage_d_rows["model_type"] = model_type
            all_stage_d_rows.append(stage_d_rows)
            stage_d_best = _select_best_by_lambda(stage_d_rows, lambda_map)
            if not stage_d_best.empty:
                stage_d_best["model_type"] = model_type
                all_stage_d_best.append(stage_d_best)

            for lambda_case, (lambda_R, lambda_L) in lambda_map.items():
                stage_e_rows = _run_stage_e(
                    model_context,
                    scenario_def,
                    candidate_line_ids,
                    risk_line_ids,
                    p_env,
                    baseline_R,
                    baseline_loading,
                    c_by_line,
                    lambda_case,
                    float(lambda_R),
                    float(lambda_L),
                    max_deenergized_lines,
                    stage_e_budget,
                )
                stage_e_rows["model_type"] = model_type
                all_stage_e_rows.append(stage_e_rows)
            model_metadata.append(
                {
                    "model_type": model_type,
                    "candidate_line_ids": [int(line_id) for line_id in candidate_line_ids],
                    "num_candidate_lines": int(len(candidate_line_ids)),
                    "risk_line_ids": [int(line_id) for line_id in risk_line_ids],
                    "num_risk_lines": int(len(risk_line_ids)),
                }
            )

    stage_d_all = pd.concat(all_stage_d_rows, ignore_index=True, sort=False) if all_stage_d_rows else pd.DataFrame()
    stage_d_best_all = pd.concat(all_stage_d_best, ignore_index=True, sort=False) if all_stage_d_best else pd.DataFrame()
    stage_e_all = pd.concat(all_stage_e_rows, ignore_index=True, sort=False) if all_stage_e_rows else pd.DataFrame()
    stage_e_best_all = _best_stage_e(stage_e_all)
    pareto, nondominated = _pareto_tables(stage_d_all, stage_e_all)
    combined = pd.concat(
        [frame for frame in [stage_d_best_all, stage_e_best_all] if not frame.empty],
        ignore_index=True,
        sort=False,
    ) if not stage_d_best_all.empty or not stage_e_best_all.empty else pd.DataFrame()
    hypothesis = _hypothesis_comparison(scenario_defs, stage_d_best_all, stage_e_best_all)
    gap = _gap_table(stage_d_best_all, stage_e_best_all)

    scenario_payload = {
        "low_p_env": 0.05,
        "high_p_env": 1.0,
        "risk_scope": RISK_SCOPE,
        "scenarios": [scenario.to_dict() for scenario in scenario_defs],
        "s3_distinctness_note": "S3 uses [18,27,16,19,22] rather than the initial [18,27,32,36,101] because the initial set overlapped S1 too strongly.",
    }
    write_json(run_dir / "scenario_definitions.json", scenario_payload)
    write_dataframe(run_dir / "stage_d_all_topologies.csv", stage_d_all)
    write_dataframe(run_dir / "stage_d_best_by_lambda.csv", stage_d_best_all)
    write_dataframe(run_dir / "stage_e_candidate_evaluations.csv", stage_e_all)
    write_dataframe(run_dir / "stage_e_best_by_lambda.csv", stage_e_best_all)
    write_dataframe(run_dir / "combined_stage_de_results.csv", combined)
    write_dataframe(run_dir / "hypothesis_comparison.csv", hypothesis)
    write_dataframe(run_dir / "stage_e_vs_stage_d_gap.csv", gap)
    write_dataframe(run_dir / "pareto_points.csv", pareto)
    write_dataframe(run_dir / "nondominated_points.csv", nondominated)
    write_json(
        run_dir / "metadata.json",
        {
            **git_metadata(),
            "stage": "F",
            "study": "decision_quality_analysis",
            "runtime_seconds": float(time.perf_counter() - started),
            "grouping_top_fraction": float(grouping_top_fraction),
            "max_deenergized_lines": int(max_deenergized_lines),
            "stage_e_budget": int(stage_e_budget),
            "lambda_cases": {name: list(values) for name, values in lambda_map.items()},
            "model_metadata": model_metadata,
            "risk_scope": RISK_SCOPE,
            "true_risk_formula": "sum_l_in_all_lines z_l * p_env_l * loading_l^2",
            "load_service_proxy": "demand_weighted_load_shed_from_prediction",
            "pareto_plots_are_diagnostic_only": True,
            "fixed_control_u_base": True,
        },
    )
    _plot_outputs(stage_d_all, stage_d_best_all, stage_e_all, stage_e_best_all, plots_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage F decision-quality K<=2 scenario suite.")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["gnn"])
    parser.add_argument("--scenarios", nargs="+", choices=sorted([scenario.scenario_id for scenario in get_scenarios()]), default=None)
    parser.add_argument("--lambda-cases", nargs="+", choices=sorted(STAGE_E_LAMBDA_CASES), default=None)
    parser.add_argument("--grouping-top-fraction", type=float, default=0.30)
    parser.add_argument("--max-deenergized-lines", type=int, default=2)
    parser.add_argument("--stage-e-budget", type=int, default=100)
    parser.add_argument("--clear", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_stage_f_decision_quality(
        models=args.models,
        scenario_ids=args.scenarios,
        lambda_cases=args.lambda_cases,
        grouping_top_fraction=args.grouping_top_fraction,
        max_deenergized_lines=args.max_deenergized_lines,
        stage_e_budget=args.stage_e_budget,
        clear=bool(args.clear),
    )
    print(f"Wrote Stage F decision-quality results to {run_dir}")


if __name__ == "__main__":
    main()
