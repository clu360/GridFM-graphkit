from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import networkx as nx
import numpy as np
import pandas as pd

from experiments.test.wildfire_tests.shared.state_extraction import extract_state_quantities
from experiments.test.wildfire_tests.shared.wildfire_scenario import WildfireScenario
from experiments.test.wildfire_tests.gridfm_support.branch_metadata import expand_to_physical_line_ids


@dataclass
class EnvironmentalCaseResult:
    name: str
    p_env_by_line: Dict[int, float]
    largest_group_id: str | None
    largest_group_line_ids: List[int]
    manual_high_risk_group_id: str | None
    group_p_env: Dict[str, float]


def demand_weighted_service(prediction: Dict[str, np.ndarray], scenario) -> tuple[float, np.ndarray]:
    demand = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    pd_pred = np.asarray(prediction["Pd"], dtype=float)
    service_fraction = np.ones_like(demand, dtype=float)
    load_mask = demand > 1e-12
    service_fraction[load_mask] = np.clip(pd_pred[load_mask] / demand[load_mask], 0.0, 1.0)
    served = float(np.sum(demand * service_fraction))
    return served, service_fraction


def demand_weighted_load_shed_from_prediction(prediction: Dict[str, np.ndarray], scenario) -> float:
    demand = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    total = float(np.sum(demand))
    if total <= 1e-12:
        return 0.0
    _served, service_fraction = demand_weighted_service(prediction, scenario)
    return float(np.sum(demand * (1.0 - service_fraction)) / total)


def compute_fixed_line_consequence_scores(scenario, runner, u_base: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    edge_array = scenario.edge_index.cpu().numpy() if hasattr(scenario.edge_index, "cpu") else np.asarray(scenario.edge_index)
    num_lines = int(edge_array.shape[1])
    demand = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    s_d_base = float(np.sum(demand))
    if s_d_base <= 1e-12:
        raise ValueError("Cannot compute demand-weighted consequence with zero baseline demand.")

    impacts = np.zeros(num_lines, dtype=float)
    rows = []
    for line_id in range(num_lines):
        valid = True
        try:
            outage_prediction = runner.predict_with_line_outage(u_base, line_id)
            s_d_outage, _service_fraction = demand_weighted_service(outage_prediction, scenario)
            if not np.all(np.isfinite(np.asarray(outage_prediction["Pd"], dtype=float))):
                valid = False
        except Exception:
            s_d_outage = 0.0
            valid = False
        service_loss_mw = max(0.0, s_d_base - float(s_d_outage))
        impact = service_loss_mw / s_d_base
        impacts[line_id] = impact
        rows.append(
            {
                "line_id": int(line_id),
                "from_bus": int(edge_array[0, line_id]),
                "to_bus": int(edge_array[1, line_id]),
                "S_D_base": s_d_base,
                "S_D_outage": float(s_d_outage),
                "I_l": float(impact),
                "c_l": float(impact),
                "service_loss_MW": float(service_loss_mw),
                "service_loss_fraction": float(impact),
                "valid_outage_prediction": bool(valid),
            }
        )
    return impacts, pd.DataFrame(rows)


def parse_line_ids(value) -> List[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if pd.isna(value) or str(value) == "":
        return []
    return [int(item) for item in str(value).split(",") if item != ""]


def numeric_group_id(group_id: str) -> int:
    try:
        return int(str(group_id).split("_")[1])
    except Exception:
        return 10**9


def identify_largest_group(group_summary: pd.DataFrame) -> tuple[str | None, List[int]]:
    if group_summary.empty:
        return None, []
    rows = []
    for _, row in group_summary.iterrows():
        rows.append(
            {
                "group_id": str(row["group_id"]),
                "line_ids": parse_line_ids(row["line_ids"]),
                "num_lines": int(row["num_lines"]),
                "baseline_group_risk": float(row["baseline_group_risk"]),
            }
        )
    best = sorted(
        rows,
        key=lambda item: (
            -item["num_lines"],
            -item["baseline_group_risk"],
            numeric_group_id(item["group_id"]),
        ),
    )[0]
    return best["group_id"], best["line_ids"]


def apply_environmental_case(wildfire: WildfireScenario, group_summary: pd.DataFrame, case_name: str) -> EnvironmentalCaseResult:
    candidate_lines = sorted({int(line_id) for group in wildfire.line_groups for line_id in group.line_ids})
    if case_name == "auto_env":
        p_env = wildfire.hazard_vector(max(candidate_lines, default=-1) + 1 if candidate_lines else 0)
        return EnvironmentalCaseResult(
            name=case_name,
            p_env_by_line={line_id: float(p_env[line_id]) for line_id in candidate_lines},
            largest_group_id=None,
            largest_group_line_ids=[],
            manual_high_risk_group_id=None,
            group_p_env={group.name: float(p_env[group.line_ids[0]]) for group in wildfire.line_groups if group.line_ids},
        )

    if case_name != "largest_group_high":
        raise ValueError(f"Unsupported environmental risk case: {case_name}")

    largest_group_id, largest_lines = identify_largest_group(group_summary)
    rng = np.random.default_rng(30)
    p_env_by_line: Dict[int, float] = {}
    group_p_env: Dict[str, float] = {}
    for group in wildfire.line_groups:
        if group.name == largest_group_id:
            value = 1.0
        else:
            value = float(rng.uniform(0.6, 0.9))
        group_p_env[group.name] = value
        for line_id in group.line_ids:
            p_env_by_line[int(line_id)] = value

    return EnvironmentalCaseResult(
        name=case_name,
        p_env_by_line=p_env_by_line,
        largest_group_id=largest_group_id,
        largest_group_line_ids=[int(line_id) for line_id in largest_lines],
        manual_high_risk_group_id=largest_group_id,
        group_p_env=group_p_env,
    )


def psps_line_count(num_candidate_lines: int, psps_top_fraction: float) -> int:
    if num_candidate_lines <= 0:
        raise ValueError("PSPS candidate set is empty.")
    if psps_top_fraction <= 0.0 or psps_top_fraction > 1.0:
        raise ValueError(f"psps_top_fraction must be in (0, 1], got {psps_top_fraction}.")
    return int(max(1, min(num_candidate_lines, math.ceil(num_candidate_lines * float(psps_top_fraction)))))


def select_psps_lines(candidate_line_ids: Iterable[int], baseline_psps_risk: Dict[int, float], psps_top_fraction: float) -> List[int]:
    candidates = [int(line_id) for line_id in candidate_line_ids]
    count = psps_line_count(len(candidates), psps_top_fraction)
    ordered = sorted(candidates, key=lambda line_id: (-float(baseline_psps_risk[line_id]), int(line_id)))
    return ordered[:count]


def compute_line_risk_with_z(
    loading_ratio: np.ndarray,
    p_env_by_line: Dict[int, float],
    impact: np.ndarray,
    z_by_line: Dict[int, int],
    candidate_line_ids: Iterable[int],
) -> tuple[float, Dict[int, float]]:
    loading = np.asarray(loading_ratio, dtype=float)
    impacts = np.asarray(impact, dtype=float)
    risk_by_line: Dict[int, float] = {}
    total = 0.0
    for line_id in candidate_line_ids:
        line_id = int(line_id)
        z_l = int(z_by_line.get(line_id, 1))
        risk = float(z_l * p_env_by_line.get(line_id, 0.0) * loading[line_id] ** 2 * impacts[line_id])
        risk_by_line[line_id] = risk
        total += risk
    return float(total), risk_by_line


@contextmanager
def scenario_with_line_outages(scenario, line_ids: Iterable[int]):
    edge_index = scenario.edge_index
    g = scenario.G
    b = scenario.B
    rate_a = scenario.rate_a
    physical_branch_id = getattr(scenario, "physical_branch_id", None)
    canonical_line_id = getattr(scenario, "canonical_line_id", None)
    is_self_loop = getattr(scenario, "is_self_loop", None)
    branch_mapping_status = getattr(scenario, "branch_mapping_status", None)
    yf = getattr(scenario, "Yf", None)
    yt = getattr(scenario, "Yt", None)
    num_edges = int(edge_index.shape[1])
    outage_set = set(expand_to_physical_line_ids(scenario, line_ids))
    keep_mask = [idx for idx in range(num_edges) if idx not in outage_set]
    try:
        scenario.edge_index = edge_index[:, keep_mask]
        scenario.G = g[keep_mask]
        scenario.B = b[keep_mask]
        if rate_a is not None:
            scenario.rate_a = rate_a[keep_mask]
        if physical_branch_id is not None:
            scenario.physical_branch_id = physical_branch_id[keep_mask]
        if canonical_line_id is not None:
            scenario.canonical_line_id = canonical_line_id[keep_mask]
        if is_self_loop is not None:
            scenario.is_self_loop = is_self_loop[keep_mask]
        if branch_mapping_status is not None:
            scenario.branch_mapping_status = branch_mapping_status[keep_mask]
        scenario.Yf = None
        scenario.Yt = None
        yield keep_mask
    finally:
        scenario.edge_index = edge_index
        scenario.G = g
        scenario.B = b
        scenario.rate_a = rate_a
        scenario.physical_branch_id = physical_branch_id
        scenario.canonical_line_id = canonical_line_id
        scenario.is_self_loop = is_self_loop
        scenario.branch_mapping_status = branch_mapping_status
        scenario.Yf = yf
        scenario.Yt = yt


def predict_psps_state(scenario, runner, u_base: np.ndarray, deenergized_line_ids: List[int], standard_rate_a_mva: float) -> tuple[Dict, Dict]:
    num_lines = int(scenario.edge_index.shape[1])
    with scenario_with_line_outages(scenario, deenergized_line_ids):
        prediction = runner.predict(u_base)
    state = extract_state_quantities(
        scenario,
        prediction,
        standard_rate_a_mva=standard_rate_a_mva,
    )
    loading = np.asarray(state["loading_ratio"], dtype=float).copy()
    for line_id in deenergized_line_ids:
        loading[int(line_id)] = 0.0
    state["loading_ratio"] = loading
    state["apparent_flow_proxy"] = loading * float(standard_rate_a_mva)
    state["num_lines"] = num_lines
    state["max_loading_ratio"] = float(np.nanmax(loading)) if len(loading) else 0.0
    return prediction, state


def disconnected_component_count(edge_index, num_buses: int, deenergized_line_ids: Iterable[int]) -> int:
    edge_array = edge_index.cpu().numpy() if hasattr(edge_index, "cpu") else np.asarray(edge_index)
    outage_set = {int(line_id) for line_id in deenergized_line_ids}
    graph = nx.Graph()
    graph.add_nodes_from(range(num_buses))
    for line_id, (src, dst) in enumerate(edge_array.T):
        if int(line_id) in outage_set:
            continue
        graph.add_edge(int(src), int(dst))
    return int(nx.number_connected_components(graph))


def affected_buses(edge_index, deenergized_line_ids: Iterable[int]) -> List[int]:
    edge_array = edge_index.cpu().numpy() if hasattr(edge_index, "cpu") else np.asarray(edge_index)
    buses = set()
    for line_id in deenergized_line_ids:
        buses.add(int(edge_array[0, int(line_id)]))
        buses.add(int(edge_array[1, int(line_id)]))
    return sorted(buses)
