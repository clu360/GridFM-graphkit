from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from experiments.test.wildfire_tests.shared.lambda_cases import CANONICAL_LAMBDA_CASES, LAMBDA_FOLDERS

LAMBDA_CASES = CANONICAL_LAMBDA_CASES


@dataclass(frozen=True)
class BestSubset:
    lambda_case: str
    lambda_R: float
    lambda_L: float
    eval_id: int
    deenergized_line_ids: List[int]
    objective: float
    R_group: float
    R_norm: float
    L_norm: float
    demand_weighted_load_shed: float


def expected_subset_count(num_candidate_lines: int, max_deenergized_lines: int = 2) -> int:
    if num_candidate_lines < 0:
        raise ValueError("num_candidate_lines must be non-negative.")
    if max_deenergized_lines < 0:
        raise ValueError("max_deenergized_lines must be non-negative.")
    return int(
        sum(
            math.comb(num_candidate_lines, size)
            for size in range(min(num_candidate_lines, max_deenergized_lines) + 1)
        )
    )


def enumerate_deenergization_subsets(
    candidate_line_ids: Iterable[int],
    max_deenergized_lines: int = 2,
) -> List[tuple[int, ...]]:
    candidates = sorted(int(line_id) for line_id in candidate_line_ids)
    subsets: List[tuple[int, ...]] = []
    for size in range(min(len(candidates), int(max_deenergized_lines)) + 1):
        subsets.extend(tuple(items) for items in itertools.combinations(candidates, size))
    return subsets


def objective_for_lambda(row, lambda_R: float, lambda_L: float) -> float:
    return float(float(lambda_R) * float(row["R_norm"]) + float(lambda_L) * float(row["L_norm"]))


def select_best_subset(evaluations: pd.DataFrame, lambda_case: str, lambda_R: float, lambda_L: float) -> BestSubset:
    if evaluations.empty:
        raise ValueError("Cannot select a best subset from an empty evaluation table.")
    frame = evaluations.copy()
    frame["objective"] = frame.apply(lambda row: objective_for_lambda(row, lambda_R, lambda_L), axis=1)
    frame["_tie_line_ids"] = frame["deenergized_line_ids"].apply(
        lambda value: tuple(int(item) for item in value) if isinstance(value, (list, tuple)) else tuple()
    )
    frame = frame.sort_values(
        by=["objective", "L_norm", "num_deenergized_lines", "_tie_line_ids"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    best = frame.iloc[0]
    return BestSubset(
        lambda_case=lambda_case,
        lambda_R=float(lambda_R),
        lambda_L=float(lambda_L),
        eval_id=int(best["eval_id"]),
        deenergized_line_ids=[int(item) for item in best["deenergized_line_ids"]],
        objective=float(best["objective"]),
        R_group=float(best["R_group"]),
        R_norm=float(best["R_norm"]),
        L_norm=float(best["L_norm"]),
        demand_weighted_load_shed=float(best["demand_weighted_load_shed"]),
    )


def line_risk_total(
    loading_ratio: Sequence[float],
    p_env_by_line: dict[int, float],
    impact: Sequence[float],
    z_by_line: dict[int, int],
    candidate_line_ids: Iterable[int],
) -> tuple[float, dict[int, float]]:
    loading = np.asarray(loading_ratio, dtype=float)
    impacts = np.asarray(impact, dtype=float)
    risk_by_line: dict[int, float] = {}
    total = 0.0
    for line_id in candidate_line_ids:
        line_id = int(line_id)
        risk = float(int(z_by_line.get(line_id, 1)) * p_env_by_line.get(line_id, 0.0) * loading[line_id] ** 2 * impacts[line_id])
        risk_by_line[line_id] = risk
        total += risk
    return float(total), risk_by_line


def disconnected_buses_from_reference(edge_index, num_buses: int, deenergized_line_ids: Iterable[int], reference_bus: int = 0) -> List[int]:
    edge_array = edge_index.cpu().numpy() if hasattr(edge_index, "cpu") else np.asarray(edge_index)
    outage_set = {int(line_id) for line_id in deenergized_line_ids}
    graph = nx.Graph()
    graph.add_nodes_from(range(num_buses))
    for line_id, (src, dst) in enumerate(edge_array.T):
        if int(line_id) in outage_set:
            continue
        graph.add_edge(int(src), int(dst))
    if int(reference_bus) not in graph:
        return list(range(num_buses))
    connected_to_reference = nx.node_connected_component(graph, int(reference_bus))
    return sorted(int(bus) for bus in graph.nodes if int(bus) not in connected_to_reference)
