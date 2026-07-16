from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class WildfireLineGroup:
    name: str
    line_ids: List[int]
    group_weight: float = 1.0
    description: str = ""


@dataclass
class WildfireScenario:
    name: str
    line_groups: List[WildfireLineGroup]
    hazard_by_line: Dict[int, float]
    impact_by_line: Dict[int, float]
    default_hazard: float = 0.1
    default_impact: float = 1.0

    def validate(self, num_lines: int) -> None:
        for group in self.line_groups:
            for line_id in group.line_ids:
                if line_id < 0 or line_id >= num_lines:
                    raise ValueError(f"Line id {line_id} is invalid for {num_lines} lines.")

    def hazard_vector(self, num_lines: int) -> np.ndarray:
        return np.asarray(
            [self.hazard_by_line.get(i, self.default_hazard) for i in range(num_lines)],
            dtype=float,
        )

    def impact_vector(self, num_lines: int) -> np.ndarray:
        return np.asarray(
            [self.impact_by_line.get(i, self.default_impact) for i in range(num_lines)],
            dtype=float,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hazard_by_line"] = {str(k): v for k, v in self.hazard_by_line.items()}
        data["impact_by_line"] = {str(k): v for k, v in self.impact_by_line.items()}
        return data


def build_synthetic_wildfire_scenario(
    loading: np.ndarray,
    selected_line_ids: List[int] | None = None,
    selection_method: str = "top_loaded",
    num_high_risk_lines: int = 3,
    high_hazard: float = 5.0,
    default_hazard: float = 0.1,
    default_impact: float = 1.0,
    group_weight: float = 1.0,
    hazard_multiplier: float = 1.0,
) -> WildfireScenario:
    num_lines = int(len(loading))
    if selection_method == "manual_connected" and not selected_line_ids:
        raise ValueError("manual_connected wildfire selection requires selected_line_ids.")
    if selected_line_ids:
        line_ids = [int(i) for i in selected_line_ids]
    else:
        line_ids = np.argsort(np.asarray(loading))[::-1][:num_high_risk_lines].astype(int).tolist()
    hazard = {int(i): float(high_hazard * hazard_multiplier) for i in line_ids}
    impact = {int(i): float(default_impact) for i in range(num_lines)}
    scenario = WildfireScenario(
        name="synthetic_high_risk_corridor",
        line_groups=[
            WildfireLineGroup(
                name="high_risk_corridor",
                line_ids=line_ids,
                group_weight=float(group_weight),
                description=(
                    "Manual connected first-pass wildfire corridor."
                    if selection_method == "manual_connected"
                    else "Synthetic first-pass wildfire corridor selected from branch loading."
                ),
            )
        ],
        hazard_by_line=hazard,
        impact_by_line=impact,
        default_hazard=float(default_hazard),
        default_impact=float(default_impact),
    )
    scenario.validate(num_lines)
    return scenario


def _edge_array(edge_index) -> np.ndarray:
    return edge_index.cpu().numpy() if hasattr(edge_index, "cpu") else np.asarray(edge_index)


def top_fraction_line_count(num_lines: int, top_fraction: float) -> int:
    if num_lines <= 0:
        raise ValueError("Cannot select high-risk lines from an empty edge set.")
    if top_fraction <= 0.0 or top_fraction > 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}.")
    return int(max(1, min(num_lines, np.ceil(num_lines * float(top_fraction)))))


def select_top_fraction_line_ids(scores: np.ndarray, top_fraction: float) -> List[int]:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional array.")
    if len(scores) == 0:
        raise ValueError("scores must contain at least one line.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite.")
    count = top_fraction_line_count(len(scores), top_fraction)
    ordered = sorted(range(len(scores)), key=lambda line_id: (-float(scores[line_id]), int(line_id)))
    return [int(line_id) for line_id in ordered[:count]]


def canonical_physical_line_ids(edge_index) -> List[int]:
    """Return one canonical off-diagonal line ID for each physical bus pair."""

    edge_array = _edge_array(edge_index)
    pair_to_line_id: Dict[tuple[int, int], int] = {}
    for line_id, (src, dst) in enumerate(edge_array.T):
        src_i = int(src)
        dst_i = int(dst)
        if src_i == dst_i:
            continue
        pair = tuple(sorted((src_i, dst_i)))
        pair_to_line_id[pair] = min(int(line_id), pair_to_line_id.get(pair, int(line_id)))
    return sorted(pair_to_line_id.values())


def select_top_fraction_candidate_line_ids(scores: np.ndarray, candidate_line_ids: List[int], top_fraction: float) -> List[int]:
    scores = np.asarray(scores, dtype=float)
    candidates = sorted({int(line_id) for line_id in candidate_line_ids})
    if not candidates:
        raise ValueError("candidate_line_ids must contain at least one physical line.")
    if not np.all(np.isfinite(scores[candidates])):
        raise ValueError("candidate physical-line scores must be finite.")
    count = top_fraction_line_count(len(candidates), top_fraction)
    ordered = sorted(candidates, key=lambda line_id: (-float(scores[line_id]), int(line_id)))
    return [int(line_id) for line_id in ordered[:count]]


def connected_components_from_line_ids(edge_index, line_ids: List[int]) -> List[Dict]:
    if not line_ids:
        raise ValueError("At least one selected line is required for connected-component grouping.")

    edge_array = _edge_array(edge_index)
    num_edges = int(edge_array.shape[1])
    selected = sorted(int(line_id) for line_id in line_ids)
    for line_id in selected:
        if line_id < 0 or line_id >= num_edges:
            raise ValueError(f"Line id {line_id} is invalid for {num_edges} scenario edges.")

    adjacency: Dict[int, set[int]] = {}
    incident_lines: Dict[int, List[int]] = {}
    for line_id in selected:
        src = int(edge_array[0, line_id])
        dst = int(edge_array[1, line_id])
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)
        incident_lines.setdefault(src, []).append(line_id)
        incident_lines.setdefault(dst, []).append(line_id)

    visited_nodes: set[int] = set()
    components = []
    for seed_line in selected:
        src = int(edge_array[0, seed_line])
        if src in visited_nodes:
            continue
        stack = [src]
        component_nodes: set[int] = set()
        component_lines: set[int] = set()
        while stack:
            node = stack.pop()
            if node in visited_nodes:
                continue
            visited_nodes.add(node)
            component_nodes.add(node)
            component_lines.update(incident_lines.get(node, []))
            stack.extend(adjacency.get(node, set()) - visited_nodes)
        if component_lines:
            components.append(
                {
                    "line_ids": sorted(int(line_id) for line_id in component_lines),
                    "bus_ids": sorted(int(node) for node in component_nodes),
                }
            )

    return sorted(components, key=lambda item: (item["line_ids"][0], item["bus_ids"][0]))


def build_automatic_risk_component_scenario(
    edge_index,
    loading_base: np.ndarray,
    impact_base: np.ndarray,
    top_fraction: float,
    candidate_hazard: float = 1.0,
    default_hazard: float = 0.1,
    default_impact: float = 1.0,
    group_weight: float = 1.0,
) -> tuple[WildfireScenario, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Build automatic wildfire groups from baseline risk scores.

    The candidate hazard is uniform for all lines before selection. This avoids
    circularly using selected-line hazards to decide which lines are selected.
    """
    loading = np.asarray(loading_base, dtype=float)
    impact = np.asarray(impact_base, dtype=float)
    if loading.shape != impact.shape:
        raise ValueError(f"loading_base has shape {loading.shape}; impact_base has shape {impact.shape}.")
    num_lines = int(len(loading))
    if num_lines == 0:
        raise ValueError("Cannot build automatic wildfire groups for an empty edge set.")

    candidate_p_env = float(candidate_hazard)
    scores = candidate_p_env * np.square(loading) * impact
    candidate_line_ids = canonical_physical_line_ids(edge_index)
    selected_line_ids = select_top_fraction_candidate_line_ids(scores, candidate_line_ids, top_fraction)
    components = connected_components_from_line_ids(edge_index, selected_line_ids)
    group_by_line: Dict[int, str] = {}
    groups = []
    edge_array = _edge_array(edge_index)
    selected_set = set(selected_line_ids)

    for group_idx, component in enumerate(components, start=1):
        group_id = f"G_{group_idx}"
        for line_id in component["line_ids"]:
            group_by_line[int(line_id)] = group_id
        groups.append(
            WildfireLineGroup(
                name=group_id,
                line_ids=[int(line_id) for line_id in component["line_ids"]],
                group_weight=float(group_weight),
                description="Automatic connected-component wildfire group.",
            )
        )

    rank_by_line = {
        int(line_id): rank
        for rank, line_id in enumerate(
            sorted(candidate_line_ids, key=lambda idx: (-float(scores[idx]), int(idx))),
            start=1,
        )
    }
    line_rows = []
    for line_id in range(num_lines):
        line_rows.append(
            {
                "line_id": int(line_id),
                "from_bus": int(edge_array[0, line_id]),
                "to_bus": int(edge_array[1, line_id]),
                "p_env": candidate_p_env,
                "loading_base": float(loading[line_id]),
                "impact_base": float(impact[line_id]),
                "score": float(scores[line_id]),
                "rank": int(rank_by_line[line_id]) if line_id in rank_by_line else "",
                "physical_candidate": bool(line_id in set(candidate_line_ids)),
                "selected_high_risk": bool(line_id in selected_set),
                "group_id": group_by_line.get(line_id, ""),
            }
        )
    line_scores = pd.DataFrame(line_rows)

    hazard_by_line = {int(line_id): candidate_p_env for line_id in selected_line_ids}
    scenario = WildfireScenario(
        name="automatic_risk_components",
        line_groups=groups,
        hazard_by_line=hazard_by_line,
        impact_by_line={int(i): float(default_impact) for i in range(num_lines)},
        default_hazard=float(default_hazard),
        default_impact=float(default_impact),
    )
    scenario.validate(num_lines)

    num_selected = int(len(selected_line_ids))
    collapsed = int(len(groups)) == 1
    group_rows = []
    for group in groups:
        component = components[int(group.name.split("_")[1]) - 1]
        baseline_group_risk = float(np.sum(scores[group.line_ids]))
        group_rows.append(
            {
                "group_id": group.name,
                "line_ids": ",".join(str(line_id) for line_id in group.line_ids),
                "bus_ids": ",".join(str(bus_id) for bus_id in component["bus_ids"]),
                "num_lines": int(len(group.line_ids)),
                "baseline_group_risk": baseline_group_risk,
                "group_weight": float(group.group_weight),
            }
        )
    group_summary = pd.DataFrame(group_rows)
    if len(group_summary):
        largest_idx = int(group_summary["num_lines"].idxmax())
        group_summary["largest_group_flag"] = False
        group_summary.loc[largest_idx, "largest_group_flag"] = True
    else:
        group_summary["largest_group_flag"] = []

    largest_group_num_lines = int(group_summary["num_lines"].max()) if len(group_summary) else 0
    largest_fraction = float(largest_group_num_lines / num_selected) if num_selected else 0.0
    metadata = {
        "selection_method": "automatic_risk_components",
        "risk_score_formula": "p_env_times_loading_squared_times_impact",
        "threshold_method": "top_fraction",
        "requested_top_fraction": float(top_fraction),
        "top_fraction": float(top_fraction),
        "num_lines": int(num_lines),
        "num_physical_candidate_lines": int(len(candidate_line_ids)),
        "num_selected_lines": num_selected,
        "realized_selected_fraction": float(num_selected / len(candidate_line_ids)),
        "selected_high_risk_line_ids": [int(line_id) for line_id in selected_line_ids],
        "num_groups": int(len(groups)),
        "group_ids": [group.name for group in groups],
        "group_line_ids": {group.name: [int(line_id) for line_id in group.line_ids] for group in groups},
        "group_bus_ids": {
            f"G_{idx}": [int(bus_id) for bus_id in component["bus_ids"]]
            for idx, component in enumerate(components, start=1)
        },
        "collapsed_to_single_group": bool(collapsed),
        "largest_group_num_lines": largest_group_num_lines,
        "largest_group_fraction_of_selected_lines": largest_fraction,
        "candidate_p_env": candidate_p_env,
    }
    metadata["groups"] = [
        {
            "group_id": row["group_id"],
            "line_ids": [int(item) for item in str(row["line_ids"]).split(",") if item != ""],
            "bus_ids": [int(item) for item in str(row["bus_ids"]).split(",") if item != ""],
            "num_lines": int(row["num_lines"]),
            "baseline_group_risk": float(row["baseline_group_risk"]),
            "group_weight": float(row["group_weight"]),
        }
        for _, row in group_summary.iterrows()
    ]
    return scenario, line_scores, group_summary, metadata


def validate_connected_line_group(edge_index, line_ids: List[int]) -> None:
    if not line_ids:
        raise ValueError("Connected line group must contain at least one line id.")

    edge_array = edge_index.cpu().numpy() if hasattr(edge_index, "cpu") else np.asarray(edge_index)
    num_edges = int(edge_array.shape[1])
    for line_id in line_ids:
        if line_id < 0 or line_id >= num_edges:
            raise ValueError(f"Line id {line_id} is invalid for {num_edges} scenario edges.")

    adjacency: Dict[int, set[int]] = {}
    selected_edges = []
    for line_id in line_ids:
        src = int(edge_array[0, line_id])
        dst = int(edge_array[1, line_id])
        selected_edges.append((src, dst))
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    start = selected_edges[0][0]
    stack = [start]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency.get(node, set()) - visited)

    selected_nodes = {node for edge in selected_edges for node in edge}
    if not selected_nodes.issubset(visited):
        raise ValueError(
            "Selected wildfire line IDs do not form one connected corridor: "
            f"{line_ids}"
        )
