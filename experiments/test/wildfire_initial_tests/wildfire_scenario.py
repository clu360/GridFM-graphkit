from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np


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
