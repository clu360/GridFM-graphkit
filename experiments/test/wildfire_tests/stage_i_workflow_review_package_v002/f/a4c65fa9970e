from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MatpowerBranch:
    """Static MATPOWER IEEE-30 branch metadata, using zero-based bus indices."""

    from_bus: int
    to_bus: int
    r: float
    x: float
    b: float
    rate_a_mva: float
    tap: float = 0.0
    shift_deg: float = 0.0
    status: int = 1

    @property
    def pair(self) -> tuple[int, int]:
        return tuple(sorted((int(self.from_bus), int(self.to_bus))))


# Source: MATPOWER case30.m, branch columns fbus, tbus, ..., rateA.
# Bus numbering is converted from MATPOWER's one-based IDs to zero-based IDs.
MATPOWER_CASE30_BRANCHES: tuple[MatpowerBranch, ...] = (
    MatpowerBranch(0, 1, 0.02, 0.06, 0.03, 130.0),
    MatpowerBranch(0, 2, 0.05, 0.19, 0.02, 130.0),
    MatpowerBranch(1, 3, 0.06, 0.17, 0.02, 65.0),
    MatpowerBranch(2, 3, 0.01, 0.04, 0.0, 130.0),
    MatpowerBranch(1, 4, 0.05, 0.2, 0.02, 130.0),
    MatpowerBranch(1, 5, 0.06, 0.18, 0.02, 65.0),
    MatpowerBranch(3, 5, 0.01, 0.04, 0.0, 90.0),
    MatpowerBranch(4, 6, 0.05, 0.12, 0.01, 70.0),
    MatpowerBranch(5, 6, 0.03, 0.08, 0.01, 130.0),
    MatpowerBranch(5, 7, 0.01, 0.04, 0.0, 32.0),
    MatpowerBranch(5, 8, 0.0, 0.21, 0.0, 65.0),
    MatpowerBranch(5, 9, 0.0, 0.56, 0.0, 32.0),
    MatpowerBranch(8, 10, 0.0, 0.21, 0.0, 65.0),
    MatpowerBranch(8, 9, 0.0, 0.11, 0.0, 65.0),
    MatpowerBranch(3, 11, 0.0, 0.26, 0.0, 65.0),
    MatpowerBranch(11, 12, 0.0, 0.14, 0.0, 65.0),
    MatpowerBranch(11, 13, 0.12, 0.26, 0.0, 32.0),
    MatpowerBranch(11, 14, 0.07, 0.13, 0.0, 32.0),
    MatpowerBranch(11, 15, 0.09, 0.2, 0.0, 32.0),
    MatpowerBranch(13, 14, 0.22, 0.2, 0.0, 16.0),
    MatpowerBranch(15, 16, 0.08, 0.19, 0.0, 16.0),
    MatpowerBranch(14, 17, 0.11, 0.22, 0.0, 16.0),
    MatpowerBranch(17, 18, 0.06, 0.13, 0.0, 16.0),
    MatpowerBranch(18, 19, 0.03, 0.07, 0.0, 32.0),
    MatpowerBranch(9, 19, 0.09, 0.21, 0.0, 32.0),
    MatpowerBranch(9, 16, 0.03, 0.08, 0.0, 32.0),
    MatpowerBranch(9, 20, 0.03, 0.07, 0.0, 32.0),
    MatpowerBranch(9, 21, 0.07, 0.15, 0.0, 32.0),
    MatpowerBranch(20, 21, 0.01, 0.02, 0.0, 32.0),
    MatpowerBranch(14, 22, 0.1, 0.2, 0.0, 16.0),
    MatpowerBranch(21, 23, 0.12, 0.18, 0.0, 16.0),
    MatpowerBranch(22, 23, 0.13, 0.27, 0.0, 16.0),
    MatpowerBranch(23, 24, 0.19, 0.33, 0.0, 16.0),
    MatpowerBranch(24, 25, 0.25, 0.38, 0.0, 16.0),
    MatpowerBranch(24, 26, 0.11, 0.21, 0.0, 16.0),
    MatpowerBranch(27, 26, 0.0, 0.4, 0.0, 65.0),
    MatpowerBranch(26, 28, 0.22, 0.42, 0.0, 16.0),
    MatpowerBranch(26, 29, 0.32, 0.6, 0.0, 16.0),
    MatpowerBranch(28, 29, 0.24, 0.45, 0.0, 16.0),
    MatpowerBranch(7, 27, 0.06, 0.2, 0.02, 32.0),
    MatpowerBranch(5, 27, 0.02, 0.06, 0.01, 32.0),
)


def matpower_rate_by_pair() -> dict[tuple[int, int], float]:
    rates: dict[tuple[int, int], float] = {}
    for branch in MATPOWER_CASE30_BRANCHES:
        rates[branch.pair] = float(branch.rate_a_mva)
    return rates


def matpower_branch_by_pair() -> dict[tuple[int, int], MatpowerBranch]:
    branches: dict[tuple[int, int], MatpowerBranch] = {}
    for branch in MATPOWER_CASE30_BRANCHES:
        branches[branch.pair] = branch
    return branches


def _angles_to_radians(Va: np.ndarray) -> np.ndarray:
    angles = np.asarray(Va, dtype=float)
    if len(angles) and np.nanmax(np.abs(angles)) > (2.0 * np.pi + 1e-9):
        return np.deg2rad(angles)
    return angles


def compute_matpower_ac_loading_by_line(
    scenario,
    Vm: np.ndarray,
    Va: np.ndarray,
    base_mva: float = 100.0,
) -> np.ndarray:
    """Compute full MATPOWER AC branch loading mapped to scenario line IDs."""

    edge_index = scenario.edge_index.cpu().numpy() if hasattr(scenario.edge_index, "cpu") else np.asarray(scenario.edge_index)
    V = np.asarray(Vm, dtype=float) * np.exp(1j * _angles_to_radians(Va))
    branches_by_pair = matpower_branch_by_pair()
    loading = np.zeros(int(edge_index.shape[1]), dtype=float)

    for line_id, (src, dst) in enumerate(edge_index.T):
        src_i = int(src)
        dst_i = int(dst)
        if src_i == dst_i:
            continue
        pair = tuple(sorted((src_i, dst_i)))
        branch = branches_by_pair.get(pair)
        if branch is None or branch.status == 0 or branch.rate_a_mva <= 0:
            loading[line_id] = np.nan
            continue

        y = 1.0 / complex(branch.r, branch.x)
        b_shunt = 1j * float(branch.b) / 2.0
        tap_mag = float(branch.tap) if float(branch.tap) != 0.0 else 1.0
        tap = tap_mag * np.exp(1j * np.deg2rad(float(branch.shift_deg)))

        yff = (y + b_shunt) / (tap * np.conj(tap))
        yft = -y / np.conj(tap)
        ytf = -y / tap
        ytt = y + b_shunt

        # Evaluate in the physical branch orientation, then map the max loading
        # back to both directed scenario edges for compatibility.
        f = int(branch.from_bus)
        t = int(branch.to_bus)
        If = yff * V[f] + yft * V[t]
        It = ytf * V[f] + ytt * V[t]
        Sf = abs(V[f] * np.conj(If)) * float(base_mva)
        St = abs(V[t] * np.conj(It)) * float(base_mva)
        loading[line_id] = max(Sf, St) / float(branch.rate_a_mva)

    physical_branch_id = getattr(scenario, "physical_branch_id", None)
    if physical_branch_id is not None:
        physical_branch_id = np.asarray(physical_branch_id, dtype=int)
        for branch_id in sorted(int(item) for item in np.unique(physical_branch_id) if int(item) >= 0):
            branch_mask = physical_branch_id == branch_id
            branch_loading = float(np.nanmax(loading[branch_mask])) if np.any(branch_mask) else 0.0
            loading[branch_mask] = branch_loading
    return loading


def attach_case30_branch_metadata(scenario) -> None:
    """Attach physical branch metadata to a ScenarioData-like object in-place."""

    edge_index = scenario.edge_index.cpu().numpy() if hasattr(scenario.edge_index, "cpu") else np.asarray(scenario.edge_index)
    num_edges = int(edge_index.shape[1])
    rates_by_pair = matpower_rate_by_pair()

    pair_to_physical_id: dict[tuple[int, int], int] = {}
    pair_to_directed_ids: dict[tuple[int, int], list[int]] = {}
    line_to_physical_branch = np.full(num_edges, -1, dtype=int)
    canonical_line_id = np.full(num_edges, -1, dtype=int)
    rate_a = np.full(num_edges, np.nan, dtype=float)
    is_self_loop = np.zeros(num_edges, dtype=bool)
    mapping_status: list[str] = []

    for line_id, (src, dst) in enumerate(edge_index.T):
        src_i = int(src)
        dst_i = int(dst)
        if src_i == dst_i:
            is_self_loop[line_id] = True
            mapping_status.append("self_loop")
            continue

        pair = tuple(sorted((src_i, dst_i)))
        if pair not in pair_to_physical_id:
            pair_to_physical_id[pair] = len(pair_to_physical_id)
            pair_to_directed_ids[pair] = []
        physical_id = pair_to_physical_id[pair]
        pair_to_directed_ids[pair].append(int(line_id))
        line_to_physical_branch[line_id] = int(physical_id)
        rate = rates_by_pair.get(pair)
        if rate is None:
            mapping_status.append("unmapped")
        else:
            rate_a[line_id] = float(rate)
            mapping_status.append("mapped")

    for directed_ids in pair_to_directed_ids.values():
        canonical = int(min(directed_ids))
        for line_id in directed_ids:
            canonical_line_id[int(line_id)] = canonical

    scenario.rate_a = rate_a
    scenario.physical_branch_id = line_to_physical_branch
    scenario.canonical_line_id = canonical_line_id
    scenario.is_self_loop = is_self_loop
    scenario.branch_mapping_status = np.asarray(mapping_status, dtype=object)
    scenario.physical_branch_pairs = {
        int(physical_id): pair for pair, physical_id in pair_to_physical_id.items()
    }
    scenario.physical_branch_directed_line_ids = {
        int(pair_to_physical_id[pair]): sorted(int(line_id) for line_id in directed_ids)
        for pair, directed_ids in pair_to_directed_ids.items()
    }


def physical_line_ids(scenario) -> list[int]:
    """Return canonical line IDs for mapped off-diagonal physical branches."""

    canonical = getattr(scenario, "canonical_line_id", None)
    status = getattr(scenario, "branch_mapping_status", None)
    if canonical is None:
        edge_index = scenario.edge_index.cpu().numpy() if hasattr(scenario.edge_index, "cpu") else np.asarray(scenario.edge_index)
        return [
            int(line_id)
            for line_id, (src, dst) in enumerate(edge_index.T)
            if int(src) != int(dst)
        ]
    ids = sorted(
        {
            int(line_id)
            for line_id in canonical
            if int(line_id) >= 0
            and (status is None or str(status[int(line_id)]) == "mapped")
        }
    )
    return ids


def expand_to_physical_line_ids(scenario, line_ids: Iterable[int]) -> list[int]:
    """Expand any directed line IDs to all directed IDs in their physical branch."""

    directed_by_branch = getattr(scenario, "physical_branch_directed_line_ids", None)
    physical_ids = getattr(scenario, "physical_branch_id", None)
    if directed_by_branch is None or physical_ids is None:
        return sorted({int(line_id) for line_id in line_ids})

    expanded: set[int] = set()
    for line_id in line_ids:
        line_id = int(line_id)
        if line_id < 0 or line_id >= len(physical_ids):
            expanded.add(line_id)
            continue
        physical_id = int(physical_ids[line_id])
        if physical_id < 0:
            expanded.add(line_id)
            continue
        expanded.update(int(item) for item in directed_by_branch.get(physical_id, [line_id]))
    return sorted(expanded)


def canonicalize_line_ids(scenario, line_ids: Iterable[int]) -> list[int]:
    """Map directed line IDs to canonical physical branch line IDs."""

    canonical = getattr(scenario, "canonical_line_id", None)
    if canonical is None:
        return sorted({int(line_id) for line_id in line_ids})
    values: set[int] = set()
    for line_id in line_ids:
        line_id = int(line_id)
        if 0 <= line_id < len(canonical) and int(canonical[line_id]) >= 0:
            values.add(int(canonical[line_id]))
        else:
            values.add(line_id)
    return sorted(values)
