"""Stage K service accounting and topology-relative GridSFM load selection."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping

from .identity import StageKIdentity, source_less_load_ids


def effective_alpha(
    identity: StageKIdentity,
    offline_branch_ids: Iterable[int],
    requested: Mapping[int, float],
) -> tuple[dict[int, float], tuple[int, ...]]:
    islanded = source_less_load_ids(identity, offline_branch_ids)
    islanded_set = set(islanded)
    out: dict[int, float] = {}
    for load_id in identity.loads["canonical_load_id"].astype(int):
        value = float(requested.get(int(load_id), 1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"alpha[{load_id}]={value} outside [0, 1]")
        out[int(load_id)] = 0.0 if int(load_id) in islanded_set else value
    return out, islanded


def select_topology_relative_loads(
    identity: StageKIdentity,
    offline_branch_ids: Iterable[int],
    *,
    q: int,
    exclude_load_ids: Iterable[int] = (),
) -> tuple[int, ...]:
    """Reuse the Stage J endpoint-distance, demand, ID ordering."""

    offline = {int(value) for value in offline_branch_ids}
    excluded = {int(value) for value in exclude_load_ids}
    branches = identity.branches
    endpoints: set[int] = set()
    adjacency = {int(bus): [] for bus in identity.buses["bus_id"]}
    for row in branches.itertuples(index=False):
        branch_id = int(row.canonical_branch_id)
        if branch_id in offline:
            endpoints.update((int(row.from_bus_id), int(row.to_bus_id)))
            continue
        adjacency[int(row.from_bus_id)].append(int(row.to_bus_id))
        adjacency[int(row.to_bus_id)].append(int(row.from_bus_id))

    distance = {bus: 0 for bus in endpoints}
    queue = deque(sorted(endpoints))
    while queue:
        bus = queue.popleft()
        for neighbor in adjacency.get(bus, []):
            if neighbor not in distance:
                distance[neighbor] = distance[bus] + 1
                queue.append(neighbor)

    candidates = []
    for row in identity.loads.itertuples(index=False):
        load_id = int(row.canonical_load_id)
        if load_id in excluded:
            continue
        candidates.append(
            (
                distance.get(int(row.bus_id), 10**12) if endpoints else 0,
                -float(row.pd_requested_mw),
                load_id,
            )
        )
    return tuple(item[2] for item in sorted(candidates)[: int(q)])
