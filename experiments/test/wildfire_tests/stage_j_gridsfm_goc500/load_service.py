"""Stage J load-service accounting.

Stage J locks one requested alpha for every load. Source-less islands override
the request in the effective alpha vector, and the resulting load shedding is
reported as control versus topology-forced island shedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class LoadSheddingBreakdown:
    """Model-independent Stage J load shedding terms."""

    l_shed_total: float
    l_shed_control: float
    l_shed_island: float
    alpha_requested: dict[int, float]
    alpha_effective: dict[int, float]
    source_less_load_ids: tuple[int, ...]


def _coerce_by_load(load_ids: Iterable[int], values: Mapping[int, float] | Iterable[float], name: str) -> dict[int, float]:
    ids = [int(load_id) for load_id in load_ids]
    if isinstance(values, Mapping):
        missing = [load_id for load_id in ids if load_id not in values]
        if missing:
            raise KeyError(f"{name} is missing load ids: {missing}")
        return {load_id: float(values[load_id]) for load_id in ids}

    seq = [float(value) for value in values]
    if len(seq) != len(ids):
        raise ValueError(f"{name} length {len(seq)} does not match load_ids length {len(ids)}")
    return dict(zip(ids, seq))


def compute_alpha_effective(
    load_ids: Iterable[int],
    alpha_requested: Mapping[int, float] | Iterable[float],
    source_less_load_ids: Iterable[int],
) -> dict[int, float]:
    """Apply hard source-less island enforcement to a full per-load alpha vector."""

    ids = [int(load_id) for load_id in load_ids]
    requested = _coerce_by_load(ids, alpha_requested, "alpha_requested")
    source_less = {int(load_id) for load_id in source_less_load_ids}

    unknown_source_less = sorted(source_less.difference(ids))
    if unknown_source_less:
        raise KeyError(f"source_less_load_ids are not in load_ids: {unknown_source_less}")

    effective: dict[int, float] = {}
    for load_id in ids:
        value = requested[load_id]
        if value < 0.0 or value > 1.0:
            raise ValueError(f"alpha_requested[{load_id}] must be in [0, 1], got {value}")
        effective[load_id] = 0.0 if load_id in source_less else value
    return effective


def compute_load_shedding(
    load_ids: Iterable[int],
    pd_pre: Mapping[int, float] | Iterable[float],
    alpha_requested: Mapping[int, float] | Iterable[float],
    source_less_load_ids: Iterable[int],
    *,
    epsilon: float = 1e-6,
) -> LoadSheddingBreakdown:
    """Compute total, control, and island load shedding with no double counting."""

    ids = [int(load_id) for load_id in load_ids]
    source_less = tuple(sorted(int(load_id) for load_id in source_less_load_ids))
    source_less_set = set(source_less)
    demand = _coerce_by_load(ids, pd_pre, "pd_pre")
    requested = _coerce_by_load(ids, alpha_requested, "alpha_requested")
    effective = compute_alpha_effective(ids, requested, source_less)

    total_demand = float(sum(max(0.0, demand[load_id]) for load_id in ids))
    if total_demand <= epsilon:
        raise ValueError(f"total positive demand must exceed epsilon={epsilon}, got {total_demand}")

    island = float(sum(max(0.0, demand[load_id]) for load_id in source_less) / total_demand)
    control = float(
        sum(
            max(0.0, demand[load_id]) * (1.0 - requested[load_id])
            for load_id in ids
            if load_id not in source_less_set
        )
        / total_demand
    )
    total = float(sum(max(0.0, demand[load_id]) * (1.0 - effective[load_id]) for load_id in ids) / total_demand)

    if not np.isclose(total, control + island, atol=1e-9):
        raise AssertionError("load shedding decomposition failed: total != control + island")

    return LoadSheddingBreakdown(
        l_shed_total=total,
        l_shed_control=control,
        l_shed_island=island,
        alpha_requested={load_id: requested[load_id] for load_id in ids},
        alpha_effective=effective,
        source_less_load_ids=source_less,
    )
