"""Deterministic exact-K Stage K proxy candidate generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ProxyCandidate:
    rank: int
    topology_key: str
    offline_branch_ids: tuple[int, ...]
    parent_topology_key: str
    parent_rank: int | None
    proxy_objective: float
    r_proxy: float
    l_proxy: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def topology_key(offline_branch_ids: Iterable[int]) -> str:
    values = tuple(sorted({int(value) for value in offline_branch_ids}))
    return "intact" if not values else ";".join(str(value) for value in values)


def proxy_components(
    offline: Iterable[int],
    *,
    line_ids: Iterable[int],
    weights: Mapping[int, float],
    c_by_line: Mapping[int, float],
    lambda_r: float,
) -> tuple[float, float, float]:
    ids = tuple(sorted(int(value) for value in line_ids))
    opened = {int(value) for value in offline}
    denom = sum(float(weights[i]) for i in ids)
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError("proxy denominator must be positive and finite")
    r_proxy = sum(float(weights[i]) for i in ids if i not in opened) / denom
    l_proxy = sum(float(c_by_line[i]) for i in opened)
    objective = float(lambda_r) * r_proxy + (1.0 - float(lambda_r)) * l_proxy
    return float(objective), float(r_proxy), float(l_proxy)


def additive_open_score(
    line_id: int, *, weights: Mapping[int, float], c_by_line: Mapping[int, float],
    denominator: float, lambda_r: float,
) -> float:
    """Per-open-line score proving exact-K proxy separability."""

    return float((1.0 - lambda_r) * c_by_line[line_id] - lambda_r * weights[line_id] / denominator)


def exact_k_candidates(
    *,
    line_ids: Iterable[int],
    weights: Mapping[int, float],
    c_by_line: Mapping[int, float],
    lambda_r: float,
    k: int,
    count: int,
    required_open: Iterable[int] = (),
    excluded_keys: Iterable[str] = (),
    parent_rank: int | None = None,
) -> list[ProxyCandidate]:
    ids = tuple(sorted({int(value) for value in line_ids}))
    required = tuple(sorted({int(value) for value in required_open}))
    if len(required) > k or not set(required).issubset(ids):
        raise ValueError("required_open must be a subset of line_ids with size <= k")
    remaining_k = int(k) - len(required)
    if remaining_k < 0:
        raise ValueError("k must be nonnegative")
    denominator = sum(float(weights[i]) for i in ids)
    scores = {
        i: additive_open_score(
            i, weights=weights, c_by_line=c_by_line, denominator=denominator, lambda_r=float(lambda_r)
        )
        for i in ids
    }
    available = [i for i in ids if i not in required]
    excluded = set(excluded_keys)
    if remaining_k == 0:
        combos = [required]
    elif remaining_k == 1:
        combos = [tuple(sorted(required + (i,))) for i in sorted(available, key=lambda x: (scores[x], x))]
    else:
        combos = [tuple(sorted(required + combo)) for combo in combinations(available, remaining_k)]
        combos.sort(key=lambda combo: (sum(scores[i] for i in combo), combo))

    rows: list[ProxyCandidate] = []
    for combo in combos:
        key = topology_key(combo)
        if key in excluded:
            continue
        objective, r_proxy, l_proxy = proxy_components(
            combo, line_ids=ids, weights=weights, c_by_line=c_by_line, lambda_r=lambda_r
        )
        rows.append(
            ProxyCandidate(
                rank=len(rows) + 1,
                topology_key=key,
                offline_branch_ids=combo,
                parent_topology_key=topology_key(required),
                parent_rank=parent_rank,
                proxy_objective=objective,
                r_proxy=r_proxy,
                l_proxy=l_proxy,
            )
        )
        if len(rows) >= int(count):
            break
    return rows


def k2_children(
    *,
    parents: Iterable[tuple[int, int]],
    line_ids: Iterable[int],
    weights: Mapping[int, float],
    c_by_line: Mapping[int, float],
    lambda_r: float,
    children_per_parent: int,
    max_unique: int,
) -> list[ProxyCandidate]:
    """Generate exact-K2 children with global deduplication and refill."""

    parent_rows = list(parents)
    streams = []
    for parent_rank, parent_line in parent_rows:
        streams.append(
            exact_k_candidates(
                line_ids=line_ids,
                weights=weights,
                c_by_line=c_by_line,
                lambda_r=lambda_r,
                k=2,
                count=len(tuple(line_ids)),
                required_open=(parent_line,),
                parent_rank=parent_rank,
            )
        )
    cursors = [0] * len(streams)
    accepted_by_parent = [0] * len(streams)
    seen: set[str] = set()
    out: list[ProxyCandidate] = []
    progress = True
    while len(out) < int(max_unique) and progress:
        progress = False
        for idx, stream in enumerate(streams):
            if accepted_by_parent[idx] >= int(children_per_parent):
                continue
            while cursors[idx] < len(stream):
                item = stream[cursors[idx]]
                cursors[idx] += 1
                if item.topology_key in seen:
                    continue
                seen.add(item.topology_key)
                accepted_by_parent[idx] += 1
                out.append(
                    ProxyCandidate(
                        rank=len(out) + 1,
                        topology_key=item.topology_key,
                        offline_branch_ids=item.offline_branch_ids,
                        parent_topology_key=item.parent_topology_key,
                        parent_rank=item.parent_rank,
                        proxy_objective=item.proxy_objective,
                        r_proxy=item.r_proxy,
                        l_proxy=item.l_proxy,
                    )
                )
                progress = True
                break
            if len(out) >= int(max_unique):
                break
    return out
