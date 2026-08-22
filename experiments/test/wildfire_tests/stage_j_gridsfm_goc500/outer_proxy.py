"""Shared Stage J outer wildfire topology proxy with no-good cuts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ProxyTopology:
    """One topology proposed by the shared outer proxy master."""

    rank: int
    shutoff_branch_ids: tuple[int, ...]
    proxy_objective: float | None
    r_proxy: float | None
    l_proxy: float | None


def solve_proxy_topology_pool(
    *,
    candidate_branch_ids: Iterable[int],
    p_env_by_line: Mapping[int, float],
    baseline_loading: Mapping[int, float],
    c_by_line: Mapping[int, float],
    lambda_r_proxy: float,
    k: int = 2,
    pool_size: int = 10,
) -> list[ProxyTopology]:
    """Solve repeated binary proxy masters with no-good cuts."""

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"gurobipy is required for Stage J outer proxy: {exc}") from exc

    candidate_ids = sorted(int(value) for value in candidate_branch_ids)
    if not candidate_ids:
        raise ValueError("candidate_branch_ids must not be empty")
    if k < 0:
        raise ValueError("k must be nonnegative")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    lambda_r_proxy = float(lambda_r_proxy)
    if lambda_r_proxy < 0.0 or lambda_r_proxy > 1.0:
        raise ValueError("lambda_r_proxy must be in [0, 1]")
    lambda_l_proxy = 1.0 - lambda_r_proxy

    weights = {
        line_id: float(p_env_by_line.get(line_id, 0.0)) * float(baseline_loading[line_id]) ** 2
        for line_id in candidate_ids
    }
    denom = sum(weights.values())
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError(f"proxy wildfire denominator must be positive and finite, got {denom}")

    try:
        model = gp.Model("stage_j_outer_proxy")
    except Exception as exc:  # pragma: no cover - license/environment dependent
        raise RuntimeError(f"Gurobi environment is unavailable for Stage J outer proxy: {exc}") from exc
    model.Params.OutputFlag = 0
    y = {line_id: model.addVar(vtype=GRB.BINARY, name=f"y[{line_id}]") for line_id in candidate_ids}
    model.addConstr(gp.quicksum(y.values()) <= int(k), name="k_budget")
    r_proxy_expr = gp.quicksum(weights[line_id] * (1.0 - y[line_id]) for line_id in candidate_ids) / denom
    l_proxy_expr = gp.quicksum(float(c_by_line.get(line_id, 0.0)) * y[line_id] for line_id in candidate_ids)
    model.setObjective(lambda_r_proxy * r_proxy_expr + lambda_l_proxy * l_proxy_expr, GRB.MINIMIZE)
    model.update()

    out: list[ProxyTopology] = []
    seen: set[tuple[int, ...]] = set()
    for rank in range(1, int(pool_size) + 1):
        model.optimize()
        if model.Status != GRB.OPTIMAL or model.SolCount <= 0:
            break
        selected = tuple(line_id for line_id in candidate_ids if y[line_id].X > 0.5)
        if selected in seen:
            break
        seen.add(selected)
        r_proxy = sum(weights[line_id] * (0.0 if line_id in selected else 1.0) for line_id in candidate_ids) / denom
        l_proxy = sum(float(c_by_line.get(line_id, 0.0)) for line_id in selected)
        out.append(
            ProxyTopology(
                rank=rank,
                shutoff_branch_ids=selected,
                proxy_objective=float(lambda_r_proxy * r_proxy + lambda_l_proxy * l_proxy),
                r_proxy=float(r_proxy),
                l_proxy=float(l_proxy),
            )
        )
        model.addConstr(
            gp.quicksum((1.0 - y[line_id]) if line_id in selected else y[line_id] for line_id in candidate_ids) >= 1.0,
            name=f"no_good_{rank}",
        )
    return out
