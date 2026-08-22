from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    DEFAULT_PROXY_TYPE,
    SUPPORTED_PROXY_TYPES,
    compute_proxy_metrics,
    proxy_denominator,
)


class GurobiUnavailableError(RuntimeError):
    """Raised when gurobipy is not importable or cannot solve the requested model."""


def import_gurobipy():
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        raise GurobiUnavailableError(f"gurobipy is unavailable: {exc}") from exc
    return gp, GRB


def solve_tiny_gurobi_smoke() -> bool:
    gp, GRB = import_gurobipy()
    model = gp.Model("stage_e_smoke")
    model.Params.OutputFlag = 0
    x = model.addVar(vtype=GRB.BINARY, name="x")
    model.setObjective(x, GRB.MINIMIZE)
    model.optimize()
    return bool(model.Status == GRB.OPTIMAL)


def _risk_coefficients(
    candidate_line_ids: Iterable[int],
    p_env_by_line: Dict[int, float],
    baseline_loading,
    proxy_type: str,
) -> Dict[int, float]:
    if proxy_type not in SUPPORTED_PROXY_TYPES:
        raise ValueError(f"Unsupported proxy_type={proxy_type}.")
    loading = np.asarray(baseline_loading, dtype=float)
    coeffs: Dict[int, float] = {}
    for line_id in candidate_line_ids:
        line_id = int(line_id)
        if proxy_type == "env_loading_base":
            coeffs[line_id] = float(p_env_by_line.get(line_id, 0.0)) * float(loading[line_id]) ** 2
        else:
            coeffs[line_id] = float(p_env_by_line.get(line_id, 0.0))
    return coeffs


def solve_gurobi_master_next_candidate(
    candidate_line_ids: Iterable[int],
    p_env_by_line: Dict[int, float],
    baseline_loading,
    c_by_line: Dict[int, float],
    lambda_R_master: float,
    lambda_L_master: float,
    max_deenergized_lines: int | None,
    evaluated_y_vectors: List[Dict[int, int]] | None = None,
    proxy_type: str = DEFAULT_PROXY_TYPE,
) -> Dict:
    gp, GRB = import_gurobipy()
    candidates = sorted(int(line_id) for line_id in candidate_line_ids)
    if not candidates:
        raise ValueError("Cannot solve Gurobi master with an empty candidate line set.")
    if max_deenergized_lines is not None and int(max_deenergized_lines) < 0:
        raise ValueError("max_deenergized_lines must be nonnegative.")

    evaluated_y_vectors = evaluated_y_vectors or []
    denominator = proxy_denominator(candidates, p_env_by_line, baseline_loading, proxy_type)
    risk_coeffs = _risk_coefficients(candidates, p_env_by_line, baseline_loading, proxy_type)

    model = gp.Model("stage_e_gurobi_master")
    model.Params.OutputFlag = 0
    y = {line_id: model.addVar(vtype=GRB.BINARY, name=f"y_{line_id}") for line_id in candidates}
    if max_deenergized_lines is not None:
        model.addConstr(gp.quicksum(y[line_id] for line_id in candidates) <= int(max_deenergized_lines), name="shutoff_budget")

    for cut_idx, previous in enumerate(evaluated_y_vectors):
        model.addConstr(
            gp.quicksum((1 - y[line_id]) if int(previous.get(line_id, 0)) == 1 else y[line_id] for line_id in candidates) >= 1,
            name=f"no_good_{cut_idx}",
        )

    proxy_R = gp.quicksum(risk_coeffs[line_id] * (1 - y[line_id]) for line_id in candidates) / denominator
    proxy_L = gp.quicksum(float(c_by_line.get(line_id, 0.0)) * y[line_id] for line_id in candidates)
    model.setObjective(float(lambda_R_master) * proxy_R + float(lambda_L_master) * proxy_L, GRB.MINIMIZE)
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        raise GurobiUnavailableError(f"Gurobi master did not return a usable solution; status={model.Status}.")

    y_by_line = {line_id: int(round(float(y[line_id].X))) for line_id in candidates}
    metrics = compute_proxy_metrics(
        candidates,
        p_env_by_line,
        baseline_loading,
        c_by_line,
        y_by_line,
        lambda_R_master,
        lambda_L_master,
        proxy_type=proxy_type,
    )
    return {
        "y_by_line": y_by_line,
        "status": "ok",
        "gurobi_status": int(model.Status),
        "gurobi_objective": float(model.ObjVal),
        **metrics,
    }
