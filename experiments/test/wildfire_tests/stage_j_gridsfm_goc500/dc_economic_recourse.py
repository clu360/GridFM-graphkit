"""Guided-DC fixed-(z, alpha) economic recourse contract.

The full GOC-500 solver integration is intentionally separated from the
contract because it depends on the chosen optimization backend. This module
captures the Stage J semantics that every solver implementation must follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .goc500_adapter import (
    AC_LINE_FAMILY,
    AC_LINE_RATE_A_IDX,
    LOAD_PD_IDX,
    TRANSFORMER_FAMILY,
    TR_RATE_A_IDX,
    TR_SHIFT_IDX,
    TR_TAP_IDX,
    build_goc500_identity,
    source_less_load_ids,
)
from .load_service import compute_alpha_effective
from .schemas import EvaluationStatus


@dataclass(frozen=True)
class FixedTopologyDCEconomicRequest:
    """Externally supplied wildfire-side candidate for Guided-DC recourse."""

    topology_z: Mapping[int, int]
    alpha_requested: Mapping[int, float]
    electrical_scenario_id: str
    wildfire_scenario_id: str

    def validate_no_binary_inner_decisions(self) -> None:
        for line_id, z_value in self.topology_z.items():
            if int(z_value) not in (0, 1):
                raise ValueError(f"topology_z[{line_id}] must be 0 or 1, got {z_value}")
        for load_id, alpha in self.alpha_requested.items():
            if float(alpha) < 0.0 or float(alpha) > 1.0:
                raise ValueError(f"alpha_requested[{load_id}] must be in [0, 1], got {alpha}")


@dataclass(frozen=True)
class FixedTopologyDCEconomicResult:
    """Result status for a fixed-topology economic DC-OPF recourse solve."""

    evaluation_status: EvaluationStatus
    objective_cost: float | None = None
    pg_by_generator: Mapping[int, float] = field(default_factory=dict)
    theta_by_bus: Mapping[int, float] = field(default_factory=dict)
    flow_by_line: Mapping[int, float] = field(default_factory=dict)
    rejection_merit: float | None = None
    message: str = ""

    @staticmethod
    def infeasible(message: str) -> "FixedTopologyDCEconomicResult":
        return FixedTopologyDCEconomicResult(
            evaluation_status=EvaluationStatus.DC_INFEASIBLE,
            objective_cost=None,
            rejection_merit=float("inf"),
            message=message,
        )


def solve_fixed_topology_economic_dc_opf(*args, **kwargs) -> FixedTopologyDCEconomicResult:
    """Solve fixed-`(z, alpha)` economic DC-OPF for a GridSFM raw case.

    Required keyword arguments:

    `raw_case`
        GridSFM raw `.pyg.json` dict.
    `offline_branch_ids`
        Canonical branch IDs fixed offline by the wildfire-side decision.
    `alpha_requested`
        Full per-load alpha vector keyed by canonical load ID.

    Optional keyword arguments:

    `identity`
        Prebuilt `GOC500Identity`. If omitted it is built from `raw_case`.
    `time_limit_seconds`
        Optional Gurobi time limit.
    """

    if args:
        raise TypeError("solve_fixed_topology_economic_dc_opf requires keyword arguments")
    missing = [name for name in ("raw_case", "alpha_requested") if name not in kwargs]
    if missing:
        raise TypeError(f"solve_fixed_topology_economic_dc_opf missing required keyword arguments: {missing}")
    raw_case: Mapping[str, Any] = kwargs["raw_case"]
    offline_branch_ids: Iterable[int] = kwargs.get("offline_branch_ids", [])
    alpha_requested: Mapping[int, float] = kwargs["alpha_requested"]
    identity = kwargs.get("identity") or build_goc500_identity(raw_case)
    time_limit_seconds = kwargs.get("time_limit_seconds")

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"gurobipy is required for Stage J Guided-DC recourse: {exc}") from exc

    offline = {int(branch_id) for branch_id in offline_branch_ids}
    branch_by_id = identity.branch_by_id
    unknown_offline = sorted(offline.difference(branch_by_id))
    if unknown_offline:
        raise KeyError(f"offline branches are not in identity: {unknown_offline}")

    source_less = source_less_load_ids(identity, offline)
    load_ids = [load.canonical_load_id for load in identity.loads]
    alpha_effective = compute_alpha_effective(load_ids, alpha_requested, source_less)

    bus_indices = list(range(len(identity.bus_ids)))
    bus_id_to_index = {bus_id: idx for idx, bus_id in enumerate(identity.bus_ids)}
    load_pd_by_bus_idx = {idx: 0.0 for idx in bus_indices}
    for load in identity.loads:
        served = alpha_effective[load.canonical_load_id] * load.pd_pre
        load_pd_by_bus_idx[load.bus_index] += served

    gen_rows = raw_case["grid"]["nodes"].get("generator", [])
    gen_bus_ids = list(identity.generator_bus_ids)
    if len(gen_rows) != len(gen_bus_ids):
        raise ValueError("generator row count does not match metadata gen_bus_map")

    gen_by_bus_idx: dict[int, list[int]] = {idx: [] for idx in bus_indices}
    for gen_idx, bus_id in enumerate(gen_bus_ids):
        if int(bus_id) not in bus_id_to_index:
            raise KeyError(f"generator {gen_idx} maps to unknown bus id {bus_id}")
        gen_by_bus_idx[bus_id_to_index[int(bus_id)]].append(gen_idx)

    active_branches = [branch for branch in identity.branches if branch.canonical_branch_id not in offline]
    components = _bus_components(identity.bus_ids, active_branches)
    reference_bus_indices = sorted(min(bus_id_to_index[bus_id] for bus_id in comp) for comp in components)

    try:
        model = gp.Model("stage_j_fixed_topology_economic_dc_opf")
    except Exception as exc:  # pragma: no cover - license/environment dependent
        raise RuntimeError(f"Gurobi environment is unavailable for Stage J Guided-DC recourse: {exc}") from exc
    model.Params.OutputFlag = 0
    if time_limit_seconds is not None:
        model.Params.TimeLimit = float(time_limit_seconds)

    theta = {idx: model.addVar(lb=-np.pi, ub=np.pi, name=f"theta[{idx}]") for idx in bus_indices}
    flow = {
        branch.canonical_branch_id: model.addVar(lb=-branch.rate_a, ub=branch.rate_a, name=f"f[{branch.canonical_branch_id}]")
        for branch in active_branches
    }
    pg = {}
    for gen_idx, row in enumerate(gen_rows):
        pmin = float(row[2])
        pmax = float(row[3])
        pg[gen_idx] = model.addVar(lb=pmin, ub=pmax, name=f"Pg[{gen_idx}]")

    model.update()

    for ref_idx in reference_bus_indices:
        model.addConstr(theta[ref_idx] == 0.0, name=f"theta_ref[{ref_idx}]")

    for branch in active_branches:
        attrs = raw_case["grid"]["edges"][branch.edge_family]["features"][branch.family_index]
        if branch.edge_family == AC_LINE_FAMILY:
            x = float(attrs[5])
            tap = 1.0
            shift = 0.0
        elif branch.edge_family == TRANSFORMER_FAMILY:
            x = float(attrs[3])
            tap = float(attrs[TR_TAP_IDX]) if float(attrs[TR_TAP_IDX]) != 0.0 else 1.0
            shift = float(attrs[TR_SHIFT_IDX])
        else:  # pragma: no cover - identity builder only emits the two families
            continue
        if abs(x) <= 1e-10:
            return FixedTopologyDCEconomicResult.infeasible(f"branch {branch.canonical_branch_id} has near-zero reactance")
        b = 1.0 / (x * tap)
        model.addConstr(
            flow[branch.canonical_branch_id]
            == b * (theta[branch.from_bus_index] - theta[branch.to_bus_index] - shift),
            name=f"dc_flow[{branch.canonical_branch_id}]",
        )

    outgoing: dict[int, list[int]] = {idx: [] for idx in bus_indices}
    incoming: dict[int, list[int]] = {idx: [] for idx in bus_indices}
    for branch in active_branches:
        outgoing[branch.from_bus_index].append(branch.canonical_branch_id)
        incoming[branch.to_bus_index].append(branch.canonical_branch_id)

    for bus_idx in bus_indices:
        model.addConstr(
            gp.quicksum(pg[g] for g in gen_by_bus_idx[bus_idx])
            - load_pd_by_bus_idx[bus_idx]
            - gp.quicksum(flow[line_id] for line_id in outgoing[bus_idx])
            + gp.quicksum(flow[line_id] for line_id in incoming[bus_idx])
            == 0.0,
            name=f"balance[{bus_idx}]",
        )

    objective = gp.QuadExpr()
    for gen_idx, row in enumerate(gen_rows):
        objective += float(row[8]) * pg[gen_idx] * pg[gen_idx] + float(row[9]) * pg[gen_idx] + float(row[10])
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
        return FixedTopologyDCEconomicResult.infeasible(f"Gurobi status {model.Status}")
    if model.SolCount <= 0:
        return FixedTopologyDCEconomicResult(
            evaluation_status=EvaluationStatus.METHODOLOGY_FAILURE,
            rejection_merit=float("inf"),
            message=f"Gurobi produced no incumbent, status {model.Status}",
        )

    return FixedTopologyDCEconomicResult(
        evaluation_status=EvaluationStatus.OK,
        objective_cost=float(model.ObjVal),
        pg_by_generator={int(gen_idx): float(var.X) for gen_idx, var in pg.items()},
        theta_by_bus={int(identity.bus_ids[idx]): float(var.X) for idx, var in theta.items()},
        flow_by_line={int(line_id): float(var.X) for line_id, var in flow.items()},
        rejection_merit=None,
        message=f"Gurobi status {model.Status}",
    )


def _bus_components(bus_ids: Iterable[int], branches) -> list[set[int]]:
    bus_set = {int(bus_id) for bus_id in bus_ids}
    parent = {bus_id: bus_id for bus_id in bus_set}

    def find(bus_id: int) -> int:
        while parent[bus_id] != bus_id:
            parent[bus_id] = parent[parent[bus_id]]
            bus_id = parent[bus_id]
        return bus_id

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    for branch in branches:
        union(branch.from_bus_id, branch.to_bus_id)

    comps: dict[int, set[int]] = {}
    for bus_id in bus_set:
        comps.setdefault(find(bus_id), set()).add(bus_id)
    return list(comps.values())
