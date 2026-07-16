from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import networkx as nx
import numpy as np

from experiments.test.wildfire_tests.gridfm_support.branch_metadata import (
    MATPOWER_CASE30_BRANCHES,
    MatpowerBranch,
    canonicalize_line_ids,
    matpower_branch_by_pair,
    physical_line_ids,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.gurobi_master import (
    import_gurobipy,
    solve_gurobi_master_next_candidate,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.physics_infeasibility_evaluator import (
    source_less_island_buses,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    DEFAULT_PROXY_TYPE,
    deenergized_from_y,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_decision_quality import (
    FIXED_T0P30_CANDIDATE_LINE_IDS,
    _consequence_by_line,
    _edge_array,
    _scenario_baseline_exposure,
    _validate_candidate_line_ids,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_physics_decision_quality import (
    _canonicalize_decision_quality_scenario,
    _lambda_case,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import (
    DecisionQualityScenario,
    get_scenarios,
)
from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_scenario_baseline_physics_sensitivity import (
    BASELINE_LOADING_SOURCE,
    P_ENV_MODE_TARGET_MARGIN,
    build_scenario_baseline_loading_ranking,
    p_env_for_target_margin_scenario,
    remove_margin_excluded_targets,
    scenario_baseline_loading_vector,
)


STAGE_I_A = "stage_i_a_dc_k2"
STAGE_I_B = "stage_i_b_dc_miqp_k2"
STAGE_I_A_LABEL = "Stage I-a DC guided K2"
STAGE_I_B_LABEL = "Stage I-b DC MIQP K2"
TH_TOP1 = "th_top1"
TH_TOP2 = "th_top2"
AH_K2 = "ah_k2_budgeted"
TH_TOP1_LABEL = "TH top 1"
TH_TOP2_LABEL = "TH top 2"
AH_K2_LABEL = "AH K2 budgeted"

EPSILON = 1e-6
THETA_MAX = math.pi
MIP_GAP = 1e-4
MIQP_TIME_LIMIT_SECONDS = 600.0


@dataclass(frozen=True)
class DCBranch:
    line_id: int
    from_bus: int
    to_bus: int
    x: float
    rate_a_mva: float
    tau: float
    phi_rad: float
    susceptance_mw_per_rad: float


@dataclass(frozen=True)
class DCNetwork:
    branches: tuple[DCBranch, ...]
    branch_audit: dict
    generator_buses: tuple[int, ...]
    load_buses: tuple[int, ...]
    candidate_line_ids: tuple[int, ...]
    physical_line_ids: tuple[int, ...]
    total_pd: float


def _line_key(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in sorted({int(item) for item in values}))


def _parse_line_key(value) -> List[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    return sorted({int(part) for part in text.split(",") if part.strip()})


def _safe_json(values) -> str:
    return json.dumps(values, sort_keys=True)


def _scenario_edge_pair(scenario, line_id: int) -> tuple[int, int]:
    edge = _edge_array(scenario)
    src = int(edge[0, int(line_id)])
    dst = int(edge[1, int(line_id)])
    return tuple(sorted((src, dst)))


def build_dc_network(scenario, candidate_line_ids: Sequence[int] | None = None) -> DCNetwork:
    """Build the canonical physical-branch DC model data from the scenario."""

    base_mva = float(getattr(scenario, "sn_mva", 100.0))
    physical_ids = tuple(int(v) for v in physical_line_ids(scenario))
    candidates = tuple(
        int(v)
        for v in canonicalize_line_ids(
            scenario,
            FIXED_T0P30_CANDIDATE_LINE_IDS if candidate_line_ids is None else candidate_line_ids,
        )
    )
    _validate_candidate_line_ids(candidates, int(_edge_array(scenario).shape[1]))
    branches_by_pair: Dict[tuple[int, int], MatpowerBranch] = matpower_branch_by_pair()
    rows: List[DCBranch] = []
    unmapped: List[int] = []
    for line_id in physical_ids:
        pair = _scenario_edge_pair(scenario, line_id)
        branch = branches_by_pair.get(pair)
        if branch is None:
            unmapped.append(int(line_id))
            continue
        tau = float(branch.tap) if abs(float(branch.tap)) > 0.0 else 1.0
        phi = float(branch.shift_deg) * math.pi / 180.0
        if abs(float(branch.x)) <= 1e-12:
            raise ValueError(f"Cannot build DC branch for line_id={line_id}; x is zero.")
        b = base_mva / (float(branch.x) * tau)
        rows.append(
            DCBranch(
                line_id=int(line_id),
                from_bus=int(branch.from_bus),
                to_bus=int(branch.to_bus),
                x=float(branch.x),
                rate_a_mva=float(branch.rate_a_mva),
                tau=float(tau),
                phi_rad=float(phi),
                susceptance_mw_per_rad=float(b),
            )
        )

    pg_base = np.asarray(scenario.Pg_base, dtype=float)
    pg_min = np.asarray(scenario.Pg_min, dtype=float)
    pg_max = np.asarray(scenario.Pg_max, dtype=float)
    pv = np.asarray(getattr(scenario, "PV_mask", np.zeros_like(pg_base, dtype=bool)), dtype=bool)
    ref = np.asarray(getattr(scenario, "REF_mask", np.zeros_like(pg_base, dtype=bool)), dtype=bool)
    generator_mask = ((pg_max > pg_min) & (pg_max > 1e-9)) | (pg_base > 1e-9) | ref | pv
    generator_buses = tuple(int(v) for v in np.where(generator_mask)[0])
    pd_base = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    load_buses = tuple(int(v) for v in np.where(pd_base > 1e-9)[0])

    nonunity_taps = [branch for branch in MATPOWER_CASE30_BRANCHES if abs((branch.tap or 0.0) - 0.0) > 0.0 and abs(float(branch.tap) - 1.0) > 1e-12]
    nonzero_shifts = [branch for branch in MATPOWER_CASE30_BRANCHES if abs(float(branch.shift_deg)) > 1e-12]
    branch_audit = {
        "base_mva": base_mva,
        "num_physical_branches": int(len(physical_ids)),
        "num_dc_branches": int(len(rows)),
        "num_unmapped_branches": int(len(unmapped)),
        "unmapped_line_ids": _line_key(unmapped),
        "num_nonunity_taps": int(len(nonunity_taps)),
        "num_nonzero_phase_shifts": int(len(nonzero_shifts)),
        "dc_branch_model_used": "matpower_tap_shift" if nonunity_taps or nonzero_shifts else "trivial_tap_shift",
        "branch_model_warning": "" if not unmapped else f"Unmapped physical line ids: {_line_key(unmapped)}",
    }
    return DCNetwork(
        branches=tuple(rows),
        branch_audit=branch_audit,
        generator_buses=generator_buses,
        load_buses=load_buses,
        candidate_line_ids=candidates,
        physical_line_ids=physical_ids,
        total_pd=float(np.sum(pd_base[list(load_buses)])) if load_buses else 0.0,
    )


def canonical_scenarios_for_context(scenario_ids: Sequence[str] | None, grid_scenario) -> List[DecisionQualityScenario]:
    return [
        remove_margin_excluded_targets(_canonicalize_decision_quality_scenario(scenario_def, grid_scenario))
        for scenario_def in get_scenarios(scenario_ids)
    ]


def _objective_components_from_solution(
    scenario,
    network: DCNetwork,
    p_env_by_line: Dict[int, float],
    baseline_r: float,
    flow_by_line: Dict[int, float],
    service_by_bus: Dict[int, float],
) -> dict:
    pd = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    total_pd = max(float(np.sum(pd[list(network.load_buses)])), EPSILON)
    l_shed = float(sum(float(pd[i]) * (1.0 - float(service_by_bus.get(i, 1.0))) for i in network.load_buses) / total_pd)
    r_raw = 0.0
    exposure_by_line: Dict[int, float] = {}
    for branch in network.branches:
        flow = float(flow_by_line.get(int(branch.line_id), 0.0))
        value = float(p_env_by_line.get(int(branch.line_id), 0.0)) * (flow / float(branch.rate_a_mva)) ** 2
        exposure_by_line[int(branch.line_id)] = value
        r_raw += value
    r_norm = float(r_raw / max(float(baseline_r), EPSILON))
    return {
        "R_raw": float(r_raw),
        "R_norm": float(r_norm),
        "L_shed": float(l_shed),
        "L_shed_cmd": float(l_shed),
        "L_shed_gridfm_raw": np.nan,
        "L_shed_gridfm_effective": np.nan,
        "L_shed_hybrid": float(l_shed),
        "exposure_by_line": exposure_by_line,
    }


def _dc_residuals(
    scenario,
    network: DCNetwork,
    active_line_ids: Iterable[int],
    theta_by_bus: Dict[int, float],
    flow_by_line: Dict[int, float],
    pg_by_bus: Dict[int, float],
    service_by_bus: Dict[int, float],
) -> dict:
    active = {int(v) for v in active_line_ids}
    max_angle = 0.0
    max_limit = 0.0
    max_balance = 0.0
    pd = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    for branch in network.branches:
        flow = float(flow_by_line.get(branch.line_id, 0.0))
        if branch.line_id in active:
            expected = branch.susceptance_mw_per_rad * (
                float(theta_by_bus.get(branch.from_bus, 0.0))
                - float(theta_by_bus.get(branch.to_bus, 0.0))
                - float(branch.phi_rad)
            )
            max_angle = max(max_angle, abs(flow - expected))
            max_limit = max(max_limit, max(0.0, abs(flow) - float(branch.rate_a_mva)))
        else:
            max_limit = max(max_limit, abs(flow))
    outgoing: Dict[int, float] = {int(i): 0.0 for i in range(int(scenario.num_buses))}
    incoming: Dict[int, float] = {int(i): 0.0 for i in range(int(scenario.num_buses))}
    for branch in network.branches:
        flow = float(flow_by_line.get(branch.line_id, 0.0))
        outgoing[branch.from_bus] += flow
        incoming[branch.to_bus] += flow
    for bus in range(int(scenario.num_buses)):
        service = float(service_by_bus.get(bus, 1.0 if pd[bus] > 1e-9 else 0.0))
        residual = float(pg_by_bus.get(bus, 0.0)) - service * float(pd[bus]) - outgoing[bus] + incoming[bus]
        max_balance = max(max_balance, abs(residual))
    return {
        "max_abs_nodal_balance_residual": float(max_balance),
        "max_abs_angle_flow_residual": float(max_angle),
        "max_branch_flow_limit_violation": float(max_limit),
    }


def _source_less_served_fraction(scenario, service_by_bus: Dict[int, float], source_less_buses: Iterable[int]) -> float:
    pd = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    total = max(float(np.sum(pd)), EPSILON)
    return float(sum(float(pd[int(bus)]) * float(service_by_bus.get(int(bus), 0.0)) for bus in source_less_buses) / total)


def common_operational_diagnostic(
    scenario,
    network: DCNetwork,
    shutoff_line_ids: Iterable[int],
    flow_by_line: Dict[int, float],
    pg_by_bus: Dict[int, float],
    service_by_bus: Dict[int, float],
    source_less_buses: Iterable[int],
) -> dict:
    off = {int(v) for v in shutoff_line_ids}
    active_branches = [branch for branch in network.branches if int(branch.line_id) not in off]
    off_branches = [branch for branch in network.branches if int(branch.line_id) in off]
    thermal = float(
        np.mean(
            [
                max(0.0, abs(float(flow_by_line.get(branch.line_id, 0.0))) / float(branch.rate_a_mva) - 1.0) ** 2
                for branch in active_branches
            ]
        )
    ) if active_branches else 0.0
    pg_min = np.asarray(scenario.Pg_min, dtype=float)
    pg_max = np.asarray(scenario.Pg_max, dtype=float)
    gen_terms = []
    for bus in network.generator_buses:
        scale = max(float(pg_max[bus] - pg_min[bus]), EPSILON, 1.0)
        pg = float(pg_by_bus.get(int(bus), 0.0))
        gen_terms.append(max(0.0, (pg - float(pg_max[bus])) / scale) ** 2 + max(0.0, (float(pg_min[bus]) - pg) / scale) ** 2)
    gen = float(np.mean(gen_terms)) if gen_terms else 0.0
    load = float(
        np.mean([max(0.0, float(service_by_bus.get(bus, 1.0)) - 1.0) ** 2 + max(0.0, -float(service_by_bus.get(bus, 1.0))) ** 2 for bus in network.load_buses])
    ) if network.load_buses else 0.0
    source_less = _source_less_served_fraction(scenario, service_by_bus, source_less_buses)
    offline = float(
        np.mean([(abs(float(flow_by_line.get(branch.line_id, 0.0))) / float(branch.rate_a_mva)) ** 2 for branch in off_branches])
    ) if off_branches else 0.0
    total = float(thermal + gen + load + source_less + offline)
    return {
        "PAC_common_op_overlap": total,
        "PAC_common_thermal_overlap": thermal,
        "PAC_common_gen_P_overlap": gen,
        "PAC_common_load_service_bounds": load,
        "PAC_common_source_less_service": source_less,
        "PAC_common_topology_offline_flow": offline,
    }


def solve_stage_ia_dc_recourse(
    scenario,
    network: DCNetwork,
    p_env_by_line: Dict[int, float],
    baseline_r: float,
    shutoff_line_ids: Sequence[int],
    lambda_r: float,
    result_id: int = 0,
) -> dict:
    gp, GRB = import_gurobipy()
    started = time.perf_counter()
    removed = set(canonicalize_line_ids(scenario, shutoff_line_ids))
    active = [branch.line_id for branch in network.branches if int(branch.line_id) not in removed]
    source_less = source_less_island_buses(scenario, removed)

    model = gp.Model("stage_i_a_dc_recourse")
    model.Params.OutputFlag = 0
    theta = {bus: model.addVar(lb=-THETA_MAX, ub=THETA_MAX, name=f"theta_{bus}") for bus in range(int(scenario.num_buses))}
    pg_min = np.asarray(scenario.Pg_min, dtype=float)
    pg_max = np.asarray(scenario.Pg_max, dtype=float)
    pg = {
        bus: model.addVar(lb=float(pg_min[bus]), ub=float(pg_max[bus]), name=f"Pg_{bus}")
        for bus in network.generator_buses
    }
    s = {bus: model.addVar(lb=0.0, ub=1.0, name=f"s_{bus}") for bus in network.load_buses}
    for bus in source_less:
        if int(bus) in s:
            s[int(bus)].UB = 0.0
    f = {}
    for branch in network.branches:
        if int(branch.line_id) in removed:
            f[branch.line_id] = model.addVar(lb=0.0, ub=0.0, name=f"f_{branch.line_id}")
        else:
            f[branch.line_id] = model.addVar(lb=-float(branch.rate_a_mva), ub=float(branch.rate_a_mva), name=f"f_{branch.line_id}")
            model.addConstr(
                f[branch.line_id]
                == float(branch.susceptance_mw_per_rad)
                * (theta[branch.from_bus] - theta[branch.to_bus] - float(branch.phi_rad)),
                name=f"dc_flow_{branch.line_id}",
            )
    ref_bus = int(scenario.get_ref_bus()) if hasattr(scenario, "get_ref_bus") and scenario.get_ref_bus() is not None else 0
    model.addConstr(theta[ref_bus] == 0.0, name="reference_angle")
    pd = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    for bus in range(int(scenario.num_buses)):
        out_expr = gp.quicksum(f[branch.line_id] for branch in network.branches if branch.from_bus == bus)
        in_expr = gp.quicksum(f[branch.line_id] for branch in network.branches if branch.to_bus == bus)
        gen_expr = pg[bus] if bus in pg else 0.0
        load_expr = float(pd[bus]) * s[bus] if bus in s else 0.0
        model.addConstr(gen_expr - load_expr - out_expr + in_expr == 0.0, name=f"balance_{bus}")

    total_pd = max(float(network.total_pd), EPSILON)
    risk_expr = gp.QuadExpr()
    for branch in network.branches:
        coeff = float(p_env_by_line.get(branch.line_id, 0.0)) / (float(branch.rate_a_mva) ** 2) / max(float(baseline_r), EPSILON)
        risk_expr.add(coeff * f[branch.line_id] * f[branch.line_id])
    load_expr = gp.quicksum(float(pd[bus]) * (1.0 - s[bus]) for bus in network.load_buses) / total_pd
    model.setObjective(float(lambda_r) * risk_expr + (1.0 - float(lambda_r)) * load_expr, GRB.MINIMIZE)
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        return {
            "result_id": int(result_id),
            "status": "infeasible_or_no_solution",
            "solver_status": int(model.Status),
            "J_true": np.inf,
            "J_no_phys": np.inf,
            "R_raw": np.nan,
            "R_norm": np.nan,
            "L_shed": np.nan,
            "runtime_seconds": float(time.perf_counter() - started),
        }

    flow_by_line = {int(line_id): float(var.X) for line_id, var in f.items()}
    theta_by_bus = {int(bus): float(var.X) for bus, var in theta.items()}
    pg_by_bus = {int(bus): float(var.X) for bus, var in pg.items()}
    service_by_bus = {int(bus): float(var.X) for bus, var in s.items()}
    components = _objective_components_from_solution(scenario, network, p_env_by_line, baseline_r, flow_by_line, service_by_bus)
    lambda_l = 1.0 - float(lambda_r)
    j_no_phys = float(lambda_r) * float(components["R_norm"]) + lambda_l * float(components["L_shed"])
    residuals = _dc_residuals(scenario, network, active, theta_by_bus, flow_by_line, pg_by_bus, service_by_bus)
    common = common_operational_diagnostic(scenario, network, removed, flow_by_line, pg_by_bus, service_by_bus, source_less)
    return {
        "result_id": int(result_id),
        "status": "ok",
        "solver_status": int(model.Status),
        "objective_value": float(model.ObjVal),
        "J_true": float(j_no_phys),
        "J_no_phys": float(j_no_phys),
        "R_raw": float(components["R_raw"]),
        "R_norm": float(components["R_norm"]),
        "L_shed": float(components["L_shed"]),
        "L_shed_cmd": float(components["L_shed_cmd"]),
        "L_shed_gridfm_raw": np.nan,
        "L_shed_gridfm_effective": np.nan,
        "L_shed_hybrid": float(components["L_shed_hybrid"]),
        "PAC_total": 0.0,
        "PAC_operational": 0.0,
        "PAC_AC": 0.0,
        "PAC_model_consistency": 0.0,
        "risk_contribution": float(lambda_r) * float(components["R_norm"]),
        "load_contribution": lambda_l * float(components["L_shed"]),
        "physics_contribution": 0.0,
        "max_loading_ratio": float(max([abs(flow_by_line[b.line_id]) / b.rate_a_mva for b in network.branches if b.line_id in active], default=0.0)),
        "runtime_seconds": float(time.perf_counter() - started),
        "gridfm_calls": 0,
        "call_budget": 0,
        "budget_exhausted": False,
        "termination_reason": "dc_qp_solved",
        "source_less_bus_ids": _line_key(source_less),
        "dc_flow_json": _safe_json({str(k): v for k, v in sorted(flow_by_line.items())}),
        "dc_theta_json": _safe_json({str(k): v for k, v in sorted(theta_by_bus.items())}),
        "dc_pg_json": _safe_json({str(k): v for k, v in sorted(pg_by_bus.items())}),
        "dc_service_json": _safe_json({str(k): v for k, v in sorted(service_by_bus.items())}),
        "dc_exposure_by_line_json": _safe_json({str(k): v for k, v in sorted(components["exposure_by_line"].items())}),
        **residuals,
        **common,
    }


def solve_stage_ib_dc_miqp(
    scenario,
    network: DCNetwork,
    p_env_by_line: Dict[int, float],
    baseline_r: float,
    lambda_r: float,
    result_id: int = 0,
    time_limit_seconds: float = MIQP_TIME_LIMIT_SECONDS,
    mip_gap: float = MIP_GAP,
) -> dict:
    gp, GRB = import_gurobipy()
    started = time.perf_counter()
    model = gp.Model("stage_i_b_dc_miqp")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = float(mip_gap)
    model.Params.TimeLimit = float(time_limit_seconds)

    candidate_set = set(int(v) for v in network.candidate_line_ids)
    theta = {bus: model.addVar(lb=-THETA_MAX, ub=THETA_MAX, name=f"theta_{bus}") for bus in range(int(scenario.num_buses))}
    pg_min = np.asarray(scenario.Pg_min, dtype=float)
    pg_max = np.asarray(scenario.Pg_max, dtype=float)
    pg = {
        bus: model.addVar(lb=float(pg_min[bus]), ub=float(pg_max[bus]), name=f"Pg_{bus}")
        for bus in network.generator_buses
    }
    s = {bus: model.addVar(lb=0.0, ub=1.0, name=f"s_{bus}") for bus in network.load_buses}
    z = {}
    y = {}
    f = {}
    for branch in network.branches:
        line_id = int(branch.line_id)
        if line_id in candidate_set:
            z[line_id] = model.addVar(vtype=GRB.BINARY, name=f"z_{line_id}")
            y[line_id] = model.addVar(vtype=GRB.BINARY, name=f"y_{line_id}")
            model.addConstr(z[line_id] + y[line_id] == 1, name=f"zy_{line_id}")
        else:
            z[line_id] = model.addVar(lb=1.0, ub=1.0, name=f"z_{line_id}")
        f[line_id] = model.addVar(lb=-float(branch.rate_a_mva), ub=float(branch.rate_a_mva), name=f"f_{line_id}")
        big_m = float(branch.rate_a_mva) + abs(float(branch.susceptance_mw_per_rad)) * (2.0 * THETA_MAX + abs(float(branch.phi_rad)))
        flow_expr = f[line_id] - float(branch.susceptance_mw_per_rad) * (
            theta[branch.from_bus] - theta[branch.to_bus] - float(branch.phi_rad)
        )
        model.addConstr(flow_expr <= big_m * (1 - z[line_id]), name=f"dc_flow_hi_{line_id}")
        model.addConstr(flow_expr >= -big_m * (1 - z[line_id]), name=f"dc_flow_lo_{line_id}")
        model.addConstr(f[line_id] <= float(branch.rate_a_mva) * z[line_id], name=f"flow_limit_hi_{line_id}")
        model.addConstr(f[line_id] >= -float(branch.rate_a_mva) * z[line_id], name=f"flow_limit_lo_{line_id}")
    model.addConstr(gp.quicksum(y[line_id] for line_id in sorted(y)) <= 2, name="shutoff_budget_k2")
    ref_bus = int(scenario.get_ref_bus()) if hasattr(scenario, "get_ref_bus") and scenario.get_ref_bus() is not None else 0
    model.addConstr(theta[ref_bus] == 0.0, name="reference_angle")
    pd = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    for bus in range(int(scenario.num_buses)):
        out_expr = gp.quicksum(f[branch.line_id] for branch in network.branches if branch.from_bus == bus)
        in_expr = gp.quicksum(f[branch.line_id] for branch in network.branches if branch.to_bus == bus)
        gen_expr = pg[bus] if bus in pg else 0.0
        load_expr = float(pd[bus]) * s[bus] if bus in s else 0.0
        model.addConstr(gen_expr - load_expr - out_expr + in_expr == 0.0, name=f"balance_{bus}")

    total_pd = max(float(network.total_pd), EPSILON)
    risk_expr = gp.QuadExpr()
    for branch in network.branches:
        coeff = float(p_env_by_line.get(branch.line_id, 0.0)) / (float(branch.rate_a_mva) ** 2) / max(float(baseline_r), EPSILON)
        risk_expr.add(coeff * f[branch.line_id] * f[branch.line_id])
    load_expr = gp.quicksum(float(pd[bus]) * (1.0 - s[bus]) for bus in network.load_buses) / total_pd
    model.setObjective(float(lambda_r) * risk_expr + (1.0 - float(lambda_r)) * load_expr, GRB.MINIMIZE)
    model.optimize()

    has_solution = int(getattr(model, "SolCount", 0)) > 0
    if not has_solution:
        return {
            "result_id": int(result_id),
            "status": "no_incumbent",
            "solver_status": int(model.Status),
            "J_true": np.inf,
            "J_no_phys": np.inf,
            "R_raw": np.nan,
            "R_norm": np.nan,
            "L_shed": np.nan,
            "runtime_seconds": float(time.perf_counter() - started),
            "best_bound": float(model.ObjBound) if hasattr(model, "ObjBound") else np.nan,
            "mip_gap": np.nan,
            "node_count": float(getattr(model, "NodeCount", np.nan)),
            "solution_count": int(getattr(model, "SolCount", 0)),
            "time_limit_reached": bool(model.Status == GRB.TIME_LIMIT),
            "optimality_certified": False,
        }

    shutoff = sorted(int(line_id) for line_id, var in y.items() if int(round(float(var.X))) == 1)
    active = [branch.line_id for branch in network.branches if int(branch.line_id) not in set(shutoff)]
    flow_by_line = {int(line_id): float(var.X) for line_id, var in f.items()}
    theta_by_bus = {int(bus): float(var.X) for bus, var in theta.items()}
    pg_by_bus = {int(bus): float(var.X) for bus, var in pg.items()}
    service_by_bus = {int(bus): float(var.X) for bus, var in s.items()}
    components = _objective_components_from_solution(scenario, network, p_env_by_line, baseline_r, flow_by_line, service_by_bus)
    lambda_l = 1.0 - float(lambda_r)
    j_no_phys = float(lambda_r) * float(components["R_norm"]) + lambda_l * float(components["L_shed"])
    source_less = source_less_island_buses(scenario, shutoff)
    residuals = _dc_residuals(scenario, network, active, theta_by_bus, flow_by_line, pg_by_bus, service_by_bus)
    common = common_operational_diagnostic(scenario, network, shutoff, flow_by_line, pg_by_bus, service_by_bus, source_less)
    model_gap = float(model.MIPGap) if np.isfinite(getattr(model, "MIPGap", np.nan)) else np.nan
    certified = bool(model.Status in {GRB.OPTIMAL, GRB.SUBOPTIMAL} and np.isfinite(model_gap) and model_gap <= float(mip_gap) + 1e-12)
    return {
        "result_id": int(result_id),
        "status": "ok",
        "solver_status": int(model.Status),
        "objective_value": float(model.ObjVal),
        "best_bound": float(model.ObjBound),
        "mip_gap": model_gap,
        "runtime_seconds": float(time.perf_counter() - started),
        "node_count": float(model.NodeCount),
        "solution_count": int(model.SolCount),
        "time_limit_reached": bool(model.Status == GRB.TIME_LIMIT),
        "optimality_certified": certified,
        "J_true": float(j_no_phys),
        "J_no_phys": float(j_no_phys),
        "R_raw": float(components["R_raw"]),
        "R_norm": float(components["R_norm"]),
        "L_shed": float(components["L_shed"]),
        "L_shed_cmd": float(components["L_shed_cmd"]),
        "L_shed_gridfm_raw": np.nan,
        "L_shed_gridfm_effective": np.nan,
        "L_shed_hybrid": float(components["L_shed_hybrid"]),
        "PAC_total": 0.0,
        "PAC_operational": 0.0,
        "PAC_AC": 0.0,
        "PAC_model_consistency": 0.0,
        "risk_contribution": float(lambda_r) * float(components["R_norm"]),
        "load_contribution": lambda_l * float(components["L_shed"]),
        "physics_contribution": 0.0,
        "max_loading_ratio": float(max([abs(flow_by_line[b.line_id]) / b.rate_a_mva for b in network.branches if b.line_id in active], default=0.0)),
        "gridfm_calls": 0,
        "call_budget": 0,
        "budget_exhausted": False,
        "termination_reason": "dc_miqp_solved",
        "source_less_bus_ids": _line_key(source_less),
        "shutoff_line_ids": _line_key(shutoff),
        "line_id_key": _line_key(shutoff),
        "num_shutoff_lines": int(len(shutoff)),
        "dc_flow_json": _safe_json({str(k): v for k, v in sorted(flow_by_line.items())}),
        "dc_theta_json": _safe_json({str(k): v for k, v in sorted(theta_by_bus.items())}),
        "dc_pg_json": _safe_json({str(k): v for k, v in sorted(pg_by_bus.items())}),
        "dc_service_json": _safe_json({str(k): v for k, v in sorted(service_by_bus.items())}),
        "dc_exposure_by_line_json": _safe_json({str(k): v for k, v in sorted(components["exposure_by_line"].items())}),
        **residuals,
        **common,
    }


def build_stage_ia_topology_pool(
    context: dict,
    scenario_ids: Sequence[str] | None,
    lambda_values: Sequence[float],
    budget: int,
    proxy_lambda_values: Sequence[float] | None = None,
) -> tuple[object, object, object, List[DecisionQualityScenario], dict]:
    import pandas as pd

    scenario = context["scenario"]
    candidate_line_ids = canonicalize_line_ids(scenario, FIXED_T0P30_CANDIDATE_LINE_IDS)
    physical_ids = physical_line_ids(scenario)
    _validate_candidate_line_ids(candidate_line_ids, int(_edge_array(scenario).shape[1]))
    baseline_loading = scenario_baseline_loading_vector(context)
    c_by_line = _consequence_by_line(context["consequence_df"])
    canonical_scenarios = canonical_scenarios_for_context(scenario_ids, scenario)
    rows = []
    p_env_frames = []
    for scenario_def in canonical_scenarios:
        p_env, p_env_frame = p_env_for_target_margin_scenario(
            scenario_def,
            int(_edge_array(scenario).shape[1]),
            physical_ids,
            baseline_loading,
        )
        p_env_frames.append(p_env_frame)
        baseline_r, _ = _scenario_baseline_exposure(baseline_loading, p_env, physical_ids)
        lambda_pairs = [(float(v), float(v)) for v in lambda_values]
        if proxy_lambda_values is not None:
            lambda_pairs = [(float(inner), float(proxy)) for proxy in proxy_lambda_values for inner in lambda_values]
        for lambda_r, lambda_r_proxy in lambda_pairs:
            lambda_l = 1.0 - float(lambda_r)
            lambda_l_proxy = 1.0 - float(lambda_r_proxy)
            evaluated_y = []
            for iteration in range(int(budget)):
                proposal = solve_gurobi_master_next_candidate(
                    candidate_line_ids,
                    p_env,
                    baseline_loading,
                    c_by_line,
                    float(lambda_r_proxy),
                    lambda_l_proxy,
                    max_deenergized_lines=2,
                    evaluated_y_vectors=evaluated_y,
                    proxy_type=DEFAULT_PROXY_TYPE,
                )
                y_by_line = {int(key): int(value) for key, value in proposal["y_by_line"].items()}
                evaluated_y.append(y_by_line)
                rows.append(
                    {
                        "scenario_id": scenario_def.scenario_id,
                        "scenario_name": scenario_def.scenario_name,
                        "stage": STAGE_I_A,
                        "stage_label": STAGE_I_A_LABEL,
                        "method_family": "Stage I-a",
                        "lambda_R": float(lambda_r),
                        "lambda_L": lambda_l,
                        "lambda_case": _lambda_case(lambda_r),
                        "lambda_R_proxy": float(lambda_r_proxy),
                        "lambda_L_proxy": lambda_l_proxy,
                        "lambda_proxy_case": _lambda_case(lambda_r_proxy),
                        "topology_iteration": int(iteration + 1),
                        "proposal_method": "stage_i_a_independent_gurobi_proxy_limited",
                        "shutoff_line_ids": _line_key(deenergized_from_y(y_by_line)),
                        "baseline_R": float(baseline_r),
                        "p_env_json": json.dumps({str(int(k)): float(v) for k, v in p_env.items()}),
                        "proxy_objective": proposal.get("proxy_objective", np.nan),
                        "stage_i_a_proxy_R_hat": proposal.get("proxy_R_hat", np.nan),
                        "stage_i_a_proxy_L_hat": proposal.get("proxy_L_hat", np.nan),
                        "stage_i_a_gurobi_status": proposal.get("gurobi_status", np.nan),
                        "topology_proxy_excludes_pac": True,
                    }
                )
    p_env_table = pd.concat(p_env_frames, ignore_index=True, sort=False)
    ranking = build_scenario_baseline_loading_ranking(context, baseline_loading, p_env_frames)
    metadata = {
        "candidate_line_ids": candidate_line_ids,
        "physical_line_ids": physical_ids,
        "baseline_loading": baseline_loading.tolist(),
        "topology_budget": int(budget),
        "proxy_lambda_values": [float(v) for v in (lambda_values if proxy_lambda_values is None else proxy_lambda_values)],
    }
    return pd.DataFrame(rows).reset_index(drop=True), p_env_table, ranking, canonical_scenarios, metadata


def build_budgeted_heuristic_pool(
    context: dict,
    scenario_ids: Sequence[str] | None,
    lambda_values: Sequence[float],
    proxy_lambda_values: Sequence[float] | None = None,
) -> tuple[object, object]:
    import pandas as pd

    from experiments.test.wildfire_tests.stage_h_heuristic_comparison.run_stage_h_heuristic_baseline_comparison import (
        _candidate_scores,
        area_heuristic_groups,
        select_top_k_lines,
    )

    scenario = context["scenario"]
    candidate_line_ids = canonicalize_line_ids(scenario, FIXED_T0P30_CANDIDATE_LINE_IDS)
    physical_ids = physical_line_ids(scenario)
    baseline_loading = scenario_baseline_loading_vector(context)
    canonical_scenarios = canonical_scenarios_for_context(scenario_ids, scenario)
    rows = []
    audit_rows = []
    lambda_pairs = [(float(v), float(v)) for v in lambda_values]
    if proxy_lambda_values is not None:
        lambda_pairs = [(float(inner), float(proxy)) for proxy in proxy_lambda_values for inner in lambda_values]
    for scenario_def in canonical_scenarios:
        p_env, _ = p_env_for_target_margin_scenario(
            scenario_def,
            int(_edge_array(scenario).shape[1]),
            physical_ids,
            baseline_loading,
        )
        baseline_r, _ = _scenario_baseline_exposure(baseline_loading, p_env, physical_ids)
        scores = _candidate_scores(candidate_line_ids, p_env, baseline_loading)
        for k, stage, label in [(1, TH_TOP1, TH_TOP1_LABEL), (2, TH_TOP2, TH_TOP2_LABEL)]:
            selected = select_top_k_lines(scores, k)
            for lambda_r, lambda_r_proxy in lambda_pairs:
                rows.append(_heuristic_row(scenario_def, p_env, baseline_r, selected, lambda_r, stage, label, "TH", f"top_{k}", lambda_r_proxy))
        groups = area_heuristic_groups(_edge_array(scenario), scores, 0.30)
        chosen = groups[groups["ah_selected_group"].astype(bool)].iloc[0]
        group_lines = _parse_line_key(chosen["ah_group_line_ids"])
        selected_ah = sorted(group_lines, key=lambda line_id: (-float(scores[int(line_id)]), int(line_id)))[:2]
        audit_rows.append(
            {
                "scenario_id": scenario_def.scenario_id,
                "ah_group_line_ids": chosen["ah_group_line_ids"],
                "ah_k2_selected_line_ids": _line_key(selected_ah),
                "ah_group_average_score": float(chosen["ah_group_average_score"]),
                "ah_group_total_score": float(chosen["ah_group_total_score"]),
            }
        )
        for lambda_r, lambda_r_proxy in lambda_pairs:
            rows.append(_heuristic_row(scenario_def, p_env, baseline_r, selected_ah, lambda_r, AH_K2, AH_K2_LABEL, "AH", "connected_top30_then_top2", lambda_r_proxy))
    return pd.DataFrame(rows).reset_index(drop=True), pd.DataFrame(audit_rows)


def _heuristic_row(
    scenario_def: DecisionQualityScenario,
    p_env: Dict[int, float],
    baseline_r: float,
    selected: Sequence[int],
    lambda_r: float,
    stage: str,
    label: str,
    method: str,
    variant: str,
    lambda_r_proxy: float | None = None,
) -> dict:
    lambda_r_proxy = float(lambda_r if lambda_r_proxy is None else lambda_r_proxy)
    return {
        "scenario_id": scenario_def.scenario_id,
        "scenario_name": scenario_def.scenario_name,
        "stage": stage,
        "stage_label": label,
        "method_family": method,
        "lambda_R": float(lambda_r),
        "lambda_L": 1.0 - float(lambda_r),
        "lambda_case": _lambda_case(lambda_r),
        "lambda_R_proxy": float(lambda_r_proxy),
        "lambda_L_proxy": 1.0 - float(lambda_r_proxy),
        "lambda_proxy_case": _lambda_case(lambda_r_proxy),
        "topology_iteration": 1,
        "proposal_method": f"stage_h_{method.lower()}_{variant}",
        "heuristic_method": method,
        "method_variant": variant,
        "shutoff_line_ids": _line_key(selected),
        "baseline_R": float(baseline_r),
        "p_env_json": json.dumps({str(int(k)): float(v) for k, v in p_env.items()}),
        "topology_proxy_excludes_pac": True,
    }


def evaluate_stage_ia_pool(scenario, network: DCNetwork, topology_pool, rho_values: Sequence[float]) -> object:
    import pandas as pd

    rows = []
    for idx, source in enumerate(topology_pool.itertuples(index=False)):
        p_env = {int(k): float(v) for k, v in json.loads(source.p_env_json).items()}
        recourse = solve_stage_ia_dc_recourse(
            scenario,
            network,
            p_env,
            float(source.baseline_R),
            _parse_line_key(source.shutoff_line_ids),
            float(source.lambda_R),
            result_id=int(idx),
        )
        for rho in rho_values:
            rows.append(
                {
                    **source._asdict(),
                    **recourse,
                    "rho_phys": float(rho),
                    "model_type": "dc",
                    "candidate_set": "t0p30",
                    "p_env_mode": P_ENV_MODE_TARGET_MARGIN,
                    "baseline_loading_source": BASELINE_LOADING_SOURCE,
                    "R_base_s_source": BASELINE_LOADING_SOURCE,
                    "post_topology_evaluation_source": "fixed_topology_dc_recourse",
                    "continuous_recourse_optimized": True,
                    "line_id_key": source.shutoff_line_ids,
                    "num_shutoff_lines": len(_parse_line_key(source.shutoff_line_ids)),
                }
            )
    return pd.DataFrame(rows)


def evaluate_stage_ib_grid(scenario, network: DCNetwork, topology_pool, rho_values: Sequence[float]) -> object:
    import pandas as pd

    rows = []
    unique_cols = ["scenario_id", "scenario_name", "lambda_R", "lambda_L", "lambda_case", "baseline_R", "p_env_json"]
    for optional in ["lambda_R_proxy", "lambda_L_proxy", "lambda_proxy_case"]:
        if optional in topology_pool.columns:
            unique_cols.append(optional)
    unique = topology_pool[unique_cols].drop_duplicates()
    for idx, source in enumerate(unique.itertuples(index=False)):
        p_env = {int(k): float(v) for k, v in json.loads(source.p_env_json).items()}
        solved = solve_stage_ib_dc_miqp(
            scenario,
            network,
            p_env,
            float(source.baseline_R),
            float(source.lambda_R),
            result_id=int(idx),
        )
        for rho in rho_values:
            rows.append(
                {
                    "scenario_id": source.scenario_id,
                    "scenario_name": source.scenario_name,
                    "stage": STAGE_I_B,
                    "stage_label": STAGE_I_B_LABEL,
                    "method_family": "Stage I-b",
                    "lambda_R": float(source.lambda_R),
                    "lambda_L": float(source.lambda_L),
                    "lambda_case": source.lambda_case,
                    "lambda_R_proxy": float(getattr(source, "lambda_R_proxy", source.lambda_R)),
                    "lambda_L_proxy": float(getattr(source, "lambda_L_proxy", 1.0 - float(source.lambda_R))),
                    "lambda_proxy_case": getattr(source, "lambda_proxy_case", source.lambda_case),
                    "topology_iteration": 1,
                    "proposal_method": "stage_i_b_direct_dc_miqp",
                    "baseline_R": float(source.baseline_R),
                    "p_env_json": source.p_env_json,
                    "topology_proxy_excludes_pac": False,
                    **solved,
                    "rho_phys": float(rho),
                    "model_type": "dc",
                    "candidate_set": "t0p30",
                    "p_env_mode": P_ENV_MODE_TARGET_MARGIN,
                    "baseline_loading_source": BASELINE_LOADING_SOURCE,
                    "R_base_s_source": BASELINE_LOADING_SOURCE,
                    "post_topology_evaluation_source": "joint_dc_miqp",
                    "continuous_recourse_optimized": True,
                }
            )
    return pd.DataFrame(rows)
