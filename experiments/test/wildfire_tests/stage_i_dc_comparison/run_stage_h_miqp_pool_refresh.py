from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.reporting import write_dataframe, write_json
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.gurobi_master import import_gurobipy
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.physics_infeasibility_evaluator import (
    source_less_island_buses,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_physics_decision_quality import (
    _lambda_case,
)
from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_revised_continuous_implementation import (
    _make_continuous_context,
)
from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_scenario_baseline_physics_sensitivity import (
    BASELINE_LOADING_SOURCE,
    P_ENV_MODE_TARGET_MARGIN,
)
from experiments.test.wildfire_tests.stage_i_dc_comparison.dc_formulation import (
    EPSILON,
    MIP_GAP,
    STAGE_I_B,
    STAGE_I_B_LABEL,
    THETA_MAX,
    _dc_residuals,
    _line_key,
    _objective_components_from_solution,
    build_dc_network,
    common_operational_diagnostic,
)
from experiments.test.wildfire_tests.stage_i_dc_comparison.run_stage_h_dc_comparison import RESULT_ROOT


RUNS = ("main_results/r11", "MLD/r5")
EXCLUDE_GRIDFM_STAGE = "stage_e_k2"
POOL_TABLE = "stage_i_b_solution_pool.csv"
POOL_RAW_TABLE = "stage_i_b_solution_pool_raw.csv"
AUGMENTED_POINTS_TABLE = "all_evaluated_stage_h_points_with_miqp_pool.csv"
AUGMENTED_CONVERGENCE_TABLE = "traditional_lambda_objective_convergence_with_miqp_pool.csv"

STAGE_ORDER = [
    "stage_e_k2",
    "stage_i_a_dc_k2",
    "stage_i_b_dc_miqp_k2",
    "th_top1",
    "th_top2",
    "ah_k2_budgeted",
]
STAGE_LABELS = {
    "stage_e_k2": "Stage E K2 GridFM",
    "stage_i_a_dc_k2": "Stage I-a DC guided K2",
    "stage_i_b_dc_miqp_k2": "Stage I-b DC MIQP K2 + pool",
    "th_top1": "TH top-1",
    "th_top2": "TH top-2",
    "ah_k2_budgeted": "AH K2",
}
COLORS = {
    "stage_e_k2": "#4C78A8",
    "stage_i_a_dc_k2": "#F58518",
    "stage_i_b_dc_miqp_k2": "#54A24B",
    "th_top1": "#B279A2",
    "th_top2": "#E45756",
    "ah_k2_budgeted": "#72B7B2",
}
MARKERS = {
    "stage_e_k2": "o",
    "stage_i_a_dc_k2": "s",
    "stage_i_b_dc_miqp_k2": "D",
    "th_top1": "^",
    "th_top2": "v",
    "ah_k2_budgeted": "P",
}


def _long_path(path: Path) -> str:
    text = str(Path(path).resolve())
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        return "\\\\?\\" + text
    return text


def _mkdir(path: Path) -> None:
    os.makedirs(_long_path(path), exist_ok=True)


def _savefig(fig, path: Path, **kwargs) -> None:
    _mkdir(path.parent)
    staging = Path("tmp") / "stage_h_miqp_pool_plot_staging"
    staging.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=Path(path).suffix or ".png", dir=str(staging))
    tmp_path = Path(handle.name)
    handle.close()
    try:
        fig.savefig(tmp_path, **kwargs)
        os.replace(str(tmp_path), _long_path(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _parse_lines(value) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    return sorted({int(float(part.strip())) for part in text.split(",") if part.strip()})


def _nondominated_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series([], dtype=bool, index=frame.index)
    points = frame[["L_shed", "R_norm"]].astype(float).to_numpy()
    keep = np.ones(len(points), dtype=bool)
    for idx, (x, y) in enumerate(points):
        dominated = (
            (points[:, 0] <= x + 1e-12)
            & (points[:, 1] <= y + 1e-12)
            & ((points[:, 0] < x - 1e-12) | (points[:, 1] < y - 1e-12))
        )
        keep[idx] = not bool(np.any(dominated))
    return pd.Series(keep, index=frame.index)


def _finite(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    mask = np.ones(len(out), dtype=bool)
    for column in columns:
        mask &= np.isfinite(out[column].to_numpy(dtype=float))
    return out[mask].copy()


def _solution_components(
    scenario,
    network,
    p_env: dict[int, float],
    baseline_r: float,
    lambda_r: float,
    shutoff: Iterable[int],
    flow_by_line: dict[int, float],
    theta_by_bus: dict[int, float],
    pg_by_bus: dict[int, float],
    service_by_bus: dict[int, float],
) -> dict:
    active = [branch.line_id for branch in network.branches if int(branch.line_id) not in set(int(v) for v in shutoff)]
    components = _objective_components_from_solution(
        scenario,
        network,
        p_env,
        float(baseline_r),
        flow_by_line,
        service_by_bus,
    )
    lambda_l = 1.0 - float(lambda_r)
    j_no_phys = float(lambda_r) * float(components["R_norm"]) + lambda_l * float(components["L_shed"])
    source_less = source_less_island_buses(scenario, shutoff)
    residuals = _dc_residuals(scenario, network, active, theta_by_bus, flow_by_line, pg_by_bus, service_by_bus)
    common = common_operational_diagnostic(scenario, network, shutoff, flow_by_line, pg_by_bus, service_by_bus, source_less)
    return {
        "J_true": float(j_no_phys),
        "J_no_phys": float(j_no_phys),
        "J_traditional_no_phys": float(j_no_phys),
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
        "max_loading_ratio": float(
            max(
                [abs(flow_by_line[b.line_id]) / b.rate_a_mva for b in network.branches if b.line_id in active],
                default=0.0,
            )
        ),
        "gridfm_calls": 0,
        "call_budget": 0,
        "budget_exhausted": False,
        "termination_reason": "dc_miqp_solution_pool",
        "source_less_bus_ids": _line_key(source_less),
        "dc_flow_json": json.dumps({str(k): v for k, v in sorted(flow_by_line.items())}, sort_keys=True),
        "dc_theta_json": json.dumps({str(k): v for k, v in sorted(theta_by_bus.items())}, sort_keys=True),
        "dc_pg_json": json.dumps({str(k): v for k, v in sorted(pg_by_bus.items())}, sort_keys=True),
        "dc_service_json": json.dumps({str(k): v for k, v in sorted(service_by_bus.items())}, sort_keys=True),
        "dc_exposure_by_line_json": json.dumps(
            {str(k): v for k, v in sorted(components["exposure_by_line"].items())},
            sort_keys=True,
        ),
        **residuals,
        **common,
    }


def solve_stage_ib_solution_pool(
    scenario,
    network,
    p_env: dict[int, float],
    baseline_r: float,
    lambda_r: float,
    pool_solutions: int,
    pool_gap: float,
    time_limit_seconds: float,
) -> tuple[pd.DataFrame, dict]:
    gp, GRB = import_gurobipy()
    started = time.perf_counter()
    model = gp.Model("stage_i_b_dc_miqp_pool")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = float(MIP_GAP)
    model.Params.TimeLimit = float(time_limit_seconds)
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = int(pool_solutions)
    model.Params.PoolGap = float(pool_gap)

    candidate_set = set(int(v) for v in network.candidate_line_ids)
    theta = {bus: model.addVar(lb=-THETA_MAX, ub=THETA_MAX, name=f"theta_{bus}") for bus in range(int(scenario.num_buses))}
    pg_min = np.asarray(scenario.Pg_min, dtype=float)
    pg_max = np.asarray(scenario.Pg_max, dtype=float)
    pg = {
        bus: model.addVar(lb=float(pg_min[bus]), ub=float(pg_max[bus]), name=f"Pg_{bus}")
        for bus in network.generator_buses
    }
    service = {bus: model.addVar(lb=0.0, ub=1.0, name=f"s_{bus}") for bus in network.load_buses}
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
        big_m = float(branch.rate_a_mva) + abs(float(branch.susceptance_mw_per_rad)) * (
            2.0 * THETA_MAX + abs(float(branch.phi_rad))
        )
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
    pd_base = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    for bus in range(int(scenario.num_buses)):
        out_expr = gp.quicksum(f[branch.line_id] for branch in network.branches if branch.from_bus == bus)
        in_expr = gp.quicksum(f[branch.line_id] for branch in network.branches if branch.to_bus == bus)
        gen_expr = pg[bus] if bus in pg else 0.0
        load_expr = float(pd_base[bus]) * service[bus] if bus in service else 0.0
        model.addConstr(gen_expr - load_expr - out_expr + in_expr == 0.0, name=f"balance_{bus}")

    total_pd = max(float(network.total_pd), EPSILON)
    risk_expr = gp.QuadExpr()
    for branch in network.branches:
        coeff = float(p_env.get(branch.line_id, 0.0)) / (float(branch.rate_a_mva) ** 2) / max(float(baseline_r), EPSILON)
        risk_expr.add(coeff * f[branch.line_id] * f[branch.line_id])
    load_expr = gp.quicksum(float(pd_base[bus]) * (1.0 - service[bus]) for bus in network.load_buses) / total_pd
    model.setObjective(float(lambda_r) * risk_expr + (1.0 - float(lambda_r)) * load_expr, GRB.MINIMIZE)
    model.optimize()

    solve_metadata = {
        "solver_status": int(model.Status),
        "best_bound": float(model.ObjBound) if hasattr(model, "ObjBound") else np.nan,
        "mip_gap": float(model.MIPGap) if np.isfinite(getattr(model, "MIPGap", np.nan)) else np.nan,
        "runtime_seconds": float(time.perf_counter() - started),
        "node_count": float(getattr(model, "NodeCount", np.nan)),
        "solution_count": int(getattr(model, "SolCount", 0)),
        "time_limit_reached": bool(model.Status == GRB.TIME_LIMIT),
        "optimality_certified": bool(
            model.Status in {GRB.OPTIMAL, GRB.SUBOPTIMAL}
            and np.isfinite(getattr(model, "MIPGap", np.nan))
            and float(model.MIPGap) <= float(MIP_GAP) + 1e-12
        ),
    }
    rows = []
    for sol_no in range(int(getattr(model, "SolCount", 0))):
        model.Params.SolutionNumber = int(sol_no)
        shutoff = sorted(int(line_id) for line_id, var in y.items() if int(round(float(var.Xn))) == 1)
        flow_by_line = {int(line_id): float(var.Xn) for line_id, var in f.items()}
        theta_by_bus = {int(bus): float(var.Xn) for bus, var in theta.items()}
        pg_by_bus = {int(bus): float(var.Xn) for bus, var in pg.items()}
        service_by_bus = {int(bus): float(var.Xn) for bus, var in service.items()}
        components = _solution_components(
            scenario,
            network,
            p_env,
            baseline_r,
            lambda_r,
            shutoff,
            flow_by_line,
            theta_by_bus,
            pg_by_bus,
            service_by_bus,
        )
        rows.append(
            {
                "pool_solution_rank": int(sol_no + 1),
                "pool_obj_val": float(model.PoolObjVal),
                "pool_solution_count": int(model.SolCount),
                "status": "ok",
                "objective_value": float(model.PoolObjVal),
                "shutoff_line_ids": _line_key(shutoff),
                "line_id_key": _line_key(shutoff),
                "num_shutoff_lines": int(len(shutoff)),
                **solve_metadata,
                **components,
            }
        )
    return pd.DataFrame(rows), solve_metadata


def _run_pool_for_result(run_dir: Path, pool_solutions: int, pool_gap: float, time_limit_seconds: float) -> pd.DataFrame:
    context = _make_continuous_context("gnn", 5.0)
    scenario = context["scenario"]
    network = build_dc_network(scenario)
    tables = run_dir / "tables"
    best = pd.read_csv(_long_path(tables / "best_by_rho_scenario_lambda_stage.csv"))
    ib = best[best["stage"].astype(str).eq(STAGE_I_B)].copy()
    key_cols = ["scenario_id", "scenario_name", "lambda_R", "lambda_L", "lambda_case", "baseline_R", "p_env_json"]
    for optional in ["lambda_R_proxy", "lambda_L_proxy", "lambda_proxy_case"]:
        if optional in ib.columns:
            key_cols.append(optional)
    rho_values = sorted(float(v) for v in ib["rho_phys"].dropna().unique())
    unique = ib[key_cols].drop_duplicates().sort_values(["scenario_id", "lambda_R"], kind="mergesort").reset_index(drop=True)
    raw_rows = []
    for idx, source in enumerate(unique.itertuples(index=False), start=1):
        p_env = {int(k): float(v) for k, v in json.loads(source.p_env_json).items()}
        pool, metadata = solve_stage_ib_solution_pool(
            scenario,
            network,
            p_env,
            float(source.baseline_R),
            float(source.lambda_R),
            int(pool_solutions),
            float(pool_gap),
            float(time_limit_seconds),
        )
        if pool.empty:
            continue
        for rho in rho_values:
            frame = pool.copy()
            frame.insert(0, "scenario_id", source.scenario_id)
            frame.insert(1, "scenario_name", source.scenario_name)
            frame["stage"] = STAGE_I_B
            frame["stage_label"] = STAGE_I_B_LABEL
            frame["method_family"] = "Stage I-b"
            frame["lambda_R"] = float(source.lambda_R)
            frame["lambda_L"] = float(source.lambda_L)
            frame["lambda_case"] = source.lambda_case
            frame["lambda_R_proxy"] = float(getattr(source, "lambda_R_proxy", source.lambda_R))
            frame["lambda_L_proxy"] = float(getattr(source, "lambda_L_proxy", 1.0 - float(source.lambda_R)))
            frame["lambda_proxy_case"] = getattr(source, "lambda_proxy_case", source.lambda_case)
            frame["topology_iteration"] = frame["pool_solution_rank"].astype(int)
            frame["proposal_method"] = "stage_i_b_direct_dc_miqp_solution_pool"
            frame["baseline_R"] = float(source.baseline_R)
            frame["p_env_json"] = source.p_env_json
            frame["topology_proxy_excludes_pac"] = False
            frame["rho_phys"] = float(rho)
            frame["model_type"] = "dc"
            frame["candidate_set"] = "t0p30"
            frame["p_env_mode"] = P_ENV_MODE_TARGET_MARGIN
            frame["baseline_loading_source"] = BASELINE_LOADING_SOURCE
            frame["R_base_s_source"] = BASELINE_LOADING_SOURCE
            frame["post_topology_evaluation_source"] = "joint_dc_miqp_solution_pool"
            frame["continuous_recourse_optimized"] = True
            frame["pool_solve_index"] = int(idx)
            raw_rows.append(frame)
    raw = pd.concat(raw_rows, ignore_index=True, sort=False) if raw_rows else pd.DataFrame()
    write_dataframe(tables / POOL_RAW_TABLE, raw)
    if raw.empty:
        write_dataframe(tables / POOL_TABLE, raw)
        return raw
    sort_cols = ["scenario_id", "rho_phys", "lambda_R", "line_id_key", "J_no_phys", "pool_solution_rank"]
    dedup = (
        raw.sort_values(sort_cols, kind="mergesort")
        .drop_duplicates(["scenario_id", "rho_phys", "lambda_R", "line_id_key"], keep="first")
        .sort_values(["scenario_id", "rho_phys", "lambda_R", "pool_solution_rank"], kind="mergesort")
        .reset_index(drop=True)
    )
    dedup["stage_label"] = STAGE_LABELS[STAGE_I_B]
    write_dataframe(tables / POOL_TABLE, dedup)
    return dedup


def _augment_points(run_dir: Path, pool: pd.DataFrame) -> pd.DataFrame:
    tables = run_dir / "tables"
    all_eval = pd.read_csv(_long_path(tables / "all_evaluated_stage_h_points.csv"))
    if pool.empty:
        augmented = all_eval.copy()
    else:
        base = all_eval.copy()
        common = sorted(set(base.columns).union(pool.columns))
        augmented = pd.concat([base.reindex(columns=common), pool.reindex(columns=common)], ignore_index=True, sort=False)
    if "stage_order" not in augmented.columns:
        augmented["stage_order"] = augmented["stage"].map({stage: idx for idx, stage in enumerate(STAGE_ORDER)}).fillna(99)
    write_dataframe(tables / AUGMENTED_POINTS_TABLE, augmented)
    return augmented


def _build_convergence(augmented: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    rows = []
    work = _finite(augmented, ["lambda_R", "rho_phys", "topology_iteration", "J_traditional_no_phys", "L_shed", "R_norm"])
    for (scenario_id, rho, lambda_r, stage), local in work.groupby(["scenario_id", "rho_phys", "lambda_R", "stage"], sort=True):
        local = local.sort_values(["topology_iteration", "J_traditional_no_phys", "line_id_key"], kind="mergesort")
        stage_label = STAGE_LABELS.get(str(stage), str(local["stage_label"].iloc[0]) if "stage_label" in local else str(stage))
        iter_best = local.groupby("topology_iteration", as_index=False).first().sort_values("topology_iteration", kind="mergesort")
        best = np.inf
        for _, row in iter_best.iterrows():
            best = min(best, float(row["J_traditional_no_phys"]))
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "rho_phys": float(rho),
                    "lambda_R": float(lambda_r),
                    "stage": stage,
                    "stage_label": stage_label,
                    "topology_iteration": int(row["topology_iteration"]),
                    "pool_solution_rank": row.get("pool_solution_rank", np.nan),
                    "line_id_key": row.get("line_id_key", ""),
                    "J_traditional_no_phys": float(row["J_traditional_no_phys"]),
                    "best_so_far_traditional_no_phys": float(best),
                    "L_shed": float(row["L_shed"]),
                    "R_norm": float(row["R_norm"]),
                }
            )
    convergence = pd.DataFrame(rows)
    write_dataframe(run_dir / "tables" / AUGMENTED_CONVERGENCE_TABLE, convergence)
    return convergence


def _plot_pareto(local: pd.DataFrame, out: Path, title: str, filename: str, include_gridfm: bool) -> None:
    local = _finite(local, ["L_shed", "R_norm", "lambda_R"])
    if not include_gridfm:
        local = local[~local["stage"].astype(str).eq(EXCLUDE_GRIDFM_STAGE)].copy()
    fig, ax = plt.subplots(figsize=(10.2, 6.8), constrained_layout=True)
    for stage in STAGE_ORDER:
        frame = local[local["stage"].astype(str).eq(stage)].copy()
        if frame.empty:
            continue
        label = STAGE_LABELS.get(stage, stage)
        alpha = 0.18 if stage in {"stage_e_k2", "stage_i_a_dc_k2"} else 0.72
        size = 26 if stage in {"stage_e_k2", "stage_i_a_dc_k2"} else 68
        ax.scatter(
            frame["L_shed"],
            frame["R_norm"],
            s=size,
            marker=MARKERS.get(stage, "o"),
            color=COLORS.get(stage, "#666666"),
            alpha=alpha,
            edgecolor=COLORS.get(stage, "#666666"),
            linewidth=0.6,
            label=f"{label} evaluated",
        )
        unique = frame.drop_duplicates(["L_shed", "R_norm", "line_id_key"]).copy()
        nd = unique[_nondominated_mask(unique)].sort_values(["L_shed", "R_norm"], kind="mergesort")
        if not nd.empty:
            ax.plot(nd["L_shed"], nd["R_norm"], color=COLORS.get(stage, "#666666"), linewidth=2.0, alpha=0.95)
            ax.scatter(
                nd["L_shed"],
                nd["R_norm"],
                s=92,
                marker=MARKERS.get(stage, "o"),
                color=COLORS.get(stage, "#666666"),
                edgecolor="#202020",
                linewidth=0.8,
                label=f"{label} nondominated",
            )
    ax.set_xlabel("L_shed")
    ax.set_ylabel("R_norm")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    _savefig(fig, out / filename, dpi=190)
    plt.close(fig)


def _plot_convergence(local: pd.DataFrame, out: Path, title: str, filename: str, include_gridfm: bool) -> None:
    local = _finite(local, ["lambda_R", "topology_iteration", "best_so_far_traditional_no_phys", "J_traditional_no_phys"])
    if not include_gridfm:
        local = local[~local["stage"].astype(str).eq(EXCLUDE_GRIDFM_STAGE)].copy()
    lambdas = sorted(float(v) for v in local["lambda_R"].dropna().unique())
    if not lambdas:
        return
    fig, axes = plt.subplots(1, len(lambdas), figsize=(max(4.3 * len(lambdas), 7.2), 5.9), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    y_values = []
    legend = {}
    for ax, lambda_r in zip(axes, lambdas):
        sub = local[np.isclose(local["lambda_R"].astype(float), float(lambda_r))]
        for stage in STAGE_ORDER:
            frame = sub[sub["stage"].astype(str).eq(stage)].copy()
            if frame.empty:
                continue
            frame = frame.sort_values(["topology_iteration", "J_traditional_no_phys", "line_id_key"], kind="mergesort")
            y = frame["best_so_far_traditional_no_phys"].astype(float).cummin()
            y_values.extend(y.tolist())
            label = STAGE_LABELS.get(stage, stage)
            if len(frame) == 1:
                artist = ax.scatter(
                    frame["topology_iteration"],
                    y,
                    color=COLORS.get(stage, "#666666"),
                    marker=MARKERS.get(stage, "o"),
                    s=74,
                    edgecolor="#202020",
                    linewidth=0.7,
                    label=label,
                    zorder=3,
                )
            else:
                artist = ax.step(
                    frame["topology_iteration"],
                    y,
                    where="post",
                    color=COLORS.get(stage, "#666666"),
                    linewidth=2.0,
                    label=label,
                )[0]
                ax.scatter(
                    frame["topology_iteration"].iloc[-1],
                    y.iloc[-1],
                    color=COLORS.get(stage, "#666666"),
                    marker=MARKERS.get(stage, "o"),
                    s=48,
                    edgecolor="#202020",
                    linewidth=0.5,
                    zorder=3,
                )
            legend[label] = artist
        ax.set_title(f"lambda_R={lambda_r:g}", fontsize=10)
        ax.set_xlabel("Topology / pool iteration")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Best-so-far J")
    if y_values:
        vals = [float(v) for v in y_values if np.isfinite(v)]
        if vals:
            ymin, ymax = min(vals), max(vals)
            pad = max((ymax - ymin) * 0.1, 1e-4)
            for ax in axes:
                ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    fig.suptitle(title + "\nJ = lambda_R R_norm + (1 - lambda_R) L_shed", fontsize=12)
    if legend:
        fig.legend(list(legend.values()), list(legend.keys()), loc="lower center", ncols=min(len(legend), 6), fontsize=8, bbox_to_anchor=(0.5, -0.015))
    _savefig(fig, out / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _refresh_plots(run_dir: Path, augmented: pd.DataFrame, convergence: pd.DataFrame) -> dict:
    plots = run_dir / "plots"
    run_label = "Main" if "main_results" in str(run_dir) else "MLD"
    counts = {
        "pareto_frontier_scatter": 0,
        "traditional_lambda_objective_convergence": 0,
        "pareto_frontier_scatter_no_gridfm": 0,
        "traditional_lambda_objective_convergence_no_gridfm": 0,
    }
    for (rho, scenario_id), local in augmented.groupby(["rho_phys", "scenario_id"], sort=True):
        out = plots / "per_rho" / f"rho{float(rho):g}" / str(scenario_id)
        _plot_pareto(
            local,
            out,
            f"{run_label} {scenario_id}, rho={float(rho):g}: Pareto Scatter With MIQP Pool",
            "pareto_frontier_scatter.png",
            include_gridfm=True,
        )
        _plot_pareto(
            local,
            out,
            f"{run_label} {scenario_id}, rho={float(rho):g}: Pareto Scatter Excluding Stage E GridFM With MIQP Pool",
            "pareto_frontier_scatter_no_gridfm.png",
            include_gridfm=False,
        )
        counts["pareto_frontier_scatter"] += 1
        counts["pareto_frontier_scatter_no_gridfm"] += 1
    for (rho, scenario_id), local in convergence.groupby(["rho_phys", "scenario_id"], sort=True):
        out = plots / "per_rho" / f"rho{float(rho):g}" / str(scenario_id)
        _plot_convergence(
            local,
            out,
            f"{run_label} {scenario_id}, rho={float(rho):g}: Traditional Objective Convergence With MIQP Pool",
            "traditional_lambda_objective_convergence.png",
            include_gridfm=True,
        )
        _plot_convergence(
            local,
            out,
            f"{run_label} {scenario_id}, rho={float(rho):g}: Traditional Objective Convergence Excluding Stage E GridFM With MIQP Pool",
            "traditional_lambda_objective_convergence_no_gridfm.png",
            include_gridfm=False,
        )
        counts["traditional_lambda_objective_convergence"] += 1
        counts["traditional_lambda_objective_convergence_no_gridfm"] += 1
    return counts


def refresh_run(run_dir: Path, pool_solutions: int, pool_gap: float, time_limit_seconds: float) -> dict:
    pool = _run_pool_for_result(run_dir, pool_solutions, pool_gap, time_limit_seconds)
    augmented = _augment_points(run_dir, pool)
    convergence = _build_convergence(augmented, run_dir)
    counts = _refresh_plots(run_dir, augmented, convergence)
    summary = {
        "run_dir": str(run_dir),
        "pool_solutions_requested": int(pool_solutions),
        "pool_gap": float(pool_gap),
        "num_pool_rows_dedup_by_rho": int(len(pool)),
        "num_unique_pool_topologies": int(pool["line_id_key"].astype(str).nunique()) if not pool.empty else 0,
        "num_augmented_points": int(len(augmented)),
        "num_augmented_convergence_rows": int(len(convergence)),
        "plot_counts": counts,
    }
    write_json(run_dir / "tables" / "stage_i_b_solution_pool_refresh_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Stage H plots with Stage I-b MIQP solution-pool topologies.")
    parser.add_argument("--run", action="append", dest="runs", help="Run subfolder under the Stage H comparison root.")
    parser.add_argument("--pool-solutions", type=int, default=50)
    parser.add_argument("--pool-gap", type=float, default=1e9)
    parser.add_argument("--time-limit", type=float, default=600.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summaries = []
    for run in args.runs or RUNS:
        summaries.append(refresh_run(RESULT_ROOT / run, int(args.pool_solutions), float(args.pool_gap), float(args.time_limit)))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
