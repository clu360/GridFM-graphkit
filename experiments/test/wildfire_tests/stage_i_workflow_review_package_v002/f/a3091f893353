from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments.test.wildfire_tests.gridfm_support.branch_metadata import expand_to_physical_line_ids
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.physics_infeasibility_evaluator import (
    source_less_island_buses,
)
from experiments.test.wildfire_tests.stage_i_dc_comparison.dc_formulation import DCNetwork, _parse_line_key


EPSILON = 1e-6


@dataclass(frozen=True)
class ProjectionIndex:
    generator_buses: tuple[int, ...]
    load_buses: tuple[int, ...]
    n_bus: int
    n_gen: int
    n_load: int
    pg_start: int
    qg_start: int
    s_start: int


def solution_identity(row: pd.Series | dict) -> str:
    data = dict(row)
    payload = {
        "stage": str(data.get("stage", "")),
        "scenario_id": str(data.get("scenario_id", "")),
        "lambda_R": float(data.get("lambda_R", 0.0)),
        "shutoff_line_ids": str(data.get("shutoff_line_ids", data.get("line_id_key", ""))),
        "dc_flow_json": str(data.get("dc_flow_json", "")),
        "dc_pg_json": str(data.get("dc_pg_json", "")),
        "dc_service_json": str(data.get("dc_service_json", "")),
        "u_best_json": str(data.get("u_best_json", "")),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _edge_array(scenario) -> np.ndarray:
    return scenario.edge_index.cpu().numpy() if hasattr(scenario.edge_index, "cpu") else np.asarray(scenario.edge_index)


def _angles_to_radians(values: Sequence[float]) -> np.ndarray:
    angles = np.asarray(values, dtype=float)
    if len(angles) and np.nanmax(np.abs(angles)) > (2.0 * np.pi + 1e-9):
        return np.deg2rad(angles)
    return angles


def _json_dict(value) -> Dict[int, float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return {}
    loaded = json.loads(text)
    return {int(k): float(v) for k, v in loaded.items()}


def _gridfm_commands(row: pd.Series | dict, context: dict) -> tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    data = dict(row)
    decision = context["decision_vector"]
    scenario = context["scenario"]
    try:
        u = np.asarray(json.loads(str(data.get("u_best_json", "[]"))), dtype=float)
    except Exception:
        u = np.asarray([], dtype=float)
    if len(u) != len(np.asarray(decision.u_base, dtype=float)):
        return {}, {}, {}
    delta_pg, delta_qg, alpha = decision.split_decision_vector(u)
    pg_cmd = {
        int(bus): float(np.asarray(scenario.Pg_base, dtype=float)[int(bus)] + float(delta_pg[idx]))
        for idx, bus in enumerate(np.asarray(decision.selected_generator_buses, dtype=int))
    }
    qg_cmd = {
        int(bus): float(np.asarray(scenario.Qg_base, dtype=float)[int(bus)] + float(delta_qg[idx]))
        for idx, bus in enumerate(np.asarray(decision.selected_generator_buses, dtype=int))
    }
    s_cmd = {
        int(bus): float(alpha[idx])
        for idx, bus in enumerate(np.asarray(decision.selected_load_buses, dtype=int))
    }
    return pg_cmd, qg_cmd, s_cmd


def _dc_commands(row: pd.Series | dict) -> tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    data = dict(row)
    pg_cmd = _json_dict(data.get("dc_pg_json", ""))
    service_cmd = _json_dict(data.get("dc_service_json", ""))
    flow_cmd = _json_dict(data.get("dc_flow_json", ""))
    return pg_cmd, service_cmd, flow_cmd


def _projection_index(network: DCNetwork, scenario) -> ProjectionIndex:
    gen = tuple(int(v) for v in network.generator_buses)
    load = tuple(int(v) for v in network.load_buses)
    n_bus = int(scenario.num_buses)
    pg_start = 2 * n_bus
    qg_start = pg_start + len(gen)
    s_start = qg_start + len(gen)
    return ProjectionIndex(gen, load, n_bus, len(gen), len(load), pg_start, qg_start, s_start)


def _split(x: np.ndarray, index: ProjectionIndex) -> tuple[np.ndarray, np.ndarray, Dict[int, float], Dict[int, float], Dict[int, float]]:
    vm = np.asarray(x[: index.n_bus], dtype=float)
    va = np.asarray(x[index.n_bus : 2 * index.n_bus], dtype=float)
    pg_vals = np.asarray(x[index.pg_start : index.qg_start], dtype=float)
    qg_vals = np.asarray(x[index.qg_start : index.s_start], dtype=float)
    s_vals = np.asarray(x[index.s_start : index.s_start + index.n_load], dtype=float)
    pg = {int(bus): float(pg_vals[pos]) for pos, bus in enumerate(index.generator_buses)}
    qg = {int(bus): float(qg_vals[pos]) for pos, bus in enumerate(index.generator_buses)}
    service = {int(bus): float(s_vals[pos]) for pos, bus in enumerate(index.load_buses)}
    return vm, va, pg, qg, service


def _pack_initial(
    scenario,
    index: ProjectionIndex,
    pg_anchor: Dict[int, float],
    qg_anchor: Dict[int, float],
    service_anchor: Dict[int, float],
    source_less_buses: Iterable[int],
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    vm = np.clip(np.asarray(scenario.Vm_base, dtype=float), 0.9, 1.1)
    va = np.clip(_angles_to_radians(np.asarray(scenario.Va_base, dtype=float)), -np.pi, np.pi)
    pg_min = np.asarray(scenario.Pg_min, dtype=float)
    pg_max = np.asarray(scenario.Pg_max, dtype=float)
    pg = np.asarray(
        [np.clip(float(pg_anchor.get(bus, scenario.Pg_base[bus])), float(pg_min[bus]), float(pg_max[bus])) for bus in index.generator_buses],
        dtype=float,
    )
    qg = np.asarray([float(qg_anchor.get(bus, scenario.Qg_base[bus])) for bus in index.generator_buses], dtype=float)
    source_less = {int(bus) for bus in source_less_buses}
    service = np.asarray(
        [0.0 if int(bus) in source_less else np.clip(float(service_anchor.get(bus, 1.0)), 0.0, 1.0) for bus in index.load_buses],
        dtype=float,
    )
    x0 = np.concatenate([vm, va, pg, qg, service])
    bounds: list[tuple[float, float]] = []
    bounds.extend([(0.9, 1.1) for _ in range(index.n_bus)])
    bounds.extend([(-np.pi, np.pi) for _ in range(index.n_bus)])
    bounds.extend([(float(pg_min[bus]), float(pg_max[bus])) for bus in index.generator_buses])
    bounds.extend([(-1.0e4, 1.0e4) for _ in index.generator_buses])
    for bus in index.load_buses:
        bounds.append((0.0, 0.0) if int(bus) in source_less else (0.0, 1.0))
    return x0, bounds


def _active_topology_arrays(scenario, shutoff_line_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge = _edge_array(scenario)
    yf = getattr(scenario, "Yf", None)
    yt = getattr(scenario, "Yt", None)
    if yf is None or yt is None:
        raise ValueError("Scenario does not expose Yf/Yt for AC projection.")
    removed = set(expand_to_physical_line_ids(scenario, shutoff_line_ids))
    canonical = getattr(scenario, "canonical_line_id", None)
    is_self_loop = getattr(scenario, "is_self_loop", None)
    keep_values = []
    for idx in range(int(edge.shape[1])):
        if int(idx) in removed:
            continue
        if is_self_loop is not None and bool(is_self_loop[int(idx)]):
            continue
        if canonical is not None and int(canonical[int(idx)]) != int(idx):
            continue
        keep_values.append(int(idx))
    keep = np.asarray(keep_values, dtype=int)
    return keep, edge[:, keep], yf[keep, :], yt[keep, :]


def _ac_network_quantities(
    scenario,
    keep: np.ndarray,
    edge_active: np.ndarray,
    yf_active,
    yt_active,
    vm: np.ndarray,
    va: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = float(getattr(scenario, "sn_mva", 100.0))
    v = np.asarray(vm, dtype=float) * np.exp(1j * np.asarray(va, dtype=float))
    if_complex = yf_active @ v
    it_complex = yt_active @ v
    bus_current = np.zeros(int(scenario.num_buses), dtype=complex)
    fbus = edge_active[0, :].astype(int)
    tbus = edge_active[1, :].astype(int)
    np.add.at(bus_current, fbus, if_complex)
    np.add.at(bus_current, tbus, it_complex)
    s_bus = v * np.conj(bus_current) * base
    sf = v[fbus] * np.conj(if_complex) * base
    st = v[tbus] * np.conj(it_complex) * base
    return s_bus, sf, st, v


def _canonical_ac_flow_from_directed(
    scenario,
    keep: np.ndarray,
    sf: np.ndarray,
    network: DCNetwork,
) -> Dict[int, float]:
    canonical = getattr(scenario, "canonical_line_id", None)
    values: Dict[int, float] = {}
    for local_idx, original_line_id in enumerate(keep):
        line_id = int(original_line_id)
        canon = int(canonical[line_id]) if canonical is not None and int(canonical[line_id]) >= 0 else line_id
        if canon in values:
            continue
        values[canon] = float(np.real(sf[local_idx]))
    return {int(branch.line_id): float(values.get(int(branch.line_id), 0.0)) for branch in network.branches}


def _projection_objective(
    x: np.ndarray,
    scenario,
    network: DCNetwork,
    index: ProjectionIndex,
    family: str,
    pg_anchor: Dict[int, float],
    qg_anchor: Dict[int, float],
    service_anchor: Dict[int, float],
    flow_anchor: Dict[int, float],
    keep: np.ndarray,
    edge_active: np.ndarray,
    yf_active,
    yt_active,
) -> tuple[float, dict]:
    vm, va, pg, qg, service = _split(x, index)
    pg_min = np.asarray(scenario.Pg_min, dtype=float)
    pg_max = np.asarray(scenario.Pg_max, dtype=float)
    pd = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    d_pg_terms = []
    buses = pg_anchor.keys() if family == "GridFM" else index.generator_buses
    for bus in buses:
        scale = max(float(pg_max[int(bus)] - pg_min[int(bus)]), 1.0, EPSILON)
        d_pg_terms.append(((float(pg.get(int(bus), 0.0)) - float(pg_anchor.get(int(bus), scenario.Pg_base[int(bus)]))) / scale) ** 2)
    d_pg = float(np.mean(d_pg_terms)) if d_pg_terms else 0.0

    d_qg = 0.0
    if family == "GridFM" and qg_anchor:
        d_qg = float(np.mean([(float(qg.get(int(bus), 0.0)) - float(target)) ** 2 / 100.0**2 for bus, target in qg_anchor.items()]))

    service_buses = service_anchor.keys() if family == "GridFM" else index.load_buses
    service_den = max(float(sum(float(pd[int(bus)]) for bus in service_buses)), EPSILON)
    d_s = float(
        sum(float(pd[int(bus)]) * (float(service.get(int(bus), 1.0)) - float(service_anchor.get(int(bus), 1.0))) ** 2 for bus in service_buses)
        / service_den
    ) if service_buses else 0.0

    d_f = 0.0
    if family == "DC" and flow_anchor:
        _s_bus, sf, _st, _v = _ac_network_quantities(scenario, keep, edge_active, yf_active, yt_active, vm, va)
        ac_flow = _canonical_ac_flow_from_directed(scenario, keep, sf, network)
        active_ids = [int(line_id) for line_id in flow_anchor if int(line_id) in ac_flow]
        if active_ids:
            rate_by_line = {int(branch.line_id): float(branch.rate_a_mva) for branch in network.branches}
            d_f = float(
                np.mean(
                    [
                        ((float(ac_flow[line_id]) - float(flow_anchor[line_id])) / max(float(rate_by_line.get(line_id, 1.0)), EPSILON)) ** 2
                        for line_id in active_ids
                    ]
                )
            )
    total = float(d_pg + d_qg + d_s + d_f)
    return total, {"D_Pg": d_pg, "D_Qg": d_qg, "D_s": d_s, "D_f": d_f}


def _balance_residual(
    x: np.ndarray,
    scenario,
    index: ProjectionIndex,
    keep: np.ndarray,
    edge_active: np.ndarray,
    yf_active,
    yt_active,
) -> np.ndarray:
    vm, va, pg, qg, service = _split(x, index)
    s_bus, _sf, _st, _v = _ac_network_quantities(scenario, keep, edge_active, yf_active, yt_active, vm, va)
    pd = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    qd = np.asarray(scenario.Qd_base, dtype=float)
    p_inj = np.zeros(index.n_bus, dtype=float)
    q_inj = np.zeros(index.n_bus, dtype=float)
    for bus, value in pg.items():
        p_inj[int(bus)] += float(value)
    for bus, value in qg.items():
        q_inj[int(bus)] += float(value)
    for bus in index.load_buses:
        s_val = float(service.get(int(bus), 1.0))
        p_inj[int(bus)] -= s_val * float(pd[int(bus)])
        q_inj[int(bus)] -= s_val * float(qd[int(bus)])
    scale = max(float(getattr(scenario, "sn_mva", 100.0)), 1.0)
    return np.concatenate([(p_inj + np.real(s_bus)) / scale, (q_inj + np.imag(s_bus)) / scale])


def _thermal_margin(
    x: np.ndarray,
    scenario,
    network: DCNetwork,
    index: ProjectionIndex,
    keep: np.ndarray,
    edge_active: np.ndarray,
    yf_active,
    yt_active,
) -> np.ndarray:
    vm, va, _pg, _qg, _service = _split(x, index)
    _s_bus, sf, st, _v = _ac_network_quantities(scenario, keep, edge_active, yf_active, yt_active, vm, va)
    canonical = getattr(scenario, "canonical_line_id", None)
    rate_by_line = {int(branch.line_id): float(branch.rate_a_mva) for branch in network.branches}
    loading_by_canon: Dict[int, float] = {}
    for local_idx, original_line_id in enumerate(keep):
        line_id = int(original_line_id)
        canon = int(canonical[line_id]) if canonical is not None and int(canonical[line_id]) >= 0 else line_id
        if canon not in rate_by_line:
            continue
        loading = max(abs(complex(sf[local_idx])), abs(complex(st[local_idx]))) / max(float(rate_by_line[canon]), EPSILON)
        loading_by_canon[canon] = max(float(loading), float(loading_by_canon.get(canon, 0.0)))
    return np.asarray([1.0 - float(value) for value in loading_by_canon.values()], dtype=float)


def project_solution_to_ac(row: pd.Series | dict, context: dict, network: DCNetwork) -> dict:
    started = time.perf_counter()
    data = dict(row)
    identity = solution_identity(data)
    stage = str(data.get("stage", ""))
    model_type = str(data.get("model_type", ""))
    family = "DC" if model_type.lower().startswith("dc") or stage.startswith("stage_i") or stage.startswith("th_") or stage.startswith("ah_") else "GridFM"
    scenario = context["scenario"]
    shutoff = _parse_line_key(data.get("shutoff_line_ids", data.get("line_id_key", "")))
    source_less = source_less_island_buses(scenario, shutoff)
    try:
        keep, edge_active, yf_active, yt_active = _active_topology_arrays(scenario, shutoff)
        index = _projection_index(network, scenario)
        if family == "GridFM":
            pg_anchor, qg_anchor, service_anchor = _gridfm_commands(data, context)
            flow_anchor: Dict[int, float] = {}
        else:
            pg_anchor, service_anchor, flow_anchor = _dc_commands(data)
            qg_anchor = {}
        x0, bounds = _pack_initial(scenario, index, pg_anchor, qg_anchor, service_anchor, source_less)

        def objective(x):
            value, _parts = _projection_objective(
                x,
                scenario,
                network,
                index,
                family,
                pg_anchor,
                qg_anchor,
                service_anchor,
                flow_anchor,
                keep,
                edge_active,
                yf_active,
                yt_active,
            )
            return value

        constraints = [
            {
                "type": "eq",
                "fun": lambda x: _balance_residual(x, scenario, index, keep, edge_active, yf_active, yt_active),
            },
            {
                "type": "ineq",
                "fun": lambda x: _thermal_margin(x, scenario, network, index, keep, edge_active, yf_active, yt_active),
            },
        ]
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-7, "disp": False},
        )
        x_best = np.asarray(result.x if result.x is not None else x0, dtype=float)
        total, parts = _projection_objective(
            x_best,
            scenario,
            network,
            index,
            family,
            pg_anchor,
            qg_anchor,
            service_anchor,
            flow_anchor,
            keep,
            edge_active,
            yf_active,
            yt_active,
        )
        balance = _balance_residual(x_best, scenario, index, keep, edge_active, yf_active, yt_active)
        thermal = _thermal_margin(x_best, scenario, network, index, keep, edge_active, yf_active, yt_active)
        success = bool(result.success and np.max(np.abs(balance)) <= 1e-5 and (len(thermal) == 0 or np.min(thermal) >= -1e-5))
        status = "solved" if success else "solver_returned_infeasible_or_residual"
        return {
            "projection_solution_id": identity,
            "scenario_id": data.get("scenario_id", ""),
            "stage": stage,
            "stage_label": data.get("stage_label", stage),
            "lambda_R": float(data.get("lambda_R", np.nan)),
            "rho_phys": float(data.get("rho_phys", np.nan)),
            "shutoff_line_ids": data.get("shutoff_line_ids", data.get("line_id_key", "")),
            "projection_backend": "scipy_slsqp_v1",
            "projection_family": family,
            "projection_status": status,
            "projection_success": success,
            "projection_message": str(result.message),
            "D_proj_total": float(total) if success else np.nan,
            "D_Pg": float(parts["D_Pg"]) if success else np.nan,
            "D_Qg": float(parts["D_Qg"]) if success else np.nan,
            "D_s": float(parts["D_s"]) if success else np.nan,
            "D_f": float(parts["D_f"]) if success else np.nan,
            "w_theta": 0.0,
            "max_abs_ac_balance_residual": float(np.max(np.abs(balance))) if len(balance) else 0.0,
            "min_thermal_margin": float(np.min(thermal)) if len(thermal) else np.nan,
            "slsqp_success": bool(result.success),
            "slsqp_status": int(result.status),
            "slsqp_iterations": int(getattr(result, "nit", -1)),
            "Qg_bounds_available": False,
            "runtime_seconds": float(time.perf_counter() - started),
        }
    except Exception as exc:
        return {
            "projection_solution_id": identity,
            "scenario_id": data.get("scenario_id", ""),
            "stage": stage,
            "stage_label": data.get("stage_label", stage),
            "lambda_R": float(data.get("lambda_R", np.nan)),
            "rho_phys": float(data.get("rho_phys", np.nan)),
            "shutoff_line_ids": data.get("shutoff_line_ids", data.get("line_id_key", "")),
            "projection_backend": "scipy_slsqp_v1",
            "projection_family": family,
            "projection_status": "projection_exception",
            "projection_success": False,
            "projection_message": str(exc),
            "D_proj_total": np.nan,
            "D_Pg": np.nan,
            "D_Qg": np.nan,
            "D_s": np.nan,
            "D_f": np.nan,
            "w_theta": 0.0,
            "max_abs_ac_balance_residual": np.nan,
            "min_thermal_margin": np.nan,
            "slsqp_success": False,
            "slsqp_status": -1,
            "slsqp_iterations": -1,
            "Qg_bounds_available": False,
            "runtime_seconds": float(time.perf_counter() - started),
        }


def project_finalists_with_cache(
    finalists: pd.DataFrame,
    context: dict,
    network: DCNetwork,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    cache: Dict[str, dict] = {}
    if cache_path is not None and cache_path.exists():
        existing = pd.read_csv(cache_path)
        for _, row in existing.iterrows():
            cached = row.to_dict()
            if str(cached.get("projection_backend", "")).startswith("scipy_slsqp_v1_pending"):
                continue
            cache[str(row["projection_solution_id"])] = cached

    rows = []
    for _, finalist in finalists.iterrows():
        identity = solution_identity(finalist)
        if identity not in cache:
            cache[identity] = project_solution_to_ac(finalist, context, network)
        projected = dict(cache[identity])
        projected["scenario_id"] = finalist.get("scenario_id", projected.get("scenario_id", ""))
        projected["lambda_R"] = float(finalist.get("lambda_R", projected.get("lambda_R", np.nan)))
        projected["rho_phys"] = float(finalist.get("rho_phys", projected.get("rho_phys", np.nan)))
        projected["stage"] = finalist.get("stage", projected.get("stage", ""))
        projected["stage_label"] = finalist.get("stage_label", projected.get("stage_label", projected["stage"]))
        rows.append(projected)

    frame = pd.DataFrame(rows)
    if cache_path is not None:
        _mkdir(cache_path.parent)
        pd.DataFrame(list(cache.values())).to_csv(_long_path(cache_path), index=False)
    return frame


def _long_path(path: Path) -> str:
    path = Path(path)
    if os.name == "nt":
        absolute = str(path.absolute())
        return absolute if absolute.startswith("\\\\?\\") else "\\\\?\\" + absolute
    return str(path)


def _mkdir(path: Path) -> None:
    path = Path(path)
    if os.name == "nt":
        Path(_long_path(path)).mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
