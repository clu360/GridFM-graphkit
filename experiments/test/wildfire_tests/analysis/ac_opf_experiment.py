from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.config import load_first_pass_config, write_config_copy
from experiments.test.wildfire_tests.shared.decision_vector import auto_select_decision_buses
from experiments.test.wildfire_tests.shared.lambda_cases import CANONICAL_LAMBDA_CASES, LAMBDA_FOLDERS, lambda_case_rows
from experiments.test.wildfire_tests.shared.paths import CONFIGS_ROOT, RESULTS_ROOT
from experiments.test.wildfire_tests.shared.plot_network_changes import plot_network_changes
from experiments.test.wildfire_tests.shared.plot_optimization_behavior import plot_optimization_behavior
from experiments.test.wildfire_tests.shared.reporting import write_dataframe, write_json
from experiments.test.wildfire_tests.shared.scenario import load_first_pass_context
from experiments.test.wildfire_tests.shared.wildfire_scenario import (
    build_synthetic_wildfire_scenario,
    validate_connected_line_group,
)


TRADEOFF_SETS = lambda_case_rows(CANONICAL_LAMBDA_CASES)


def _line_before_after(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    return before.merge(after, on="line_id", suffixes=("_before", "_after")).rename(
        columns={
            "loading_ratio_before": "loading_ratio_before",
            "loading_ratio_after": "loading_ratio_after",
            "risk_before": "risk_before",
            "risk_after": "risk_after",
        }
    )


def _group_before_after(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    return before.merge(after, on="group_name", suffixes=("_before", "_after"))


@dataclass
class ACOPFModel:
    scenario: object
    selected_load_buses: np.ndarray
    generator_buses: np.ndarray
    wildfire: object
    rate_a: np.ndarray
    ybus: np.ndarray
    ref_bus: int
    vm_min: float = 0.95
    vm_max: float = 1.05
    q_abs_bound_mvar: float = 100.0

    @property
    def nb(self) -> int:
        return int(self.scenario.num_buses)

    @property
    def ng(self) -> int:
        return int(len(self.generator_buses))

    @property
    def nq(self) -> int:
        return int(len(self.selected_load_buses))

    def initial_vector(self) -> np.ndarray:
        pg = np.asarray(self.scenario.Pg_base[self.generator_buses], dtype=float)
        qg = np.asarray(self.scenario.Qg_base[self.generator_buses], dtype=float)
        alpha = np.ones(self.nq, dtype=float)
        vm = np.clip(np.asarray(self.scenario.Vm_base, dtype=float), self.vm_min, self.vm_max)
        va = np.asarray(self.scenario.Va_base, dtype=float) - float(self.scenario.Va_base[self.ref_bus])
        return np.hstack([pg, qg, alpha, vm, va])

    def bounds(self) -> list[tuple[float, float]]:
        pg_base = np.asarray(self.scenario.Pg_base[self.generator_buses], dtype=float)
        pg_min = np.maximum(pg_base - 5.0, 0.0)
        pg_max = pg_base + 5.0
        qg_base = np.asarray(self.scenario.Qg_base[self.generator_buses], dtype=float)
        qg_min = qg_base - self.q_abs_bound_mvar
        qg_max = qg_base + self.q_abs_bound_mvar
        bounds = list(zip(pg_min, pg_max))
        bounds.extend(zip(qg_min, qg_max))
        bounds.extend([(0.0, 1.0)] * self.nq)
        bounds.extend([(self.vm_min, self.vm_max)] * self.nb)
        bounds.extend([(-np.pi, np.pi)] * self.nb)
        ref_idx = 2 * self.ng + self.nq + self.nb + self.ref_bus
        bounds[ref_idx] = (0.0, 0.0)
        return [(float(lo), float(hi)) for lo, hi in bounds]

    def unpack(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float)
        i0 = 0
        i1 = i0 + self.ng
        i2 = i1 + self.ng
        i3 = i2 + self.nq
        i4 = i3 + self.nb
        pg_gen = x[i0:i1]
        qg_gen = x[i1:i2]
        alpha_selected = x[i2:i3]
        vm = x[i3:i4]
        va = x[i4 : i4 + self.nb]
        pg = np.asarray(self.scenario.Pg_base, dtype=float).copy()
        qg = np.asarray(self.scenario.Qg_base, dtype=float).copy()
        alpha = np.ones(self.nb, dtype=float)
        pg[self.generator_buses] = pg_gen
        qg[self.generator_buses] = qg_gen
        alpha[self.selected_load_buses] = alpha_selected
        pd = np.asarray(self.scenario.Pd_base, dtype=float) * alpha
        qd = np.asarray(self.scenario.Qd_base, dtype=float) * alpha
        return {
            "pg_gen": pg_gen,
            "qg_gen": qg_gen,
            "alpha_selected": alpha_selected,
            "pg": pg,
            "qg": qg,
            "alpha": alpha,
            "pd": pd,
            "qd": qd,
            "vm": vm,
            "va": va,
        }

    def voltage(self, x: np.ndarray) -> np.ndarray:
        state = self.unpack(x)
        return state["vm"] * np.exp(1j * state["va"])

    def bus_injections_mva(self, x: np.ndarray) -> np.ndarray:
        v = self.voltage(x)
        return v * np.conj(self.ybus @ v) * float(self.scenario.sn_mva)

    def power_balance(self, x: np.ndarray) -> np.ndarray:
        state = self.unpack(x)
        s_inj = self.bus_injections_mva(x)
        p_mismatch = state["pg"] - state["pd"] - np.real(s_inj)
        q_mismatch = state["qg"] - state["qd"] - np.imag(s_inj)
        return np.hstack([p_mismatch, q_mismatch])

    def branch_flows_mva(self, x: np.ndarray) -> np.ndarray:
        edge_index = self.scenario.edge_index.cpu().numpy() if hasattr(self.scenario.edge_index, "cpu") else np.asarray(self.scenario.edge_index)
        f = edge_index[0, :].astype(int)
        t = edge_index[1, :].astype(int)
        y = np.asarray(self.scenario.G, dtype=float) + 1j * np.asarray(self.scenario.B, dtype=float)
        v = self.voltage(x)
        s_from = v[f] * np.conj((v[f] - v[t]) * y) * float(self.scenario.sn_mva)
        s_to = v[t] * np.conj((v[t] - v[f]) * y) * float(self.scenario.sn_mva)
        return np.maximum(np.abs(s_from), np.abs(s_to))

    def thermal_margin(self, x: np.ndarray) -> np.ndarray:
        return self.rate_a - self.branch_flows_mva(x)

    def loading_ratio(self, x: np.ndarray) -> np.ndarray:
        return self.branch_flows_mva(x) / np.maximum(self.rate_a, 1e-12)

    def line_impacts(self, x: np.ndarray) -> np.ndarray:
        state = self.unpack(x)
        current_service = float(np.sum(state["alpha"]))
        scale = max(current_service, 1e-12)
        edge_index = self.scenario.edge_index.cpu().numpy() if hasattr(self.scenario.edge_index, "cpu") else np.asarray(self.scenario.edge_index)
        impacts = self.wildfire.impact_vector(edge_index.shape[1])
        group_line_ids = sorted({int(line_id) for group in self.wildfire.line_groups for line_id in group.line_ids})
        for line_id in group_line_ids:
            kept = [idx for idx in range(edge_index.shape[1]) if idx != line_id]
            adjacency: dict[int, set[int]] = {i: set() for i in range(self.nb)}
            for idx in kept:
                a = int(edge_index[0, idx])
                b = int(edge_index[1, idx])
                adjacency[a].add(b)
                adjacency[b].add(a)
            gen_set = set(int(bus) for bus in self.generator_buses)
            visited: set[int] = set()
            outage_service = 0.0
            for bus in range(self.nb):
                if bus in visited:
                    continue
                stack = [bus]
                component = []
                has_gen = False
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    component.append(node)
                    has_gen = has_gen or node in gen_set
                    stack.extend(adjacency[node] - visited)
                if has_gen:
                    outage_service += float(np.sum(state["alpha"][component]))
            impacts[line_id] = max(0.0, current_service - outage_service) / scale
        return impacts

    def grouped_risk(self, x: np.ndarray) -> tuple[float, pd.DataFrame, pd.DataFrame]:
        loading = self.loading_ratio(x)
        impact = self.line_impacts(x)
        hazard = self.wildfire.hazard_vector(len(loading))
        line_risk = hazard * loading * loading * impact
        line_rows = [
            {
                "line_id": int(i),
                "loading_ratio": float(loading[i]),
                "hazard": float(hazard[i]),
                "impact": float(impact[i]),
                "risk": float(line_risk[i]),
            }
            for i in range(len(loading))
        ]
        group_rows = []
        total = 0.0
        for group in self.wildfire.line_groups:
            raw = float(np.sum(line_risk[group.line_ids]))
            weighted = float(group.group_weight * raw)
            total += weighted
            group_rows.append(
                {
                    "group_name": group.name,
                    "line_ids": ",".join(str(i) for i in group.line_ids),
                    "group_weight": float(group.group_weight),
                    "raw_group_risk": raw,
                    "weighted_group_risk": weighted,
                }
            )
        return float(total), pd.DataFrame(line_rows), pd.DataFrame(group_rows)

    def load_shedding(self, x: np.ndarray) -> float:
        return float(np.sum(1.0 - self.unpack(x)["alpha"]))


def build_ybus(scenario) -> np.ndarray:
    edge_index = scenario.edge_index.cpu().numpy() if hasattr(scenario.edge_index, "cpu") else np.asarray(scenario.edge_index)
    ybus = np.zeros((scenario.num_buses, scenario.num_buses), dtype=complex)
    y = np.asarray(scenario.G, dtype=float) + 1j * np.asarray(scenario.B, dtype=float)
    for idx in range(edge_index.shape[1]):
        f = int(edge_index[0, idx])
        t = int(edge_index[1, idx])
        ybus[f, f] += y[idx]
        ybus[t, t] += y[idx]
        ybus[f, t] -= y[idx]
        ybus[t, f] -= y[idx]
    return ybus


def build_ac_opf_model(config_path: str | Path) -> tuple[object, ACOPFModel]:
    config = load_first_pass_config(config_path)
    context = load_first_pass_context(config)
    scenario = context.scenario
    if config.decision.selected_load_buses:
        selected_l = np.asarray(config.decision.selected_load_buses, dtype=int)
    else:
        _, selected_l = auto_select_decision_buses(
            scenario,
            config.decision.auto_select_generators,
            config.decision.auto_select_loads,
        )
    generator_buses = np.where(
        (np.asarray(scenario.Pg_base, dtype=float) > 1e-9)
        | np.asarray(scenario.PV_mask, dtype=bool)
        | np.asarray(scenario.REF_mask, dtype=bool),
    )[0].astype(int)
    ref_bus = scenario.get_ref_bus()
    if ref_bus is None:
        ref_bus = int(generator_buses[0])
    ybus = build_ybus(scenario)
    # The local ScenarioData test fixture does not carry IEEE/MATPOWER rateA.
    # Keep this experiment isolated and explicit by using the configured proxy.
    rate_a = np.full(scenario.edge_index.shape[1], config.wildfire.standard_rate_a_mva, dtype=float)
    baseline_loading = np.ones(rate_a.shape, dtype=float)
    wildfire = build_synthetic_wildfire_scenario(
        baseline_loading,
        selected_line_ids=config.wildfire.selected_line_ids,
        selection_method=config.wildfire.selection_method,
        num_high_risk_lines=config.wildfire.num_high_risk_lines,
        high_hazard=config.wildfire.high_hazard,
        default_hazard=config.wildfire.default_hazard,
        default_impact=config.wildfire.default_impact,
        group_weight=config.wildfire.group_weight,
        hazard_multiplier=config.wildfire.hazard_multiplier,
    )
    if config.wildfire.selection_method == "manual_connected":
        validate_connected_line_group(scenario.edge_index, wildfire.line_groups[0].line_ids)
    model = ACOPFModel(
        scenario=scenario,
        selected_load_buses=selected_l,
        generator_buses=generator_buses,
        wildfire=wildfire,
        rate_a=rate_a,
        ybus=ybus,
        ref_bus=int(ref_bus),
    )
    return config, model


def solve_ac_opf_tradeoff(config_path: str | Path, lambda_case: str, lambda_R: float, lambda_L: float, output_dir: Path) -> dict:
    config, model = build_ac_opf_model(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_config_copy(config, output_dir / "config.yaml")
    x0 = model.initial_vector()
    baseline_risk, baseline_line_risk, baseline_group_risk = model.grouped_risk(x0)
    risk_normalizer = max(baseline_risk, 1e-12)
    load_normalizer = float(model.nb)

    def objective(x: np.ndarray) -> float:
        risk, _, _ = model.grouped_risk(x)
        load = model.load_shedding(x)
        return float(lambda_R * (risk / risk_normalizer) + lambda_L * (load / load_normalizer))

    constraints = [
        {"type": "eq", "fun": model.power_balance},
        {"type": "ineq", "fun": model.thermal_margin},
    ]
    result = opt.minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=model.bounds(),
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-8, "disp": False},
    )
    xf = result.x
    final_risk, final_line_risk, final_group_risk = model.grouped_risk(xf)
    baseline_balance = model.power_balance(x0)
    final_balance = model.power_balance(xf)
    baseline_thermal = model.thermal_margin(x0)
    final_thermal = model.thermal_margin(xf)
    rows = []
    for label, x, risk in [("baseline", x0, baseline_risk), ("final", xf, final_risk)]:
        rows.append(
            {
                "point": label,
                "objective": objective(x),
                "grouped_wildfire_risk": float(risk),
                "normalized_wildfire_risk": float(risk / risk_normalizer),
                "load_shedding": model.load_shedding(x),
                "normalized_load_shedding": model.load_shedding(x) / load_normalizer,
                "max_abs_power_balance_mva": float(np.max(np.abs(model.power_balance(x)))),
                "min_thermal_margin_mva": float(np.min(model.thermal_margin(x))),
                "max_loading_ratio": float(np.max(model.loading_ratio(x))),
            }
        )
    summary_frame = pd.DataFrame(rows)
    write_dataframe(output_dir / "summary_points.csv", summary_frame)
    write_dataframe(output_dir / "baseline_line_risk.csv", baseline_line_risk)
    write_dataframe(output_dir / "final_line_risk.csv", final_line_risk)
    write_dataframe(output_dir / "baseline_group_risk.csv", baseline_group_risk)
    write_dataframe(output_dir / "final_group_risk.csv", final_group_risk)
    write_dataframe(output_dir / "risk_by_line_before_after.csv", _line_before_after(baseline_line_risk, final_line_risk))
    write_dataframe(output_dir / "risk_by_group_before_after.csv", _group_before_after(baseline_group_risk, final_group_risk))
    decision_rows = [
        {"decision_type": "delta_pg", "bus_idx": int(bus), "value": float(model.unpack(xf)["pg"][idx] - model.unpack(x0)["pg"][idx])}
        for idx, bus in enumerate(model.generator_buses)
    ]
    decision_rows.extend(
        {"decision_type": "alpha", "bus_idx": int(bus), "value": float(model.unpack(xf)["alpha"][idx])}
        for idx, bus in enumerate(model.selected_load_buses)
    )
    write_dataframe(output_dir / "decision_vector_initial.csv", pd.DataFrame(decision_rows))
    write_dataframe(output_dir / "decision_vector_final.csv", pd.DataFrame(decision_rows))
    write_json(output_dir / "wildfire_scenario.json", model.wildfire.to_dict())
    objective_trace = summary_frame.rename(
        columns={
            "objective": "objective_total",
            "grouped_wildfire_risk": "wildfire_group_risk",
            "normalized_wildfire_risk": "normalized_wildfire_group_risk",
        }
    )
    objective_trace["eval_idx"] = range(len(objective_trace))
    objective_trace["risk_objective_term"] = float(lambda_R) * objective_trace["normalized_wildfire_group_risk"]
    objective_trace["load_shedding_objective_term"] = float(lambda_L) * objective_trace["normalized_load_shedding"]
    objective_trace["generator_movement"] = 0.0
    objective_trace["mean_alpha"] = 1.0
    objective_trace["max_abs_delta_pg"] = 0.0
    objective_trace["message"] = objective_trace["point"].astype(str)
    write_dataframe(output_dir / "objective_trace.csv", objective_trace)
    summary = {
            "stage": "ac_opf",
            "model_type": config.model.model_type.lower(),
            "lambda_case": lambda_case,
            "lambda_R": float(lambda_R),
            "lambda_L": float(lambda_L),
            "success": bool(result.success),
            "message": str(result.message),
            "num_iterations": int(result.nit),
            "objective_baseline": float(objective(x0)),
            "objective_final": float(objective(xf)),
            "risk_baseline": float(baseline_risk),
            "risk_final": float(final_risk),
            "load_shedding_final": model.load_shedding(xf),
            "risk_normalizer": float(risk_normalizer),
            "load_shedding_normalizer": float(load_normalizer),
            "max_abs_power_balance_baseline_mva": float(np.max(np.abs(baseline_balance))),
            "max_abs_power_balance_final_mva": float(np.max(np.abs(final_balance))),
            "min_thermal_margin_baseline_mva": float(np.min(baseline_thermal)),
            "min_thermal_margin_final_mva": float(np.min(final_thermal)),
            "max_loading_final": float(np.max(model.loading_ratio(xf))),
            "generator_buses": model.generator_buses.tolist(),
            "selected_load_buses": model.selected_load_buses.tolist(),
            "note": (
                "Separate hard-constraint AC-OPF experiment over local ScenarioData. "
                "Uses AC P/Q balance, voltage/reference bounds, Pg/Qg/alpha bounds, "
                "and branch apparent-flow limits with configured proxy ratings."
            ),
        }
    write_json(output_dir / "ac_opf_summary.json", summary)
    write_json(output_dir / "optimization_summary.json", summary)
    visualization_summary = {"optimization_behavior_plot": None, "topology_change_plot": None, "errors": []}
    try:
        visualization_summary["optimization_behavior_plot"] = str(plot_optimization_behavior(output_dir))
    except Exception as exc:
        visualization_summary["errors"].append(f"optimization_behavior: {exc}")
    try:
        visualization_summary["topology_change_plot"] = str(plot_network_changes(output_dir))
    except Exception as exc:
        visualization_summary["errors"].append(f"topology_change: {exc}")
    write_json(output_dir / "visualization_summary.json", visualization_summary)
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "objective_baseline": float(objective(x0)),
        "objective_final": float(objective(xf)),
        "risk_baseline": float(baseline_risk),
        "risk_final": float(final_risk),
        "load_shedding_final": model.load_shedding(xf),
        "max_abs_power_balance_final_mva": float(np.max(np.abs(final_balance))),
        "min_thermal_margin_final_mva": float(np.min(final_thermal)),
        "run_dir": str(output_dir),
    }


def plot_ac_opf_summary(summary: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(summary["tradeoff_set"], summary["objective_final"], color="#4c78a8")
    axes[0].set_ylabel("final objective")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(summary["tradeoff_set"], summary["max_abs_power_balance_final_mva"], color="#f58518")
    axes[1].set_ylabel("max |P/Q balance| (MVA)")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def run_ac_opf_tradeoff_sets(
    config_path: str | Path = CONFIGS_ROOT / "connected_corridor_gps.yaml",
    output_root: str | Path | None = None,
) -> Path:
    if output_root is None:
        output_root = RESULTS_ROOT / "analysis" / "ac_opf_connected_corridor"
    output_root = Path(output_root)
    run_root = output_root / f"ac_opf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, weights in TRADEOFF_SETS.items():
        result = solve_ac_opf_tradeoff(
            config_path,
            name,
            weights["lambda_R"],
            weights["lambda_L"],
            run_root / LAMBDA_FOLDERS.get(name, name),
        )
        rows.append({"tradeoff_set": name, **weights, **result})
    summary = pd.DataFrame(rows)
    write_dataframe(run_root / "ac_opf_tradeoff_summary.csv", summary)
    plot_ac_opf_summary(summary, run_root / "ac_opf_tradeoff_summary.png")
    write_json(
        run_root / "methodology.json",
        {
            "references": [
                "MATPOWER AC OPF: power balance equalities, branch flow inequalities, and variable bounds.",
                "PowerModels AC OPF: complex-voltage power balance, generator bounds, voltage bounds, branch thermal limits.",
            ],
            "isolation": "This directory is separable from connected_corridor GridFM results and can be deleted without changing the main first-pass workflow.",
        },
    )
    print(f"[OK] AC-OPF hard-constraint experiment written to {run_root}")
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_ROOT / "connected_corridor_gps.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    run_ac_opf_tradeoff_sets(args.config, args.output_root)


if __name__ == "__main__":
    main()
