from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_initial_tests.config import load_first_pass_config, write_config_copy
from experiments.test.wildfire_initial_tests.decision_vector import (
    FirstPassDecisionVector,
    auto_select_decision_buses,
)
from experiments.test.wildfire_initial_tests.gridfm_runner import GridFMRunner, load_gridfm_model
from experiments.test.wildfire_initial_tests.objective import compute_first_pass_objective_components
from experiments.test.wildfire_initial_tests.reporting import write_dataframe, write_json
from experiments.test.wildfire_initial_tests.scenario import load_first_pass_context
from experiments.test.wildfire_initial_tests.state_extraction import extract_state_quantities
from experiments.test.wildfire_initial_tests.wildfire_risk import (
    compute_counterfactual_line_impacts,
    compute_grouped_wildfire_risk,
)
from experiments.test.wildfire_initial_tests.wildfire_scenario import (
    build_synthetic_wildfire_scenario,
    validate_connected_line_group,
)


def _build_decision_vector(config, scenario) -> FirstPassDecisionVector:
    if config.decision.selected_generator_buses or config.decision.selected_load_buses:
        selected_g = np.asarray(config.decision.selected_generator_buses, dtype=int)
        selected_l = np.asarray(config.decision.selected_load_buses, dtype=int)
    else:
        selected_g, selected_l = auto_select_decision_buses(
            scenario,
            config.decision.auto_select_generators,
            config.decision.auto_select_loads,
        )
    return FirstPassDecisionVector(
        scenario,
        selected_g,
        selected_l,
        delta_pg_bound_mw=config.decision.delta_pg_bound_mw,
        alpha_min=config.decision.alpha_min,
        alpha_max=config.decision.alpha_max,
    )


def build_line_impact_objective_sweep(
    u: np.ndarray,
    decision_vector: FirstPassDecisionVector,
    state: dict,
    wildfire,
    base_line_impact: np.ndarray,
    line_id: int,
    impact_values: Sequence[float],
    lambda_R: float,
    lambda_L: float,
    risk_normalizer: float,
    load_shedding_normalizer: float,
    normalize_terms: bool = True,
) -> pd.DataFrame:
    line_id = int(line_id)
    base_line_impact = np.asarray(base_line_impact, dtype=float)
    if line_id < 0 or line_id >= len(base_line_impact):
        raise ValueError(f"line_id={line_id} is invalid for {len(base_line_impact)} line impacts.")

    rows = []
    for impact in impact_values:
        line_impact = base_line_impact.copy()
        line_impact[line_id] = float(impact)
        _, components = compute_first_pass_objective_components(
            u,
            decision_vector,
            state,
            wildfire,
            lambda_R,
            lambda_L,
            normalize_terms=normalize_terms,
            risk_normalizer=risk_normalizer,
            load_shedding_normalizer=load_shedding_normalizer,
            line_impact=line_impact,
        )
        rows.append(
            {
                "line_id": line_id,
                "overridden_impact": float(impact),
                "wildfire_group_risk": float(components["wildfire_group_risk"]),
                "normalized_wildfire_group_risk": float(components["normalized_wildfire_group_risk"]),
                "load_shedding": float(components["load_shedding"]),
                "normalized_load_shedding": float(components["normalized_load_shedding"]),
                "objective_total": float(components["objective_total"]),
                "risk_objective_term": float(components["risk_objective_term"]),
                "load_shedding_objective_term": float(components["load_shedding_objective_term"]),
            }
        )
    return pd.DataFrame(rows)


def plot_line_impact_objective_sweep(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame["overridden_impact"], frame["objective_total"], color="#252525", linewidth=1.8)
    ax.set_xlabel("overridden consequence I_l(u)")
    ax.set_ylabel("objective J(u)")
    ax.set_title("Frozen objective sensitivity to one line consequence")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_decision_variable_sweeps(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variables = list(frame["decision_label"].drop_duplicates())
    n_rows = max(len(variables), 1)
    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3.2 * n_rows), squeeze=False)
    for ax, label in zip(axes[:, 0], variables):
        sub = frame[frame["decision_label"] == label]
        ax.plot(sub["decision_value"], sub["objective_total"], color="#252525", linewidth=1.6, label="objective")
        ax.set_ylabel("J(u)")
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
        twin = ax.twinx()
        twin.plot(sub["decision_value"], sub["wildfire_group_risk"], color="#c43c39", linewidth=1.2, alpha=0.8, label="risk")
        twin.set_ylabel("R_group")
    axes[-1, 0].set_xlabel("decision value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _prepare_single_evaluation(config_path: Path):
    config = load_first_pass_config(config_path)
    context = load_first_pass_context(config)
    scenario = context.scenario
    decision_vector = _build_decision_vector(config, scenario)
    model = load_gridfm_model(config, context)
    runner = GridFMRunner(
        model,
        config.model.model_type,
        scenario,
        decision_vector,
        device=config.model.device,
    )
    base_prediction = runner.predict(decision_vector.u_base)
    base_state = extract_state_quantities(
        scenario,
        base_prediction,
        standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
    )
    wildfire = build_synthetic_wildfire_scenario(
        base_state["loading_ratio"],
        selected_line_ids=config.wildfire.selected_line_ids,
        selection_method=config.wildfire.selection_method,
        num_high_risk_lines=config.wildfire.num_high_risk_lines,
        high_hazard=config.wildfire.high_hazard,
        default_hazard=config.wildfire.default_hazard,
        group_weight=config.wildfire.group_weight,
        hazard_multiplier=config.wildfire.hazard_multiplier,
    )
    if config.wildfire.selection_method == "manual_connected":
        validate_connected_line_group(scenario.edge_index, wildfire.line_groups[0].line_ids)
    base_line_impact = compute_counterfactual_line_impacts(
        decision_vector.u_base,
        scenario,
        runner,
        wildfire,
        base_prediction,
    )
    base_risk, _, _ = compute_grouped_wildfire_risk(
        base_state["loading_ratio"],
        wildfire,
        impact=base_line_impact,
    )
    risk_normalizer = config.objective.risk_normalizer
    if config.objective.normalize_terms and risk_normalizer <= 0.0:
        risk_normalizer = float(max(base_risk, 1e-12))
    load_shedding_normalizer = config.objective.load_shedding_normalizer
    if config.objective.normalize_terms and load_shedding_normalizer <= 0.0:
        load_shedding_normalizer = float(max(scenario.num_buses, 1e-12))
    return config, scenario, decision_vector, runner, wildfire, risk_normalizer, load_shedding_normalizer, base_risk


def evaluate_decision_candidate(
    u: np.ndarray,
    scenario,
    decision_vector: FirstPassDecisionVector,
    runner: GridFMRunner,
    wildfire,
    config,
    risk_normalizer: float,
    load_shedding_normalizer: float,
) -> dict:
    prediction = runner.predict(u)
    state = extract_state_quantities(
        scenario,
        prediction,
        standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
    )
    line_impact = compute_counterfactual_line_impacts(
        u,
        scenario,
        runner,
        wildfire,
        prediction,
    )
    _, components = compute_first_pass_objective_components(
        u,
        decision_vector,
        state,
        wildfire,
        config.objective.lambda_R,
        config.objective.lambda_L,
        normalize_terms=config.objective.normalize_terms,
        risk_normalizer=risk_normalizer,
        load_shedding_normalizer=load_shedding_normalizer,
        line_impact=line_impact,
    )
    return {
        "objective_total": float(components["objective_total"]),
        "wildfire_group_risk": float(components["wildfire_group_risk"]),
        "load_shedding": float(components["load_shedding"]),
        "normalized_wildfire_group_risk": float(components["normalized_wildfire_group_risk"]),
        "normalized_load_shedding": float(components["normalized_load_shedding"]),
        "risk_objective_term": float(components["risk_objective_term"]),
        "load_shedding_objective_term": float(components["load_shedding_objective_term"]),
        "generator_movement": float(components["generator_movement"]),
        "max_loading_ratio": float(components["max_loading_ratio"]),
        "max_voltage": float(components["max_voltage"]),
        "min_voltage": float(components["min_voltage"]),
        "num_nan": int(components["num_nan"]),
        "num_inf": int(components["num_inf"]),
        "corridor_impacts": {
            str(line_id): float(line_impact[int(line_id)])
            for group in wildfire.line_groups
            for line_id in group.line_ids
        },
    }


def run_decision_variable_objective_sweeps(
    config_path: str | Path,
    num_points: int = 21,
    output_root: str | Path | None = None,
) -> Path:
    config_path = Path(config_path)
    (
        config,
        scenario,
        decision_vector,
        runner,
        wildfire,
        risk_normalizer,
        load_shedding_normalizer,
        base_risk,
    ) = _prepare_single_evaluation(config_path)

    if output_root is None:
        output_root = REPO_ROOT / "experiments" / "test" / "wildfire_initial_tests" / "results" / "objective_analysis" / "decision_sweeps"
    output_root = Path(output_root)
    run_dir = output_root / f"decision_sweeps_{config.model.model_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_copy(config, run_dir / "config.yaml")

    rows = []
    u_base = decision_vector.u_base.copy()
    metadata = decision_vector.metadata_frame(u_base)
    for _, meta in metadata.iterrows():
        idx = int(meta["decision_index"])
        values = np.linspace(float(meta["lower_bound"]), float(meta["upper_bound"]), int(num_points))
        for value in values:
            u = u_base.copy()
            u[idx] = float(value)
            metrics = evaluate_decision_candidate(
                u,
                scenario,
                decision_vector,
                runner,
                wildfire,
                config,
                risk_normalizer,
                load_shedding_normalizer,
            )
            delta_pg, alpha = decision_vector.split_decision_vector(u)
            row = {
                "decision_index": idx,
                "decision_type": meta["decision_type"],
                "bus_idx": int(meta["bus_idx"]),
                "decision_label": f"{meta['decision_type']}_{idx}_bus{int(meta['bus_idx'])}",
                "decision_value": float(value),
                "mean_alpha": float(np.mean(alpha)) if len(alpha) else 1.0,
                "max_abs_delta_pg": float(np.max(np.abs(delta_pg))) if len(delta_pg) else 0.0,
            }
            row.update({k: v for k, v in metrics.items() if k != "corridor_impacts"})
            for line_id, impact in metrics["corridor_impacts"].items():
                row[f"impact_line_{line_id}"] = impact
            rows.append(row)

    frame = pd.DataFrame(rows)
    write_dataframe(run_dir / "decision_variable_sweeps.csv", frame)
    plot_path = plot_decision_variable_sweeps(frame, run_dir / "decision_variable_sweeps.png")
    write_json(
        run_dir / "analysis_summary.json",
        {
            "analysis_type": "decision_variable_sweeps",
            "config_path": str(config_path),
            "model_type": config.model.model_type.lower(),
            "num_points_per_variable": int(num_points),
            "num_decision_variables": int(decision_vector.n_total),
            "risk_normalizer": float(risk_normalizer),
            "load_shedding_normalizer": float(load_shedding_normalizer),
            "baseline_grouped_wildfire_risk": float(base_risk),
            "lambda_R": float(config.objective.lambda_R),
            "lambda_L": float(config.objective.lambda_L),
            "sweep_csv": str(run_dir / "decision_variable_sweeps.csv"),
            "sweep_plot": str(plot_path),
        },
    )
    print(f"[OK] Decision-variable objective analysis written to {run_dir}")
    return run_dir


def run_line_impact_objective_sweep(
    config_path: str | Path,
    line_id: int = 23,
    impact_values: Sequence[float] | None = None,
    output_root: str | Path | None = None,
) -> Path:
    config_path = Path(config_path)
    config = load_first_pass_config(config_path)
    if impact_values is None:
        impact_values = np.linspace(0.0, 1.0, 101)
    else:
        impact_values = np.asarray(list(impact_values), dtype=float)

    if output_root is None:
        output_root = REPO_ROOT / "experiments" / "test" / "wildfire_initial_tests" / "results" / "objective_analysis"
    output_root = Path(output_root)
    run_dir = output_root / f"line_{int(line_id)}_impact_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_copy(config, run_dir / "config.yaml")

    context = load_first_pass_context(config)
    scenario = context.scenario
    decision_vector = _build_decision_vector(config, scenario)
    model = load_gridfm_model(config, context)
    runner = GridFMRunner(
        model,
        config.model.model_type,
        scenario,
        decision_vector,
        device=config.model.device,
    )

    u = decision_vector.u_base.copy()
    prediction = runner.predict(u)
    state = extract_state_quantities(
        scenario,
        prediction,
        standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
    )
    wildfire = build_synthetic_wildfire_scenario(
        state["loading_ratio"],
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

    frozen_line_impact = compute_counterfactual_line_impacts(
        u,
        scenario,
        runner,
        wildfire,
        prediction,
    )
    frozen_risk, frozen_line_risk, frozen_group_risk = compute_grouped_wildfire_risk(
        state["loading_ratio"],
        wildfire,
        impact=frozen_line_impact,
    )
    risk_normalizer = config.objective.risk_normalizer
    if config.objective.normalize_terms and risk_normalizer <= 0.0:
        risk_normalizer = float(max(frozen_risk, 1e-12))
    load_shedding_normalizer = config.objective.load_shedding_normalizer
    if config.objective.normalize_terms and load_shedding_normalizer <= 0.0:
        load_shedding_normalizer = float(max(scenario.num_buses, 1e-12))

    frame = build_line_impact_objective_sweep(
        u,
        decision_vector,
        state,
        wildfire,
        frozen_line_impact,
        line_id,
        impact_values,
        config.objective.lambda_R,
        config.objective.lambda_L,
        risk_normalizer,
        load_shedding_normalizer,
        normalize_terms=config.objective.normalize_terms,
    )

    write_dataframe(run_dir / "objective_vs_line_impact.csv", frame)
    write_dataframe(run_dir / "frozen_line_risk.csv", frozen_line_risk)
    write_dataframe(run_dir / "frozen_group_risk.csv", frozen_group_risk)
    plot_path = plot_line_impact_objective_sweep(frame, run_dir / "objective_vs_line_impact.png")
    write_json(
        run_dir / "analysis_summary.json",
        {
            "config_path": str(config_path),
            "model_type": config.model.model_type.lower(),
            "line_id": int(line_id),
            "frozen_decision": decision_vector.metadata_frame(u).to_dict(orient="records"),
            "frozen_grouped_wildfire_risk": float(frozen_risk),
            "risk_normalizer": float(risk_normalizer),
            "load_shedding_normalizer": float(load_shedding_normalizer),
            "lambda_R": float(config.objective.lambda_R),
            "lambda_L": float(config.objective.lambda_L),
            "base_line_impact": float(frozen_line_impact[int(line_id)]),
            "impact_min": float(np.min(impact_values)),
            "impact_max": float(np.max(impact_values)),
            "num_points": int(len(impact_values)),
            "sweep_csv": str(run_dir / "objective_vs_line_impact.csv"),
            "sweep_plot": str(plot_path),
        },
    )
    print(f"[OK] Objective analysis written to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["impact", "decision"], default="impact")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/test/wildfire_initial_tests/configs/connected_corridor_gps.yaml"),
    )
    parser.add_argument("--line-id", type=int, default=23)
    parser.add_argument("--num-points", type=int, default=101)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "impact":
        values = np.linspace(0.0, 1.0, args.num_points)
        run_line_impact_objective_sweep(
            args.config,
            line_id=args.line_id,
            impact_values=values,
            output_root=args.output_root,
        )
    else:
        run_decision_variable_objective_sweeps(
            args.config,
            num_points=args.num_points,
            output_root=args.output_root,
        )


if __name__ == "__main__":
    main()
