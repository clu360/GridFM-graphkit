from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .reporting import write_dataframe, write_json
from .wildfire_risk import compute_counterfactual_line_impacts
from .wildfire_scenario import (
    WildfireLineGroup,
    WildfireScenario,
    build_automatic_risk_component_scenario,
    build_synthetic_wildfire_scenario,
    validate_connected_line_group,
)


def _num_lines(edge_index) -> int:
    return int(edge_index.shape[1])


def _candidate_p_env(config) -> float:
    risk_score = config.wildfire.risk_score or {}
    if "candidate_p_env" in risk_score:
        return float(risk_score["candidate_p_env"])
    if "uniform_p_env" in risk_score:
        return float(risk_score["uniform_p_env"])
    return float(config.wildfire.high_hazard * config.wildfire.hazard_multiplier)


def _top_fraction(config) -> float:
    risk_score = config.wildfire.risk_score or {}
    return float(risk_score.get("top_fraction", 0.15))


def _all_line_scoring_scenario(config, num_lines: int, candidate_p_env: float) -> WildfireScenario:
    return WildfireScenario(
        name="automatic_risk_components_scoring",
        line_groups=[
            WildfireLineGroup(
                name="all_candidate_lines",
                line_ids=list(range(num_lines)),
                group_weight=1.0,
                description="Temporary all-line group used only for automatic risk scoring.",
            )
        ],
        hazard_by_line={int(line_id): float(candidate_p_env) for line_id in range(num_lines)},
        impact_by_line={int(line_id): float(config.wildfire.default_impact) for line_id in range(num_lines)},
        default_hazard=float(candidate_p_env),
        default_impact=float(config.wildfire.default_impact),
    )


def build_wildfire_for_baseline(config, scenario, runner, decision_vector, baseline_prediction, baseline_state):
    selection_method = config.wildfire.selection_method
    if selection_method == "automatic_risk_components":
        num_lines = _num_lines(scenario.edge_index)
        candidate_p_env = _candidate_p_env(config)
        scoring_wildfire = _all_line_scoring_scenario(config, num_lines, candidate_p_env)
        baseline_line_impact = compute_counterfactual_line_impacts(
            decision_vector.u_base,
            scenario,
            runner,
            scoring_wildfire,
            baseline_prediction,
        )
        wildfire, line_scores, group_summary, metadata = build_automatic_risk_component_scenario(
            scenario.edge_index,
            baseline_state["loading_ratio"],
            baseline_line_impact,
            top_fraction=_top_fraction(config),
            candidate_hazard=candidate_p_env,
            default_hazard=float(candidate_p_env),
            default_impact=config.wildfire.default_impact,
            group_weight=1.0,
        )
        return wildfire, baseline_line_impact, {
            "line_scores": line_scores,
            "group_summary": group_summary,
            "metadata": metadata,
        }

    wildfire = build_synthetic_wildfire_scenario(
        baseline_state["loading_ratio"],
        selected_line_ids=config.wildfire.selected_line_ids,
        selection_method=config.wildfire.selection_method,
        num_high_risk_lines=config.wildfire.num_high_risk_lines,
        high_hazard=config.wildfire.high_hazard,
        default_hazard=config.wildfire.default_hazard,
        default_impact=config.wildfire.default_impact,
        group_weight=config.wildfire.group_weight,
        hazard_multiplier=config.wildfire.hazard_multiplier,
    )
    if selection_method == "manual_connected":
        validate_connected_line_group(scenario.edge_index, wildfire.line_groups[0].line_ids)
    baseline_line_impact = compute_counterfactual_line_impacts(
        decision_vector.u_base,
        scenario,
        runner,
        wildfire,
        baseline_prediction,
    )
    return wildfire, baseline_line_impact, None


def automatic_diagnostics(metadata: Dict, baseline_grouped_risk: float, config) -> list[str]:
    warnings = []
    diagnostics = config.wildfire.diagnostics or {}
    largest_threshold = float(diagnostics.get("warn_if_largest_group_fraction_above", 0.80))
    if int(metadata.get("num_selected_lines", 0)) <= 0:
        warnings.append("no_lines_selected")
    if int(metadata.get("num_groups", 0)) == 1 and diagnostics.get("warn_if_single_group", True):
        warnings.append("collapsed_to_single_group")
    if float(metadata.get("largest_group_fraction_of_selected_lines", 0.0)) > largest_threshold:
        warnings.append("largest_group_fraction_above_threshold")
    if not np.isfinite(float(baseline_grouped_risk)) or float(baseline_grouped_risk) <= 0.0:
        warnings.append("baseline_grouped_risk_zero_or_invalid")
    return warnings


def write_automatic_group_artifacts(run_dir: Path, artifacts: Dict, baseline_grouped_risk: float, config) -> Dict:
    if artifacts is None:
        return {}
    metadata = dict(artifacts["metadata"])
    metadata["baseline_grouped_risk"] = float(baseline_grouped_risk)
    metadata["diagnostic_warnings"] = automatic_diagnostics(metadata, baseline_grouped_risk, config)

    line_scores = artifacts["line_scores"].copy()
    group_summary = artifacts["group_summary"].copy()
    for key in ["requested_top_fraction", "num_selected_lines", "realized_selected_fraction"]:
        line_scores[key] = metadata[key]
        group_summary[key] = metadata[key]

    write_dataframe(run_dir / "automatic_line_risk_scores.csv", line_scores)
    write_dataframe(run_dir / "automatic_group_summary.csv", group_summary)
    write_json(run_dir / "automatic_wildfire_groups.json", metadata)
    return metadata


def automatic_visualization_summary_fields(metadata: Dict) -> Dict:
    if not metadata:
        return {}
    return {
        "selection_method": metadata.get("selection_method"),
        "top_fraction": metadata.get("top_fraction"),
        "requested_top_fraction": metadata.get("requested_top_fraction"),
        "realized_selected_fraction": metadata.get("realized_selected_fraction"),
        "selected_high_risk_line_ids": metadata.get("selected_high_risk_line_ids"),
        "num_selected_lines": metadata.get("num_selected_lines"),
        "num_groups": metadata.get("num_groups"),
        "group_ids": metadata.get("group_ids"),
        "group_line_ids": metadata.get("group_line_ids"),
        "group_bus_ids": metadata.get("group_bus_ids"),
        "collapsed_to_single_group": metadata.get("collapsed_to_single_group"),
        "largest_group_num_lines": metadata.get("largest_group_num_lines"),
        "largest_group_fraction_of_selected_lines": metadata.get(
            "largest_group_fraction_of_selected_lines"
        ),
    }


def load_automatic_group_metadata(run_dir: Path) -> Dict:
    path = run_dir / "automatic_wildfire_groups.json"
    if not path.exists():
        return {}
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
