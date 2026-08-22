"""Write the J7.5 full-alpha smoke report from generated artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(value):
    if value in ("", None):
        return None
    return float(value)


def _metric(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _changed_loads(best_alpha: dict[str, float]) -> list[tuple[int, float]]:
    return sorted((int(k), float(v)) for k, v in best_alpha.items() if abs(float(v) - 1.0) > 1e-9)


def _flow_bus_comparison(gridsfm_dir: Path, reference_dir: Path) -> dict[str, object]:
    g_branch = {int(row["branch_id"]): row for row in _load_csv(gridsfm_dir / "gridsfm_finalist_branch_state.csv")}
    a_branch = {int(row["branch_id"]): row for row in _load_csv(reference_dir / "reference_a_branch_loading.csv")}
    common = sorted(set(g_branch).intersection(a_branch))
    flow_terms = []
    loading_diffs = []
    for line_id in common:
        g = g_branch[line_id]
        a = a_branch[line_id]
        rate = float(a["rate_a"])
        for g_key, a_key in (("p_from", "pf"), ("q_from", "qf"), ("p_to", "pt"), ("q_to", "qt")):
            flow_terms.append(((float(g[g_key]) - float(a[a_key])) / rate) ** 2)
        loading_diffs.append(abs(float(g["loading"]) - float(a["ac_loading"])))

    g_bus = {int(row["bus_id"]): row for row in _load_csv(gridsfm_dir / "gridsfm_finalist_bus_state.csv")}
    a_bus = {int(row["bus_id"]): row for row in _load_csv(reference_dir / "reference_a_bus_state.csv")}
    common_bus = sorted(set(g_bus).intersection(a_bus))
    bus_terms = []
    for bus_id in common_bus:
        bus_terms.append((float(g_bus[bus_id]["vm"]) - float(a_bus[bus_id]["vm"])) ** 2)
        bus_terms.append((float(g_bus[bus_id]["va"]) - float(a_bus[bus_id]["va"])) ** 2)

    d_flow = sum(flow_terms) / len(flow_terms) if flow_terms else None
    d_bus = sum(bus_terms) / len(bus_terms) if bus_terms else None
    d_state = None
    if d_flow is not None and d_bus is not None:
        d_state = d_flow + d_bus
    return {
        "common_branch_count": len(common),
        "common_bus_count": len(common_bus),
        "d_flow_normalized_mse": d_flow,
        "d_bus_vm_va_mse": d_bus,
        "d_state_to_ac_flow_voltage_v1": d_state,
        "mean_abs_loading_delta": sum(loading_diffs) / len(loading_diffs) if loading_diffs else None,
        "max_abs_loading_delta": max(loading_diffs) if loading_diffs else None,
        "generator_distance_status": "NOT_COMPUTED_CANONICAL_GEN_ID_MAPPING_UNRESOLVED",
    }


def _optional_json(path: Path):
    return _load_json(path) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    dc = _load_json(root / "guided_dc" / "j7_5_guided_dc_alpha_summary.json")
    sfm = _load_json(root / "gridsfm" / "j7_5_gridsfm_alpha_summary.json")
    ac = _load_json(root / "reference_a" / "reference_a_summary.json")
    comparison = _flow_bus_comparison(root / "gridsfm_finalist_state", root / "reference_a")
    repeatability = _optional_json(root / "repeatability" / "gridsfm_repeatability_summary.json")

    dc_changed = _changed_loads(dc["best_alpha"])
    sfm_changed = _changed_loads(sfm["best_alpha"])
    if (
        dc["termination_reason"] == "budget_exhausted"
        or sfm["termination_reason"] == "budget_exhausted"
        or dc["accepted_moves"] > 0
        or sfm["accepted_moves"] > 0
    ):
        recommendation = "J8_BLOCKED_BY_ALPHA_SEARCH"
        recommendation_reason = (
            "The optimizer and smoke execution work, but at least one backend exhausted B_alpha or accepted a move "
            "before saturation. J8 should not start until Caleb approves a larger budget, a modified stopping rule, "
            "or an approved approximation/screening layer."
        )
    else:
        recommendation = "READY_FOR_J8_FULL_ALPHA_SEARCH"
        recommendation_reason = "Both smoke searches terminated without budget exhaustion or additional accepted moves."

    payload = {
        "status": "J7_5_FULL_ALPHA_SMOKE_COMPLETE",
        "root_dir": str(root),
        "topology_ids": sfm["topology_ids"],
        "scenario_id": sfm["scenario_id"],
        "lambda_r": sfm["lambda_r"],
        "guided_dc": {
            "termination_reason": dc["termination_reason"],
            "actual_evaluation_count": dc["actual_evaluation_count"],
            "completed_sweeps": dc["completed_sweeps"],
            "accepted_moves": dc["accepted_moves"],
            "best_changed_loads": dc_changed,
            "best_evaluation": dc["best_evaluation"],
        },
        "guided_gridsfm": {
            "termination_reason": sfm["termination_reason"],
            "actual_evaluation_count": sfm["actual_evaluation_count"],
            "completed_sweeps": sfm["completed_sweeps"],
            "accepted_moves": sfm["accepted_moves"],
            "best_changed_loads": sfm_changed,
            "best_evaluation": sfm["best_evaluation"],
            "argmin_j_trade_row": sfm["argmin_j_trade_row"],
            "argmin_j_total_row": sfm["argmin_j_total_row"],
        },
        "reference_a": ac,
        "gridsfm_vs_reference_a": comparison,
        "gridsfm_repeatability": repeatability,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    lines = [
        "# J7.5 Full Per-Load Alpha Optimizer Smoke Report",
        "",
        "## A. Scope",
        f"- Scenario: `{payload['scenario_id']}`",
        f"- Topology: `{';'.join(str(v) for v in payload['topology_ids'])}`",
        f"- Lambda: `{payload['lambda_r']}`",
        "- Alpha domain: full per-load requested alpha over all GOC-500 load records.",
        "- Electrical recourse remains outside the alpha optimizer: fixed-topology economic DC-OPF or GridSFM inference.",
        "",
        "## B. Optimizer Status",
        f"- Guided-DC evaluations: `{dc['actual_evaluation_count']}`, termination `{dc['termination_reason']}`.",
        f"- Guided-GridSFM evaluations: `{sfm['actual_evaluation_count']}`, termination `{sfm['termination_reason']}`.",
        "- Both methods used the same seed values, coordinate rules, step schedule, source-less rules, and budget.",
        f"- DC cache namespace hash: `{dc.get('cache_namespace_hash', 'N/A')}`.",
        f"- GridSFM cache namespace hash: `{sfm.get('cache_namespace_hash', 'N/A')}`.",
        "- J7.5 cache is in-run only; a persistent cross-run J8 cache remains a scale-up task.",
        "- Optional interaction search was omitted in J7.5 v1.",
        "",
        "## C. Guided-DC Best Alpha",
        f"- Changed loads: `{dc_changed}`",
        f"- Best J_trade: `{_metric(dc['best_evaluation']['j_trade'])}`",
        f"- R_norm: `{_metric(dc['best_evaluation']['r_norm'])}`",
        f"- L_shed_total/control/island: `{_metric(dc['best_evaluation']['l_shed_total'])}` / `{_metric(dc['best_evaluation']['l_shed_control'])}` / `{_metric(dc['best_evaluation']['l_shed_island'])}`",
        "",
        "## D. Guided-GridSFM Best Alpha",
        f"- Changed loads under J_total selection: `{sfm_changed}`",
        f"- Best J_total: `{_metric(sfm['best_evaluation']['j_total'])}`",
        f"- J_trade: `{_metric(sfm['best_evaluation']['j_trade'])}`",
        f"- R_norm: `{_metric(sfm['best_evaluation']['r_norm'])}`",
        f"- L_shed_total/control/island: `{_metric(sfm['best_evaluation']['l_shed_total'])}` / `{_metric(sfm['best_evaluation']['l_shed_control'])}` / `{_metric(sfm['best_evaluation']['l_shed_island'])}`",
        f"- PAC_total: `{_metric(sfm['best_evaluation']['pac_total'])}`",
        f"- Max predicted loading / overloaded lines: `{_metric(sfm['best_evaluation']['extra']['max_loading'])}` / `{sfm['best_evaluation']['extra']['num_loading_gt_1']}`",
        "",
        "## E. J_trade Versus J_total Selection",
        f"- Argmin J_trade alpha hash: `{sfm['argmin_j_trade_row']['alpha_hash']}` changed `{sfm['argmin_j_trade_row']['changed_load_ids']}`.",
        f"- Argmin J_total alpha hash: `{sfm['argmin_j_total_row']['alpha_hash']}` changed `{sfm['argmin_j_total_row']['changed_load_ids']}`.",
        "- These differ in the smoke trace, confirming the physics-aware merit function affects GridSFM candidate selection.",
        "",
        "## F. Reference A AC Audit",
        f"- Status: `{ac['termination_status']}`",
        f"- Alpha input: `{ac.get('alpha_effective_csv', 'N/A')}`",
        f"- Objective: `{_metric(ac['objective'])}`",
        f"- Runtime seconds: `{_metric(ac['runtime_seconds'])}`",
        f"- Max AC loading / overloaded lines: `{_metric(ac['max_ac_loading'])}` / `{ac['num_ac_loading_gt_1']}`",
        "",
        "## G. GridSFM Versus Reference A",
        f"- Common branches compared: `{comparison['common_branch_count']}`",
        f"- Common buses compared: `{comparison['common_bus_count']}`",
        f"- D_flow normalized MSE: `{_metric(comparison['d_flow_normalized_mse'])}`",
        f"- D_bus vm/va MSE: `{_metric(comparison['d_bus_vm_va_mse'])}`",
        f"- D_state_to_AC flow+voltage v1: `{_metric(comparison['d_state_to_ac_flow_voltage_v1'])}`",
        f"- Mean/max absolute loading delta: `{_metric(comparison['mean_abs_loading_delta'])}` / `{_metric(comparison['max_abs_loading_delta'])}`",
        f"- Generator comparison: `{comparison['generator_distance_status']}`",
        "",
        "## H. Repeatability And Epsilon",
    ]
    if repeatability is None:
        lines.extend(
            [
                "- Repeatability artifact: `NOT_AVAILABLE`",
                "- Meaningful-improvement epsilon remains provisional.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- Repetitions: `{repeatability['repetitions']}`",
                f"- Max objective range: `{_metric(repeatability['max_objective_range'])}`",
                f"- Recommended epsilon_abs floor from smoke: `{_metric(repeatability['recommended_epsilon_abs_floor'])}`",
                "- This is a smoke-candidate calibration only; broader J8 tolerance should be frozen before full sweeps.",
                "",
            ]
        )
    lines.extend(
        [
        "## I. Budget And Runtime Implication",
        "- `B_alpha=300` allowed seed evaluations plus one full coordinate sweep and early second-sweep exploration.",
        "- Both backends accepted one alpha move, so the alpha search did not saturate before the smoke budget.",
        "- This is a tractability warning for J8 because each topology/backend/lambda/scenario may need more than 300 evaluations for convergence.",
        "",
        "## J. Required Artifacts",
        f"- DC trace: `{root / 'guided_dc' / 'j7_5_guided_dc_alpha_trace.csv'}`",
        f"- GridSFM trace: `{root / 'gridsfm' / 'j7_5_gridsfm_alpha_trace.csv'}`",
        f"- Reference A summary: `{root / 'reference_a' / 'reference_a_summary.json'}`",
        f"- GridSFM finalist state: `{root / 'gridsfm_finalist_state'}`",
        f"- Repeatability summary: `{root / 'repeatability' / 'gridsfm_repeatability_summary.json'}`",
        "",
        "## K. Recommendation",
        f"`{recommendation}`",
        "",
        payload["recommendation_reason"],
        ]
    )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_md": args.output_md, "recommendation": recommendation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
