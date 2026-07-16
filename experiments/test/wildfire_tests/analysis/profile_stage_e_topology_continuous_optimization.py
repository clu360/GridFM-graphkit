from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import git_metadata
from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import _build_model_context
from experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps import apply_environmental_case
from experiments.test.wildfire_tests.stage_e_gurobi_implementation import physics_infeasibility_evaluator as evaluator


RESULT_ROOT = (
    RESULTS_ROOT
    / "leq"
    / "stage_e"
    / "analysis"
    / "topology_continuous_optimization_profile"
)


def _long_path(path: Path) -> str:
    text = str(path)
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return text
    return "\\\\?\\" + str(path.resolve())


def _make_run_dir() -> Path:
    run_dir = RESULT_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(_long_path(run_dir), exist_ok=True)
    return run_dir


def _write_json(path: Path, data: Dict) -> None:
    os.makedirs(_long_path(path.parent), exist_ok=True)
    with open(_long_path(path), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    os.makedirs(_long_path(path.parent), exist_ok=True)
    frame.to_csv(_long_path(path), index=False)


def _candidate_line_ids(wildfire) -> List[int]:
    return sorted({int(line_id) for group in wildfire.line_groups for line_id in group.line_ids})


def _baseline_exposure(model_context: dict, candidate_line_ids: List[int], p_env_by_line: Dict[int, float]) -> float:
    scenario = model_context["scenario"]
    loading = np.asarray(model_context["baseline_state"]["loading_ratio"], dtype=float)
    z_all_on = {line_id: 1 for line_id in range(int(scenario.edge_index.shape[1]))}
    raw, _by_line = compute_operational_wildfire_exposure(
        loading,
        p_env_by_line,
        z_all_on,
        candidate_line_ids,
    )
    return float(raw)


def profile_topology_continuous_optimization(
    shutoff_line_ids: List[int],
    lambda_R: float = 0.8,
    rho_phys: float = 0.0,
    optimizer_maxiter: int = 10,
) -> Path:
    run_dir = _make_run_dir()
    setup_started = time.perf_counter()
    model_context = _build_model_context("gnn", grouping_top_fraction=0.30)
    wildfire = model_context["wildfire"]
    group_summary = model_context["automatic_artifacts"]["group_summary"]
    env_case = apply_environmental_case(wildfire, group_summary, "auto_env")
    candidate_line_ids = _candidate_line_ids(wildfire)
    baseline_R = _baseline_exposure(model_context, candidate_line_ids, env_case.p_env_by_line)
    setup_seconds = float(time.perf_counter() - setup_started)

    runner = model_context["runner"]
    original_predict = runner.predict
    original_topology_predict = evaluator._predict_topology_state
    predict_seconds: List[float] = []
    topology_pipeline_seconds: List[float] = []

    def timed_predict(u):
        started = time.perf_counter()
        try:
            return original_predict(u)
        finally:
            predict_seconds.append(float(time.perf_counter() - started))

    def timed_topology_predict(context, u, removed):
        started = time.perf_counter()
        try:
            return original_topology_predict(context, u, removed)
        finally:
            topology_pipeline_seconds.append(float(time.perf_counter() - started))

    runner.predict = timed_predict
    evaluator._predict_topology_state = timed_topology_predict
    optimization_started = time.perf_counter()
    try:
        result = evaluator.evaluate_topology_with_physics_recourse(
            model_context=model_context,
            candidate_line_ids=candidate_line_ids,
            p_env_by_line=env_case.p_env_by_line,
            baseline_R_raw=baseline_R,
            shutoff_line_ids=shutoff_line_ids,
            lambda_R=float(lambda_R),
            lambda_L=float(1.0 - float(lambda_R)),
            rho_phys=float(rho_phys),
            stage="stage_e_k2_continuous_profile",
            case_name="auto_env",
            lambda_case=f"lr{float(lambda_R):.2f}".replace(".", "p"),
            eval_id=0,
            proposal_method="fixed_topology_profile",
            model_name="gnn",
            topology_budget=2,
            recourse_maxiter=int(optimizer_maxiter),
            optimize_recourse=True,
        )
    finally:
        optimization_seconds = float(time.perf_counter() - optimization_started)
        runner.predict = original_predict
        evaluator._predict_topology_state = original_topology_predict

    row = result.row
    total_predict_seconds = float(sum(predict_seconds))
    total_pipeline_seconds = float(sum(topology_pipeline_seconds))
    surrounding_pipeline_seconds = max(0.0, total_pipeline_seconds - total_predict_seconds)
    uninstrumented_overhead_seconds = max(0.0, optimization_seconds - total_pipeline_seconds)
    trace = pd.DataFrame(result.trace)
    timing_rows = pd.DataFrame(
        {
            "call_idx": np.arange(1, len(topology_pipeline_seconds) + 1),
            "topology_pipeline_seconds": topology_pipeline_seconds,
            "gridfm_predict_seconds": predict_seconds,
            "topology_state_physics_overhead_seconds": np.asarray(topology_pipeline_seconds)
            - np.asarray(predict_seconds),
        }
    )

    summary = {
        **git_metadata(),
        "study": "stage_e_topology_control_plus_continuous_optimization_profile",
        "purpose": "Runtime diagnosis for proposed Stage E topology control with continuous optimization inside one fixed topology.",
        "case_name": "auto_env",
        "model": "gnn",
        "shutoff_line_ids": [int(value) for value in sorted(shutoff_line_ids)],
        "lambda_R": float(lambda_R),
        "lambda_L": float(1.0 - float(lambda_R)),
        "rho_phys": float(rho_phys),
        "optimizer_maxiter": int(optimizer_maxiter),
        "setup_seconds": setup_seconds,
        "continuous_optimization_seconds": optimization_seconds,
        "gridfm_calls": int(row["gridfm_calls"]),
        "scipy_success": bool(row["scipy_success"]),
        "scipy_message": str(row["scipy_message"]),
        "gridfm_status": str(row["gridfm_status"]),
        "total_gridfm_predict_seconds": total_predict_seconds,
        "mean_gridfm_predict_seconds": float(np.mean(predict_seconds)) if predict_seconds else 0.0,
        "total_topology_pipeline_seconds": total_pipeline_seconds,
        "mean_topology_pipeline_seconds": float(np.mean(topology_pipeline_seconds))
        if topology_pipeline_seconds
        else 0.0,
        "topology_state_physics_overhead_seconds": surrounding_pipeline_seconds,
        "scipy_and_python_overhead_seconds": uninstrumented_overhead_seconds,
        "fixed_or_final_J_true": float(row["J_true"]),
        "R_norm": float(row["R_norm"]),
        "L_shed": float(row["L_shed"]),
        "PAC_total": float(row["PAC_total"]),
        "max_abs_delta_pg": float(row["max_abs_delta_pg"]),
        "mean_alpha": float(row["mean_alpha"]),
        "min_alpha": float(row["min_alpha"]),
        "continuous_recourse_optimized": True,
        "load_shedding_definition": str(row["L_shed_definition"]),
    }

    _write_dataframe(run_dir / "objective_call_trace.csv", trace)
    _write_dataframe(run_dir / "timing_by_gridfm_call.csv", timing_rows)
    _write_dataframe(run_dir / "final_evaluation_row.csv", pd.DataFrame([row]))
    _write_json(run_dir / "profile_summary.json", summary)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile proposed Stage E topology control plus continuous optimization for one fixed topology."
    )
    parser.add_argument("--shutoff-line-ids", nargs="+", type=int, default=[18, 23])
    parser.add_argument("--lambda-r", type=float, default=0.8)
    parser.add_argument("--rho-phys", type=float, default=0.0)
    parser.add_argument("--optimizer-maxiter", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = profile_topology_continuous_optimization(
        shutoff_line_ids=args.shutoff_line_ids,
        lambda_R=args.lambda_r,
        rho_phys=args.rho_phys,
        optimizer_maxiter=args.optimizer_maxiter,
    )
    print(f"Wrote Stage E topology-plus-continuous profile to {run_dir}")


if __name__ == "__main__":
    main()
