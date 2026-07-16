from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import write_dataframe, write_json
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import MODEL_CONFIGS
from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import get_scenarios
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_physics_decision_quality import (
    _canonicalize_decision_quality_scenario,
    _expected_vs_observed,
    _plot_scenario_outputs,
    run_stage_f_physics_decision_quality,
)


RESULT_ROOT = RESULTS_ROOT / "leq" / "stage_g" / "matpower_30"
WITH_PHYSICS_ROOT = RESULT_ROOT / "with_physics_infeasibility" / "rho100"
WITHOUT_PHYSICS_ROOT = RESULT_ROOT / "without_physics_infeasibility" / "rho0"

# Canonical physical branch IDs from the Stage G loading audit whose GridFM-predicted
# loadings are dominated by the current model-state limitation rather than baseline physics.
DEFAULT_GRIDFM_HEAVY_LOADING_LINE_IDS = (23, 26, 40, 41, 69, 72, 77, 81, 84)
DEFAULT_SUPPRESSED_P_ENV = 0.005


def run_stage_g_matpower_30_decision_quality(
    models: List[str] | None = None,
    scenario_ids: List[str] | None = None,
    lambda_step: float = 0.05,
    max_deenergized_lines: int = 2,
    stage_e_budget: int = 100,
    stage_d_limit: int | None = None,
    suppressed_p_env: float = DEFAULT_SUPPRESSED_P_ENV,
    suppress_heavy_loading_line_ids: List[int] | None = None,
    run_mode: str = "both",
    clear: bool = False,
) -> Dict[str, Path]:
    suppressed_ids = (
        list(DEFAULT_GRIDFM_HEAVY_LOADING_LINE_IDS)
        if suppress_heavy_loading_line_ids is None
        else [int(line_id) for line_id in suppress_heavy_loading_line_ids]
    )
    run_dirs: Dict[str, Path] = {}

    if run_mode in {"both", "without"}:
        run_dirs["without_physics_infeasibility"] = run_stage_f_physics_decision_quality(
            models=models,
            scenario_ids=scenario_ids,
            rho_phys=0.0,
            lambda_step=lambda_step,
            max_deenergized_lines=max_deenergized_lines,
            stage_e_budget=stage_e_budget,
            stage_d_limit=stage_d_limit,
            clear=clear,
            result_root=WITHOUT_PHYSICS_ROOT,
            stage_name="G",
            study_name="stage_g_matpower_30_without_physics_infeasibility",
            p_env_suppressed_value=suppressed_p_env,
            p_env_extra_suppressed_line_ids=suppressed_ids,
        )

    if run_mode in {"both", "with"}:
        run_dirs["with_physics_infeasibility"] = run_stage_f_physics_decision_quality(
            models=models,
            scenario_ids=scenario_ids,
            rho_phys=100.0,
            lambda_step=lambda_step,
            max_deenergized_lines=max_deenergized_lines,
            stage_e_budget=stage_e_budget,
            stage_d_limit=stage_d_limit,
            clear=clear,
            result_root=WITH_PHYSICS_ROOT,
            stage_name="G",
            study_name="stage_g_matpower_30_with_physics_infeasibility",
            p_env_suppressed_value=suppressed_p_env,
            p_env_extra_suppressed_line_ids=suppressed_ids,
        )

    return run_dirs


def _latest_run(root: Path) -> Path:
    runs = sorted(Path(root).glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No run_* folders found under {root}")
    return runs[-1]


def repair_expected_match_artifacts(run_dir: Path, model_type: str = "gnn") -> Path:
    from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import _build_model_context

    run_dir = Path(run_dir)
    context = _build_model_context(model_type, grouping_top_fraction=0.30)
    canonical_scenarios = [
        _canonicalize_decision_quality_scenario(scenario_def, context["scenario"])
        for scenario_def in get_scenarios()
    ]

    scored = pd.read_csv(run_dir / "all_physics_aware_evaluations.csv")
    best = pd.read_csv(run_dir / "best_by_scenario_lambda_stage.csv")
    pareto_points = pd.read_csv(run_dir / "pareto_unique_points.csv")
    frontier = pd.read_csv(run_dir / "pareto_frontier_points.csv")
    expected = _expected_vs_observed(best, canonical_scenarios)
    write_dataframe(run_dir / "expected_vs_observed_line_subsets.csv", expected)
    write_json(
        run_dir / "scenario_definitions.json",
        {
            "low_p_env": 0.05,
            "high_p_env": 1.0,
            "suppressed_p_env": DEFAULT_SUPPRESSED_P_ENV,
            "extra_suppressed_line_ids": list(DEFAULT_GRIDFM_HEAVY_LOADING_LINE_IDS),
            "scenarios": [scenario.to_dict() for scenario in canonical_scenarios],
            "canonicalized_after_run": True,
        },
    )
    rho_phys = float(best["rho_phys"].dropna().iloc[0])
    for scenario_def in canonical_scenarios:
        _plot_scenario_outputs(
            scenario_def.scenario_id,
            scored,
            best,
            expected,
            pareto_points,
            frontier,
            run_dir / "plots" / scenario_def.scenario_id,
            rho_phys,
        )
    return run_dir


def repair_latest_stage_g_expected_matches(model_type: str = "gnn") -> Dict[str, Path]:
    runs = {
        "without_physics_infeasibility": _latest_run(WITHOUT_PHYSICS_ROOT),
        "with_physics_infeasibility": _latest_run(WITH_PHYSICS_ROOT),
    }
    return {label: repair_expected_match_artifacts(run_dir, model_type=model_type) for label, run_dir in runs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage G MATPOWER-30 decision quality results with the Stage F "
            "physics-aware plot suite for both with/without physics-infeasibility variants."
        )
    )
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["gnn"])
    parser.add_argument("--scenarios", nargs="+", choices=[scenario.scenario_id for scenario in get_scenarios()], default=None)
    parser.add_argument("--lambda-step", type=float, default=0.05)
    parser.add_argument("--max-deenergized-lines", type=int, default=2)
    parser.add_argument("--stage-e-budget", type=int, default=100)
    parser.add_argument("--stage-d-limit", type=int, default=None)
    parser.add_argument("--suppressed-p-env", type=float, default=DEFAULT_SUPPRESSED_P_ENV)
    parser.add_argument(
        "--suppress-heavy-loading-line-ids",
        nargs="+",
        type=int,
        default=list(DEFAULT_GRIDFM_HEAVY_LOADING_LINE_IDS),
    )
    parser.add_argument("--run-mode", choices=["both", "with", "without"], default="both")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument(
        "--repair-existing-matches",
        action="store_true",
        help="Recompute expected-vs-observed match CSVs and plots for the latest completed Stage G runs without rerunning optimization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repair_existing_matches:
        run_dirs = repair_latest_stage_g_expected_matches(model_type=args.models[0])
        for label, run_dir in run_dirs.items():
            print(f"Repaired canonical expected-match artifacts for {label} at {run_dir}")
        return
    run_dirs = run_stage_g_matpower_30_decision_quality(
        models=args.models,
        scenario_ids=args.scenarios,
        lambda_step=args.lambda_step,
        max_deenergized_lines=args.max_deenergized_lines,
        stage_e_budget=args.stage_e_budget,
        stage_d_limit=args.stage_d_limit,
        suppressed_p_env=args.suppressed_p_env,
        suppress_heavy_loading_line_ids=args.suppress_heavy_loading_line_ids,
        run_mode=args.run_mode,
        clear=args.clear,
    )
    for label, run_dir in run_dirs.items():
        print(f"Wrote Stage G MATPOWER-30 {label} decision-quality results to {run_dir}")


if __name__ == "__main__":
    main()
