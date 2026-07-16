from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.config import load_first_pass_config, write_config_copy
from experiments.test.wildfire_tests.shared.lambda_cases import CANONICAL_LAMBDA_CASES, lambda_case_rows
from experiments.test.wildfire_tests.shared.paths import CONFIGS_ROOT, RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import write_dataframe, write_json
from experiments.test.wildfire_tests.stage_b_multigroup.run_multistart_optimization import run_multistart_optimization


TRADEOFF_SETS = lambda_case_rows(CANONICAL_LAMBDA_CASES)

MODEL_CONFIGS = {
    "gps": "automatic_multigroup_gps.yaml",
    "gnn": "automatic_multigroup_gnn.yaml",
}

DEFAULT_TOP_FRACTIONS = [0.10, 0.125, 0.15, 0.175, 0.20]


def threshold_label(top_fraction: float) -> str:
    value = float(top_fraction)
    if abs(value * 100.0 - round(value * 100.0)) < 1e-9:
        text = f"{value:.2f}"
    else:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "threshold_" + text.replace(".", "p")


def _multi_group_root() -> Path:
    return RESULTS_ROOT / "stage_b" / "multi_group"


def run_threshold_sensitivity(
    top_fractions: list[float],
    models: list[str],
    tradeoff_cases: list[str],
    num_seed_points: int = 11,
    max_seeds: int = 5,
    clear: bool = False,
    output_root: Path | None = None,
    generated_config_root: Path | None = None,
) -> Path:
    root = _multi_group_root() if output_root is None else Path(output_root)
    if clear and root.exists():
        resolved_root = root.resolve()
        expected_parent = RESULTS_ROOT.resolve()
        if resolved_root.parent != expected_parent:
            raise ValueError(f"Refusing to delete unexpected path: {resolved_root}")
        shutil.rmtree(resolved_root)
    root.mkdir(parents=True, exist_ok=True)

    generated_configs = (
        CONFIGS_ROOT / "generated_multi_group"
        if generated_config_root is None
        else Path(generated_config_root)
    )
    generated_configs.mkdir(parents=True, exist_ok=True)

    rows = []
    for top_fraction in top_fractions:
        label = threshold_label(top_fraction)
        for tradeoff_case in tradeoff_cases:
            weights = TRADEOFF_SETS[tradeoff_case]
            for model_type in models:
                base_config_path = CONFIGS_ROOT / MODEL_CONFIGS[model_type]
                output_root = root / label / tradeoff_case / model_type
                row = {
                    "requested_top_fraction": float(top_fraction),
                    "top_fraction": float(top_fraction),
                    "threshold_label": label,
                    "model_type": model_type,
                    "tradeoff_case": tradeoff_case,
                    "lambda_R": float(weights["lambda_R"]),
                    "lambda_L": float(weights["lambda_L"]),
                    "status": "ok",
                    "error": "",
                }
                try:
                    config = load_first_pass_config(base_config_path)
                    config.model.model_type = model_type
                    config.wildfire.selection_method = "automatic_risk_components"
                    config.wildfire.risk_score = {
                        **(config.wildfire.risk_score or {}),
                        "formula": "p_env_times_loading_squared_times_impact",
                        "threshold_method": "top_fraction",
                        "top_fraction": float(top_fraction),
                        "candidate_p_env": float((config.wildfire.risk_score or {}).get("candidate_p_env", 1.0)),
                    }
                    config.wildfire.grouping = {
                        **(config.wildfire.grouping or {}),
                        "method": "connected_components",
                        "allow_single_group": True,
                        "min_group_size": 1,
                        "group_weighting": "equal",
                    }
                    config.wildfire.diagnostics = {
                        **(config.wildfire.diagnostics or {}),
                        "warn_if_single_group": True,
                        "warn_if_largest_group_fraction_above": 0.80,
                    }
                    config.objective.lambda_R = float(weights["lambda_R"])
                    config.objective.lambda_L = float(weights["lambda_L"])
                    config.objective.risk_normalizer = 0.0
                    config.objective.load_shedding_normalizer = 0.0
                    config.output.output_root = str(output_root)
                    config.output.run_name = f"multi_group_{label}_{tradeoff_case}_{model_type}"

                    run_config_path = generated_configs / f"{label}_{tradeoff_case}_{model_type}.yaml"
                    write_config_copy(config, run_config_path)
                    run_dir = run_multistart_optimization(
                        run_config_path,
                        num_seed_points=num_seed_points,
                        max_seeds=max_seeds,
                        output_root=output_root,
                    )
                    with open(run_dir / "analysis_summary.json", "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    row.update(
                        {
                            "num_selected_lines": summary.get("num_selected_lines"),
                            "realized_selected_fraction": summary.get("realized_selected_fraction"),
                            "num_groups": summary.get("num_groups"),
                            "largest_group_num_lines": summary.get("largest_group_num_lines"),
                            "largest_group_fraction_of_selected_lines": summary.get(
                                "largest_group_fraction_of_selected_lines"
                            ),
                            "collapsed_to_single_group": summary.get("collapsed_to_single_group"),
                            "baseline_grouped_risk": summary.get("baseline_grouped_risk", summary.get("risk_normalizer")),
                            "baseline_objective": summary.get("baseline_objective"),
                            "best_seed_objective": summary.get("best_seed_objective"),
                            "best_start_index": summary.get("best_start_index"),
                            "best_objective": summary.get("best_objective"),
                            "best_grouped_risk": summary.get("best_grouped_risk"),
                            "best_load_shedding": summary.get("best_load_shedding"),
                            "optimizer_success": summary.get("optimizer_success"),
                            "optimizer_message": summary.get("optimizer_message"),
                            "run_dir": str(run_dir),
                        }
                    )
                except Exception as exc:
                    row.update({"status": "failed", "error": str(exc), "run_dir": ""})
                rows.append(row)

    summary_frame = pd.DataFrame(rows)
    summary_csv = root / "multi_group_threshold_sensitivity_summary.csv"
    summary_json = root / "multi_group_threshold_sensitivity_summary.json"
    write_dataframe(summary_csv, summary_frame)
    write_json(
        summary_json,
        {
            "num_runs": int(len(rows)),
            "num_successful_runs": int((summary_frame["status"] == "ok").sum()) if len(summary_frame) else 0,
            "top_fractions": [float(item) for item in top_fractions],
            "models": models,
            "tradeoff_cases": tradeoff_cases,
            "num_seed_points_per_variable": int(num_seed_points),
            "max_seeds": int(max_seeds),
            "tradeoff_sets": TRADEOFF_SETS,
            "summary_csv": str(summary_csv),
        },
    )
    print(f"[OK] Multi-group threshold sensitivity summary written to {summary_csv}")
    return summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-fractions", nargs="+", type=float, default=DEFAULT_TOP_FRACTIONS)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["gps", "gnn"])
    parser.add_argument("--tradeoff-cases", nargs="+", choices=sorted(TRADEOFF_SETS), default=list(TRADEOFF_SETS))
    parser.add_argument("--num-seed-points", type=int, default=11)
    parser.add_argument("--max-seeds", type=int, default=5)
    parser.add_argument("--clear", action="store_true", help="Delete existing results/stage_b/multi_group first.")
    args = parser.parse_args()
    run_threshold_sensitivity(
        top_fractions=args.top_fractions,
        models=args.models,
        tradeoff_cases=args.tradeoff_cases,
        num_seed_points=args.num_seed_points,
        max_seeds=args.max_seeds,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
