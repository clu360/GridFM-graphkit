from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_initial_tests.config import load_first_pass_config, write_config_copy
from experiments.test.wildfire_initial_tests.reporting import write_dataframe, write_json
from experiments.test.wildfire_initial_tests.run_basic_case import run_basic_case


TRADEOFF_SETS = {
    "risk": {
        "lambda_R": 0.999001,
        "lambda_L": 0.000999,
        "description": "Near-one weight on normalized wildfire risk.",
    },
    "balanced": {
        "lambda_R": 0.5,
        "lambda_L": 0.5,
        "description": "Equal weights on normalized wildfire risk and load shedding.",
    },
    "shed": {
        "lambda_R": 0.000999,
        "lambda_L": 0.999001,
        "description": "Near-one weight on normalized load-shedding penalty.",
    },
}


MODEL_CONFIGS = {
    "gnn": "connected_corridor_gnn.yaml",
    "gps": "connected_corridor_gps.yaml",
}


def _connected_corridor_root() -> Path:
    return REPO_ROOT / "experiments" / "test" / "wildfire_initial_tests" / "results" / "connected_corridor"


def run_tradeoff_sets(clear: bool = False) -> Path:
    root = _connected_corridor_root()
    if clear and root.exists():
        resolved_root = root.resolve()
        expected_parent = (REPO_ROOT / "experiments" / "test" / "wildfire_initial_tests" / "results").resolve()
        if resolved_root.parent != expected_parent:
            raise ValueError(f"Refusing to delete unexpected path: {resolved_root}")
        shutil.rmtree(resolved_root)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    generated_configs = REPO_ROOT / "experiments" / "test" / "wildfire_initial_tests" / "configs" / "generated_tradeoffs"
    generated_configs.mkdir(parents=True, exist_ok=True)

    for set_name, weights in TRADEOFF_SETS.items():
        for model_type, config_name in MODEL_CONFIGS.items():
            base_config_path = REPO_ROOT / "experiments" / "test" / "wildfire_initial_tests" / "configs" / config_name
            config = load_first_pass_config(base_config_path)
            config.model.model_type = model_type
            config.decision.alpha_min = 0.0
            config.objective.lambda_R = float(weights["lambda_R"])
            config.objective.lambda_L = float(weights["lambda_L"])
            config.objective.risk_normalizer = 0.0
            config.objective.load_shedding_normalizer = 0.0
            config.output.output_root = str(root / set_name / model_type)
            config.output.run_name = f"{set_name}_{model_type}"

            run_config_path = generated_configs / f"{set_name}_{model_type}.yaml"
            write_config_copy(config, run_config_path)
            result = run_basic_case(run_config_path)
            rows.append(
                {
                    "tradeoff_set": set_name,
                    "description": weights["description"],
                    "model_type": model_type,
                    "lambda_R": float(weights["lambda_R"]),
                    "lambda_L": float(weights["lambda_L"]),
                    "baseline_objective": result.get("baseline_objective"),
                    "final_objective": result.get("final_objective"),
                    "baseline_grouped_wildfire_risk": result.get("baseline_grouped_wildfire_risk"),
                    "final_grouped_wildfire_risk": result.get("final_grouped_wildfire_risk"),
                    "mean_alpha": result.get("mean_alpha"),
                    "min_alpha": result.get("min_alpha"),
                    "max_abs_delta_pg": result.get("max_abs_delta_pg"),
                    "optimizer_success": result.get("optimizer_success"),
                    "optimizer_message": result.get("optimizer_message"),
                    "num_objective_evals": result.get("num_objective_evals"),
                    "run_dir": result.get("run_dir"),
                }
            )

    summary_csv = root / "connected_corridor_tradeoff_summary.csv"
    summary_json = root / "connected_corridor_tradeoff_summary.json"
    write_dataframe(summary_csv, pd.DataFrame(rows))
    write_json(
        summary_json,
        {
            "alpha_min": 0.0,
            "alpha_max": 1.0,
            "num_runs": len(rows),
            "summary_csv": str(summary_csv),
            "tradeoff_sets": TRADEOFF_SETS,
        },
    )
    print(f"[OK] Connected-corridor tradeoff summary written to {summary_csv}")
    return summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Delete existing connected_corridor results first.")
    args = parser.parse_args()
    run_tradeoff_sets(clear=args.clear)


if __name__ == "__main__":
    main()
