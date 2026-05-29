from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_initial_tests.config import load_first_pass_config, write_config_copy
from experiments.test.wildfire_initial_tests.reporting import make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_initial_tests.run_basic_case import run_basic_case


def run_stability_sweep(config_path: Path) -> Path:
    config = load_first_pass_config(config_path)
    sweep_dir = make_run_dir(config.output_root_path(), config.output.run_name)
    write_config_copy(config, sweep_dir / "config.yaml")
    rows = []

    for model_type, seed, lambda_R, lambda_L, hazard_multiplier, initialization in itertools.product(
        config.sweep.model_types,
        config.sweep.seeds,
        config.sweep.lambda_R_values,
        config.sweep.lambda_L_values,
        config.sweep.hazard_multipliers,
        config.sweep.initializations,
    ):
        run_config = load_first_pass_config(config_path)
        run_config.model.model_type = model_type
        run_config.random_seed = seed
        run_config.objective.lambda_R = lambda_R
        run_config.objective.lambda_L = lambda_L
        run_config.wildfire.hazard_multiplier = hazard_multiplier
        run_config.output.output_root = str(sweep_dir / "runs")
        run_config.output.run_name = f"sweep_{model_type}_seed{seed}_r{lambda_R}_l{lambda_L}_h{hazard_multiplier}_{initialization}"
        temp_config = sweep_dir / f"{run_config.output.run_name}.yaml"
        write_config_copy(run_config, temp_config)
        try:
            result = run_basic_case(temp_config)
            initial_objective = result.get("baseline_objective", float("nan"))
            final_objective = result.get("final_objective", float("nan"))
            initial_risk = result.get("baseline_grouped_wildfire_risk", float("nan"))
            final_risk = result.get("final_grouped_wildfire_risk", float("nan"))
            rows.append(
                {
                    "model_type": model_type,
                    "seed": seed,
                    "lambda_R": lambda_R,
                    "lambda_L": lambda_L,
                    "hazard_multiplier": hazard_multiplier,
                    "initialization": initialization,
                    "initial_objective": initial_objective,
                    "final_objective": final_objective,
                    "objective_reduction_pct": 100.0 * (initial_objective - final_objective) / initial_objective if initial_objective else float("nan"),
                    "initial_group_risk": initial_risk,
                    "final_group_risk": final_risk,
                    "group_risk_reduction_pct": 100.0 * (initial_risk - final_risk) / initial_risk if initial_risk else float("nan"),
                    "mean_alpha_final": result.get("mean_alpha", float("nan")),
                    "min_alpha_final": result.get("min_alpha", float("nan")),
                    "max_abs_delta_pg": result.get("max_abs_delta_pg", float("nan")),
                    "optimizer_success": result.get("optimizer_success", False),
                    "optimizer_message": result.get("optimizer_message", ""),
                    "num_objective_evals": result.get("num_objective_evals", 0),
                    "prediction_nan_count": 0,
                    "max_voltage_violation": float("nan"),
                    "max_thermal_violation": float("nan"),
                    "run_dir": result.get("run_dir", ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_type": model_type,
                    "seed": seed,
                    "lambda_R": lambda_R,
                    "lambda_L": lambda_L,
                    "hazard_multiplier": hazard_multiplier,
                    "initialization": initialization,
                    "optimizer_success": False,
                    "optimizer_message": f"failed: {exc}",
                }
            )

    out = sweep_dir / "stability_sweep_summary.csv"
    write_dataframe(out, pd.DataFrame(rows))
    write_json(sweep_dir / "stability_sweep_summary.json", {"num_runs": len(rows), "summary_csv": str(out)})
    print(f"[OK] Stability sweep written to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_stability_sweep(args.config)


if __name__ == "__main__":
    main()
