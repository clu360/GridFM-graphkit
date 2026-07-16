from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from experiments.test.wildfire_tests.shared.config import load_first_pass_config, write_config_copy
from experiments.test.wildfire_tests.shared.lambda_cases import (
    lambda_case_rows,
)
from experiments.test.wildfire_tests.shared.paths import CONFIGS_ROOT, RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import write_dataframe, write_json
from experiments.test.wildfire_tests.stage_a_first_pass.run_basic_case import run_basic_case
from experiments.test.wildfire_tests.stage_a_first_pass.run_connected_corridor_tradeoffs import MODEL_CONFIGS
from experiments.test.wildfire_tests.stage_b_multigroup import run_multi_group_threshold_sensitivity as stage_b_threshold
from experiments.test.wildfire_tests.stage_b_multigroup.run_multistart_optimization import run_multistart_optimization
from experiments.test.wildfire_tests.stage_c_psps_baseline import run_stage_c_psps_baseline as stage_c_runner
from experiments.test.wildfire_tests.stage_d_deenergization import run_stage_d_deenergization as stage_d_runner


LEGACY_ROOT = RESULTS_ROOT / "leq"
STANDARDIZED_LAMBDA_CASES = {
    "wildfire_risk": (0.9, 0.1),
    "balanced": (0.5, 0.5),
    "service_leaning": (0.1, 0.9),
}
STANDARDIZED_CASE_ROWS = lambda_case_rows(STANDARDIZED_LAMBDA_CASES)
CASE_FOLDERS = {
    "risk": "risk",
    "wildfire_risk": "risk",
    "balanced": "bal",
    "shed": "svc",
    "risk_leaning": "risk",
    "service_leaning": "svc",
}


def _safe_clear(path: Path) -> None:
    resolved = path.resolve()
    expected_parent = RESULTS_ROOT.resolve()
    if resolved.parent != expected_parent:
        raise ValueError(f"Refusing to delete unexpected path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _write_manifest(root: Path, rows: list[dict]) -> Path:
    manifest = {
        "output_root": str(root),
        "purpose": "Legacy-equivalent refactored harness outputs with standardized lambda cases for parity/methodology checks.",
        "standardized_lambda_cases": {
            key: {"lambda_R": value[0], "lambda_L": value[1]}
            for key, value in STANDARDIZED_LAMBDA_CASES.items()
        },
        "runs": rows,
    }
    path = root / "legacy_equivalent_manifest.json"
    write_json(path, manifest)
    write_dataframe(root / "legacy_equivalent_manifest.csv", pd.DataFrame(rows))
    return path


def generate_stage_a(root: Path) -> list[dict]:
    stage_root = root / "stage_a" / "connected_corridor"
    generated_configs = root / "generated_configs" / "stage_a" / "connected_corridor"
    rows: list[dict] = []
    for case_name, weights in STANDARDIZED_CASE_ROWS.items():
        case_folder = CASE_FOLDERS.get(case_name, case_name)
        for model_type, config_name in MODEL_CONFIGS.items():
            config = load_first_pass_config(CONFIGS_ROOT / config_name)
            config.model.model_type = model_type
            config.decision.alpha_min = 0.0
            config.objective.lambda_R = float(weights["lambda_R"])
            config.objective.lambda_L = float(weights["lambda_L"])
            config.objective.risk_normalizer = 0.0
            config.objective.load_shedding_normalizer = 0.0
            config.output.output_root = str(stage_root / case_folder / model_type)
            config.output.run_name = f"{case_folder}_{model_type}"
            config_path = generated_configs / f"{case_folder}_{model_type}.yaml"
            write_config_copy(config, config_path)
            result = run_basic_case(config_path)
            rows.append(
                {
                    "stage": "stage_a_connected_corridor",
                    "case": case_name,
                    "model_type": model_type,
                    "lambda_R": float(weights["lambda_R"]),
                    "lambda_L": float(weights["lambda_L"]),
                    "run_dir": result.get("run_dir"),
                    "status": "ok",
                    "error": "",
                }
            )
    return rows


def generate_stage_b_multistart(root: Path, num_seed_points: int, max_seeds: int) -> list[dict]:
    stage_root = root / "stage_b" / "multistart"
    generated_configs = root / "generated_configs" / "stage_b" / "multistart"
    rows: list[dict] = []
    for case_name, weights in STANDARDIZED_CASE_ROWS.items():
        case_folder = CASE_FOLDERS.get(case_name, case_name)
        for model_type, config_name in MODEL_CONFIGS.items():
            config = load_first_pass_config(CONFIGS_ROOT / config_name)
            config.model.model_type = model_type
            config.decision.alpha_min = 0.0
            config.objective.lambda_R = float(weights["lambda_R"])
            config.objective.lambda_L = float(weights["lambda_L"])
            config.objective.risk_normalizer = 0.0
            config.objective.load_shedding_normalizer = 0.0
            config.output.output_root = str(stage_root / case_folder / model_type)
            config.output.run_name = f"ms_{case_folder}_{model_type}"
            config_path = generated_configs / f"{case_folder}_{model_type}.yaml"
            write_config_copy(config, config_path)
            run_dir = run_multistart_optimization(
                config_path,
                num_seed_points=num_seed_points,
                max_seeds=max_seeds,
                output_root=stage_root / case_folder / model_type,
            )
            rows.append(
                {
                    "stage": "stage_b_multistart",
                    "case": case_name,
                    "model_type": model_type,
                    "lambda_R": float(weights["lambda_R"]),
                    "lambda_L": float(weights["lambda_L"]),
                    "run_dir": str(run_dir),
                    "status": "ok",
                    "error": "",
                }
            )
    return rows


def generate_stage_b_multigroup(
    root: Path,
    top_fractions: list[float],
    num_seed_points: int,
    max_seeds: int,
) -> list[dict]:
    original_tradeoff_sets = stage_b_threshold.TRADEOFF_SETS
    try:
        stage_b_threshold.TRADEOFF_SETS = STANDARDIZED_CASE_ROWS
        summary_csv = stage_b_threshold.run_threshold_sensitivity(
            top_fractions=top_fractions,
            models=["gps", "gnn"],
            tradeoff_cases=list(STANDARDIZED_CASE_ROWS),
            num_seed_points=num_seed_points,
            max_seeds=max_seeds,
            output_root=root / "stage_b" / "multi_group",
            generated_config_root=root / "generated_configs" / "stage_b" / "multi_group",
        )
    finally:
        stage_b_threshold.TRADEOFF_SETS = original_tradeoff_sets
    return [
        {
            "stage": "stage_b_multi_group_threshold_sensitivity",
            "case": "standardized",
            "model_type": "gps,gnn",
            "lambda_R": "",
            "lambda_L": "",
            "run_dir": str(summary_csv.parent),
            "summary_csv": str(summary_csv),
            "status": "ok",
            "error": "",
        }
    ]


def generate_stage_c(root: Path) -> list[dict]:
    original_results_root = stage_c_runner.RESULTS_ROOT
    original_cases = stage_c_runner.STAGE_C_LAMBDA_CASES
    try:
        stage_c_runner.RESULTS_ROOT = root
        stage_c_runner.STAGE_C_LAMBDA_CASES = STANDARDIZED_LAMBDA_CASES
        summary_csv = stage_c_runner.run_stage_c_psps_baseline(
            grouping_top_fraction=0.30,
            psps_top_fraction=0.10,
            models=["gps", "gnn"],
            cases=list(stage_c_runner.CASES),
            lambda_cases=list(STANDARDIZED_LAMBDA_CASES),
        )
    finally:
        stage_c_runner.RESULTS_ROOT = original_results_root
        stage_c_runner.STAGE_C_LAMBDA_CASES = original_cases
    return [
        {
            "stage": "stage_c_psps",
            "case": "standardized",
            "model_type": "gps,gnn",
            "lambda_R": "",
            "lambda_L": "",
            "run_dir": str(summary_csv.parent),
            "summary_csv": str(summary_csv),
            "status": "ok",
            "error": "",
        }
    ]


def generate_stage_d(root: Path) -> list[dict]:
    original_results_root = stage_d_runner.RESULTS_ROOT
    original_cases = stage_d_runner.LAMBDA_CASES
    try:
        stage_d_runner.RESULTS_ROOT = root
        stage_d_runner.LAMBDA_CASES = STANDARDIZED_LAMBDA_CASES
        summary_csv = stage_d_runner.run_stage_d_deenergization(
            grouping_top_fraction=0.30,
            models=["gps", "gnn"],
            cases=list(stage_d_runner.CASES),
            lambda_cases=list(STANDARDIZED_LAMBDA_CASES),
            evaluation_mode=stage_d_runner.EVALUATION_MODE,
            max_deenergized_lines=2,
        )
    finally:
        stage_d_runner.RESULTS_ROOT = original_results_root
        stage_d_runner.LAMBDA_CASES = original_cases
    return [
        {
            "stage": "stage_d_deenergization",
            "case": "standardized",
            "model_type": "gps,gnn",
            "lambda_R": "",
            "lambda_L": "",
            "run_dir": str(summary_csv.parent),
            "summary_csv": str(summary_csv),
            "status": "ok",
            "error": "",
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["stage-a", "stage-b-multistart", "stage-b-multigroup", "stage-c", "stage-d"],
        choices=["stage-a", "stage-b-multistart", "stage-b-multigroup", "stage-c", "stage-d"],
    )
    parser.add_argument("--top-fractions", nargs="+", type=float, default=[0.10, 0.20, 0.30])
    parser.add_argument("--num-seed-points", type=int, default=11)
    parser.add_argument("--max-seeds", type=int, default=5)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if args.clear:
        _safe_clear(LEGACY_ROOT)
    LEGACY_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if "stage-a" in args.stages:
        rows.extend(generate_stage_a(LEGACY_ROOT))
    if "stage-b-multistart" in args.stages:
        rows.extend(generate_stage_b_multistart(LEGACY_ROOT, args.num_seed_points, args.max_seeds))
    if "stage-b-multigroup" in args.stages:
        rows.extend(generate_stage_b_multigroup(LEGACY_ROOT, args.top_fractions, args.num_seed_points, args.max_seeds))
    if "stage-c" in args.stages:
        rows.extend(generate_stage_c(LEGACY_ROOT))
    if "stage-d" in args.stages:
        rows.extend(generate_stage_d(LEGACY_ROOT))

    manifest_path = _write_manifest(LEGACY_ROOT, rows)
    print(f"[OK] Legacy-equivalent manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
