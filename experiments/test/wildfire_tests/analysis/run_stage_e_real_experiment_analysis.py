from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import write_dataframe, write_json
from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import (
    CASES,
    MODEL_CONFIGS,
    _build_model_context,
)
from experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps import (
    apply_environmental_case,
    demand_weighted_load_shed_from_prediction,
    predict_psps_state,
)
from experiments.test.wildfire_tests.stage_d_deenergization.stage_d_deenergization import (
    enumerate_deenergization_subsets,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    STAGE_E_LAMBDA_CASES,
    compute_true_metrics,
    z_from_y,
)


STAGE_E_ROOT = RESULTS_ROOT / "leq" / "stage_e" / "gurobi_gridfm"
SUMMARY_ROOT = STAGE_E_ROOT / "experiment_summaries"


def _parse_line_ids(value) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [int(item) for item in parsed]
    except Exception:
        pass
    return [int(item) for item in text.replace("[", "").replace("]", "").split(",") if item.strip()]


def _line_id_key(line_ids: Iterable[int]) -> str:
    return ",".join(str(int(line_id)) for line_id in sorted(int(item) for item in line_ids))


def _candidate_y(candidate_line_ids: Iterable[int], deenergized_line_ids: Iterable[int]) -> dict[int, int]:
    deenergized = {int(line_id) for line_id in deenergized_line_ids}
    return {int(line_id): int(int(line_id) in deenergized) for line_id in candidate_line_ids}


def _setup_gridfm_calls_estimate(context: dict) -> int:
    num_lines = int(context["scenario"].edge_index.shape[1])
    # _build_model_context performs one baseline prediction, automatic-group
    # counterfactual line impacts, and fixed c_l consequence scoring.
    return int(1 + num_lines + num_lines)


def _stage_e_summary_path(stage_e_root: Path) -> Path:
    path = stage_e_root / "stage_e_gurobi_gridfm_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage E summary: {path}")
    return path


def _load_candidate_rows(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "gurobi_candidate_evaluations.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage E candidate table: {path}")
    frame = pd.read_csv(path)
    frame["run_dir"] = str(run_dir)
    return frame


def _load_baseline_metrics(run_dir: Path) -> dict:
    path = run_dir / "baseline_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline metrics: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stage_e_call_rows(stage_e_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in stage_e_summary.iterrows():
        if str(row.get("status", "")) != "ok":
            continue
        run_dir = Path(str(row["run_dir"]))
        candidates = _load_candidate_rows(run_dir)
        ok = candidates[candidates["status"].astype(str) == "ok"].copy()
        baseline_reuse = int((ok["num_deenergized_lines"].astype(int) == 0).sum()) if len(ok) else 0
        baseline = _load_baseline_metrics(run_dir)
        num_lines = len(baseline.get("baseline_loading_by_line", {}))
        setup_calls = int(1 + num_lines + num_lines)
        search_calls = int(row.get("num_gridfm_calls", 0))
        runtime_seconds = float(ok["runtime_seconds"].max()) if len(ok) and "runtime_seconds" in ok.columns else np.nan
        rows.append(
            {
                **row.to_dict(),
                "setup_gridfm_calls": setup_calls,
                "search_gridfm_calls": search_calls,
                "total_gridfm_calls": int(setup_calls + search_calls),
                "total_candidate_evaluations": int(len(ok)),
                "baseline_reuse_evaluations": baseline_reuse,
                "runtime_seconds": runtime_seconds,
                "setup_call_estimate_note": "baseline + automatic impact counterfactuals + fixed c_l counterfactuals",
            }
        )
    return pd.DataFrame(rows)


def _proxy_alignment(stage_e_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in stage_e_summary.iterrows():
        if str(row.get("status", "")) != "ok":
            continue
        run_dir = Path(str(row["run_dir"]))
        candidates = _load_candidate_rows(run_dir)
        ok = candidates[candidates["status"].astype(str) == "ok"].copy()
        if ok.empty:
            continue
        proxy_best = ok.sort_values("proxy_objective", kind="mergesort").iloc[0]
        true_best = ok.sort_values("true_objective", kind="mergesort").iloc[0]
        correlation = np.nan
        if len(ok) >= 3 and ok["proxy_objective"].nunique() > 1 and ok["true_objective"].nunique() > 1:
            correlation = float(ok["proxy_objective"].corr(ok["true_objective"], method="spearman"))
        rows.append(
            {
                "model_type": row["model_type"],
                "environmental_risk_case": row["environmental_risk_case"],
                "lambda_case": row["lambda_case"],
                "K": int(row["K"]),
                "run_dir": str(run_dir),
                "num_candidates": int(len(ok)),
                "proxy_true_spearman": correlation,
                "proxy_best_deenergized_line_ids": proxy_best["deenergized_line_ids"],
                "true_best_deenergized_line_ids": true_best["deenergized_line_ids"],
                "proxy_best_proxy_objective": float(proxy_best["proxy_objective"]),
                "proxy_best_true_objective": float(proxy_best["true_objective"]),
                "true_best_proxy_objective": float(true_best["proxy_objective"]),
                "true_best_true_objective": float(true_best["true_objective"]),
                "proxy_best_is_true_best": bool(
                    _line_id_key(_parse_line_ids(proxy_best["deenergized_line_ids"]))
                    == _line_id_key(_parse_line_ids(true_best["deenergized_line_ids"]))
                ),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_revised_stage_d_candidates(context: dict, case_name: str, max_deenergized_lines: int) -> tuple[pd.DataFrame, dict]:
    config = context["config"]
    scenario = context["scenario"]
    runner = context["runner"]
    decision_vector = context["decision_vector"]
    wildfire = context["wildfire"]
    group_summary = context["automatic_artifacts"]["group_summary"]
    baseline_prediction = context["baseline_prediction"]
    baseline_state = context["baseline_state"]
    baseline_loading = np.asarray(baseline_state["loading_ratio"], dtype=float)
    candidate_line_ids = sorted({int(line_id) for group in wildfire.line_groups for line_id in group.line_ids})
    env = apply_environmental_case(wildfire, group_summary, case_name)
    baseline_z = {line_id: 1 for line_id in range(len(baseline_loading))}
    baseline_R_raw, _baseline_by_line = compute_operational_wildfire_exposure(
        baseline_loading,
        env.p_env_by_line,
        baseline_z,
        candidate_line_ids,
    )
    if baseline_R_raw <= 1e-12:
        raise ValueError(f"Cannot compare Stage D with zero revised baseline exposure for {case_name}.")

    rows = []
    search_calls = 0
    start = time.perf_counter()
    for eval_id, subset in enumerate(enumerate_deenergization_subsets(candidate_line_ids, max_deenergized_lines)):
        deenergized = [int(line_id) for line_id in subset]
        y_by_line = _candidate_y(candidate_line_ids, deenergized)
        z_by_line = z_from_y(y_by_line, int(scenario.edge_index.shape[1]))
        status = "ok"
        error = ""
        try:
            if deenergized:
                prediction, state = predict_psps_state(
                    scenario,
                    runner,
                    decision_vector.u_base,
                    deenergized,
                    standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
                )
                search_calls += 1
            else:
                prediction = baseline_prediction
                state = baseline_state
            loading = np.asarray(state["loading_ratio"], dtype=float)
            load_shed = demand_weighted_load_shed_from_prediction(prediction, scenario) if deenergized else 0.0
            true = compute_true_metrics(
                loading,
                env.p_env_by_line,
                z_by_line,
                candidate_line_ids,
                baseline_R_raw,
                true_L_shed=load_shed,
                true_P_AC=0.0,
                lambda_R_true=1.0,
                lambda_L_true=0.0,
                lambda_P=0.0,
            )
        except Exception as exc:
            status = "failed"
            error = str(exc)
            true = {
                "true_R_raw": np.nan,
                "true_R_norm": np.nan,
                "true_L_shed": np.nan,
                "true_P_AC": np.nan,
                "true_objective": np.nan,
            }
        rows.append(
            {
                "eval_id": int(eval_id),
                "deenergized_line_ids": deenergized,
                "line_id_key": _line_id_key(deenergized),
                "num_deenergized_lines": int(len(deenergized)),
                "true_R_raw": float(true["true_R_raw"]) if np.isfinite(true["true_R_raw"]) else np.nan,
                "true_R_norm": float(true["true_R_norm"]) if np.isfinite(true["true_R_norm"]) else np.nan,
                "true_L_shed": float(true["true_L_shed"]) if np.isfinite(true["true_L_shed"]) else np.nan,
                "true_P_AC": float(true["true_P_AC"]) if np.isfinite(true["true_P_AC"]) else np.nan,
                "status": status,
                "error": error,
            }
        )
    metadata = {
        "candidate_line_ids": [int(line_id) for line_id in candidate_line_ids],
        "baseline_R_raw_new": float(baseline_R_raw),
        "baseline_R_norm": 1.0,
        "baseline_L_shed": float(demand_weighted_load_shed_from_prediction(baseline_prediction, scenario)),
        "baseline_P_AC": 0.0,
        "setup_gridfm_calls": _setup_gridfm_calls_estimate(context),
        "search_gridfm_calls": int(search_calls),
        "total_candidate_evaluations": int(len(rows)),
        "runtime_seconds": float(time.perf_counter() - start),
    }
    metadata["total_gridfm_calls"] = int(metadata["setup_gridfm_calls"] + metadata["search_gridfm_calls"])
    return pd.DataFrame(rows), metadata


def _best_revised_stage_d_rows(evaluations: pd.DataFrame, metadata: dict, model_type: str, case_name: str, k: int) -> list[dict]:
    rows = []
    ok = evaluations[evaluations["status"].astype(str) == "ok"].copy()
    for lambda_case, (lambda_R, lambda_L) in STAGE_E_LAMBDA_CASES.items():
        frame = ok.copy()
        frame["true_objective"] = float(lambda_R) * frame["true_R_norm"].astype(float) + float(lambda_L) * frame["true_L_shed"].astype(float)
        best = frame.sort_values(
            by=["true_objective", "true_L_shed", "num_deenergized_lines", "line_id_key"],
            ascending=[True, True, True, True],
            kind="mergesort",
        ).iloc[0]
        rows.append(
            {
                "method": "stage_d_revised_enumeration",
                "model_type": model_type,
                "environmental_risk_case": case_name,
                "lambda_case": lambda_case,
                "lambda_R": float(lambda_R),
                "lambda_L": float(lambda_L),
                "K": int(k),
                "best_deenergized_line_ids": best["deenergized_line_ids"],
                "best_line_id_key": best["line_id_key"],
                "best_true_R_raw": float(best["true_R_raw"]),
                "best_true_R_norm": float(best["true_R_norm"]),
                "best_true_L_shed": float(best["true_L_shed"]),
                "best_true_P_AC": float(best["true_P_AC"]),
                "best_true_objective": float(best["true_objective"]),
                "baseline_R_raw_new": float(metadata["baseline_R_raw_new"]),
                "baseline_L_shed": float(metadata["baseline_L_shed"]),
                "setup_gridfm_calls": int(metadata["setup_gridfm_calls"]),
                "search_gridfm_calls": int(metadata["search_gridfm_calls"]),
                "total_gridfm_calls": int(metadata["total_gridfm_calls"]),
                "total_candidate_evaluations": int(metadata["total_candidate_evaluations"]),
                "runtime_seconds": float(metadata["runtime_seconds"]),
                "status": "ok",
            }
        )
    return rows


def _stage_e_best_rows(stage_e_calls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in stage_e_calls.iterrows():
        rows.append(
            {
                "method": "stage_e_gurobi_master_gridfm",
                "model_type": row["model_type"],
                "environmental_risk_case": row["environmental_risk_case"],
                "lambda_case": row["lambda_case"],
                "lambda_R": float(row["lambda_R_true"]),
                "lambda_L": float(row["lambda_L_true"]),
                "K": int(row["K"]),
                "best_deenergized_line_ids": _parse_line_ids(row["best_deenergized_line_ids"]),
                "best_line_id_key": _line_id_key(_parse_line_ids(row["best_deenergized_line_ids"])),
                "best_true_R_raw": float(row["best_true_R_raw"]),
                "best_true_R_norm": float(row["best_true_R_norm"]),
                "best_true_L_shed": float(row["best_true_L_shed"]),
                "best_true_P_AC": float(row["best_true_P_AC"]),
                "best_true_objective": float(row["best_true_objective"]),
                "baseline_R_raw_new": float(row["baseline_R_raw_new"]),
                "baseline_L_shed": float(row["baseline_L_shed"]),
                "setup_gridfm_calls": int(row["setup_gridfm_calls"]),
                "search_gridfm_calls": int(row["search_gridfm_calls"]),
                "total_gridfm_calls": int(row["total_gridfm_calls"]),
                "total_candidate_evaluations": int(row["total_candidate_evaluations"]),
                "runtime_seconds": float(row["runtime_seconds"]) if "runtime_seconds" in row and pd.notna(row["runtime_seconds"]) else np.nan,
                "status": row["status"],
                "run_dir": row["run_dir"],
            }
        )
    return pd.DataFrame(rows)


def _compare_stage_e_to_stage_d(stage_e_best: pd.DataFrame, stage_d_best: pd.DataFrame) -> pd.DataFrame:
    merged = stage_e_best.merge(
        stage_d_best,
        on=["model_type", "environmental_risk_case", "lambda_case", "K"],
        suffixes=("_stage_e", "_stage_d"),
    )
    if merged.empty:
        return merged
    merged["objective_gap"] = (
        merged["best_true_objective_stage_e"].astype(float) - merged["best_true_objective_stage_d"].astype(float)
    ) / merged["best_true_objective_stage_d"].astype(float).abs().clip(lower=1e-12)
    merged["line_overlap_count"] = [
        len(set(_parse_line_ids(left)).intersection(set(_parse_line_ids(right))))
        for left, right in zip(merged["best_deenergized_line_ids_stage_e"], merged["best_deenergized_line_ids_stage_d"])
    ]
    merged["same_best_topology"] = merged["best_line_id_key_stage_e"].astype(str) == merged["best_line_id_key_stage_d"].astype(str)
    return merged


def run_stage_e_real_experiment_analysis(
    models: list[str] | None = None,
    cases: list[str] | None = None,
    max_deenergized_lines: list[int] | None = None,
    stage_e_root: Path = STAGE_E_ROOT,
    output_root: Path = SUMMARY_ROOT,
) -> Path:
    models = ["gps"] if models is None else models
    cases = list(CASES) if cases is None else cases
    max_deenergized_lines = [1, 2] if max_deenergized_lines is None else [int(item) for item in max_deenergized_lines]
    output_root.mkdir(parents=True, exist_ok=True)

    stage_e_summary = pd.read_csv(_stage_e_summary_path(stage_e_root))
    stage_e_summary = stage_e_summary[stage_e_summary["status"].astype(str) == "ok"].copy()
    stage_e_calls = _stage_e_call_rows(stage_e_summary)
    stage_e_best = _stage_e_best_rows(stage_e_calls)
    proxy_alignment = _proxy_alignment(stage_e_summary)

    revised_candidate_paths = []
    stage_d_best_rows = []
    for model_type in models:
        context = _build_model_context(model_type, 0.30)
        for case_name in cases:
            for k in max_deenergized_lines:
                evaluations, metadata = _evaluate_revised_stage_d_candidates(context, case_name, int(k))
                candidate_path = output_root / f"stage_d_revised_candidates_{model_type}_{case_name}_k{int(k)}.csv"
                write_dataframe(candidate_path, evaluations)
                revised_candidate_paths.append(str(candidate_path))
                stage_d_best_rows.extend(_best_revised_stage_d_rows(evaluations, metadata, model_type, case_name, int(k)))

    stage_d_best = pd.DataFrame(stage_d_best_rows)
    comparison = _compare_stage_e_to_stage_d(stage_e_best, stage_d_best)
    best_decisions = pd.concat([stage_e_best, stage_d_best], ignore_index=True, sort=False)
    runtime = best_decisions[
        [
            "method",
            "model_type",
            "environmental_risk_case",
            "lambda_case",
            "K",
            "setup_gridfm_calls",
            "search_gridfm_calls",
            "total_gridfm_calls",
            "total_candidate_evaluations",
            "runtime_seconds",
            "status",
        ]
    ].copy()

    write_dataframe(output_root / "stage_e_real_experiment_summary.csv", stage_e_calls)
    write_dataframe(output_root / "stage_e_vs_stage_d_comparison.csv", comparison)
    write_dataframe(output_root / "proxy_vs_true_alignment.csv", proxy_alignment)
    write_dataframe(output_root / "best_decisions_by_lambda.csv", best_decisions)
    write_dataframe(output_root / "runtime_and_call_count_summary.csv", runtime)
    manifest = {
        "stage_e_root": str(stage_e_root),
        "output_root": str(output_root),
        "models": models,
        "cases": cases,
        "max_deenergized_lines": [int(item) for item in max_deenergized_lines],
        "lambda_cases": {
            name: {"lambda_R": values[0], "lambda_L": values[1]}
            for name, values in STAGE_E_LAMBDA_CASES.items()
        },
        "revised_stage_d_objective": "lambda_R * sum_l(z_l * p_env_l * loading_l^2) / baseline + lambda_L * L_shed",
        "stage_d_revised_candidate_files": revised_candidate_paths,
        "summary_files": {
            "stage_e_real_experiment_summary": str(output_root / "stage_e_real_experiment_summary.csv"),
            "stage_e_vs_stage_d_comparison": str(output_root / "stage_e_vs_stage_d_comparison.csv"),
            "proxy_vs_true_alignment": str(output_root / "proxy_vs_true_alignment.csv"),
            "best_decisions_by_lambda": str(output_root / "best_decisions_by_lambda.csv"),
            "runtime_and_call_count_summary": str(output_root / "runtime_and_call_count_summary.csv"),
        },
    }
    write_json(output_root / "stage_e_real_experiment_summary.json", manifest)
    print(f"[OK] Stage E experiment analysis written to {output_root}")
    return output_root / "stage_e_vs_stage_d_comparison.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["gps"])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--max-deenergized-lines", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--stage-e-root", type=Path, default=STAGE_E_ROOT)
    parser.add_argument("--output-root", type=Path, default=SUMMARY_ROOT)
    args = parser.parse_args()
    run_stage_e_real_experiment_analysis(
        models=args.models,
        cases=args.cases,
        max_deenergized_lines=args.max_deenergized_lines,
        stage_e_root=args.stage_e_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
