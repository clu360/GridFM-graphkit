from __future__ import annotations

import argparse
import ast
import shutil
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import write_dataframe, write_json
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import (
    CASE_FOLDERS,
    CASES,
    MODEL_CONFIGS,
    _build_model_context,
    compact_fraction_label,
)
from experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps import apply_environmental_case
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_gurobi_gridfm_unconstrained import (
    EVALUATION_MODE,
    UNCONSTRAINED_METHOD_NAME,
    _run_unconstrained_lambda_case,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import DEFAULT_PROXY_TYPE


def _frontier_root() -> Path:
    return RESULTS_ROOT / "leq" / "stage_e" / "unconstrained_frontier"


def _lambda_grid(step: float) -> list[float]:
    if step <= 0.0 or step > 1.0:
        raise ValueError(f"lambda_step must be in (0, 1], got {step}.")
    count = int(round(1.0 / float(step)))
    values = [round(i * float(step), 10) for i in range(count + 1)]
    if abs(values[-1] - 1.0) > 1e-9:
        values.append(1.0)
    return [float(min(1.0, max(0.0, value))) for value in values]


def _lambda_name(lambda_r: float) -> str:
    return f"lr{lambda_r:.2f}".replace(".", "p")


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


def _line_id_key(value) -> str:
    return ",".join(str(int(item)) for item in sorted(_parse_line_ids(value)))


def _collect_run_candidates(summary_rows: list[dict]) -> pd.DataFrame:
    frames = []
    for summary in summary_rows:
        if summary.get("status") != "ok" or not summary.get("run_dir"):
            continue
        run_dir = Path(str(summary["run_dir"]))
        path = run_dir / "gurobi_candidate_evaluations.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["status"].eq("ok")].copy()
        if frame.empty:
            continue
        frame["study"] = "unconstrained_frontier"
        frame["environmental_risk_case"] = summary["environmental_risk_case"]
        frame["lambda_case"] = summary["lambda_case"]
        frame["lambda_R"] = float(summary["lambda_R_true"])
        frame["lambda_L"] = float(summary["lambda_L_true"])
        frame["run_dir"] = str(run_dir)
        frame["line_id_key"] = frame["deenergized_line_ids"].apply(_line_id_key)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _unique_candidate_points(all_points: pd.DataFrame) -> pd.DataFrame:
    if all_points.empty:
        return all_points.copy()
    sort_cols = [
        "environmental_risk_case",
        "line_id_key",
        "true_R_norm",
        "true_L_shed",
        "num_deenergized_lines",
        "lambda_R",
    ]
    unique = all_points.sort_values(sort_cols, kind="mergesort").drop_duplicates(
        subset=["environmental_risk_case", "y_vector"],
        keep="first",
    )
    return unique.reset_index(drop=True)


def _pareto_frontier(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return points.copy()
    rows = []
    for case_name, case_points in points.groupby("environmental_risk_case", sort=True):
        values = case_points[["true_R_norm", "true_L_shed"]].astype(float).to_numpy()
        nondominated = np.ones(len(case_points), dtype=bool)
        for idx, candidate in enumerate(values):
            dominates = (
                (values[:, 0] <= candidate[0] + 1e-12)
                & (values[:, 1] <= candidate[1] + 1e-12)
                & ((values[:, 0] < candidate[0] - 1e-12) | (values[:, 1] < candidate[1] - 1e-12))
            )
            if np.any(dominates):
                nondominated[idx] = False
        frontier = case_points.loc[nondominated].copy()
        frontier["pareto_rank_scope"] = case_name
        rows.append(frontier)
    if not rows:
        return pd.DataFrame()
    frontier = pd.concat(rows, ignore_index=True)
    frontier = frontier.sort_values(
        ["environmental_risk_case", "true_L_shed", "true_R_norm", "num_deenergized_lines"],
        kind="mergesort",
    ).reset_index(drop=True)
    frontier["frontier_point_index"] = frontier.groupby("environmental_risk_case").cumcount()
    return frontier


def _plot_frontier(all_points: pd.DataFrame, frontier: pd.DataFrame, output_path: Path) -> Path:
    if all_points.empty or frontier.empty:
        raise ValueError("Cannot plot frontier without all points and Pareto points.")
    cases = sorted(all_points["environmental_risk_case"].unique())
    fig, axes = plt.subplots(1, len(cases), figsize=(7.0 * len(cases), 5.5), squeeze=False, constrained_layout=True)
    for ax, case_name in zip(axes[0], cases):
        case_points = all_points[all_points["environmental_risk_case"].eq(case_name)]
        case_frontier = frontier[frontier["environmental_risk_case"].eq(case_name)]
        ax.scatter(
            case_points["true_L_shed"],
            case_points["true_R_norm"],
            s=18,
            color="#B8B8B8",
            alpha=0.28,
            linewidths=0,
            label="Evaluated topologies",
        )
        ax.scatter(
            case_frontier["true_L_shed"],
            case_frontier["true_R_norm"],
            s=48,
            color="#D62728",
            alpha=0.92,
            edgecolors="white",
            linewidths=0.6,
            label="Nondominated frontier",
        )
        ordered = case_frontier.sort_values(["true_L_shed", "true_R_norm"], kind="mergesort")
        ax.plot(ordered["true_L_shed"], ordered["true_R_norm"], color="#D62728", linewidth=1.4, alpha=0.8)
        ax.set_title(f"Unconstrained Stage E Frontier: {case_name}")
        ax.set_xlabel("True load shed, L_shed")
        ax.set_ylabel("True wildfire exposure, R_norm")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Pareto Frontier From Lambda Sweep, lambda_R = 0.00..1.00 step 0.05", y=1.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return output_path


def run_stage_e_unconstrained_frontier(
    grouping_top_fraction: float = 0.30,
    models: list[str] | None = None,
    cases: list[str] | None = None,
    lambda_step: float = 0.05,
    evaluation_budget: int = 100,
    proxy_type: str = DEFAULT_PROXY_TYPE,
    clear: bool = False,
) -> Path:
    models = ["gps"] if models is None else models
    cases = list(CASES) if cases is None else cases
    lambdas = _lambda_grid(lambda_step)
    root = _frontier_root()
    if clear and root.exists():
        resolved_root = root.resolve()
        expected = (RESULTS_ROOT / "leq" / "stage_e").resolve()
        if resolved_root.parent != expected:
            raise ValueError(f"Refusing to delete unexpected path: {resolved_root}")
        shutil.rmtree(resolved_root)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    threshold_root = root / compact_fraction_label("t", grouping_top_fraction)
    for model_type in models:
        try:
            context = _build_model_context(model_type, grouping_top_fraction)
            wildfire = context["wildfire"]
            group_summary = context["automatic_artifacts"]["group_summary"]
            for case_name in cases:
                env = apply_environmental_case(wildfire, group_summary, case_name)
                for lambda_R in lambdas:
                    lambda_L = float(1.0 - float(lambda_R))
                    lambda_case_name = _lambda_name(lambda_R)
                    output_root = (
                        threshold_root
                        / model_type
                        / CASE_FOLDERS.get(case_name, case_name)
                        / lambda_case_name
                    )
                    rows.append(
                        _run_unconstrained_lambda_case(
                            context,
                            case_name,
                            env,
                            output_root,
                            grouping_top_fraction,
                            int(evaluation_budget),
                            lambda_case_name,
                            float(lambda_R),
                            lambda_L,
                            proxy_type,
                            generate_run_figures=False,
                        )
                    )
        except Exception as exc:
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            for case_name in cases:
                for lambda_R in lambdas:
                    rows.append(
                        {
                            "stage": "E",
                            "study": "unconstrained_frontier",
                            "method_name": UNCONSTRAINED_METHOD_NAME,
                            "evaluation_mode": EVALUATION_MODE,
                            "model_type": model_type,
                            "environmental_risk_case": case_name,
                            "lambda_case": _lambda_name(lambda_R),
                            "lambda_R": float(lambda_R),
                            "lambda_L": float(1.0 - float(lambda_R)),
                            "lambda_R_master": float(lambda_R),
                            "lambda_L_master": float(1.0 - float(lambda_R)),
                            "lambda_R_true": float(lambda_R),
                            "lambda_L_true": float(1.0 - float(lambda_R)),
                            "lambda_P": 0.0,
                            "grouping_top_fraction": float(grouping_top_fraction),
                            "K": "unconstrained",
                            "evaluation_budget": int(evaluation_budget),
                            "proxy_type": proxy_type,
                            "status": "failed",
                            "error": error_text,
                            "run_dir": "",
                        }
                    )

    summary_frame = pd.DataFrame(rows)
    all_points = _collect_run_candidates(rows)
    unique_points = _unique_candidate_points(all_points)
    frontier = _pareto_frontier(unique_points)

    summary_csv = root / "unconstrained_frontier_summary.csv"
    all_points_csv = root / "all_candidate_points.csv"
    unique_points_csv = root / "unique_candidate_points.csv"
    frontier_csv = root / "pareto_frontier_points.csv"
    write_dataframe(summary_csv, summary_frame)
    write_dataframe(all_points_csv, all_points)
    write_dataframe(unique_points_csv, unique_points)
    write_dataframe(frontier_csv, frontier)

    figure_path = root / "figures" / "unconstrained_pareto_frontier_scatter.png"
    plot_error = ""
    try:
        _plot_frontier(unique_points, frontier, figure_path)
    except Exception as exc:
        plot_error = str(exc)

    write_json(
        root / "unconstrained_frontier_summary.json",
        {
            "study": "unconstrained_frontier",
            "num_runs": int(len(rows)),
            "num_successful_runs": int((summary_frame["status"] == "ok").sum()) if len(summary_frame) else 0,
            "models": models,
            "cases": cases,
            "grouping_top_fraction": float(grouping_top_fraction),
            "lambda_R_values": [float(value) for value in lambdas],
            "lambda_L_rule": "1 - lambda_R",
            "evaluation_budget": int(evaluation_budget),
            "proxy_type": proxy_type,
            "all_candidate_points": int(len(all_points)),
            "unique_candidate_points": int(len(unique_points)),
            "pareto_frontier_points": int(len(frontier)),
            "summary_csv": str(summary_csv),
            "all_candidate_points_csv": str(all_points_csv),
            "unique_candidate_points_csv": str(unique_points_csv),
            "pareto_frontier_points_csv": str(frontier_csv),
            "frontier_scatter_plot": str(figure_path) if not plot_error else None,
            "plot_error": plot_error,
        },
    )
    print(f"[OK] Stage E unconstrained frontier summary written to {summary_csv}")
    return summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouping-top-fraction", type=float, default=0.30)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["gps"])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--lambda-step", type=float, default=0.05)
    parser.add_argument("--evaluation-budget", type=int, default=100)
    parser.add_argument("--proxy-type", choices=["env_loading_base", "env_only"], default=DEFAULT_PROXY_TYPE)
    parser.add_argument("--clear", action="store_true", help="Delete existing unconstrained_frontier results first.")
    args = parser.parse_args()
    run_stage_e_unconstrained_frontier(
        grouping_top_fraction=args.grouping_top_fraction,
        models=args.models,
        cases=args.cases,
        lambda_step=args.lambda_step,
        evaluation_budget=args.evaluation_budget,
        proxy_type=args.proxy_type,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
