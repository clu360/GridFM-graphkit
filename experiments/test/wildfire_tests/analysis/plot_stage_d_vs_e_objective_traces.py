from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    STAGE_E_LAMBDA_CASES,
)


STAGE_E_ROOT = RESULTS_ROOT / "leq" / "stage_e" / "gurobi_gridfm"
SUMMARY_ROOT = STAGE_E_ROOT / "experiment_summaries"
FIGURE_ROOT = SUMMARY_ROOT / "figures"


def _stage_e_summary_path(stage_e_root: Path) -> Path:
    candidates = [
        stage_e_root / "stage_e_gurobi_gridfm_summary.csv",
        stage_e_root / "stage_e_real_experiment_summary.csv",
        SUMMARY_ROOT / "stage_e_real_experiment_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a Stage E summary CSV. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _stage_d_candidates_path(summary_root: Path, model: str, case: str, k: int) -> Path:
    path = summary_root / f"stage_d_revised_candidates_{model}_{case}_k{k}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing revised Stage D candidate file: {path}")
    return path


def _load_stage_e_candidates(row: pd.Series) -> pd.DataFrame:
    run_dir = Path(str(row["run_dir"]))
    path = run_dir / "gurobi_candidate_evaluations.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage E candidate file: {path}")
    candidates = pd.read_csv(path)
    candidates = candidates[candidates["status"].eq("ok")].copy()
    candidates["objective"] = candidates["true_objective"].astype(float)
    candidates["best_so_far"] = candidates["objective"].cummin()
    candidates["candidate_eval"] = range(1, len(candidates) + 1)
    return candidates


def _load_stage_d_candidates(
    summary_root: Path,
    model: str,
    case: str,
    k: int,
    lambda_r: float,
    lambda_l: float,
) -> pd.DataFrame:
    path = _stage_d_candidates_path(summary_root, model, case, k)
    candidates = pd.read_csv(path)
    candidates = candidates[candidates["status"].eq("ok")].copy()
    candidates["objective"] = (
        float(lambda_r) * candidates["true_R_norm"].astype(float)
        + float(lambda_l) * candidates["true_L_shed"].astype(float)
    )
    candidates["best_so_far"] = candidates["objective"].cummin()
    candidates["candidate_eval"] = range(1, len(candidates) + 1)
    return candidates


def _plot_case_k_lambda(
    *,
    stage_e_row: pd.Series,
    stage_d: pd.DataFrame,
    stage_e: pd.DataFrame,
    output_path: Path,
) -> dict:
    model = str(stage_e_row["model_type"])
    case = str(stage_e_row["environmental_risk_case"])
    lambda_case = str(stage_e_row["lambda_case"])
    lambda_r = float(stage_e_row.get("lambda_R_true", stage_e_row.get("lambda_R")))
    lambda_l = float(stage_e_row.get("lambda_L_true", stage_e_row.get("lambda_L")))
    k = int(stage_e_row["K"])

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    ax.plot(
        stage_d["candidate_eval"],
        stage_d["best_so_far"],
        color="#4C78A8",
        linewidth=2.0,
        label="Stage D revised enumeration",
    )
    ax.scatter(
        stage_d["candidate_eval"],
        stage_d["objective"],
        color="#4C78A8",
        alpha=0.18,
        s=12,
        linewidths=0,
    )
    ax.plot(
        stage_e["candidate_eval"],
        stage_e["best_so_far"],
        color="#F58518",
        linewidth=2.0,
        label="Stage E Gurobi + GridFM",
    )
    ax.scatter(
        stage_e["candidate_eval"],
        stage_e["objective"],
        color="#F58518",
        alpha=0.55,
        s=22,
        linewidths=0,
    )
    ax.set_title(
        f"{model} / {case} / {lambda_case} / K={k}\n"
        f"J = {lambda_r:g} R_norm + {lambda_l:g} L_shed"
    )
    ax.set_xlabel("Candidate topology evaluation")
    ax.set_ylabel("Best-so-far true objective")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "model_type": model,
        "environmental_risk_case": case,
        "lambda_case": lambda_case,
        "lambda_R_true": lambda_r,
        "lambda_L_true": lambda_l,
        "K": k,
        "stage_d_evaluations": int(len(stage_d)),
        "stage_e_evaluations": int(len(stage_e)),
        "stage_d_final_best": float(stage_d["best_so_far"].iloc[-1]),
        "stage_e_final_best": float(stage_e["best_so_far"].iloc[-1]),
        "output_path": str(output_path),
    }


def _plot_k_overview(k: int, records: list[dict], output_path: Path) -> None:
    if not records:
        return
    cases = sorted({record["environmental_risk_case"] for record in records})
    lambdas = [name for name in STAGE_E_LAMBDA_CASES if any(r["lambda_case"] == name for r in records)]
    fig, axes = plt.subplots(
        len(cases),
        len(lambdas),
        figsize=(5.2 * len(lambdas), 3.8 * len(cases)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, case in enumerate(cases):
        for col_idx, lambda_case in enumerate(lambdas):
            ax = axes[row_idx][col_idx]
            match = next(
                (
                    record
                    for record in records
                    if record["environmental_risk_case"] == case
                    and record["lambda_case"] == lambda_case
                ),
                None,
            )
            if match is None:
                ax.axis("off")
                continue
            stage_d = match["_stage_d"]
            stage_e = match["_stage_e"]
            ax.plot(stage_d["candidate_eval"], stage_d["best_so_far"], color="#4C78A8", linewidth=2.0)
            ax.plot(stage_e["candidate_eval"], stage_e["best_so_far"], color="#F58518", linewidth=2.0)
            ax.scatter(stage_e["candidate_eval"], stage_e["objective"], color="#F58518", alpha=0.45, s=16)
            ax.set_title(
                f"{case} / {lambda_case}\n"
                f"J = {match['lambda_R_true']:g} R_norm + {match['lambda_L_true']:g} L_shed"
            )
            ax.set_xlabel("Evaluation")
            ax.set_ylabel("Best objective")
            ax.grid(True, alpha=0.25)

    handles = [
        plt.Line2D([0], [0], color="#4C78A8", linewidth=2.0, label="Stage D revised"),
        plt.Line2D([0], [0], color="#F58518", linewidth=2.0, label="Stage E"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2)
    fig.suptitle(f"Objective Trace Comparison, K={k}", y=1.02)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def generate_objective_trace_plots(
    *,
    stage_e_root: Path = STAGE_E_ROOT,
    summary_root: Path = SUMMARY_ROOT,
    figure_root: Path = FIGURE_ROOT,
) -> list[dict]:
    figure_root.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(_stage_e_summary_path(stage_e_root))
    if "K" not in summary.columns and "max_deenergized_lines" in summary.columns:
        summary["K"] = summary["max_deenergized_lines"]
    summary = summary[summary["status"].eq("ok")].copy()

    records: list[dict] = []
    overview_records_by_k: dict[int, list[dict]] = {}
    for _, row in summary.iterrows():
        model = str(row["model_type"])
        case = str(row["environmental_risk_case"])
        lambda_case = str(row["lambda_case"])
        k = int(row["K"])
        lambda_r = float(row.get("lambda_R_true", row.get("lambda_R", STAGE_E_LAMBDA_CASES[lambda_case][0])))
        lambda_l = float(row.get("lambda_L_true", row.get("lambda_L", STAGE_E_LAMBDA_CASES[lambda_case][1])))

        stage_d = _load_stage_d_candidates(summary_root, model, case, k, lambda_r, lambda_l)
        stage_e = _load_stage_e_candidates(row)
        output_path = (
            figure_root
            / f"objective_trace_{_safe_name(model)}_{_safe_name(case)}_{_safe_name(lambda_case)}_k{k}.png"
        )
        record = _plot_case_k_lambda(
            stage_e_row=row,
            stage_d=stage_d,
            stage_e=stage_e,
            output_path=output_path,
        )
        records.append(record)
        overview_record = dict(record)
        overview_record["_stage_d"] = stage_d
        overview_record["_stage_e"] = stage_e
        overview_records_by_k.setdefault(k, []).append(overview_record)

    for k, k_records in sorted(overview_records_by_k.items()):
        overview_path = figure_root / f"objective_trace_overview_k{k}.png"
        _plot_k_overview(k, k_records, overview_path)
        records.append(
            {
                "plot_type": "k_overview",
                "K": int(k),
                "output_path": str(overview_path),
                "num_panels": int(len(k_records)),
            }
        )

    manifest_path = figure_root / "objective_trace_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot revised Stage D vs Stage E objective traces.")
    parser.add_argument("--stage-e-root", type=Path, default=STAGE_E_ROOT)
    parser.add_argument("--summary-root", type=Path, default=SUMMARY_ROOT)
    parser.add_argument("--figure-root", type=Path, default=FIGURE_ROOT)
    args = parser.parse_args()

    records = generate_objective_trace_plots(
        stage_e_root=args.stage_e_root,
        summary_root=args.summary_root,
        figure_root=args.figure_root,
    )
    print(f"Wrote {len(records)} objective trace plot records to {args.figure_root}")


if __name__ == "__main__":
    main()
