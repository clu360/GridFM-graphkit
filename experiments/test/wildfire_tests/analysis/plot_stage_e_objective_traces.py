from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT


STAGE_E_ROOT = RESULTS_ROOT / "leq" / "stage_e" / "gurobi_gridfm"
SUMMARY_ROOT = STAGE_E_ROOT / "experiment_summaries"
FIGURE_ROOT = SUMMARY_ROOT / "figures" / "stage_e_only"


def _safe_name(value: str) -> str:
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")


def _case_code(value: str) -> str:
    mapping = {"auto_env": "auto", "largest_group_high": "lgh"}
    return mapping.get(str(value), _safe_name(str(value))[:12])


def _lambda_code(value: str) -> str:
    mapping = {"risk_priority": "risk", "balanced": "bal", "load_priority": "load"}
    return mapping.get(str(value), _safe_name(str(value))[:8])


def _stage_e_summary_path(stage_e_root: Path) -> Path:
    candidates = [
        stage_e_root / "stage_e_gurobi_gridfm_summary.csv",
        SUMMARY_ROOT / "stage_e_real_experiment_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find Stage E summary CSV.")


def _short_topology_label(value: object) -> str:
    text = str(value).strip()
    if text == "[]" or not text:
        return "none"
    return text.replace(" ", "")


def _load_candidates(row: pd.Series) -> pd.DataFrame:
    path = Path(str(row["run_dir"])) / "gurobi_candidate_evaluations.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage E candidates: {path}")
    candidates = pd.read_csv(path)
    candidates = candidates[candidates["status"].eq("ok")].copy()
    candidates["candidate_eval"] = range(1, len(candidates) + 1)
    candidates["objective"] = candidates["true_objective"].astype(float)
    candidates["best_so_far"] = candidates["objective"].cummin()
    candidates["topology_label"] = candidates["deenergized_line_ids"].apply(_short_topology_label)
    return candidates


def _annotate_key_points(ax, candidates: pd.DataFrame) -> None:
    best_indices = candidates.index[candidates["best_so_far"].diff().fillna(-1).lt(0)].tolist()
    final_best_index = int(candidates["objective"].idxmin())
    key_indices = sorted(set(best_indices + [final_best_index, int(candidates.index[0])]))
    for idx in key_indices:
        row = candidates.loc[idx]
        ax.annotate(
            row["topology_label"],
            xy=(row["candidate_eval"], row["objective"]),
            xytext=(4, 8),
            textcoords="offset points",
            fontsize=8,
            rotation=25,
            ha="left",
            va="bottom",
        )


def _plot_single_run(row: pd.Series, candidates: pd.DataFrame, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = str(row["model_type"])
    case = str(row["environmental_risk_case"])
    lambda_case = str(row["lambda_case"])
    k = int(row["K"])
    lambda_r = float(row.get("lambda_R_true", row.get("lambda_R")))
    lambda_l = float(row.get("lambda_L_true", row.get("lambda_L")))
    best_row = candidates.loc[candidates["objective"].idxmin()]

    fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    ax.plot(
        candidates["candidate_eval"],
        candidates["objective"],
        color="#7A7A7A",
        linewidth=1.25,
        alpha=0.55,
        label="Evaluated topology objective",
    )
    ax.scatter(
        candidates["candidate_eval"],
        candidates["objective"],
        color="#F58518",
        s=30,
        alpha=0.8,
        label="Stage E GridFM evaluations",
    )
    ax.step(
        candidates["candidate_eval"],
        candidates["best_so_far"],
        where="post",
        color="#1F77B4",
        linewidth=2.4,
        label="Best objective so far",
    )
    ax.scatter(
        [best_row["candidate_eval"]],
        [best_row["objective"]],
        color="#D62728",
        s=70,
        zorder=5,
        label="Final best topology",
    )
    _annotate_key_points(ax, candidates)

    ax.set_title(
        f"Stage E Objective Trace: {model} / {case} / {lambda_case} / K={k}\n"
        f"J = {lambda_r:g} R_norm + {lambda_l:g} L_shed; final best = {best_row['topology_label']}"
    )
    ax.set_xlabel("Stage E candidate topology evaluation")
    ax.set_ylabel("True GridFM objective")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "model_type": model,
        "environmental_risk_case": case,
        "lambda_case": lambda_case,
        "K": k,
        "lambda_R_true": lambda_r,
        "lambda_L_true": lambda_l,
        "num_evaluations": int(len(candidates)),
        "final_best_topology": str(best_row["deenergized_line_ids"]),
        "final_best_iteration": int(best_row["iteration"]),
        "final_best_objective": float(best_row["objective"]),
        "output_path": str(output_path),
    }


def generate_stage_e_only_traces(
    *,
    stage_e_root: Path = STAGE_E_ROOT,
    figure_root: Path = FIGURE_ROOT,
) -> list[dict]:
    figure_root.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(_stage_e_summary_path(stage_e_root))
    if "K" not in summary.columns and "max_deenergized_lines" in summary.columns:
        summary["K"] = summary["max_deenergized_lines"]
    summary = summary[summary["status"].eq("ok")].copy()

    records: list[dict] = []
    for _, row in summary.iterrows():
        candidates = _load_candidates(row)
        output_path = (
            figure_root
            / f"e_trace_{_safe_name(row['model_type'])}_{_case_code(row['environmental_risk_case'])}_{_lambda_code(row['lambda_case'])}_k{int(row['K'])}.png"
        )
        records.append(_plot_single_run(row, candidates, output_path))

    manifest_path = figure_root / "stage_e_objective_trace_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage E-only objective traces.")
    parser.add_argument("--stage-e-root", type=Path, default=STAGE_E_ROOT)
    parser.add_argument("--figure-root", type=Path, default=FIGURE_ROOT)
    args = parser.parse_args()
    records = generate_stage_e_only_traces(stage_e_root=args.stage_e_root, figure_root=args.figure_root)
    print(f"Wrote {len(records)} Stage E objective trace plots to {args.figure_root}")


if __name__ == "__main__":
    main()
