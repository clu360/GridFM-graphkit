"""Aggregate, validate, and plot the five-model FT7 refined fine-tuning study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.stage_j_gridsfm_goc500 import summarize_stage_j_complete as base
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.finetune import plot_ft3_ft4_model_comparison as comparison


METHOD_SPECS = {
    "dc": {
        "label": "Guided-DC", "source": "frozen", "source_method": "Guided-DC",
        "selection": "dc", "variant": "guided_dc", "sha256": "",
    },
    "m0": {
        "label": "Guided-GridSFM (released v1.1)", "source": "frozen",
        "source_method": "Guided-GridSFM", "selection": "frozen",
        "variant": "released_v1_1",
        "sha256": "F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831",
    },
    "m1": {
        "label": "Guided-GridSFM (FullTop-1000)", "source": "m1",
        "source_method": "Guided-GridSFM", "selection": "ft",
        "variant": "fulltop_ft_n1000",
        "sha256": "A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD",
    },
    "m2": {
        "label": "Guided-GridSFM (FullTop-1500)", "source": "m2",
        "source_method": "Guided-GridSFM", "selection": "ft",
        "variant": "fulltop_ft_n1000_then_fulltop_n500",
        "sha256": "08EDA70270F787DB42C48B751BECC9DA2062B0482550A5C2A761B4EBA0C94FF6",
    },
    "m3": {
        "label": "Guided-GridSFM (FullTop+N-1)", "source": "m3",
        "source_method": "Guided-GridSFM", "selection": "ft",
        "variant": "fulltop_ft_n1000_then_n1_n500",
        "sha256": "4EC89D36DE80081BE2FC26C14A1BC5A5369D7423B557D71B9DD4A254462F302D",
    },
}
METHOD_ORDER = [spec["label"] for spec in METHOD_SPECS.values()]
METHOD_COLORS = {
    METHOD_SPECS["dc"]["label"]: "#0072B2",
    METHOD_SPECS["m0"]["label"]: "#D55E00",
    METHOD_SPECS["m1"]["label"]: "#6A3D9A",
    METHOD_SPECS["m2"]["label"]: "#009E73",
    METHOD_SPECS["m3"]["label"]: "#CC79A7",
}
CORE_COUNTS = {
    "topology_objectives_all.csv": 7500,
    "candidate_evaluations_all.csv": 150000,
    "method_finalists_all.csv": 75,
    "reference_a_all.csv": 75,
    "reference_b_all.csv": 150,
    "state_fidelity_all.csv": 600,
}
NUMERIC_COLUMNS = {
    "topology_objectives_all.csv": [
        "topology_rank", "j_trade", "r_norm", "l_shed_total", "lambda_r",
    ],
    "candidate_evaluations_all.csv": [
        "topology_rank", "candidate_index", "j_trade", "r_norm",
        "l_shed_total", "lambda_r", "runtime_seconds",
    ],
    "method_finalists_all.csv": [
        "lambda_r", "j_trade", "r_norm", "l_shed_total", "num_shutoffs",
        "pac_total", "j_total",
    ],
    "reference_a_all.csv": [
        "lambda_r", "reference_a_objective", "native_r_norm", "reference_a_r_norm_ac",
        "delta_r_norm_native_minus_ac", "native_l_shed_total",
        "reference_a_l_shed_total", "delta_j_true_native_minus_ac",
        "reference_a_runtime_seconds",
    ],
    "reference_b_all.csv": [
        "lambda_r", "solver_objective", "runtime_seconds", "l_shed_ac_mld", "r_norm_ac_mld",
    ],
    "state_fidelity_all.csv": ["lambda_r", "nmae", "nrmse"],
}
WARM_ORDER = [
    "cold_start", "dc_partial_warm", "gridsfm_m0_full_warm",
    "gridsfm_m1_full_warm", "gridsfm_m2_full_warm",
    "gridsfm_m3_full_warm", "gt_warm",
]
WARM_LABELS = {
    "cold_start": "Cold", "dc_partial_warm": "DC",
    "gridsfm_m0_full_warm": "GridSFM released v1.1",
    "gridsfm_m1_full_warm": "GridSFM FullTop-1000",
    "gridsfm_m2_full_warm": "GridSFM FullTop-1500",
    "gridsfm_m3_full_warm": "GridSFM FullTop+N-1", "gt_warm": "Exact A",
}
WARM_COLORS = {
    "cold_start": "#666666", "dc_partial_warm": "#0072B2",
    "gridsfm_m0_full_warm": "#D55E00", "gridsfm_m1_full_warm": "#6A3D9A",
    "gridsfm_m2_full_warm": "#009E73", "gridsfm_m3_full_warm": "#CC79A7",
    "gt_warm": "#111111",
}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _configure_plot_modules() -> None:
    base.METHOD_ORDER = METHOD_ORDER
    base.METHOD_COLORS = METHOD_COLORS
    comparison.METHOD_ORDER = METHOD_ORDER
    comparison.METHOD_COLORS = METHOD_COLORS
    comparison.WARM_START_ORDER = WARM_ORDER
    comparison.WARM_START_LABELS = WARM_LABELS
    comparison.WARM_START_COLORS = WARM_COLORS


def _roots(config: dict[str, Any]) -> dict[str, Path]:
    return {
        "frozen": (REPO_ROOT / config["frozen_result_root"]).resolve(),
        "m1": (REPO_ROOT / config["m1_result_root"]).resolve(),
        "m2": Path(config["models"]["m2"]["working_result_root"]).resolve(),
        "m3": Path(config["models"]["m3"]["working_result_root"]).resolve(),
    }


def _combine_table(filename: str, roots: dict[str, Path]) -> pd.DataFrame:
    pieces = []
    for method_id, spec in METHOD_SPECS.items():
        path = roots[str(spec["source"])] / "core_results" / filename
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[frame["method"] == spec["source_method"]].copy()
        frame["source_checkpoint_sha256"] = (
            frame["checkpoint_sha256"].fillna("") if "checkpoint_sha256" in frame else ""
        )
        frame["source_model_variant"] = (
            frame["model_variant"].fillna("") if "model_variant" in frame else ""
        )
        frame["method"] = spec["label"]
        frame["comparison_variant"] = method_id
        frame["source_package"] = spec["source"]
        frame["model_selection"] = spec["selection"]
        frame["model_variant"] = spec["variant"]
        frame["checkpoint_sha256"] = spec["sha256"]
        pieces.append(frame)
    combined = pd.concat(pieces, ignore_index=True, sort=False)
    for column in NUMERIC_COLUMNS[filename]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def _write_core(config: dict[str, Any]) -> tuple[Path, dict[str, pd.DataFrame]]:
    working = Path(config["working_root"]).resolve()
    core = working / "combined/core_results"
    core.mkdir(parents=True, exist_ok=True)
    roots = _roots(config)
    tables = {filename: _combine_table(filename, roots) for filename in CORE_COUNTS}
    for filename, frame in tables.items():
        frame.to_csv(core / filename, index=False)
    tables["method_finalists_all.csv"].to_csv(core / "finalist_outcomes_all.csv", index=False)
    runtime = (
        tables["candidate_evaluations_all.csv"]
        .groupby(["scenario_id", "lambda_r", "method"], dropna=False)
        .agg(
            candidate_evaluations=("candidate_index", "size"),
            candidate_runtime_seconds=("runtime_seconds", "sum"),
            best_j_trade=("j_trade", "min"),
        ).reset_index()
    )
    runtime.to_csv(core / "candidate_runtime_summary.csv", index=False)
    return working, tables


def _validate_core(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    observed: dict[str, int] = {}
    expected_methods = set(METHOD_ORDER)
    for filename, expected in CORE_COUNTS.items():
        frame = tables[filename]
        observed[filename] = len(frame)
        checks[f"{filename}_rows"] = len(frame) == expected
        checks[f"{filename}_five_methods"] = set(frame["method"]) == expected_methods
        checks[f"{filename}_no_th"] = not frame["method"].astype(str).str.startswith("TH-").any()
        checks[f"{filename}_15_settings_per_method"] = all(
            len(group[["scenario_id", "lambda_r"]].drop_duplicates()) == 15
            for _, group in frame.groupby("method")
        )
        checks[f"{filename}_provenance"] = all(
            (frame.loc[frame["comparison_variant"] == method_id, "checkpoint_sha256"].fillna("")
             == spec["sha256"]).all()
            for method_id, spec in METHOD_SPECS.items()
        )
        checks[f"{filename}_observed_source_provenance"] = all(
            (
                frame.loc[frame["comparison_variant"] == method_id, "source_checkpoint_sha256"]
                == spec["sha256"]
            ).all()
            and (
                frame.loc[frame["comparison_variant"] == method_id, "source_model_variant"]
                == spec["variant"]
            ).all()
            for method_id, spec in METHOD_SPECS.items() if method_id in {"m1", "m2", "m3"}
        )
    candidates = tables["candidate_evaluations_all.csv"]
    topology = tables["topology_objectives_all.csv"]
    checks["candidate_rows_30000_per_method"] = all(
        count == 30000 for count in candidates.groupby("method").size()
    )
    checks["topology_rows_1500_per_method"] = all(
        count == 1500 for count in topology.groupby("method").size()
    )
    checks["candidate_primary_keys_unique"] = not candidates.duplicated(
        ["scenario_id", "lambda_r", "method", "topology_rank", "candidate_index"]
    ).any()
    checks["topology_primary_keys_unique"] = not topology.duplicated(
        ["scenario_id", "lambda_r", "method", "topology_rank"]
    ).any()
    finalists = tables["method_finalists_all.csv"]
    checks["finalist_primary_keys_unique"] = not finalists.duplicated(
        ["scenario_id", "lambda_r", "method"]
    ).any()
    reference_a = tables["reference_a_all.csv"]
    reference_b = tables["reference_b_all.csv"]
    checks["reference_a_all_successful"] = reference_a["reference_a_status"].isin(
        ["LOCALLY_SOLVED", "OPTIMAL"]
    ).all()
    checks["reference_b_all_successful"] = reference_b["status"].isin(
        ["LOCALLY_SOLVED", "OPTIMAL"]
    ).all()
    b2 = reference_b[reference_b["reference_b_stage"] == "B2_cost_tiebreak"]
    checks["reference_b2_service_lock_verified"] = (
        b2["service_lock_satisfied_with_solver_tolerance"].astype(str).str.lower().eq("true").all()
    )
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "FT7_P2_EXACT_AND_CORE_PASS" if all(checks.values()) else "FT7_CORE_BLOCKED",
        "checks": checks, "expected_rows": CORE_COUNTS, "observed_rows": observed,
        "method_order": METHOD_ORDER,
    }


def _pareto_and_primary_plots(
    working: Path, tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    core = working / "combined/core_results"
    derived = core / "derived_visual_summaries"
    figures = working / "combined/figures"
    derived.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    topology = tables["topology_objectives_all.csv"].copy()
    topology["best_found"] = topology["best_found"].astype(str).str.lower().eq("true")
    candidates = tables["candidate_evaluations_all.csv"]
    frontiers = base.build_evaluated_pareto_frontiers(candidates)
    base._validate_pareto_frontiers(candidates, frontiers)
    frontiers.to_csv(derived / "evaluated_native_pareto_frontiers.csv", index=False)
    for scenario in sorted(topology["scenario_id"].dropna().unique()):
        base._save_topology_figure(topology, scenario, figures)
        base._save_convergence_figure(candidates, scenario, figures)
        base._save_pareto_figure(frontiers, scenario, figures)
    finite = candidates[
        candidates["evaluation_status"].astype(str).str.lower().isin(base.PARETO_ELIGIBLE_STATUSES)
        & np.isfinite(candidates["r_norm"]) & np.isfinite(candidates["l_shed_total"])
    ]
    expected_groups = {(scenario, method) for scenario in ("J-S1", "J-S2", "J-S3") for method in METHOD_ORDER}
    observed_groups = set(map(tuple, frontiers[["scenario_id", "method"]].drop_duplicates().to_numpy()))
    metadata = {
        "status": "FT7_P3_ALL_CANDIDATE_PARETO_PASS" if observed_groups == expected_groups else "FT7_PARETO_BLOCKED",
        "scope": "all eligible evaluated candidates pooled across lambda per scenario and model",
        "eligible_rows": len(finite), "frontier_rows": len(frontiers),
        "groups": len(observed_groups), "expected_groups": len(expected_groups),
        "group_counts": {
            f"{scenario}|{method}": int(len(group))
            for (scenario, method), group in frontiers.groupby(["scenario_id", "method"])
        },
    }
    return frontiers, metadata


def _reference_plots(
    working: Path, tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    core = working / "combined/core_results"
    derived = core / "derived_visual_summaries"
    figures = working / "combined/figures"
    data = {
        "reference_a": tables["reference_a_all.csv"],
        "reference_b": tables["reference_b_all.csv"],
        "fidelity": tables["state_fidelity_all.csv"],
    }
    outputs = {
        "guided_reference_a_discrepancy_distance.csv": comparison._plot_reference_a(data, figures),
        "guided_reference_b_mld.csv": comparison._plot_reference_b(data, figures),
        "guided_state_distance_by_metric.csv": comparison._plot_fidelity(data, figures),
    }
    for filename, frame in outputs.items():
        frame.to_csv(derived / filename, index=False)
    return outputs


def _warm_plots(working: Path) -> dict[str, int | str]:
    controlled = working / "warm_start_iteration_rerun/warm_start_iteration_rerun.csv"
    source = controlled if controlled.is_file() else working / "warm_start/warm_start_crossed.csv"
    if not source.is_file():
        return {"warm_start_rows": 0, "warm_start_figures": 0}
    if source == controlled:
        validation = _read_json(
            controlled.parent / "FT7_ITERATION_RERUN_VALIDATION.json"
        )
        if validation.get("status") != "FT7_ITERATION_RERUN_PASS":
            raise RuntimeError(f"controlled warm-start rerun is not validated: {validation}")
    core = working / "combined/core_results"
    derived = core / "derived_visual_summaries"
    figures = working / "combined/figures"
    rows = pd.read_csv(source, low_memory=False)
    for column in (
        "lambda_r", "solver_runtime_seconds", "wall_seconds",
        "start_construction_seconds", "end_to_end_seconds", "iteration_count",
    ):
        if column in rows:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows.to_csv(core / "warm_start_crossed.csv", index=False)
    warm_rows, summary, paired = comparison._plot_warm_start({"warm": rows}, figures)
    warm_rows.to_csv(derived / "guided_warm_start_speedups.csv", index=False)
    summary.to_csv(derived / "guided_warm_start_summary.csv", index=False)
    paired.to_csv(derived / "guided_warm_start_paired_comparisons.csv", index=False)
    overall, win_loss = comparison._plot_ipopt_timing_summary(warm_rows, figures)
    overall.to_csv(derived / "ipopt_warm_start_summary.csv", index=False)
    win_loss.to_csv(derived / "ipopt_warm_start_win_loss.csv", index=False)
    iteration_summary, iteration_comparison = comparison._plot_ipopt_iteration_summary(
        warm_rows, figures
    )
    iteration_summary.to_csv(derived / "ipopt_iteration_summary.csv", index=False)
    iteration_comparison.to_csv(derived / "ipopt_iteration_paired_comparisons.csv", index=False)
    return {
        "warm_start_rows": len(rows), "warm_start_figures": 3,
        "warm_start_source": "controlled_iteration_rerun" if source == controlled else "legacy_crossed",
    }


def run(config_path: Path, *, skip_warm: bool) -> dict[str, Any]:
    config = _read_json(config_path)
    _configure_plot_modules()
    working, tables = _write_core(config)
    core_validation = _validate_core(tables)
    _write_json(working / "combined/FT7_CORE_VALIDATION.json", core_validation)
    if core_validation["status"] != "FT7_P2_EXACT_AND_CORE_PASS":
        raise RuntimeError(f"FT7 core validation failed: {core_validation['checks']}")
    frontiers, pareto = _pareto_and_primary_plots(working, tables)
    _write_json(working / "combined/FT7_PARETO_VALIDATION.json", pareto)
    if pareto["status"] != "FT7_P3_ALL_CANDIDATE_PARETO_PASS":
        raise RuntimeError(f"FT7 Pareto validation failed: {pareto}")
    reference = _reference_plots(working, tables)
    warm = {"warm_start_rows": 0, "warm_start_figures": 0} if skip_warm else _warm_plots(working)
    if not skip_warm and warm["warm_start_rows"] != 525:
        raise RuntimeError(f"FT7 warm-start source is incomplete: {warm}")
    figure_count = len(list((working / "combined/figures").glob("*.png")))
    status = {
        "status": "FT7_P3_FIGURES_PASS" if skip_warm else "FT7_P4_SUMMARY_PASS",
        "core_validation": core_validation["status"], "pareto_validation": pareto["status"],
        "pareto_frontier_rows": len(frontiers), "reference_summary_tables": len(reference),
        **warm, "figure_count": figure_count, "methods": METHOD_ORDER, "th_included": False,
    }
    _write_json(working / "combined/FT7_SUMMARY_STATUS.json", status)
    print(json.dumps(status, indent=2))
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--skip-warm", action="store_true")
    args = parser.parse_args()
    run(args.config.resolve(), skip_warm=args.skip_warm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
