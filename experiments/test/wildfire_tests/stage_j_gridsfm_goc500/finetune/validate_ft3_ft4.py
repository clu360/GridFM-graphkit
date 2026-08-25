"""Validate linked FT3 raw evidence and FT4 reproduced artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {
    "topology_objectives_all.csv": 1530,
    "candidate_evaluations_all.csv": 30600,
    "method_finalists_all.csv": 45,
    "reference_a_all.csv": 45,
    "reference_b_all.csv": 90,
    "warm_start_all.csv": 225,
    "state_fidelity_all.csv": 360,
}
EXPECTED_FIGURES = [
    *(f"evaluated_risk_load_scatter_j-s{i}.png" for i in (1, 2, 3)),
    *(f"j_trade_by_topology_rank_j-s{i}.png" for i in (1, 2, 3)),
    *(f"j_trade_convergence_j-s{i}.png" for i in (1, 2, 3)),
    "stage_j_guided_reference_a_discrepancy_distance.png",
    "stage_j_guided_reference_b_mld.png",
    "stage_j_guided_state_distance_heatmap.png",
    "stage_j_guided_warm_start_study.png",
    "stage_j_ipopt_warm_start_summary.png",
]
EXPECTED_COMPARISON_METHODS = {
    "Guided-DC",
    "Guided-GridSFM (frozen)",
    "Guided-GridSFM (fine-tuned)",
    "TH-GridSFM-top1",
    "TH-GridSFM-top2",
}
EXPECTED_COMPARISON_VARIANTS = {"dc", "frozen", "ft", "th_frozen"}
EXPECTED_CROSSED_STARTS = {
    "cold_start",
    "dc_partial_warm",
    "gridsfm_frozen_full_warm",
    "gridsfm_ft_full_warm",
    "gt_warm",
}
COMPARISON_FILES = [
    "compare_topology.csv",
    "compare_candidates.csv",
    "compare_ref_a.csv",
    "compare_ref_b.csv",
    "compare_fidelity.csv",
    "compare_warm.csv",
]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate(root: Path) -> dict[str, Any]:
    status = _read_json(root / "RUN_STATUS.json")
    config = _read_json(root / "RUN_CONFIG.json")
    pools = _read_json(root / "TOPOLOGY_POOL_MANIFEST.json")
    checks: dict[str, bool] = {
        "ft3_complete": status.get("status") == "COMPLETE",
        "model_selection_ft": config.get("model_selection") == "ft",
        "model_variant_correct": config.get("model_variant") == "fulltop_ft_n1000",
        "all_30_pools_verified": len(pools.get("records", [])) == 30 and all(
            row.get("ordered_byte_identical") is True for row in pools.get("records", [])
        ),
    }
    observed_counts: dict[str, int] = {}
    metadata_failures: list[str] = []
    for filename, expected in EXPECTED_COUNTS.items():
        rows = _read_csv(root / "core_results" / filename)
        observed_counts[filename] = len(rows)
        checks[f"count_{filename}"] = len(rows) == expected
        for index, row in enumerate(rows):
            if row.get("model_selection") != "ft" or row.get("model_variant") != "fulltop_ft_n1000":
                metadata_failures.append(f"{filename}:{index}")
                break
    checks["all_core_rows_identify_ft"] = not metadata_failures

    comparison_failures: dict[str, dict[str, list[str]]] = {}
    for filename in COMPARISON_FILES:
        path = root / "core_results" / filename
        rows = _read_csv(path) if path.is_file() else []
        methods = {row.get("method", "") for row in rows}
        variants = {row.get("comparison_variant", "") for row in rows}
        checks[f"comparison_{filename}"] = (
            methods == EXPECTED_COMPARISON_METHODS
            and variants == EXPECTED_COMPARISON_VARIANTS
        )
        if not checks[f"comparison_{filename}"]:
            comparison_failures[filename] = {
                "methods": sorted(methods),
                "variants": sorted(variants),
            }

    crossed_path = root / "core_results" / "warm_start_crossed.csv"
    crossed_rows = _read_csv(crossed_path) if crossed_path.is_file() else []
    crossed_combinations = {
        (row.get("method", ""), row.get("warm_start_type", "")) for row in crossed_rows
    }
    checks["crossed_warm_start_rows_375"] = len(crossed_rows) == 375
    checks["crossed_warm_start_five_by_five"] = (
        {row.get("method", "") for row in crossed_rows} == EXPECTED_COMPARISON_METHODS
        and {row.get("warm_start_type", "") for row in crossed_rows} == EXPECTED_CROSSED_STARTS
        and len(crossed_combinations) == 25
        and all(
            sum(
                row.get("method") == method and row.get("warm_start_type") == warm_type
                for row in crossed_rows
            ) == 15
            for method, warm_type in crossed_combinations
        )
    )
    checks["crossed_warm_start_validation_passed"] = (
        (root / "FT5_CROSSED_WARM_START_VALIDATION.json").is_file()
        and _read_json(root / "FT5_CROSSED_WARM_START_VALIDATION.json").get("status") == "PASS"
    )
    ipopt_validation_path = root / "FT5_IPOPT_TIMING_VALIDATION.json"
    ipopt_timing_path = root / "core_results" / "warm_start_ipopt_timing.csv"
    ipopt_rows = _read_csv(ipopt_timing_path) if ipopt_timing_path.is_file() else []
    checks["ipopt_timing_rows_375"] = len(ipopt_rows) == 375
    checks["ipopt_timing_validation_passed"] = (
        ipopt_validation_path.is_file()
        and _read_json(ipopt_validation_path).get("status") == "PASS"
    )
    checks["canonical_primary_metric_is_ipopt"] = bool(crossed_rows) and all(
        row.get("runtime_metric") == "ipopt_solve_time_seconds"
        and row.get("solver_runtime_seconds") == row.get("ipopt_solve_time_seconds")
        for row in crossed_rows
    )

    figures = {
        name: (root / "figures" / name).stat().st_size
        if (root / "figures" / name).is_file() else -1
        for name in EXPECTED_FIGURES
    }
    checks["all_expected_figures_nonempty"] = all(size > 0 for size in figures.values())
    derived = root / "core_results" / "derived_visual_summaries"
    checks["derived_visual_tables_present"] = derived.is_dir() and len(list(derived.glob("*.csv"))) >= 8
    pareto_path = derived / "evaluated_native_pareto_frontiers.csv"
    pareto_rows = _read_csv(pareto_path) if pareto_path.is_file() else []
    pareto_groups = {
        (row.get("scenario_id", ""), row.get("method", "")) for row in pareto_rows
    }
    expected_pareto_groups = {
        (scenario, method)
        for scenario in ("J-S1", "J-S2", "J-S3")
        for method in EXPECTED_COMPARISON_METHODS
    }
    checks["evaluated_native_pareto_rows_343"] = len(pareto_rows) == 343
    checks["evaluated_native_pareto_five_methods_per_scenario"] = (
        pareto_groups == expected_pareto_groups
    )
    checks["evaluated_native_pareto_metadata_consistent"] = bool(pareto_rows) and all(
        row.get("frontier_scope") == "scenario_method_all_lambda_evaluated_native"
        and row.get("evaluation_status") in {"ok", "model_output_penalized"}
        and int(row.get("pareto_front_size", 0))
        == sum(
            other.get("scenario_id") == row.get("scenario_id")
            and other.get("method") == row.get("method")
            for other in pareto_rows
        )
        for row in pareto_rows
    )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "expected_counts": EXPECTED_COUNTS,
        "observed_counts": observed_counts, "metadata_failures": metadata_failures,
        "comparison_failures": comparison_failures,
        "expected_comparison_methods": sorted(EXPECTED_COMPARISON_METHODS),
        "expected_comparison_variants": sorted(EXPECTED_COMPARISON_VARIANTS),
        "figure_size_bytes": figures,
        "evaluated_native_pareto_rows": len(pareto_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    root = Path(args.result_root).expanduser()
    if not root.is_absolute():
        root = root.resolve()
    payload = validate(root)
    _write_json(root / "FT4_VALIDATION.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
