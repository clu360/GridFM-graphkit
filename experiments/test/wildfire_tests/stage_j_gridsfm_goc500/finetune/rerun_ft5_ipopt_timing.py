"""Rerun the established FT5 starts and isolate IPOPT-native solve time."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
STAGE_J_ROOT = Path(__file__).resolve().parents[1]
if str(STAGE_J_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_J_ROOT))

from run_j9_j10_reference_smoke import _line_ids, _read_json, _run_julia


SUCCESS = {"LOCALLY_SOLVED", "OPTIMAL"}
START_ORDER = [
    "cold_start",
    "dc_partial_warm",
    "gridsfm_frozen_full_warm",
    "gridsfm_ft_full_warm",
    "gt_warm",
]
START_STUBS = {
    "cold_start": "cold",
    "dc_partial_warm": "dc",
    "gridsfm_frozen_full_warm": "gridsfm_frozen",
    "gridsfm_ft_full_warm": "gridsfm_ft",
    "gt_warm": "exact",
}
FROZEN_LABELS = {
    "Guided-DC": "Guided-DC",
    "Guided-GridSFM": "Guided-GridSFM (frozen)",
    "TH-GridSFM-top1": "TH-GridSFM-top1",
    "TH-GridSFM-top2": "TH-GridSFM-top2",
}
FT_LABELS = {"Guided-GridSFM": "Guided-GridSFM (fine-tuned)"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty timing table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _setting_dirs(root: Path) -> list[Path]:
    settings = sorted(path for path in root.glob("s*_l*") if (path / "finalists.json").is_file())
    if len(settings) != 15:
        raise RuntimeError(f"expected 15 settings under {root}, observed {len(settings)}")
    return settings


def _state_directory(method_dir: Path, source_package: str, warm_type: str) -> Path | None:
    if warm_type == "cold_start":
        return None
    if warm_type == "dc_partial_warm":
        return method_dir / "wdc"
    if warm_type == "gt_warm":
        return method_dir / "wgt"
    if warm_type == "gridsfm_frozen_full_warm":
        return (
            method_dir / "wgf"
            if source_package == "frozen"
            else method_dir / "crossed_warm" / "released_v1_1" / "state"
        )
    return (
        method_dir / "wgf"
        if source_package == "ft"
        else method_dir / "crossed_warm" / "fulltop_ft_n1000" / "state"
    )


def _tasks(args: argparse.Namespace) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for source_package, cache_root, labels in (
        ("frozen", args.frozen_cache, FROZEN_LABELS),
        ("ft", args.ft_cache, FT_LABELS),
    ):
        for setting in _setting_dirs(cache_root):
            manifest = _read_json(setting / "finalists.json")
            for finalist in manifest["finalists"]:
                raw_method = str(finalist["method"])
                if raw_method not in labels:
                    continue
                best = finalist["best_topology"]
                method_dir = setting / "refs" / str(finalist["artifact_stub"])
                alpha_csv = method_dir / "alpha_effective_full.csv"
                if not alpha_csv.is_file():
                    raise FileNotFoundError(alpha_csv)
                for warm_type in START_ORDER:
                    start_dir = _state_directory(method_dir, source_package, warm_type)
                    if start_dir is not None:
                        for filename in ("gen_start.csv", "bus_start.csv"):
                            if not (start_dir / filename).is_file():
                                raise FileNotFoundError(start_dir / filename)
                    tasks.append({
                        "setting_code": setting.name,
                        "scenario_id": str(best["scenario_id"]),
                        "lambda_r": finalist.get("lambda_r", best.get("lambda_r")),
                        "source_package": source_package,
                        "raw_method": raw_method,
                        "method": labels[raw_method],
                        "warm_start_type": warm_type,
                        "alpha_csv": alpha_csv,
                        "offline_branch_ids": _line_ids(str(best.get("topology_id") or "")),
                        "source_less_load_ids": _line_ids(str(best.get("source_less_load_ids") or "")),
                        "start_dir": start_dir,
                        "output_dir": method_dir / "ipopt_timing" / START_STUBS[warm_type],
                    })
    if len(tasks) != 375:
        raise RuntimeError(f"expected 375 timing tasks, observed {len(tasks)}")
    return tasks


def _summary_path(task: dict[str, object]) -> Path:
    return Path(task["output_dir"]) / "reference_a_summary.json"


def _complete(task: dict[str, object]) -> bool:
    path = _summary_path(task)
    if not path.is_file():
        return False
    summary = _read_json(path)
    return (
        summary.get("termination_status") in SUCCESS
        and isinstance(summary.get("ipopt_solve_time_seconds"), (int, float))
        and float(summary["ipopt_solve_time_seconds"]) >= 0.0
        and isinstance(summary.get("power_models_total_seconds"), (int, float))
        and float(summary["power_models_total_seconds"]) >= float(summary["ipopt_solve_time_seconds"])
    )


def _record(task: dict[str, object], *, wall_seconds: float, cached: bool) -> dict[str, object]:
    summary = _read_json(_summary_path(task))
    return {
        "setting_code": task["setting_code"],
        "scenario_id": task["scenario_id"],
        "lambda_r": task["lambda_r"],
        "source_package": task["source_package"],
        "method": task["method"],
        "warm_start_type": task["warm_start_type"],
        "status": summary.get("termination_status"),
        "objective": summary.get("objective"),
        "ipopt_solve_time_seconds": summary.get("ipopt_solve_time_seconds"),
        "power_models_total_seconds": summary.get("power_models_total_seconds"),
        "model_and_result_overhead_seconds": summary.get("model_and_result_overhead_seconds"),
        "timing_rerun_wall_seconds": wall_seconds,
        "timing_result_cached": cached,
        "solver_start_source": summary.get("start_source"),
        "start_bus_count": summary.get("start_bus_count"),
        "start_gen_count": summary.get("start_gen_count"),
    }


def _run_task(args: argparse.Namespace, task: dict[str, object]) -> dict[str, object]:
    if _complete(task):
        return _record(task, wall_seconds=0.0, cached=True)
    ok, stdout, stderr, wall = _run_julia(
        julia_exe=args.julia_exe,
        julia_depot_path=args.julia_depot,
        script=STAGE_J_ROOT / "stage_j_ac_reference.jl",
        mode="reference_a",
        case_path=args.case_path,
        alpha_csv=Path(task["alpha_csv"]),
        offline_branch_ids=task["offline_branch_ids"],
        source_less_load_ids_=task["source_less_load_ids"],
        output_dir=Path(task["output_dir"]),
        start_dir=None if task["start_dir"] is None else Path(task["start_dir"]),
        timeout_seconds=args.timeout_seconds,
    )
    if not ok or not _complete(task):
        raise RuntimeError(
            f"IPOPT timing solve failed for {task['setting_code']} {task['method']} "
            f"{task['warm_start_type']}: stdout={stdout!r}, stderr={stderr!r}"
        )
    return _record(task, wall_seconds=wall, cached=False)


def _validate(rows: list[dict[str, object]], prior: pd.DataFrame, tolerance: float) -> dict[str, object]:
    keys = ["setting_code", "method", "warm_start_type"]
    timing = pd.DataFrame(rows)
    merged = prior[keys + ["objective"]].merge(
        timing[keys + ["objective"]], on=keys, suffixes=("_prior", "_timing"), validate="one_to_one"
    )
    merged["objective_delta"] = (
        pd.to_numeric(merged["objective_prior"], errors="coerce")
        - pd.to_numeric(merged["objective_timing"], errors="coerce")
    ).abs()
    spread = (
        timing.assign(objective=pd.to_numeric(timing["objective"], errors="coerce"))
        .groupby(["setting_code", "method"])["objective"]
        .agg(lambda values: values.max() - values.min())
    )
    combinations = timing.groupby(["method", "warm_start_type"]).size()
    checks = {
        "row_count_375": len(timing) == 375,
        "unique_keys_375": len(timing.drop_duplicates(keys)) == 375,
        "all_25_combinations_have_15_rows": len(combinations) == 25 and bool((combinations == 15).all()),
        "all_solves_successful": bool(timing["status"].isin(SUCCESS).all()),
        "all_ipopt_times_nonnegative": bool(
            (pd.to_numeric(timing["ipopt_solve_time_seconds"], errors="coerce") >= 0.0).all()
        ),
        "outer_time_not_less_than_ipopt_time": bool((
            pd.to_numeric(timing["power_models_total_seconds"], errors="coerce")
            >= pd.to_numeric(timing["ipopt_solve_time_seconds"], errors="coerce")
        ).all()),
        "objective_spread_within_tolerance": float(spread.max()) <= tolerance,
        "objectives_match_prior_canonical": float(merged["objective_delta"].max()) <= tolerance,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "rows": len(timing),
        "max_objective_spread_across_starts": float(spread.max()),
        "max_objective_delta_vs_prior": float(merged["objective_delta"].max()),
        "objective_tolerance": tolerance,
        "primary_metric": "ipopt_solve_time_seconds",
        "ipopt_metric_source": "PowerModels result['solve_time'] backed by MOI.SolveTimeSec",
    }


def _merge_canonical(rows: list[dict[str, object]], canonical_path: Path) -> pd.DataFrame:
    keys = ["setting_code", "method", "warm_start_type"]
    canonical = pd.read_csv(canonical_path, low_memory=False)
    timing = pd.DataFrame(rows)
    timing_columns = keys + [
        "ipopt_solve_time_seconds",
        "power_models_total_seconds",
        "model_and_result_overhead_seconds",
        "timing_rerun_wall_seconds",
        "objective",
    ]
    timing = timing[timing_columns].rename(columns={"objective": "timing_rerun_objective"})
    for column in timing_columns[3:]:
        if column in canonical:
            canonical = canonical.drop(columns=column)
    if "original_power_models_runtime_seconds" not in canonical:
        canonical["original_power_models_runtime_seconds"] = canonical["solver_runtime_seconds"]
    merged = canonical.merge(timing, on=keys, how="left", validate="one_to_one")
    merged["solver_runtime_seconds"] = merged["ipopt_solve_time_seconds"]
    merged["runtime_metric"] = "ipopt_solve_time_seconds"
    merged.to_csv(canonical_path, index=False)
    return merged


def run(args: argparse.Namespace) -> dict[str, object]:
    tasks = _tasks(args)
    if args.smoke:
        record = _run_task(args, tasks[0])
        if not isinstance(record.get("ipopt_solve_time_seconds"), (int, float)):
            raise RuntimeError("smoke did not produce IPOPT solve time")
        print(json.dumps(record, indent=2, sort_keys=True))
        return {"status": "SMOKE_PASS", "rows": 1}

    started = time.time()
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_task, args, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            rows.append(future.result())
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(json.dumps({
                    "completed": completed,
                    "total": len(tasks),
                    "elapsed_seconds": time.time() - started,
                }), flush=True)
    rows.sort(key=lambda row: (
        str(row["setting_code"]), str(row["method"]), START_ORDER.index(str(row["warm_start_type"]))
    ))
    canonical_path = args.result_root / "core_results" / "warm_start_crossed.csv"
    prior = pd.read_csv(canonical_path, low_memory=False)
    validation = _validate(rows, prior, args.objective_tolerance)
    _write_csv(args.result_root / "core_results" / "warm_start_ipopt_timing.csv", rows)
    if validation["status"] == "PASS":
        _merge_canonical(rows, canonical_path)
    validation["runtime_seconds"] = time.time() - started
    validation["workers"] = args.workers
    _write_json(args.result_root / "FT5_IPOPT_TIMING_VALIDATION.json", validation)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-cache", type=Path,
        default=Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\complete_run_v001"),
    )
    parser.add_argument(
        "--ft-cache", type=Path,
        default=Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\ft3_fulltop_ft_n1000_v002"),
    )
    parser.add_argument(
        "--result-root", type=Path,
        default=Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\results\stage_j\fulltop_ft_n1000_complete_run"),
    )
    parser.add_argument(
        "--julia-exe", type=Path,
        default=Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\tools\julia-1.10.11\bin\julia.exe"),
    )
    parser.add_argument(
        "--julia-depot", type=Path,
        default=Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\julia_depot"),
    )
    parser.add_argument(
        "--case-path", type=Path,
        default=Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\repos\pglib-opf\pglib_opf_case500_goc.m"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--objective-tolerance", type=float, default=1e-3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for field in (
        "frozen_cache", "ft_cache", "result_root", "julia_exe", "julia_depot", "case_path"
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"PASS", "SMOKE_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
