"""Rerun the FT7 warm-start matrix with IPOPT iteration statistics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.stage_j_gridsfm_goc500 import (
    run_j9_j10_reference_smoke as reference,
)
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.finetune import (
    run_ft7_warm_starts as ft7,
)


RUN_CONTRACT = "ft7_ipopt_iteration_rerun_v1"
MODEL_VARIANTS = {key: value["variant"] for key, value in ft7.MODEL_SPECS.items()}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _offline_ids(topology_id: object) -> list[int]:
    return [int(value) for value in str(topology_id or "").split(";") if value.strip()]


def _entry(setting: Path, family_id: str) -> dict[str, object]:
    method = ft7.FAMILY_SPECS[family_id]["method"]
    finalists = _read_json(setting / "finalists.json")["finalists"]
    return next(row for row in finalists if row["method"] == method)


def _method_dir(setting: Path, family_id: str) -> Path:
    return setting / "refs" / str(_entry(setting, family_id)["artifact_stub"])


def _start_dir(setting: Path, family_id: str, warm_type: str) -> Path | None:
    method_dir = _method_dir(setting, family_id)
    if warm_type == "cold_start":
        return None
    if warm_type == "dc_partial_warm":
        return method_dir / "wdc"
    if warm_type == "gt_warm":
        return method_dir / "wgt"
    model_id = warm_type.split("_")[1]
    package_id = ft7.FAMILY_SPECS[family_id]["package"]
    native_model = "m0" if package_id == "frozen" else package_id
    if model_id == native_model:
        return method_dir / "wgf"
    return method_dir / "crossed_warm" / MODEL_VARIANTS[model_id] / "state"


def _ordered_starts(group_index: int) -> list[str]:
    offset = group_index % len(ft7.START_ORDER)
    return ft7.START_ORDER[offset:] + ft7.START_ORDER[:offset]


def _complete(output: Path) -> bool:
    summary_path = output / "reference_a_summary.json"
    metadata_path = output / "run_metadata.json"
    if not summary_path.is_file() or not metadata_path.is_file():
        return False
    summary = _read_json(summary_path)
    metadata = _read_json(metadata_path)
    return (
        metadata.get("run_contract") == RUN_CONTRACT
        and summary.get("termination_status") in ft7.SUCCESS
        and summary.get("ipopt_solve_time_seconds") is not None
        and summary.get("iteration_count") is not None
    )


def _run_group(
    args: argparse.Namespace,
    setting: Path,
    family_id: str,
    group_index: int,
) -> dict[str, object]:
    context = ft7._context(setting, family_id, args)
    entry = _entry(setting, family_id)
    best = entry["best_topology"]
    method_dir = _method_dir(setting, family_id)
    starts = _ordered_starts(group_index)
    group_started = time.time()
    executed = 0
    for position, warm_type in enumerate(starts, start=1):
        output = args.output_root / "solves" / setting.name / family_id / warm_type
        if _complete(output):
            continue
        start_dir = _start_dir(setting, family_id, warm_type)
        if start_dir is not None and not (
            (start_dir / "bus_start.csv").is_file()
            and (start_dir / "gen_start.csv").is_file()
        ):
            raise FileNotFoundError(f"missing warm-start state for {family_id}/{warm_type}: {start_dir}")
        ok, stdout, stderr, wall = reference._run_julia(
            julia_exe=args.julia_exe,
            julia_depot_path=args.julia_depot_path,
            script=args.julia_script,
            mode="reference_a",
            case_path=args.case_path,
            alpha_csv=method_dir / "alpha_effective_full.csv",
            offline_branch_ids=_offline_ids(best.get("topology_id")),
            source_less_load_ids_=[],
            output_dir=output,
            start_dir=start_dir,
            timeout_seconds=args.timeout_seconds,
        )
        summary = _read_json(output / "reference_a_summary.json") if (
            output / "reference_a_summary.json"
        ).is_file() else {}
        metadata = {
            "run_contract": RUN_CONTRACT,
            "group_index": group_index,
            "execution_position": position,
            "execution_order": starts,
            "warm_start_type": warm_type,
            "wall_seconds": wall,
            "subprocess_success": ok,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            **context,
            **ft7._warm_provenance(warm_type, args),
        }
        _write_json(output / "run_metadata.json", metadata)
        if not _complete(output):
            raise RuntimeError(
                f"iteration timing solve failed for {setting.name}/{family_id}/{warm_type}: "
                f"status={summary.get('termination_status')}, iterations={summary.get('iteration_count')}"
            )
        executed += 1
    return {
        "setting_code": setting.name,
        "family_id": family_id,
        "group_index": group_index,
        "executed_solves": executed,
        "runtime_seconds": time.time() - group_started,
        "status": "PASS",
    }


def _row(args: argparse.Namespace, setting: Path, family_id: str, warm_type: str) -> dict[str, object]:
    output = args.output_root / "solves" / setting.name / family_id / warm_type
    summary = _read_json(output / "reference_a_summary.json")
    metadata = _read_json(output / "run_metadata.json")
    start_payload = {
        "cold_start": "V=1,theta=0,Pg=(Pmin+Pmax)/2,Qg=0",
        "dc_partial_warm": "Pg,theta",
        "gt_warm": "exact_Pg,Qg,V,theta",
    }.get(warm_type, "Pg,Qg,V,theta")
    return {
        "warm_start_type": warm_type,
        "warm_start_label": ft7.START_LABELS[warm_type],
        "warm_start_order": ft7.START_ORDER.index(warm_type),
        "execution_position": metadata["execution_position"],
        "execution_order": json.dumps(metadata["execution_order"]),
        "group_index": metadata["group_index"],
        "run_contract": RUN_CONTRACT,
        "source_package": "ft7_iteration_rerun",
        "same_reference_a_instance": True,
        "status": summary["termination_status"],
        "solver_runtime_seconds": summary["ipopt_solve_time_seconds"],
        "ipopt_solve_time_seconds": summary["ipopt_solve_time_seconds"],
        "power_models_total_seconds": summary["power_models_total_seconds"],
        "model_and_result_overhead_seconds": summary["model_and_result_overhead_seconds"],
        "wall_seconds": metadata["wall_seconds"],
        "iteration_count": summary["iteration_count"],
        "objective": summary["objective"],
        "solver_start_source": summary["start_source"],
        "start_payload": start_payload,
        "start_bus_count": summary["start_bus_count"],
        "start_gen_count": summary["start_gen_count"],
        "start_vm_min": summary["start_vm_min"],
        "start_vm_max": summary["start_vm_max"],
        "start_va_abs_max": summary["start_va_abs_max"],
        "start_pg_midpoint_max_abs_error": summary["start_pg_midpoint_max_abs_error"],
        "start_qg_abs_max": summary["start_qg_abs_max"],
        **ft7._context(setting, family_id, args),
        **ft7._warm_provenance(warm_type, args),
    }


def _assemble_and_validate(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    group_index = 0
    for family_id, family in ft7.FAMILY_SPECS.items():
        for setting in ft7._settings(args.caches[family["package"]]):
            for warm_type in ft7.START_ORDER:
                rows.append(_row(args, setting, family_id, warm_type))
            group_index += 1
    frame = pd.DataFrame(rows)
    objective_spreads = frame.groupby("reference_a_instance_id")["objective"].agg(
        lambda values: float(values.max() - values.min())
    )
    position_counts = frame.groupby(["warm_start_type", "execution_position"]).size()
    checks = {
        "rows_525": len(frame) == 525,
        "unique_cells_525": not frame.duplicated(
            ["setting_code", "comparison_variant", "warm_start_type"]
        ).any(),
        "all_solves_successful": frame["status"].isin(ft7.SUCCESS).all(),
        "all_iteration_counts_positive": pd.to_numeric(
            frame["iteration_count"], errors="coerce"
        ).gt(0).all(),
        "ipopt_metric_exact": frame["solver_runtime_seconds"].eq(
            frame["ipopt_solve_time_seconds"]
        ).all(),
        "objective_spread_within_tolerance": objective_spreads.max() <= 1e-3,
        "all_positions_balanced": position_counts.groupby(level=0).agg(
            lambda values: values.max() - values.min()
        ).le(1).all(),
        "all_contracts_current": frame["run_contract"].eq(RUN_CONTRACT).all(),
    }
    validation = {
        "status": "FT7_ITERATION_RERUN_PASS" if all(checks.values()) else "FT7_ITERATION_RERUN_BLOCKED",
        "checks": {key: bool(value) for key, value in checks.items()},
        "rows": len(frame),
        "groups": int(frame["reference_a_instance_id"].nunique()),
        "max_objective_spread": float(objective_spreads.max()),
        "iteration_min": int(frame["iteration_count"].min()),
        "iteration_max": int(frame["iteration_count"].max()),
        "execution_position_counts": {
            f"{start}|{position}": int(count)
            for (start, position), count in position_counts.items()
        },
    }
    return frame, validation


def run(args: argparse.Namespace) -> dict[str, object]:
    groups = []
    group_index = 0
    for family_id, family in ft7.FAMILY_SPECS.items():
        for setting in ft7._settings(args.caches[family["package"]]):
            groups.append((setting, family_id, group_index))
            group_index += 1
    started = time.time()
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_group, args, setting, family_id, index): (setting, family_id)
            for setting, family_id, index in groups
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            elapsed = time.time() - started
            eta = elapsed / completed * (len(groups) - completed)
            print(json.dumps({
                **record,
                "completed_groups": completed,
                "total_groups": len(groups),
                "elapsed_minutes": round(elapsed / 60, 1),
                "eta_minutes": round(eta / 60, 1),
            }), flush=True)
    frame, validation = _assemble_and_validate(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_root / "warm_start_iteration_rerun.csv", index=False)
    validation["execution_records"] = records
    validation["runtime_seconds"] = time.time() - started
    _write_json(args.output_root / "FT7_ITERATION_RERUN_VALIDATION.json", validation)
    print(json.dumps({
        "status": validation["status"],
        "rows": validation["rows"],
        "runtime_minutes": round(validation["runtime_seconds"] / 60, 1),
    }, indent=2))
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    config = _read_json(args.config.resolve())
    args.caches = {
        "frozen": Path(config["frozen_cache_root"]).resolve(),
        "m1": Path(config["m1_cache_root"]).resolve(),
        "m2": Path(config["models"]["m2"]["cache_root"]).resolve(),
        "m3": Path(config["models"]["m3"]["cache_root"]).resolve(),
    }
    args.checkpoints = {
        "m0": Path(config["gridsfm_root"]).resolve() / "model/checkpoints/gridsfm_open_v1.1.pt",
        "m1": Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\checkpoints\stage_j_finetune\gridsfm_goc500_fulltop_ft_n1000.pt"),
        "m2": Path(config["models"]["m2"]["checkpoint"]).resolve(),
        "m3": Path(config["models"]["m3"]["checkpoint"]).resolve(),
    }
    args.output_root = Path(config["working_root"]).resolve() / "warm_start_iteration_rerun"
    args.julia_exe = Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\tools\julia-1.10.11\bin\julia.exe")
    args.julia_depot_path = Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\julia_depot")
    args.julia_script = Path(__file__).resolve().parents[1] / "stage_j_ac_reference.jl"
    args.case_path = Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\repos\pglib-opf\pglib_opf_case500_goc.m")
    validation = run(args)
    return 0 if validation["status"] == "FT7_ITERATION_RERUN_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
