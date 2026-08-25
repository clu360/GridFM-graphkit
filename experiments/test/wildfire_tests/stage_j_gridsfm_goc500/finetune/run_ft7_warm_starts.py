"""Build the FT7 five-finalist by seven-initialization IPOPT timing study."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[5]
RUNNER = Path(__file__).resolve().parents[1] / "run_j9_j10_reference_smoke.py"
SUCCESS = {"LOCALLY_SOLVED", "OPTIMAL"}
MODEL_SPECS = {
    "m0": {
        "selection": "frozen",
        "variant": "released_v1_1",
        "sha256": "F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831",
    },
    "m1": {
        "selection": "ft",
        "variant": "fulltop_ft_n1000",
        "sha256": "A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD",
    },
    "m2": {
        "selection": "ft",
        "variant": "fulltop_ft_n1000_then_fulltop_n500",
        "sha256": "08EDA70270F787DB42C48B751BECC9DA2062B0482550A5C2A761B4EBA0C94FF6",
    },
    "m3": {
        "selection": "ft",
        "variant": "fulltop_ft_n1000_then_n1_n500",
        "sha256": "4EC89D36DE80081BE2FC26C14A1BC5A5369D7423B557D71B9DD4A254462F302D",
    },
}
FAMILY_SPECS = {
    "dc": {"label": "Guided-DC", "package": "frozen", "method": "Guided-DC"},
    "m0": {
        "label": "Guided-GridSFM (released v1.1)",
        "package": "frozen",
        "method": "Guided-GridSFM",
    },
    "m1": {
        "label": "Guided-GridSFM (FullTop-1000)",
        "package": "m1",
        "method": "Guided-GridSFM",
    },
    "m2": {
        "label": "Guided-GridSFM (FullTop-1500)",
        "package": "m2",
        "method": "Guided-GridSFM",
    },
    "m3": {
        "label": "Guided-GridSFM (FullTop+N-1)",
        "package": "m3",
        "method": "Guided-GridSFM",
    },
}
START_ORDER = [
    "cold_start", "dc_partial_warm", "gridsfm_m0_full_warm",
    "gridsfm_m1_full_warm", "gridsfm_m2_full_warm",
    "gridsfm_m3_full_warm", "gt_warm",
]
START_LABELS = {
    "cold_start": "Cold",
    "dc_partial_warm": "DC",
    "gridsfm_m0_full_warm": "GridSFM released v1.1",
    "gridsfm_m1_full_warm": "GridSFM FullTop-1000",
    "gridsfm_m2_full_warm": "GridSFM FullTop-1500",
    "gridsfm_m3_full_warm": "GridSFM FullTop+N-1",
    "gt_warm": "Exact A",
}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
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


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def _canonical_alpha(value: object) -> str:
    parsed = value if isinstance(value, Mapping) else json.loads(str(value or "{}"))
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


def _settings(root: Path) -> list[Path]:
    rows = sorted(path for path in root.glob("s*_l*") if (path / "finalists.json").is_file())
    if len(rows) != 15:
        raise RuntimeError(f"expected 15 cache settings under {root}, observed {len(rows)}")
    return rows


def _source_methods(package_id: str) -> set[str]:
    return {spec["method"] for spec in FAMILY_SPECS.values() if spec["package"] == package_id}


def _cross_path(setting: Path, warm_id: str) -> Path:
    return setting / "refs" / f"reference_a_crossed_full_warm_{warm_id}.csv"


def _cross_complete(setting: Path, package_id: str, warm_id: str) -> bool:
    path = _cross_path(setting, warm_id)
    if not path.is_file():
        return False
    rows = _read_csv(path)
    expected_methods = _source_methods(package_id)
    return (
        len(rows) == len(expected_methods)
        and {row.get("method") for row in rows} == expected_methods
        and {row.get("warm_start_type") for row in rows} == {f"gridsfm_{warm_id}_full_warm"}
        and all(row.get("status") in SUCCESS for row in rows)
        and all(row.get("start_payload") == "Pg,Qg,V,theta" for row in rows)
        and all(row.get("checkpoint_sha256") == MODEL_SPECS[warm_id]["sha256"] for row in rows)
    )


def _command(args: argparse.Namespace, setting: Path, package_id: str, warm_id: str) -> list[str]:
    finalists = _read_json(setting / "finalists.json")["finalists"]
    scenarios = {str(row["best_topology"].get("scenario_id")) for row in finalists}
    if len(scenarios) != 1:
        raise RuntimeError(f"ambiguous scenario in {setting}: {scenarios}")
    model = MODEL_SPECS[warm_id]
    command = [
        str(args.python), "-B", str(RUNNER),
        "--j8-root", str(setting / "main"),
        "--finalists-manifest", str(setting / "finalists.json"),
        "--input-dir", str(args.input_dir),
        "--gridsfm-root", str(args.gridsfm_root),
        "--model-selection", str(model["selection"]),
        "--scenario-id", next(iter(scenarios)),
        "--output-dir", str(setting / "refs"),
        "--full-gridsfm-only", "--warm-start-id", warm_id,
        "--expected-checkpoint-sha256", str(model["sha256"]),
        "--timeout-seconds", str(args.timeout_seconds),
    ]
    if warm_id != "m0":
        command.extend(["--checkpoint", str(args.checkpoints[warm_id])])
    for method in sorted(_source_methods(package_id)):
        command.extend(["--method-filter", method])
    return command


def _run_one(
    args: argparse.Namespace, setting: Path, package_id: str, warm_id: str,
) -> dict[str, object]:
    if _cross_complete(setting, package_id, warm_id):
        return {"setting_code": setting.name, "package": package_id, "warm_id": warm_id,
                "status": "SKIPPED_COMPLETE", "runtime_seconds": 0.0}
    started = time.time()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        _command(args, setting, package_id, warm_id), cwd=REPO_ROOT,
        env=environment, text=True, capture_output=True, check=False,
    )
    complete = _cross_complete(setting, package_id, warm_id)
    record = {
        "setting_code": setting.name, "package": package_id, "warm_id": warm_id,
        "status": "PASS" if proc.returncode == 0 and complete else "FAIL",
        "returncode": proc.returncode, "runtime_seconds": time.time() - started,
        "stdout": proc.stdout, "stderr": proc.stderr,
    }
    _write_json(setting / "logs" / f"ft7_cross_{warm_id}.json", record)
    return record


def _run_crosses(args: argparse.Namespace) -> list[dict[str, object]]:
    tasks: list[tuple[Path, str, str]] = []
    for package_id, cache in args.caches.items():
        native = "m0" if package_id == "frozen" else package_id
        for warm_id in MODEL_SPECS:
            if warm_id == native:
                continue
            for setting in _settings(cache):
                tasks.append((setting, package_id, warm_id))
    # M1 already has M0 from FT5; frozen already has M1 from FT5.
    tasks = [task for task in tasks if not (
        (task[1] == "m1" and task[2] == "m0")
        or (task[1] == "frozen" and task[2] == "m1")
    )]
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_one, args, *task): task for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({key: record[key] for key in (
                "setting_code", "package", "warm_id", "status", "runtime_seconds"
            )}), flush=True)
            if record["status"] == "FAIL":
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"FT7 crossed warm start failed: {record}")
    return records


def _collect_execution_records(args: argparse.Namespace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    expected = {
        "frozen": ("m2", "m3"),
        "m1": ("m2", "m3"),
        "m2": ("m0", "m1", "m3"),
        "m3": ("m0", "m1", "m2"),
    }
    for package_id, warm_ids in expected.items():
        for setting in _settings(args.caches[package_id]):
            for warm_id in warm_ids:
                record = _read_json(setting / "logs" / f"ft7_cross_{warm_id}.json")
                records.append({
                    key: record.get(key) for key in (
                        "setting_code", "package", "warm_id", "status",
                        "returncode", "runtime_seconds",
                    )
                })
    if len(records) != 150 or not all(row["status"] == "PASS" for row in records):
        raise RuntimeError("FT7 crossed execution logs do not satisfy the 150-task PASS contract")
    return records


def _context(
    setting: Path, family_id: str, args: argparse.Namespace,
) -> dict[str, object]:
    family = FAMILY_SPECS[family_id]
    entries = _read_json(setting / "finalists.json")["finalists"]
    entry = next(row for row in entries if row["method"] == family["method"])
    best = entry["best_topology"]
    alpha = _canonical_alpha(best.get("best_alpha_selected"))
    model = MODEL_SPECS.get(family_id)
    instance = json.dumps({
        "setting_code": setting.name, "finalist_family": family["label"],
        "topology_id": str(best.get("topology_id") or ""),
        "alpha_selected": json.loads(alpha),
    }, sort_keys=True, separators=(",", ":"))
    return {
        "setting_code": setting.name,
        "scenario_id": str(best.get("scenario_id")),
        "lambda_r": entry.get("lambda_r", best.get("lambda_r")),
        "method": family["label"], "finalist_family": family["label"],
        "comparison_variant": family_id,
        "finalist_backend": str(entry.get("backend", "")),
        "finalist_model_selection": "dc" if family_id == "dc" else model["selection"],
        "finalist_model_variant": "guided_dc" if family_id == "dc" else model["variant"],
        "finalist_checkpoint_path": "" if family_id == "dc" else str(
            args.checkpoints[family_id]
        ),
        "finalist_checkpoint_sha256": "" if family_id == "dc" else model["sha256"],
        "finalist_topology_id": str(best.get("topology_id") or ""),
        "finalist_topology_rank": best.get("topology_rank", ""),
        "finalist_num_shutoffs": best.get("num_shutoffs", ""),
        "finalist_alpha_selected": alpha, "finalist_alpha_sha256": _digest(alpha),
        "reference_a_instance_id": _digest(instance)[:20],
    }


def _warm_provenance(warm_type: str, args: argparse.Namespace) -> dict[str, object]:
    if warm_type.startswith("gridsfm_m"):
        model_id = warm_type.split("_")[1]
        model = MODEL_SPECS[model_id]
        return {
            "warm_start_model_selection": model["selection"],
            "warm_start_model_variant": model["variant"],
            "warm_start_checkpoint_path": str(args.checkpoints[model_id]),
            "warm_start_checkpoint_sha256": model["sha256"],
        }
    selection, variant = {
        "cold_start": ("cold", "explicit_generic"),
        "dc_partial_warm": ("dc", "economic_dc_partial"),
        "gt_warm": ("exact", "reference_a_solution"),
    }[warm_type]
    return {
        "warm_start_model_selection": selection, "warm_start_model_variant": variant,
        "warm_start_checkpoint_path": "", "warm_start_checkpoint_sha256": "",
    }


def _canonical(
    raw: Mapping[str, object], context: Mapping[str, object], warm_type: str,
    args: argparse.Namespace, source_package: str,
) -> dict[str, object]:
    excluded = {
        "method", "model_selection", "model_variant", "checkpoint_path", "checkpoint_sha256",
        "finalist_model_selection", "finalist_model_variant", "finalist_checkpoint_path",
        "finalist_checkpoint_sha256", "warm_start_model_selection", "warm_start_model_variant",
        "warm_start_checkpoint_path", "warm_start_checkpoint_sha256",
    }
    row = {key: value for key, value in raw.items() if key not in excluded}
    row.update(context)
    row.update({
        "warm_start_type": warm_type, "warm_start_label": START_LABELS[warm_type],
        "warm_start_order": START_ORDER.index(warm_type), "source_package": source_package,
        "runtime_metric": "ipopt_solve_time_seconds",
        "ipopt_solve_time_seconds": raw.get("solver_runtime_seconds", ""),
        "objective": raw.get("objective") or raw.get("timing_rerun_objective", ""),
    })
    row.update(_warm_provenance(warm_type, args))
    return row


def _reuse_ft5(args: argparse.Namespace) -> list[dict[str, object]]:
    old_families = {
        "Guided-DC": "dc", "Guided-GridSFM (frozen)": "m0",
        "Guided-GridSFM (fine-tuned)": "m1",
    }
    old_starts = {
        "cold_start": "cold_start", "dc_partial_warm": "dc_partial_warm",
        "gridsfm_frozen_full_warm": "gridsfm_m0_full_warm",
        "gridsfm_ft_full_warm": "gridsfm_m1_full_warm", "gt_warm": "gt_warm",
    }
    rows = []
    for raw in _read_csv(args.ft5_reuse):
        family_id = old_families.get(raw.get("finalist_family", ""))
        warm_type = old_starts.get(raw.get("warm_start_type", ""))
        if family_id is None or warm_type is None:
            continue
        setting = args.caches[FAMILY_SPECS[family_id]["package"]] / str(raw["setting_code"])
        rows.append(_canonical(raw, _context(setting, family_id, args), warm_type, args, "ft5_reuse"))
    if len(rows) != 225:
        raise RuntimeError(f"expected 225 reusable FT5 rows, observed {len(rows)}")
    return rows


def _assemble(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = _reuse_ft5(args)
    for family_id in ("m2", "m3"):
        package_id = FAMILY_SPECS[family_id]["package"]
        for setting in _settings(args.caches[package_id]):
            context = _context(setting, family_id, args)
            base = _read_csv(setting / "refs" / "reference_a_warm_start_summary.csv")
            for raw in base:
                if raw.get("method") != "Guided-GridSFM":
                    continue
                warm_type = {
                    "cold_start": "cold_start", "dc_partial_warm": "dc_partial_warm",
                    "gridsfm_full_warm": f"gridsfm_{family_id}_full_warm",
                    "gt_warm": "gt_warm",
                }.get(raw.get("warm_start_type", ""))
                if warm_type:
                    rows.append(_canonical(raw, context, warm_type, args, package_id))

    for family_id, family in FAMILY_SPECS.items():
        package_id = family["package"]
        for setting in _settings(args.caches[package_id]):
            context = _context(setting, family_id, args)
            present = {
                str(row["warm_start_type"]) for row in rows
                if row["setting_code"] == setting.name and row["comparison_variant"] == family_id
            }
            for warm_id in MODEL_SPECS:
                warm_type = f"gridsfm_{warm_id}_full_warm"
                if warm_type in present:
                    continue
                for raw in _read_csv(_cross_path(setting, warm_id)):
                    if raw.get("method") == family["method"]:
                        rows.append(_canonical(raw, context, warm_type, args, package_id))
    return rows


def _truth(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _validate(rows: list[dict[str, object]], tolerance: float) -> dict[str, object]:
    keys = {(row["setting_code"], row["comparison_variant"], row["warm_start_type"]) for row in rows}
    cells = {
        f"{family}|{start}": sum(
            row["comparison_variant"] == family and row["warm_start_type"] == start
            for row in rows
        )
        for family in FAMILY_SPECS for start in START_ORDER
    }
    by_instance: dict[str, list[float]] = {}
    for row in rows:
        by_instance.setdefault(str(row["reference_a_instance_id"]), []).append(float(row["objective"]))
    max_spread = max(max(values) - min(values) for values in by_instance.values())
    cold = [row for row in rows if row["warm_start_type"] == "cold_start"]
    full = [row for row in rows if str(row["warm_start_type"]).startswith("gridsfm_m")]
    checks = {
        "rows_525": len(rows) == 525, "unique_keys_525": len(keys) == 525,
        "five_families": {row["comparison_variant"] for row in rows} == set(FAMILY_SPECS),
        "seven_starts": {row["warm_start_type"] for row in rows} == set(START_ORDER),
        "all_35_cells_have_15_rows": all(count == 15 for count in cells.values()),
        "all_solves_successful": all(row["status"] in SUCCESS for row in rows),
        "same_reference_instance": all(_truth(row["same_reference_a_instance"]) for row in rows),
        "objective_spread_within_tolerance": max_spread <= tolerance,
        "cold_policy": all(
            row["solver_start_source"] == "explicit_generic_V1_theta0_Pg_midpoint_Qg0"
            and abs(float(row["start_vm_min"]) - 1.0) <= 1e-12
            and abs(float(row["start_vm_max"]) - 1.0) <= 1e-12
            and abs(float(row["start_va_abs_max"])) <= 1e-12
            and abs(float(row["start_pg_midpoint_max_abs_error"])) <= 1e-12
            and abs(float(row["start_qg_abs_max"])) <= 1e-12 for row in cold
        ),
        "full_state_payload": all(row["start_payload"] == "Pg,Qg,V,theta" for row in full),
        "all_checkpoint_provenance": all(
            row["warm_start_checkpoint_sha256"] == MODEL_SPECS[str(row["warm_start_type"]).split("_")[1]]["sha256"]
            for row in full
        ),
        "native_ipopt_metric": all(row["runtime_metric"] == "ipopt_solve_time_seconds" for row in rows),
    }
    return {
        "status": "FT7_P4_WARM_START_PASS" if all(checks.values()) else "FT7_WARM_START_BLOCKED",
        "checks": checks, "rows": len(rows), "cell_counts": cells,
        "max_objective_spread": max_spread, "objective_spread_tolerance": tolerance,
        "primary_metric": "ipopt_solve_time_seconds",
        "cold_start_policy": "V=1,theta=0,Pg=(Pmin+Pmax)/2,Qg=0",
        "reused_ft5_rows": 225, "new_m2_m3_base_rows": 120,
        "new_crossed_task_groups": 150, "new_crossed_solver_rows": 180,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--objective-spread-tolerance", type=float, default=1e-3)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    config = _read_json(args.config.resolve())
    args.python = Path(config["gridsfm_python"]).resolve()
    args.gridsfm_root = Path(config["gridsfm_root"]).resolve()
    args.input_dir = Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\inputs\case500_goc_e0")
    args.caches = {
        "frozen": Path(config["frozen_cache_root"]).resolve(),
        "m1": Path(config["m1_cache_root"]).resolve(),
        "m2": Path(config["models"]["m2"]["cache_root"]).resolve(),
        "m3": Path(config["models"]["m3"]["cache_root"]).resolve(),
    }
    args.checkpoints = {
        "m0": args.gridsfm_root / "model/checkpoints/gridsfm_open_v1.1.pt",
        "m1": Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\checkpoints\stage_j_finetune\gridsfm_goc500_fulltop_ft_n1000.pt"),
        "m2": Path(config["models"]["m2"]["checkpoint"]).resolve(),
        "m3": Path(config["models"]["m3"]["checkpoint"]).resolve(),
    }
    args.ft5_reuse = Path(config["m1_result_root"]).resolve() / "core_results/warm_start_crossed.csv"
    args.working_root = Path(config["working_root"]).resolve()
    records = _collect_execution_records(args) if args.aggregate_only else _run_crosses(args)
    rows = _assemble(args)
    output = args.working_root / "warm_start"
    _write_csv(output / "warm_start_crossed.csv", rows)
    validation = _validate(rows, args.objective_spread_tolerance)
    validation["execution_records"] = records
    _write_json(output / "FT7_WARM_START_VALIDATION.json", validation)
    print(json.dumps({"status": validation["status"], "rows": len(rows)}, indent=2))
    return 0 if validation["status"] == "FT7_P4_WARM_START_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
