"""Augment completed Stage J packages with explicit cold and full GridSFM starts."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
RUNNER = Path(__file__).resolve().parents[1] / "run_j9_j10_reference_smoke.py"
EXPECTED_WARM_TYPES = {
    "cold_start",
    "dc_partial_warm",
    "gridsfm_partial_warm",
    "gridsfm_full_warm",
    "gt_warm",
}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty aggregate: {path}")
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


def _setting_context(setting_dir: Path) -> dict[str, dict[str, object]]:
    payload = _read_json(setting_dir / "finalists.json")
    context: dict[str, dict[str, object]] = {}
    for finalist in payload["finalists"]:
        best = dict(finalist["best_topology"])
        context[str(finalist["method"])] = {
            "setting_code": setting_dir.name,
            "scenario_id": best.get("scenario_id", payload.get("scenario_id")),
            "lambda_r": finalist.get("lambda_r", best.get("lambda_r", payload.get("lambda_r"))),
            "finalist_backend": finalist.get("backend"),
            "model_selection": finalist.get("model_selection"),
            "model_variant": finalist.get("model_variant"),
            "finalist_topology_id": best.get("topology_id", ""),
            "finalist_topology_rank": best.get("topology_rank", ""),
            "finalist_num_shutoffs": best.get("num_shutoffs", ""),
        }
    return context


def _is_complete(setting_dir: Path) -> bool:
    context = _setting_context(setting_dir)
    path = setting_dir / "refs" / "reference_a_warm_start_summary.csv"
    if not path.is_file():
        return False
    rows = _read_csv(path)
    if len(rows) != 5 * len(context):
        return False
    if {row.get("warm_start_type") for row in rows} != EXPECTED_WARM_TYPES:
        return False
    for row in rows:
        if row.get("status") not in {"LOCALLY_SOLVED", "OPTIMAL"}:
            return False
        if row.get("warm_start_type") == "cold_start":
            if row.get("solver_start_source") != "explicit_generic_V1_theta0_Pg_midpoint_Qg0":
                return False
            for field in (
                "start_vm_min", "start_vm_max", "start_va_abs_max",
                "start_pg_midpoint_max_abs_error", "start_qg_abs_max",
            ):
                expected = 1.0 if field in {"start_vm_min", "start_vm_max"} else 0.0
                if abs(float(row[field]) - expected) > 1e-12:
                    return False
    return True


def _command(args: argparse.Namespace, setting_dir: Path) -> list[str]:
    context = _setting_context(setting_dir)
    scenario_ids = {str(row["scenario_id"]) for row in context.values()}
    if len(scenario_ids) != 1:
        raise ValueError(f"ambiguous scenario in {setting_dir}: {scenario_ids}")
    command = [
        str(args.python), "-B", str(RUNNER),
        "--j8-root", str(setting_dir / "main"),
        "--finalists-manifest", str(setting_dir / "finalists.json"),
        "--input-dir", str(args.input_dir),
        "--gridsfm-root", str(args.gridsfm_root),
        "--model-selection", args.model_selection,
        "--scenario-id", next(iter(scenario_ids)),
        "--output-dir", str(setting_dir / "refs"),
        "--warm-starts-only",
        "--timeout-seconds", str(args.timeout_seconds),
    ]
    if args.checkpoint:
        command.extend(["--checkpoint", str(args.checkpoint)])
    if args.expected_checkpoint_sha256:
        command.extend(["--expected-checkpoint-sha256", args.expected_checkpoint_sha256])
    return command


def _run_setting(args: argparse.Namespace, setting_dir: Path) -> dict[str, object]:
    if not args.force and _is_complete(setting_dir):
        return {"setting_code": setting_dir.name, "status": "SKIPPED_COMPLETE", "runtime_seconds": 0.0}
    started = time.time()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        _command(args, setting_dir), cwd=REPO_ROOT, env=environment,
        text=True, capture_output=True, check=False,
    )
    record = {
        "setting_code": setting_dir.name,
        "status": "PASS" if proc.returncode == 0 and _is_complete(setting_dir) else "FAIL",
        "returncode": proc.returncode,
        "runtime_seconds": time.time() - started,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    _write_json(setting_dir / "logs" / "full_warm_start_augmentation.json", record)
    return record


def _objective_spread(rows: list[dict[str, str]]) -> float:
    by_method: dict[str, list[float]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(float(row["objective"]))
    return max(max(values) - min(values) for values in by_method.values())


def _aggregate(args: argparse.Namespace, settings: list[Path], records: list[dict[str, object]]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    setting_sync: dict[str, bool] = {}
    max_objective_spread = 0.0
    for setting_dir in settings:
        context = _setting_context(setting_dir)
        source = setting_dir / "refs" / "reference_a_warm_start_summary.csv"
        setting_rows = _read_csv(source)
        max_objective_spread = max(max_objective_spread, _objective_spread(setting_rows))
        for row in setting_rows:
            rows.append({**context[row["method"]], **row})
        destination_dir = args.final_root / "settings" / setting_dir.name / "ac_audits"
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination_dir / source.name)
            shutil.copy2(
                setting_dir / "refs" / "j9_j10_reference_smoke_summary.json",
                destination_dir / "j9_j10_reference_smoke_summary.json",
            )
            setting_sync[setting_dir.name] = True
        except (FileNotFoundError, OSError, PermissionError):
            setting_sync[setting_dir.name] = False

    _write_csv(args.final_root / "core_results" / "warm_start_all.csv", rows)
    expected_rows = sum(5 * len(_setting_context(setting)) for setting in settings)
    status = "PASS" if all(record["status"] in {"PASS", "SKIPPED_COMPLETE"} for record in records) else "FAIL"
    if len(rows) != expected_rows or max_objective_spread > args.objective_spread_tolerance:
        status = "FAIL"

    count_path = args.final_root / "ARTIFACT_COUNT_VALIDATION.json"
    if count_path.is_file():
        counts = _read_json(count_path)
        counts.setdefault("expected", {})["warm_start_all.csv"] = expected_rows
        counts.setdefault("observed", {})["warm_start_all.csv"] = len(rows)
        counts["status"] = "PASS" if counts.get("expected") == counts.get("observed") else "FAIL"
        _write_json(count_path, counts)

    manifest = {
        "status": status,
        "model_selection": args.model_selection,
        "checkpoint": "" if args.checkpoint is None else str(args.checkpoint),
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256 or "",
        "cold_start_policy": "V=1,theta=0,Pg=(Pmin+Pmax)/2,Qg=0",
        "warm_start_types": sorted(EXPECTED_WARM_TYPES),
        "settings": len(settings),
        "warm_start_rows": len(rows),
        "max_objective_spread_across_starts": max_objective_spread,
        "objective_spread_tolerance": args.objective_spread_tolerance,
        "setting_artifact_sync": setting_sync,
        "records": sorted(records, key=lambda row: str(row["setting_code"])),
    }
    _write_json(args.final_root / "FULL_WARM_START_AUGMENTATION.json", manifest)
    return manifest


def augment(args: argparse.Namespace) -> dict[str, object]:
    settings = sorted(path for path in args.cache_root.glob("s*_l*") if (path / "finalists.json").is_file())
    if not settings:
        raise FileNotFoundError(f"no Stage J settings found under {args.cache_root}")
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_setting, args, setting): setting for setting in settings}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({key: record[key] for key in ("setting_code", "status", "runtime_seconds")}))
            if record["status"] == "FAIL":
                raise RuntimeError(f"warm-start augmentation failed for {record['setting_code']}")
    return _aggregate(args, settings, records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--gridsfm-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-selection", choices=["frozen", "ft"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--objective-spread-tolerance", type=float, default=1e-3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for field in ("python", "cache_root", "final_root", "gridsfm_root", "input_dir"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest = augment(args)
    print(json.dumps({key: manifest[key] for key in ("status", "settings", "warm_start_rows")}, indent=2))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
