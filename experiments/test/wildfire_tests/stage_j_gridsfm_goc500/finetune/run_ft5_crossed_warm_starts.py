"""Run and validate the crossed frozen/FT GridSFM warm-start study."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
RUNNER = Path(__file__).resolve().parents[1] / "run_j9_j10_reference_smoke.py"
FROZEN_SHA256 = "F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831"
FT_SHA256 = "A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD"
SUCCESS = {"LOCALLY_SOLVED", "OPTIMAL"}
START_ORDER = [
    "cold_start",
    "dc_partial_warm",
    "gridsfm_frozen_full_warm",
    "gridsfm_ft_full_warm",
    "gt_warm",
]
START_LABELS = {
    "cold_start": "Cold",
    "dc_partial_warm": "DC",
    "gridsfm_frozen_full_warm": "GridSFM frozen",
    "gridsfm_ft_full_warm": "GridSFM fine-tuned",
    "gt_warm": "Exact",
}
FROZEN_FAMILY_LABELS = {
    "Guided-DC": "Guided-DC",
    "Guided-GridSFM": "Guided-GridSFM (frozen)",
    "TH-GridSFM-top1": "TH-GridSFM-top1",
    "TH-GridSFM-top2": "TH-GridSFM-top2",
}
FT_FAMILY_LABELS = {"Guided-GridSFM": "Guided-GridSFM (fine-tuned)"}
FAMILY_ORDER = [
    "Guided-DC",
    "Guided-GridSFM (frozen)",
    "Guided-GridSFM (fine-tuned)",
    "TH-GridSFM-top1",
    "TH-GridSFM-top2",
]


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


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def _setting_dirs(cache_root: Path) -> list[Path]:
    settings = sorted(
        path for path in cache_root.glob("s*_l*") if (path / "finalists.json").is_file()
    )
    if len(settings) != 15:
        raise RuntimeError(f"expected 15 settings under {cache_root}, observed {len(settings)}")
    return settings


def _expected_methods(source_package: str) -> set[str]:
    return set(FROZEN_FAMILY_LABELS if source_package == "frozen" else FT_FAMILY_LABELS)


def _cross_table(setting: Path, warm_model: str) -> Path:
    return setting / "refs" / f"reference_a_crossed_full_warm_{warm_model}.csv"


def _cross_complete(
    setting: Path,
    *,
    warm_model: str,
    source_package: str,
    expected_sha256: str,
) -> bool:
    path = _cross_table(setting, warm_model)
    if not path.is_file():
        return False
    rows = _read_csv(path)
    expected_methods = _expected_methods(source_package)
    expected_type = f"gridsfm_{warm_model}_full_warm"
    return (
        len(rows) == len(expected_methods)
        and {row.get("method") for row in rows} == expected_methods
        and {row.get("warm_start_type") for row in rows} == {expected_type}
        and all(row.get("status") in SUCCESS for row in rows)
        and all(row.get("start_payload") == "Pg,Qg,V,theta" for row in rows)
        and all(row.get("warm_start_checkpoint_sha256") == expected_sha256 for row in rows)
    )


def _runner_command(
    args: argparse.Namespace,
    *,
    setting: Path,
    warm_model: str,
    source_package: str,
) -> list[str]:
    finalists = _read_json(setting / "finalists.json")["finalists"]
    scenario_ids = {
        str(row["best_topology"].get("scenario_id")) for row in finalists
    }
    if len(scenario_ids) != 1:
        raise RuntimeError(f"ambiguous scenario identity in {setting}: {scenario_ids}")
    command = [
        str(args.python), "-B", str(RUNNER),
        "--j8-root", str(setting / "main"),
        "--finalists-manifest", str(setting / "finalists.json"),
        "--input-dir", str(args.input_dir),
        "--gridsfm-root", str(args.gridsfm_root),
        "--model-selection", warm_model,
        "--scenario-id", next(iter(scenario_ids)),
        "--output-dir", str(setting / "refs"),
        "--full-gridsfm-only",
        "--timeout-seconds", str(args.timeout_seconds),
    ]
    if warm_model == "ft":
        command.extend([
            "--checkpoint", str(args.ft_checkpoint),
            "--expected-checkpoint-sha256", args.ft_sha256,
        ])
    else:
        command.extend(["--expected-checkpoint-sha256", FROZEN_SHA256])
    if source_package == "ft":
        command.extend(["--method-filter", "Guided-GridSFM"])
    return command


def _run_setting(
    args: argparse.Namespace,
    *,
    setting: Path,
    warm_model: str,
    source_package: str,
    expected_sha256: str,
) -> dict[str, object]:
    if _cross_complete(
        setting,
        warm_model=warm_model,
        source_package=source_package,
        expected_sha256=expected_sha256,
    ):
        return {
            "setting_code": setting.name,
            "warm_model": warm_model,
            "source_package": source_package,
            "status": "SKIPPED_COMPLETE",
            "runtime_seconds": 0.0,
        }
    started = time.time()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        _runner_command(
            args, setting=setting, warm_model=warm_model, source_package=source_package
        ),
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    complete = _cross_complete(
        setting,
        warm_model=warm_model,
        source_package=source_package,
        expected_sha256=expected_sha256,
    )
    record = {
        "setting_code": setting.name,
        "warm_model": warm_model,
        "source_package": source_package,
        "status": "PASS" if proc.returncode == 0 and complete else "FAIL",
        "returncode": proc.returncode,
        "runtime_seconds": time.time() - started,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    _write_json(
        setting / "logs" / f"ft5_crossed_{source_package}_with_{warm_model}.json",
        record,
    )
    return record


def _run_phase(
    args: argparse.Namespace,
    *,
    cache_root: Path,
    warm_model: str,
    source_package: str,
    expected_sha256: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_setting,
                args,
                setting=setting,
                warm_model=warm_model,
                source_package=source_package,
                expected_sha256=expected_sha256,
            ): setting
            for setting in _setting_dirs(cache_root)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({
                key: record[key]
                for key in ("setting_code", "source_package", "warm_model", "status", "runtime_seconds")
            }), flush=True)
            if record["status"] == "FAIL":
                raise RuntimeError(
                    f"crossed warm start failed for {record['source_package']} "
                    f"{record['setting_code']} with {record['warm_model']}"
                )
    return sorted(records, key=lambda row: str(row["setting_code"]))


def _finalist_context(
    setting: Path,
    *,
    source_package: str,
    frozen_checkpoint: Path,
    ft_checkpoint: Path,
) -> dict[str, dict[str, object]]:
    labels = FROZEN_FAMILY_LABELS if source_package == "frozen" else FT_FAMILY_LABELS
    entries = _read_json(setting / "finalists.json")["finalists"]
    context: dict[str, dict[str, object]] = {}
    for entry in entries:
        method = str(entry["method"])
        if method not in labels:
            continue
        best = entry["best_topology"]
        finalist_selection = (
            "ft" if source_package == "ft" else "dc" if method == "Guided-DC" else "frozen"
        )
        finalist_variant = {
            "ft": "fulltop_ft_n1000",
            "dc": "guided_dc",
            "frozen": "released_v1_1",
        }[finalist_selection]
        checkpoint_path = ""
        checkpoint_sha256 = ""
        if finalist_selection == "frozen":
            checkpoint_path = str(frozen_checkpoint)
            checkpoint_sha256 = FROZEN_SHA256
        elif finalist_selection == "ft":
            checkpoint_path = str(ft_checkpoint)
            checkpoint_sha256 = FT_SHA256
        alpha = str(best.get("best_alpha_selected") or "{}")
        alpha_canonical = json.dumps(json.loads(alpha), sort_keys=True, separators=(",", ":"))
        topology_id = str(best.get("topology_id") or "")
        instance_payload = json.dumps(
            {
                "setting_code": setting.name,
                "finalist_family": labels[method],
                "topology_id": topology_id,
                "alpha_selected": json.loads(alpha_canonical),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        context[method] = {
            "setting_code": setting.name,
            "scenario_id": str(best.get("scenario_id")),
            "lambda_r": entry.get("lambda_r", best.get("lambda_r")),
            "method": labels[method],
            "finalist_family": labels[method],
            "comparison_variant": {
                "Guided-DC": "dc",
                "Guided-GridSFM (frozen)": "frozen",
                "Guided-GridSFM (fine-tuned)": "ft",
                "TH-GridSFM-top1": "th_frozen",
                "TH-GridSFM-top2": "th_frozen",
            }[labels[method]],
            "finalist_backend": str(entry["backend"]),
            "finalist_model_selection": finalist_selection,
            "finalist_model_variant": finalist_variant,
            "finalist_checkpoint_path": checkpoint_path,
            "finalist_checkpoint_sha256": checkpoint_sha256,
            "finalist_topology_id": topology_id,
            "finalist_topology_rank": best.get("topology_rank", ""),
            "finalist_num_shutoffs": best.get("num_shutoffs", ""),
            "finalist_alpha_selected": alpha_canonical,
            "finalist_alpha_sha256": _digest_text(alpha_canonical),
            "reference_a_instance_id": _digest_text(instance_payload)[:20],
        }
    return context


def _warm_provenance(
    warm_type: str,
    *,
    frozen_checkpoint: Path,
    ft_checkpoint: Path,
) -> dict[str, object]:
    if warm_type == "gridsfm_frozen_full_warm":
        selection, variant = "frozen", "released_v1_1"
        checkpoint, digest = frozen_checkpoint, FROZEN_SHA256
    elif warm_type == "gridsfm_ft_full_warm":
        selection, variant = "ft", "fulltop_ft_n1000"
        checkpoint, digest = ft_checkpoint, FT_SHA256
    elif warm_type == "dc_partial_warm":
        selection, variant, checkpoint, digest = "dc", "economic_dc_partial", "", ""
    elif warm_type == "gt_warm":
        selection, variant, checkpoint, digest = "exact", "reference_a_solution", "", ""
    else:
        selection, variant, checkpoint, digest = "cold", "explicit_generic", "", ""
    return {
        "warm_start_model_selection": selection,
        "warm_start_model_variant": variant,
        "warm_start_checkpoint_path": str(checkpoint) if checkpoint else "",
        "warm_start_checkpoint_sha256": digest,
    }


def _canonical_row(
    raw: dict[str, str],
    context: dict[str, object],
    warm_type: str,
    *,
    source_package: str,
    frozen_checkpoint: Path,
    ft_checkpoint: Path,
) -> dict[str, object]:
    row: dict[str, object] = {
        key: value
        for key, value in raw.items()
        if key not in {
            "method", "model_selection", "model_variant", "checkpoint_path", "checkpoint_sha256",
            "finalist_model_selection", "finalist_model_variant", "finalist_checkpoint_path",
            "finalist_checkpoint_sha256", "warm_start_model_selection", "warm_start_model_variant",
            "warm_start_checkpoint_path", "warm_start_checkpoint_sha256",
        }
    }
    row.update(context)
    row.update({
        "warm_start_type": warm_type,
        "warm_start_label": START_LABELS[warm_type],
        "warm_start_order": START_ORDER.index(warm_type),
        "source_package": source_package,
        "source_package_model_selection": raw.get("model_selection", source_package),
    })
    row.update(_warm_provenance(
        warm_type, frozen_checkpoint=frozen_checkpoint, ft_checkpoint=ft_checkpoint
    ))
    return row


def _assemble(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    package_specs = [
        ("frozen", args.frozen_cache, FROZEN_FAMILY_LABELS),
        ("ft", args.ft_cache, FT_FAMILY_LABELS),
    ]
    base_type_map = {
        "cold_start": "cold_start",
        "dc_partial_warm": "dc_partial_warm",
        "gt_warm": "gt_warm",
    }
    for source_package, cache_root, labels in package_specs:
        for setting in _setting_dirs(cache_root):
            context = _finalist_context(
                setting,
                source_package=source_package,
                frozen_checkpoint=args.frozen_checkpoint,
                ft_checkpoint=args.ft_checkpoint,
            )
            for raw in _read_csv(setting / "refs" / "reference_a_warm_start_summary.csv"):
                method = raw["method"]
                if method not in labels or raw["warm_start_type"] == "gridsfm_partial_warm":
                    continue
                warm_type = base_type_map.get(raw["warm_start_type"])
                if raw["warm_start_type"] == "gridsfm_full_warm":
                    warm_type = f"gridsfm_{source_package}_full_warm"
                if warm_type is None:
                    continue
                rows.append(_canonical_row(
                    raw,
                    context[method],
                    warm_type,
                    source_package=source_package,
                    frozen_checkpoint=args.frozen_checkpoint,
                    ft_checkpoint=args.ft_checkpoint,
                ))

            cross_model = "ft" if source_package == "frozen" else "frozen"
            for raw in _read_csv(_cross_table(setting, cross_model)):
                method = raw["method"]
                rows.append(_canonical_row(
                    raw,
                    context[method],
                    raw["warm_start_type"],
                    source_package=source_package,
                    frozen_checkpoint=args.frozen_checkpoint,
                    ft_checkpoint=args.ft_checkpoint,
                ))
    return rows


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _validate(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    combinations: dict[str, int] = {}
    for family in FAMILY_ORDER:
        for warm_type in START_ORDER:
            combinations[f"{family}|{warm_type}"] = sum(
                row["finalist_family"] == family and row["warm_start_type"] == warm_type
                for row in rows
            )
    unique_keys = {
        (row["setting_code"], row["finalist_family"], row["warm_start_type"])
        for row in rows
    }
    by_instance: dict[str, list[float]] = {}
    for row in rows:
        by_instance.setdefault(str(row["reference_a_instance_id"]), []).append(float(row["objective"]))
    objective_spreads = {
        key: max(values) - min(values) for key, values in by_instance.items()
    }
    max_spread = max(objective_spreads.values())
    cold_rows = [row for row in rows if row["warm_start_type"] == "cold_start"]
    full_rows = [row for row in rows if "gridsfm_" in str(row["warm_start_type"])]
    checks = {
        "row_count_375": len(rows) == 375,
        "unique_primary_keys_375": len(unique_keys) == 375,
        "five_families": {row["finalist_family"] for row in rows} == set(FAMILY_ORDER),
        "five_start_types": {row["warm_start_type"] for row in rows} == set(START_ORDER),
        "all_25_combinations_have_15_settings": all(value == 15 for value in combinations.values()),
        "all_exact_solves_successful": all(row["status"] in SUCCESS for row in rows),
        "all_same_reference_a_instance": all(
            _as_bool(row["same_reference_a_instance"]) for row in rows
        ),
        "objective_spread_within_tolerance": max_spread <= args.objective_spread_tolerance,
        "cold_policy_audited": all(
            row["solver_start_source"] == "explicit_generic_V1_theta0_Pg_midpoint_Qg0"
            and abs(float(row["start_vm_min"]) - 1.0) <= 1e-12
            and abs(float(row["start_vm_max"]) - 1.0) <= 1e-12
            and abs(float(row["start_va_abs_max"])) <= 1e-12
            and abs(float(row["start_pg_midpoint_max_abs_error"])) <= 1e-12
            and abs(float(row["start_qg_abs_max"])) <= 1e-12
            for row in cold_rows
        ),
        "full_starts_supply_all_state_families": all(
            row["start_payload"] == "Pg,Qg,V,theta" for row in full_rows
        ),
        "frozen_checkpoint_provenance": all(
            row["warm_start_checkpoint_sha256"] == FROZEN_SHA256
            for row in rows if row["warm_start_type"] == "gridsfm_frozen_full_warm"
        ),
        "ft_checkpoint_provenance": all(
            row["warm_start_checkpoint_sha256"] == args.ft_sha256
            for row in rows if row["warm_start_type"] == "gridsfm_ft_full_warm"
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "rows": len(rows),
        "settings": len({row["setting_code"] for row in rows}),
        "finalist_families": FAMILY_ORDER,
        "warm_start_types": START_ORDER,
        "combination_counts": combinations,
        "objective_spread_tolerance": args.objective_spread_tolerance,
        "max_objective_spread_across_starts": max_spread,
        "cold_start_policy": "V=1,theta=0,Pg=(Pmin+Pmax)/2,Qg=0",
        "primary_metric": "solver_runtime_seconds",
        "supporting_metrics": [
            "wall_seconds", "start_construction_seconds", "end_to_end_seconds"
        ],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    records: list[dict[str, object]] = []
    if not args.aggregate_only:
        records.extend(_run_phase(
            args,
            cache_root=args.frozen_cache,
            warm_model="ft",
            source_package="frozen",
            expected_sha256=args.ft_sha256,
        ))
        records.extend(_run_phase(
            args,
            cache_root=args.ft_cache,
            warm_model="frozen",
            source_package="ft",
            expected_sha256=FROZEN_SHA256,
        ))
    rows = _assemble(args)
    core = args.result_root / "core_results"
    _write_csv(core / "warm_start_crossed.csv", rows)
    validation = _validate(rows, args)
    validation["execution_records"] = records
    validation["frozen_checkpoint_path"] = str(args.frozen_checkpoint)
    validation["frozen_checkpoint_sha256"] = FROZEN_SHA256
    validation["ft_checkpoint_path"] = str(args.ft_checkpoint)
    validation["ft_checkpoint_sha256"] = args.ft_sha256
    _write_json(args.result_root / "FT5_CROSSED_WARM_START_VALIDATION.json", validation)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python", type=Path,
        default=Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\envs\gridsfm\Scripts\python.exe"),
    )
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
        "--gridsfm-root", type=Path,
        default=Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\repos\GridSFM"),
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path(r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\inputs\case500_goc_e0"),
    )
    parser.add_argument(
        "--frozen-checkpoint", type=Path,
        default=Path(
            r"C:\Users\Caleb Lu\.gridfm_stage_j\repos\GridSFM\model\checkpoints\gridsfm_open_v1.1.pt"
        ),
    )
    parser.add_argument(
        "--ft-checkpoint", type=Path,
        default=Path(
            r"C:\Users\Caleb Lu\.gridfm_stage_j\checkpoints\stage_j_finetune\gridsfm_goc500_fulltop_ft_n1000.pt"
        ),
    )
    parser.add_argument("--ft-sha256", default=FT_SHA256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--objective-spread-tolerance", type=float, default=1e-3)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    for field in (
        "python", "frozen_cache", "ft_cache", "result_root", "gridsfm_root",
        "input_dir", "frozen_checkpoint", "ft_checkpoint",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.workers < 1:
        parser.error("--workers must be positive")
    validation = run(args)
    print(json.dumps({
        key: validation[key]
        for key in ("status", "rows", "settings", "max_objective_spread_across_starts")
    }, indent=2))
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
