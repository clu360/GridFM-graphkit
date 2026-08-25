"""Preflight and run the two approved FT7 guided-only Stage J model variants."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[5]
WILDFIRE_ROOT = Path(__file__).resolve().parents[2]
if str(WILDFIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_ROOT))

from stage_j_gridsfm_goc500.model_selection import resolve_model_selection, sha256_file


COMPLETE_RUNNER = Path(__file__).resolve().parents[1] / "run_stage_j_complete.py"
SUCCESS_STATUS = "FT7_P0_PREFLIGHT_PASS"
EXPECTED_SETTINGS = {
    f"s{scenario}_l{str(value).replace('.', 'p')}"
    for scenario in (1, 2, 3) for value in (0.0, 0.2, 0.5, 0.8, 1.0)
}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "analysis_python", "approval_path", "artifact_id", "expected_gridsfm_commit",
        "frozen_cache_root", "frozen_result_root", "gridsfm_python", "gridsfm_root",
        "m1_cache_root", "m1_result_root", "models", "publication_root", "settings",
        "working_root",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing FT7 config keys: {missing}")
    if tuple(config["models"]) != ("m2", "m3"):
        raise ValueError("FT7 model order must be m2 then m3")
    settings = config["settings"]
    if settings != {
        "continuous_eval_budget": 20,
        "lambdas": [0.0, 0.2, 0.5, 0.8, 1.0],
        "q": 5,
        "scenarios": ["J-S1", "J-S2", "J-S3"],
        "topology_budget": 100,
    }:
        raise ValueError("FT7 settings differ from the frozen Stage J contract")


def _package_settings(root: Path) -> set[str]:
    settings = root / "settings"
    return {path.name for path in settings.iterdir() if path.is_dir()}


def _preflight(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    _validate_config(config)
    approval_path = _repo_path(config["approval_path"])
    approval = _read_json(approval_path)
    if approval.get("approval") != "EXPLICIT":
        raise RuntimeError("FT7 explicit approval is absent")

    ft6_status_path = REPO_ROOT / (
        "experiments/test/wildfire_tests/stage_j_gridsfm_goc500/finetune/"
        "results/ft6/ft6_status.json"
    )
    ft6_status = _read_json(ft6_status_path)
    if ft6_status.get("status") != "FT6_COMPLETE_AWAITING_CALEB_FT7_APPROVAL":
        raise RuntimeError("FT6 terminal review gate is not complete")

    gridsfm_root = Path(config["gridsfm_root"]).expanduser().resolve()
    observed_commit = _git_commit(gridsfm_root)
    if observed_commit != config["expected_gridsfm_commit"]:
        raise RuntimeError("GridSFM commit differs from the FT7 protocol")

    model_records = {}
    for model_id, spec in config["models"].items():
        training_manifest_path = _repo_path(spec["training_manifest"])
        training = _read_json(training_manifest_path)
        required_status = f"FT6_P2{('A' if model_id == 'm2' else 'B')}_{model_id.upper()}_TRAINING_PASS"
        if training.get("status") != required_status:
            raise RuntimeError(f"{model_id} FT6 training manifest is not approved")
        selected = resolve_model_selection(
            "ft", gridsfm_root=gridsfm_root,
            checkpoint=Path(spec["checkpoint"]),
            expected_sha256=spec["expected_checkpoint_sha256"],
        )
        if selected.model_variant != spec["expected_model_variant"]:
            raise RuntimeError(f"{model_id} model variant resolution mismatch")
        if training["checkpoint_sha256"] != selected.checkpoint_sha256:
            raise RuntimeError(f"{model_id} training and FT7 checkpoint hashes differ")
        model_records[model_id] = {
            **selected.as_dict(),
            "training_manifest": str(training_manifest_path),
            "training_manifest_sha256": sha256_file(training_manifest_path),
            "cache_root": str(Path(spec["cache_root"]).expanduser().resolve()),
            "working_result_root": str(Path(spec["working_result_root"]).expanduser().resolve()),
        }

    frozen_root = _repo_path(config["frozen_result_root"])
    m1_root = _repo_path(config["m1_result_root"])
    frozen_status = _read_json(frozen_root / "RUN_STATUS.json")
    m1_status = _read_json(m1_root / "FT3_FT4_STATUS.json")
    if frozen_status.get("status") != "COMPLETE":
        raise RuntimeError("DC/released Stage J package is incomplete")
    if m1_status.get("status") != "FT3_FT4_COMPLETE":
        raise RuntimeError("M1 Stage J package is incomplete")
    if m1_status.get("checkpoint_sha256") != "A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD":
        raise RuntimeError("M1 Stage J checkpoint hash differs from protocol")
    if _package_settings(frozen_root) != EXPECTED_SETTINGS or _package_settings(m1_root) != EXPECTED_SETTINGS:
        raise RuntimeError("existing Stage J package setting coverage differs from FT7")

    pool_records = []
    for setting in sorted(EXPECTED_SETTINGS):
        path = frozen_root / "settings" / setting / "pools" / "guided.csv"
        rows = _read_csv(path)
        if len(rows) != 100:
            raise RuntimeError(f"guided topology pool does not contain 100 rows: {setting}")
        pool_records.append({
            "setting_code": setting, "path": str(path),
            "sha256": sha256_file(path), "rows": len(rows),
        })

    ft5_path = m1_root / "core_results" / "warm_start_crossed.csv"
    ft5 = _read_csv(ft5_path)
    retained_ft5 = [
        row for row in ft5
        if row.get("finalist_family") in {
            "Guided-DC", "Guided-GridSFM (frozen)", "Guided-GridSFM (fine-tuned)"
        }
    ]
    if len(ft5) != 375 or len(retained_ft5) != 225:
        raise RuntimeError("FT5 warm-start reuse rows differ from the 225-row FT7 contract")
    if {row.get("runtime_metric") for row in retained_ft5} != {"ipopt_solve_time_seconds"}:
        raise RuntimeError("FT5 reusable warm starts do not use corrected native IPOPT timing")

    checks = {
        "explicit_approval_recorded": True,
        "ft6_gate_complete": True,
        "gridsfm_commit_frozen": True,
        "m2_m3_checkpoint_paths_hashes_and_variants_verified": True,
        "existing_dc_m0_m1_packages_complete": True,
        "same_15_settings": True,
        "all_15_guided_pools_have_100_rows": True,
        "ft5_native_ipopt_reuse_rows_225": True,
        "new_execution_is_m2_m3_guided_only": True,
        "publication_root_is_refined_finetune_study": Path(config["publication_root"]).name == "refined_finetune_study",
    }
    status = SUCCESS_STATUS if all(checks.values()) else "FT7_PREFLIGHT_BLOCKED"
    return {
        "artifact_id": config["artifact_id"], "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "approval": {"path": str(approval_path), "sha256": sha256_file(approval_path)},
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "ft6_status": {"path": str(ft6_status_path), "sha256": sha256_file(ft6_status_path)},
        "gridsfm_commit": observed_commit, "models": model_records,
        "frozen_result_root": str(frozen_root), "m1_result_root": str(m1_root),
        "guided_topology_pools": pool_records,
        "warm_start_reuse": {
            "source": str(ft5_path), "source_rows": len(ft5),
            "retained_non_th_rows": len(retained_ft5),
            "runtime_metric": "ipopt_solve_time_seconds",
        },
        "output_contract": {
            "working_root": str(Path(config["working_root"]).expanduser().resolve()),
            "publication_root": str(_repo_path(config["publication_root"])),
            "methods": ["dc", "m0", "m1", "m2", "m3"],
            "pareto_source": "all eligible candidate evaluations pooled across lambda by scenario/method",
            "warm_start_rows": 525, "th_included": False,
        },
        "checks": checks,
    }


def _validate_model_run(config: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    spec = config["models"][model_id]
    root = Path(spec["working_result_root"]).expanduser().resolve()
    status = _read_json(root / "RUN_STATUS.json")
    run_config = _read_json(root / "RUN_CONFIG.json")
    counts = _read_json(root / "ARTIFACT_COUNT_VALIDATION.json")
    rows = _read_csv(root / "core_results" / "candidate_evaluations_all.csv")
    checks = {
        "run_complete": status.get("status") == "COMPLETE",
        "guided_only": run_config.get("guided_only") is True,
        "only_guided_method": run_config.get("methods") == ["Guided-GridSFM"],
        "model_variant": run_config.get("model_variant") == spec["expected_model_variant"],
        "checkpoint_sha256": run_config.get("checkpoint_sha256") == spec["expected_checkpoint_sha256"],
        "artifact_counts_pass": counts.get("status") == "PASS",
        "candidate_rows_30000": len(rows) == 30000,
        "candidate_provenance": all(
            row.get("method") == "Guided-GridSFM"
            and row.get("model_variant") == spec["expected_model_variant"]
            and row.get("checkpoint_sha256") == spec["expected_checkpoint_sha256"]
            for row in rows
        ),
        "settings_15": len(status.get("settings", [])) == 15,
        "th_not_executed": all(row.get("th_gridsfm_ok") is None for row in status.get("settings", [])),
    }
    return {
        "status": f"FT7_P1{('A' if model_id == 'm2' else 'B')}_{model_id.upper()}_OPS_PASS"
        if all(checks.values()) else "FT7_MODEL_RUN_BLOCKED",
        "model_id": model_id, "root": str(root), "checks": checks,
        "artifact_counts": counts,
    }


@contextmanager
def _exclusive_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "FT7_RUN.lock"
    handle = path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"another FT7 process holds {path}") from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _run_model(config: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    spec = config["models"][model_id]
    command = [
        str(Path(config["gridsfm_python"]).expanduser().resolve()),
        str(COMPLETE_RUNNER),
        "--run-id", spec["run_id"],
        "--cache-root", str(Path(spec["cache_root"]).expanduser().resolve()),
        "--final-root", str(Path(spec["working_result_root"]).expanduser().resolve()),
        "--gridsfm-root", str(Path(config["gridsfm_root"]).expanduser().resolve()),
        "--model-selection", "ft",
        "--gridsfm-checkpoint", str(Path(spec["checkpoint"]).expanduser().resolve()),
        "--expected-checkpoint-sha256", spec["expected_checkpoint_sha256"],
        "--pool-source-root", str(_repo_path(config["frozen_result_root"])),
        "--guided-only",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    result = _validate_model_run(config, model_id)
    if not all(result["checks"].values()):
        raise RuntimeError(f"{model_id} FT7 OPS validation failed: {result['checks']}")
    return result


def run(config_path: Path, *, preflight_only: bool, model: str | None) -> int:
    config = _read_json(config_path)
    publication_root = _repo_path(config["publication_root"])
    publication_root.mkdir(parents=True, exist_ok=True)
    status_path = publication_root / "FT7_STATUS.json"
    preflight = _preflight(config_path)
    _write_json(publication_root / "FT7_PREFLIGHT.json", preflight)
    _write_json(status_path, {"status": preflight["status"], "phase": "PREFLIGHT"})
    if preflight["status"] != SUCCESS_STATUS:
        return 1
    if preflight_only:
        print(json.dumps({"status": SUCCESS_STATUS}, indent=2))
        return 0

    models = [model] if model else ["m2", "m3"]
    working_root = Path(config["working_root"]).expanduser().resolve()
    with _exclusive_lock(working_root):
        for model_id in models:
            _write_json(status_path, {"status": "FT7_RUNNING", "phase": f"{model_id.upper()}_OPS"})
            result = _run_model(config, model_id)
            _write_json(publication_root / f"FT7_{model_id.upper()}_OPS_VALIDATION.json", result)
            _write_json(status_path, {"status": result["status"], "phase": f"{model_id.upper()}_COMPLETE"})
            print(json.dumps({"status": result["status"], "model_id": model_id}, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--model", choices=["m2", "m3"])
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    try:
        return run(config_path, preflight_only=args.preflight_only, model=args.model)
    except Exception as exc:
        try:
            config = _read_json(config_path)
            root = _repo_path(config["publication_root"])
            _write_json(root / "FT7_STATUS.json", {
                "status": "FT7_BLOCKED", "phase": "UNHANDLED_EXCEPTION",
                "exception_type": type(exc).__name__, "message": str(exc),
            })
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
