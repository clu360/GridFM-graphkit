"""Run linked FT3 Stage J execution and FT4 evidence reproduction."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
WILDFIRE_ROOT = Path(__file__).resolve().parents[2]
if str(WILDFIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_ROOT))

from stage_j_gridsfm_goc500.model_selection import resolve_model_selection


COMPLETE_RUNNER = Path(__file__).resolve().parents[1] / "run_stage_j_complete.py"
SUMMARIZER = Path(__file__).resolve().parents[1] / "summarize_stage_j_complete.py"
COMPARISON_PLOTTER = Path(__file__).with_name("plot_ft3_ft4_model_comparison.py")
VALIDATOR = Path(__file__).with_name("validate_ft3_ft4.py")
PUBLISHER = Path(__file__).with_name("publish_ft3_ft4.py")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _preflight(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "analysis_python", "artifact_id", "cache_root", "checkpoint",
        "expected_checkpoint_sha256", "expected_gridsfm_commit", "final_root",
        "ft1_manifest", "frozen_result_root", "gridsfm_root", "model_selection",
        "model_variant", "publication_root", "run_id",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing FT3/FT4 config keys: {missing}")
    if config["model_selection"] != "ft" or config["model_variant"] != "fulltop_ft_n1000":
        raise ValueError("FT3/FT4 is pinned to model_selection=ft/fulltop_ft_n1000")

    gridsfm_root = Path(config["gridsfm_root"]).expanduser().resolve()
    selected = resolve_model_selection(
        config["model_selection"], gridsfm_root=gridsfm_root,
        checkpoint=Path(config["checkpoint"]),
        expected_sha256=config["expected_checkpoint_sha256"],
    )
    if _git_commit(gridsfm_root) != config["expected_gridsfm_commit"]:
        raise RuntimeError("GridSFM commit differs from the FT3 pin")
    ft1_manifest_path = _repo_path(config["ft1_manifest"])
    ft1 = _read_json(ft1_manifest_path)
    if ft1["status"] != "FT1_TRAINED_READY_FOR_FT2":
        raise RuntimeError("FT1 manifest is not approved for FT3")
    if ft1["checkpoint_path"] != selected.checkpoint_path:
        raise RuntimeError("FT1 and FT3 checkpoint paths differ")
    if ft1["checkpoint_sha256"] != selected.checkpoint_sha256:
        raise RuntimeError("FT1 and FT3 checkpoint hashes differ")

    frozen_root = _repo_path(config["frozen_result_root"])
    frozen_status = _read_json(frozen_root / "RUN_STATUS.json")
    if frozen_status.get("status") != "COMPLETE":
        raise RuntimeError("frozen Stage J comparison package is incomplete")
    expected_settings = {
        f"s{scenario}_l{str(value).replace('.', 'p')}"
        for scenario in (1, 2, 3) for value in (0.0, 0.2, 0.5, 0.8, 1.0)
    }
    observed_settings = {
        path.name for path in (frozen_root / "settings").iterdir() if path.is_dir()
    }
    if expected_settings != observed_settings:
        raise RuntimeError("frozen Stage J setting set differs from the FT3 contract")
    for setting in sorted(expected_settings):
        for pool_name in ("guided.csv", "th.csv"):
            if not (frozen_root / "settings" / setting / "pools" / pool_name).is_file():
                raise FileNotFoundError(f"missing frozen topology pool: {setting}/{pool_name}")

    analysis_python = Path(config["analysis_python"]).expanduser().resolve()
    if not analysis_python.is_file():
        raise FileNotFoundError(f"analysis Python does not exist: {analysis_python}")
    return {
        "status": "PASS", "artifact_id": config["artifact_id"],
        **selected.as_dict(), "gridsfm_commit": config["expected_gridsfm_commit"],
        "ft1_manifest": str(ft1_manifest_path), "frozen_result_root": str(frozen_root),
        "settings": 15, "topology_pools": 30,
    }


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


@contextmanager
def _exclusive_run_lock(cache_root: Path):
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / "FT3_FT4_RUN.lock"
    handle = lock_path.open("a+b")
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
        raise RuntimeError(f"another FT3/FT4 process holds {lock_path}") from exc
    try:
        yield lock_path
    finally:
        handle.seek(0)
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def run(config_path: Path, *, preflight_only: bool = False) -> int:
    config = _read_json(config_path)
    final_root = _repo_path(config["final_root"])
    status_path = final_root / "FT3_FT4_STATUS.json"
    preflight = _preflight(config)
    final_root.mkdir(parents=True, exist_ok=True)
    _write_json(final_root / "FT3_FT4_PREFLIGHT.json", preflight)
    if preflight_only:
        _write_json(status_path, {**preflight, "status": "PREFLIGHT_PASS"})
        return 0

    cache_root = Path(config["cache_root"]).expanduser().resolve()
    with _exclusive_run_lock(cache_root):
        _write_json(status_path, {**preflight, "status": "PREFLIGHT_PASS"})
        complete_command = [
            sys.executable, str(COMPLETE_RUNNER),
            "--run-id", config["run_id"],
            "--cache-root", str(cache_root),
            "--final-root", str(final_root),
            "--gridsfm-root", str(Path(config["gridsfm_root"]).expanduser().resolve()),
            "--model-selection", "ft",
            "--gridsfm-checkpoint", str(Path(config["checkpoint"]).expanduser().resolve()),
            "--expected-checkpoint-sha256", config["expected_checkpoint_sha256"],
            "--pool-source-root", str(_repo_path(config["frozen_result_root"])),
        ]
        _write_json(status_path, {**preflight, "status": "FT3_RUNNING", "command": complete_command})
        _run(complete_command)
        run_status = _read_json(final_root / "RUN_STATUS.json")
        if run_status.get("status") != "COMPLETE":
            raise RuntimeError("FT3 complete-run status did not pass")

        analysis_python = str(Path(config["analysis_python"]).expanduser().resolve())
        _write_json(status_path, {**preflight, "status": "FT4_RUNNING"})
        frozen_result_root = str(_repo_path(config["frozen_result_root"]))
        _run([
            analysis_python, str(SUMMARIZER), "--result-root", str(final_root),
            "--comparison-root", frozen_result_root,
        ])
        _run([
            analysis_python, str(COMPARISON_PLOTTER), "--result-root", str(final_root),
            "--comparison-root", frozen_result_root,
        ])
        _run([sys.executable, str(VALIDATOR), "--result-root", str(final_root)])
        validation = _read_json(final_root / "FT4_VALIDATION.json")
        if validation.get("status") != "PASS":
            raise RuntimeError("FT4 artifact validation failed")
        _write_json(status_path, {**preflight, "status": "FT3_FT4_COMPLETE"})
        _run([analysis_python, str(PUBLISHER), "--config", str(config_path)])
        publication = _read_json(_repo_path(config["publication_root"]) / "PUBLICATION_MANIFEST.json")
        if publication.get("status") != "PASS":
            raise RuntimeError("FT3/FT4 publication validation failed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    return run(Path(args.config).expanduser().resolve(), preflight_only=args.preflight_only)


if __name__ == "__main__":
    raise SystemExit(main())
