"""Publish the validated FT7 refined study with Parquet tabular artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_extended_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _extended_path(path: Path) -> str:
    resolved = str(path.resolve())
    return f"\\\\?\\{resolved}" if os.name == "nt" else resolved


def _stat(path: Path):
    return os.stat(_extended_path(path))


def _copy_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(20):
        try:
            os.makedirs(_extended_path(destination.parent), exist_ok=True)
            shutil.copyfile(_extended_path(source), _extended_path(destination))
            return
        except (FileNotFoundError, PermissionError):
            if attempt == 19:
                raise
            time.sleep(0.5)


def publish(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    working = Path(config["working_root"]).resolve()
    source = working / "combined"
    destination = _resolve(config["publication_root"])
    core_validation = _read_json(source / "FT7_CORE_VALIDATION.json")
    pareto_validation = _read_json(source / "FT7_PARETO_VALIDATION.json")
    summary_status = _read_json(source / "FT7_SUMMARY_STATUS.json")
    warm_validation = _read_json(working / "warm_start/FT7_WARM_START_VALIDATION.json")
    gates = {
        "core": core_validation.get("status") == "FT7_P2_EXACT_AND_CORE_PASS",
        "pareto": pareto_validation.get("status") == "FT7_P3_ALL_CANDIDATE_PARETO_PASS",
        "warm": warm_validation.get("status") == "FT7_P4_WARM_START_PASS",
        "summary": summary_status.get("status") == "FT7_P4_SUMMARY_PASS",
        "figures_15": summary_status.get("figure_count") == 15,
        "warm_rows_525": summary_status.get("warm_start_rows") == 525,
        "controlled_iteration_rerun": summary_status.get("warm_start_source") == "controlled_iteration_rerun",
    }
    if not all(gates.values()):
        raise RuntimeError(f"cannot publish FT7 before all gates pass: {gates}")

    sources = sorted(path for path in source.rglob("*") if path.is_file())
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ft7_publish_") as temporary:
        staging = Path(temporary)
        for index, input_path in enumerate(sources):
            relative = input_path.relative_to(source)
            published_relative = relative
            if relative.parts[:2] == ("core_results", "derived_visual_summaries"):
                published_relative = Path("derived", *relative.parts[2:])
            if input_path.suffix.lower() == ".csv":
                output_path = destination / published_relative.with_suffix(".parquet")
                staged = staging / f"{index:04d}.parquet"
                frame = pd.read_csv(input_path, low_memory=False)
                frame.to_parquet(staged, index=False)
                restored = pd.read_parquet(staged)
                if len(restored) != len(frame) or list(restored.columns) != list(frame.columns):
                    raise RuntimeError(f"Parquet read-back differs from source: {relative}")
                file_format = "parquet"
                rows = len(frame)
            else:
                output_path = destination / published_relative
                staged = input_path
                file_format = input_path.suffix.lower().lstrip(".") or "binary"
                rows = None
            _copy_with_retry(staged, output_path)
            if _sha256(staged) != _sha256(output_path):
                raise RuntimeError(f"published byte hash differs after copy: {relative}")
            records.append({
                "source_relative_path": relative.as_posix(),
                "published_relative_path": output_path.relative_to(destination).as_posix(),
                "format": file_format, "rows": rows,
                "source_bytes": input_path.stat().st_size,
                "published_bytes": _stat(output_path).st_size,
                "source_sha256": _sha256(input_path),
                "published_sha256": _sha256(output_path),
            })

    supporting = {
        "FT7_WARM_START_VALIDATION.json": working / "warm_start/FT7_WARM_START_VALIDATION.json",
        "FT7_ITERATION_RERUN_VALIDATION.json": working / "warm_start_iteration_rerun/FT7_ITERATION_RERUN_VALIDATION.json",
        "FT7_M2_OPS_VALIDATION.json": destination / "FT7_M2_OPS_VALIDATION.json",
        "FT7_M3_OPS_VALIDATION.json": destination / "FT7_M3_OPS_VALIDATION.json",
        "FT7_PREFLIGHT.json": destination / "FT7_PREFLIGHT.json",
        "FT7_DISK_CAPACITY_CHECK.json": destination / "FT7_DISK_CAPACITY_CHECK.json",
        "FT7_APPROVAL.json": _resolve(config["approval_path"]),
        "REFINED_FINETUNE_STUDY_SUMMARY.md": destination / "REFINED_FINETUNE_STUDY_SUMMARY.md",
    }
    supporting_records = []
    for name, input_path in supporting.items():
        output_path = destination / name
        if not input_path.is_file():
            raise FileNotFoundError(f"missing FT7 supporting evidence: {input_path}")
        if input_path.resolve() == output_path.resolve():
            pass
        else:
            _copy_with_retry(input_path, output_path)
        supporting_records.append({
            "published_relative_path": name,
            "published_bytes": _stat(output_path).st_size,
            "published_sha256": _sha256(output_path),
        })

    manifest = {
        "status": "FT7_P5_PUBLICATION_PASS",
        "artifact_id": config["artifact_id"], "publication_root": str(destination),
        "source_root": str(source), "gates": gates, "files": len(records),
        "csv_files_converted_to_parquet": sum(row["format"] == "parquet" for row in records),
        "checkpoint_files_published": 0, "methods": ["dc", "m0", "m1", "m2", "m3"],
        "pareto_scope": "all eligible candidates pooled across lambda by scenario/model",
        "artifacts": records, "supporting_artifacts": supporting_records,
    }
    _write_json(destination / "PUBLICATION_MANIFEST.json", manifest)
    _write_json(destination / "FT7_STATUS.json", {
        "status": "FT7_COMPLETE", "phase": "PUBLISHED", "publication_manifest": "PUBLICATION_MANIFEST.json",
    })
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    manifest = publish(args.config.resolve())
    print(json.dumps({key: manifest[key] for key in (
        "status", "files", "csv_files_converted_to_parquet", "checkpoint_files_published"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
