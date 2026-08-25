"""Publish validated FT3/FT4 results with tabular CSV artifacts stored as Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _write_with_retry(destination: Path, write) -> None:
    for attempt in range(20):
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            write()
            return
        except (FileNotFoundError, PermissionError):
            if attempt == 19:
                raise
            time.sleep(0.5)


def _published_path(destination_root: Path, relative: Path, *, parquet: bool) -> Path:
    path_id = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:20].upper()
    suffix = ".parquet" if parquet else relative.suffix
    return destination_root / f"{path_id}{suffix}"


def publish(config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    source_root = _resolve(config["final_root"])
    destination_root = _resolve(config["publication_root"])
    validation = _read_json(source_root / "FT4_VALIDATION.json")
    status = _read_json(source_root / "FT3_FT4_STATUS.json")
    if validation.get("status") != "PASS":
        raise RuntimeError("cannot publish before FT4 validation passes")
    if status.get("status") != "FT3_FT4_COMPLETE":
        raise RuntimeError("cannot publish before linked FT3/FT4 completion")

    sources = sorted(path for path in source_root.rglob("*") if path.is_file())
    destination_parents = {
        _published_path(
            destination_root,
            source.relative_to(source_root),
            parquet=source.suffix.lower() == ".csv",
        ).parent
        for source in sources
    }
    for parent in sorted(destination_parents):
        parent.mkdir(parents=True, exist_ok=True)
    time.sleep(1.0)

    records: list[dict[str, Any]] = []
    for source in sources:
        relative = source.relative_to(source_root)
        if source.suffix.lower() == ".csv":
            destination = _published_path(destination_root, relative, parquet=True)
            frame = pd.read_csv(source, low_memory=False)
            _write_with_retry(destination, lambda: frame.to_parquet(destination, index=False))
            record = {
                "format": "parquet",
                "rows": len(frame),
                "source_relative_path": relative.as_posix(),
            }
        else:
            destination = _published_path(destination_root, relative, parquet=False)
            _write_with_retry(destination, lambda: shutil.copy2(source, destination))
            record = {
                "format": source.suffix.lower().lstrip(".") or "binary",
                "source_relative_path": relative.as_posix(),
            }
        records.append({
            **record,
            "published_relative_path": destination.relative_to(destination_root).as_posix(),
            "source_bytes": source.stat().st_size,
            "source_sha256": _sha256(source),
            "published_bytes": destination.stat().st_size,
            "published_sha256": _sha256(destination),
        })

    manifest = {
        "status": "PASS",
        "artifact_id": config["artifact_id"],
        "model_selection": config["model_selection"],
        "model_variant": config["model_variant"],
        "checkpoint_sha256": config["expected_checkpoint_sha256"],
        "source_root": str(source_root),
        "publication_root": str(destination_root),
        "files": len(records),
        "csv_files_converted_to_parquet": sum(row["format"] == "parquet" for row in records),
        "artifacts": records,
    }
    manifest_path = destination_root / "PUBLICATION_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    publish(Path(args.config).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
