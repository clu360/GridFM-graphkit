"""Hash-bound, atomic Stage K artifact utilities."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def require_new_run_dir(path: str | Path, *, resume: bool = False) -> Path:
    target = Path(path)
    complete = target / "RUN_COMPLETE.json"
    if complete.exists():
        raise FileExistsError(f"completed Stage K run is immutable: {target}")
    if target.exists() and not resume and any(target.iterdir()):
        raise FileExistsError(f"run directory exists and is nonempty; use explicit resume: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def checkpoint_matches(path: str | Path, expected: dict[str, Any]) -> bool:
    candidate = Path(path)
    if not candidate.exists():
        return False
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return False
    return all(payload.get(key) == value for key, value in expected.items())
