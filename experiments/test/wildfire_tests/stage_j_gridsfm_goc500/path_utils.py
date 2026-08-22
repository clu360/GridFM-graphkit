"""Path helpers for Stage J external environment discovery."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_gridsfm_root(explicit_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the external GridSFM checkout root without hard-coded long paths."""

    raw = explicit_path or os.environ.get("GRIDSFM_ROOT")
    if raw is None or str(raw).strip() == "":
        raise FileNotFoundError("GridSFM root is not configured; pass --gridsfm-root or set GRIDSFM_ROOT")
    root = Path(raw).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"GridSFM root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"GridSFM root is not a directory: {root}")
    return root


def resolve_graphkit_root(explicit_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the local GridFM-GraphKit checkout root."""

    raw = explicit_path or os.environ.get("GRIDFM_GRAPHKIT_ROOT")
    if raw is None or str(raw).strip() == "":
        return Path(__file__).resolve().parents[4]
    root = Path(raw).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"GridFM-GraphKit root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"GridFM-GraphKit root is not a directory: {root}")
    return root
