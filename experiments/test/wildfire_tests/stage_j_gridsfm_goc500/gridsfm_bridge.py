"""GridSFM adapter contract.

This module deliberately does not import the external GridSFM package. The
actual inference runner should live in the GridSFM environment and consume a
serialized request produced by this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .path_utils import resolve_gridsfm_root
from .schemas import EvaluationStatus


@dataclass(frozen=True)
class GridSFMEnvironmentStatus:
    gridsfm_root: Path | None
    available: bool
    evaluation_status: EvaluationStatus
    message: str


def check_gridsfm_environment(gridsfm_root: str | Path | None = None) -> GridSFMEnvironmentStatus:
    """Check whether the external GridSFM checkout is available for J1 smoke."""

    try:
        root = resolve_gridsfm_root(gridsfm_root)
    except (FileNotFoundError, NotADirectoryError) as exc:
        return GridSFMEnvironmentStatus(
            gridsfm_root=None,
            available=False,
            evaluation_status=EvaluationStatus.GRIDSFM_PREPROCESSING_FAILURE,
            message=str(exc),
        )
    return GridSFMEnvironmentStatus(
        gridsfm_root=root,
        available=True,
        evaluation_status=EvaluationStatus.OK,
        message="GridSFM root resolved; official smoke runner still must validate model loading and GOC-500 inference.",
    )
