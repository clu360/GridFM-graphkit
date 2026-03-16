"""
Filesystem helpers for the Improved Optimization experiment suite.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_results_structure(results_root: Path) -> Dict[str, Path]:
    improved_runs = ensure_directory(results_root / "improved_runs")
    pareto = ensure_directory(results_root / "pareto")
    sensitivity = ensure_directory(results_root / "sensitivity")
    figures = ensure_directory(results_root / "figures")
    return {
        "results_root": ensure_directory(results_root),
        "improved_runs": improved_runs,
        "pareto": pareto,
        "sensitivity": sensitivity,
        "figures": figures,
    }


def create_run_directory(base_dir: Path, run_name: str, timestamp: bool = True) -> Path:
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else "latest"
    return ensure_directory(base_dir / f"{run_name}_{suffix}")


def relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)
