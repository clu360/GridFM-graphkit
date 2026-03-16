"""
Reporting utilities for Improved Optimization runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .io_utils import relative_to_repo


def save_objective_history_csv(history, output_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "iteration": range(len(history.objective)),
            "objective": history.objective,
            "generator_deviation_cost": history.generator_deviation_cost,
            "load_shedding_cost": history.load_shedding_cost,
            "wildfire_cost": history.wildfire_cost,
            "total_shed_mw": history.total_shed_mw,
            "max_risky_line_loading": history.max_risky_line_loading,
            "decision_norm": history.decision_norm,
        }
    )
    df.to_csv(output_path, index=False)
    return output_path


def save_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    df.to_csv(output_path, index=False)
    return output_path


def save_optimization_summary_json(summary: Dict, output_path: Path) -> Path:
    serializable = {}
    for key, value in summary.items():
        if hasattr(value, "tolist"):
            serializable[key] = value.tolist()
        elif isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            serializable[key] = value
        else:
            serializable[key] = str(value)
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return output_path


def write_markdown_summary(output_path: Path, title: str, lines: list[str]) -> Path:
    content = [f"# {title}", ""]
    content.extend(lines)
    output_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    return output_path


def append_run_summary_block(report_path: Path, header: str, summary: Dict, repo_root: Path) -> None:
    lines = [f"## {header}", ""]
    for key, value in summary.items():
        if isinstance(value, Path):
            value = relative_to_repo(value, repo_root)
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
