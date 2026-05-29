from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def make_run_dir(output_root: Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_json_safe(data), f, indent=2)
            return
        except FileNotFoundError:
            if attempt == 2:
                raise
            time.sleep(0.1)


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            return
        except FileNotFoundError:
            if attempt == 2:
                raise
            time.sleep(0.1)


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if not isinstance(v, pd.DataFrame)}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def git_metadata() -> Dict[str, str]:
    def run_git(args):
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unavailable"

    return {
        "git_branch": run_git(["branch", "--show-current"]),
        "git_commit": run_git(["rev-parse", "HEAD"]),
    }
