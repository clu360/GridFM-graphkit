from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.test.wildfire_tests.shared.reporting import write_json


NUMERIC_TOLERANCE = 1.0e-8


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _numeric_delta(old_value, new_value) -> float | None:
    try:
        return float(new_value) - float(old_value)
    except (TypeError, ValueError):
        return None


def compare_run_directories(old_run_dir: str | Path, new_run_dir: str | Path, output_path: str | Path | None = None) -> dict:
    old_run_dir = Path(old_run_dir)
    new_run_dir = Path(new_run_dir)
    old_summary = _load_json(old_run_dir / "optimization_summary.json")
    new_summary = _load_json(new_run_dir / "optimization_summary.json")
    key_rows = []
    for key in sorted(set(old_summary) & set(new_summary)):
        delta = _numeric_delta(old_summary.get(key), new_summary.get(key))
        matches = old_summary.get(key) == new_summary.get(key)
        if delta is not None:
            matches = abs(delta) <= NUMERIC_TOLERANCE
        key_rows.append(
            {
                "key": key,
                "old": old_summary.get(key),
                "new": new_summary.get(key),
                "numeric_delta": delta,
                "matches": bool(matches),
            }
        )

    artifact_rows = []
    for relative in [
        "objective_trace.csv",
        "risk_by_group_before_after.csv",
        "risk_by_line_before_after.csv",
        "wildfire_scenario.json",
        "visualization_summary.json",
        "figures/optimization_behavior.png",
        "figures/ieee30_network_changes.png",
    ]:
        artifact_rows.append(
            {
                "artifact": relative,
                "old_exists": (old_run_dir / relative).exists(),
                "new_exists": (new_run_dir / relative).exists(),
            }
        )

    result = {
        "old_run_dir": str(old_run_dir),
        "new_run_dir": str(new_run_dir),
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "summary_key_comparisons": key_rows,
        "artifact_comparisons": artifact_rows,
        "all_summary_keys_match": all(row["matches"] for row in key_rows),
        "all_expected_new_artifacts_exist": all(row["new_exists"] for row in artifact_rows),
    }
    if output_path is not None:
        output_path = Path(output_path)
        write_json(output_path, result)
        pd.DataFrame(key_rows).to_csv(output_path.with_suffix(".summary_keys.csv"), index=False)
        pd.DataFrame(artifact_rows).to_csv(output_path.with_suffix(".artifacts.csv"), index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-run-dir", required=True, type=Path)
    parser.add_argument("--new-run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = compare_run_directories(args.old_run_dir, args.new_run_dir, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
