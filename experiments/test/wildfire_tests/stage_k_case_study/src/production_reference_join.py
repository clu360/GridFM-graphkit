"""Validate and join independently executed production reference tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .io_utils import atomic_write_json
from .production_join import EVALUATORS, lambda_slug


def join_reference_tasks(
    *, config_path: str | Path, evaluator_dir: str | Path,
    task_root: str | Path, output_dir: str | Path,
) -> dict[str, object]:
    config = load_config(config_path)
    evaluators = Path(evaluator_dir)
    tasks = Path(task_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames_a: list[pd.DataFrame] = []
    frames_b: list[pd.DataFrame] = []
    frames_d: list[pd.DataFrame] = []
    task_status: dict[str, object] = {}

    for evaluator in EVALUATORS:
        finalists = pd.read_parquet(evaluators / f"{evaluator}_finalists.parquet")
        for lambda_r in map(float, config["search"]["lambda_r"]):
            expected = finalists.loc[finalists["lambda_r"].astype(float).eq(lambda_r)]
            if len(expected) != 1:
                raise ValueError(f"expected one sealed finalist for {evaluator}/{lambda_r}")
            task = tasks / evaluator / lambda_slug(lambda_r)
            summary_path = task / "reference_run_summary.json"
            required = {
                "a": task / "reference_a.parquet",
                "b": task / "reference_b.parquet",
                "d": task / "reference_discrepancies.parquet",
            }
            if not summary_path.is_file() or any(not path.is_file() for path in required.values()):
                raise FileNotFoundError(f"reference task incomplete: {task}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
                raise ValueError(f"reference task failed: {task}")
            if summary.get("config_sha256") != config["config_sha256"]:
                raise ValueError(f"reference config hash mismatch: {task}")
            if summary.get("evaluators") != [evaluator] or summary.get("lambda_values") != [lambda_r]:
                raise ValueError(f"reference task identity mismatch: {task}")
            a = pd.read_parquet(required["a"])
            b = pd.read_parquet(required["b"])
            d = pd.read_parquet(required["d"])
            if len(a) != 1 or len(b) != 1 or len(d) != 1:
                raise ValueError(f"reference task row accounting mismatch: {task}")
            b_row = b.iloc[0]
            b2_valid = bool(b_row["b2_solver_eligible"])
            expected_source = "b2_economic_tiebreak" if b2_valid else "b1_maximum_service_fallback"
            if b_row["diagnostic_state_source"] != expected_source:
                raise ValueError(f"Reference B diagnostic source violates the frozen B2 policy: {task}")
            if b2_valid != pd.notna(b_row["economic_cost_tiebreak"]):
                raise ValueError(f"Reference B economic cost violates the frozen B2 policy: {task}")
            for frame in (a, b, d):
                row = frame.iloc[0]
                if row["evaluator"] != evaluator or float(row["lambda_r"]) != lambda_r:
                    raise ValueError(f"reference row identity mismatch: {task}")
                if str(row["topology_key"]) != str(expected.iloc[0]["topology_key"]):
                    raise ValueError(f"reference topology differs from sealed finalist: {task}")
            frames_a.append(a)
            frames_b.append(b)
            frames_d.append(d)
            task_status[f"{evaluator}:{lambda_r:.6g}"] = {
                "status": summary["status"],
                "b2_failure_count": int(summary.get("reference_b2_failure_count", 0)),
            }

    reference_a = pd.concat(frames_a, ignore_index=True)
    reference_b = pd.concat(frames_b, ignore_index=True)
    discrepancies = pd.concat(frames_d, ignore_index=True)
    keys = ["evaluator", "lambda_r"]
    if any(frame.duplicated(keys).any() for frame in (reference_a, reference_b, discrepancies)):
        raise ValueError("duplicate reference primary key after join")
    expected_count = len(EVALUATORS) * len(config["search"]["lambda_r"])
    if not all(len(frame) == expected_count for frame in (reference_a, reference_b, discrepancies)):
        raise ValueError("joined reference accounting is incomplete")
    reference_a.to_parquet(output / "reference_a.parquet", index=False)
    reference_b.to_parquet(output / "reference_b.parquet", index=False)
    discrepancies.to_parquet(output / "reference_discrepancies.parquet", index=False)
    b2_failures = sum(int(row["b2_failure_count"]) for row in task_status.values())
    summary = {
        "status": "PASS" if b2_failures == 0 else "PASS_WITH_WARNINGS",
        "reference_a_rows": len(reference_a), "reference_b_rows": len(reference_b),
        "reference_b2_failure_count": b2_failures,
        "config_sha256": config["config_sha256"], "tasks": task_status,
    }
    atomic_write_json(output / "reference_run_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluator-dir", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = join_reference_tasks(
        config_path=args.config, evaluator_dir=args.evaluator_dir,
        task_root=args.task_root, output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
