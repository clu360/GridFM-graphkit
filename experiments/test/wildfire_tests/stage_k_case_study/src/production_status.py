"""Build the machine-readable Stage K production completion watchdog."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .io_utils import atomic_write_json
from .production_join import EVALUATORS, lambda_slug


def _read_job_ids(run: Path) -> dict[str, str]:
    path = run / "submitted_job_ids.csv"
    if not path.is_file():
        return {}
    table = pd.read_csv(path, dtype=str)
    return dict(zip(table["job"], table["job_id"], strict=True))


def _slurm_records(run: Path) -> list[dict[str, object]]:
    path = run / "runtime" / "slurm_sacct.psv"
    if not path.is_file() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path, sep="|").where(pd.notna, None).to_dict(orient="records")


def _checkpoint_info(path: Path) -> tuple[int, str | None]:
    files = list(path.rglob("*.json")) if path.exists() else []
    if not files:
        return 0, None
    newest = max(files, key=lambda item: item.stat().st_mtime)
    return len(files), datetime.fromtimestamp(newest.stat().st_mtime, tz=timezone.utc).isoformat()


def build_status(*, config_path: str | Path, run_dir: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    run = Path(run_dir)
    prepared_manifest_path = run / "prepared" / "input_manifest.json"
    prepared = json.loads(prepared_manifest_path.read_text(encoding="utf-8")) if prepared_manifest_path.is_file() else {}
    job_ids = _read_job_ids(run)
    evaluator_tasks: dict[str, object] = {}
    reference_tasks: dict[str, object] = {}
    lambdas = [float(value) for value in config["search"]["lambda_r"]]

    for evaluator in EVALUATORS:
        for index, lambda_r in enumerate(lambdas):
            key = f"{evaluator}:{lambda_r:.6g}"
            chunk = run / "chunks" / evaluator / lambda_slug(lambda_r)
            summary_path = chunk / f"{evaluator}_run_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
            checkpoint_count, checkpoint_time = _checkpoint_info(chunk / "checkpoints")
            evaluator_tasks[key] = {
                "submitted": f"{evaluator}_array" in job_ids,
                "job_id": f"{job_ids.get(f'{evaluator}_array', '')}_{index}" if f"{evaluator}_array" in job_ids else None,
                "state": "completed" if summary.get("status") == "PASS" else "submitted" if f"{evaluator}_array" in job_ids else "not_submitted",
                "expected_candidates": 1 + int(config["search"]["k1_count"]) + int(config["search"]["k2_max_unique"]),
                "attempted_candidates": int(summary.get("attempted_states", 0)),
                "eligible_candidates": int(summary.get("eligible_states", 0)),
                "failed_candidates": int(summary.get("attempted_states", 0)) - int(summary.get("eligible_states", 0)),
                "checkpoint_location": str(chunk / "checkpoints"),
                "checkpoint_count": checkpoint_count,
                "checkpoint_latest_utc": checkpoint_time,
                "config_sha256": summary.get("config_sha256", config["config_sha256"]),
                "input_sha256": summary.get("input_sha256"),
                "final_status": summary.get("status"),
            }

    for task_index in range(len(EVALUATORS) * len(lambdas)):
        evaluator = EVALUATORS[task_index // len(lambdas)]
        lambda_r = lambdas[task_index % len(lambdas)]
        key = f"{evaluator}:{lambda_r:.6g}"
        task = run / "reference_tasks" / evaluator / lambda_slug(lambda_r)
        summary_path = task / "reference_run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        reference_tasks[key] = {
            "submitted": "references_array" in job_ids,
            "job_id": f"{job_ids.get('references_array', '')}_{task_index}" if "references_array" in job_ids else None,
            "state": "completed" if summary.get("status") in {"PASS", "PASS_WITH_WARNINGS"} else "submitted" if "references_array" in job_ids else "not_submitted",
            "reference_a_attempted": (task / "reference_a.parquet").is_file(),
            "reference_b1_attempted": (task / "reference_b.parquet").is_file(),
            "reference_b2_attempted": (task / "reference_b.parquet").is_file(),
            "b2_failure_count": int(summary.get("reference_b2_failure_count", 0)),
            "config_sha256": summary.get("config_sha256", config["config_sha256"]),
            "final_status": summary.get("status"),
        }

    expected_evaluator_tasks = len(EVALUATORS) * len(lambdas)
    expected_reference_tasks = expected_evaluator_tasks
    evaluator_complete = sum(row["state"] == "completed" for row in evaluator_tasks.values())
    reference_complete = sum(row["state"] == "completed" for row in reference_tasks.values())
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run.name,
        "config_sha256": config["config_sha256"],
        "prepared_config_sha256": prepared.get("config_sha256"),
        "input_sha256": prepared.get("input_sha256"),
        "jobs": job_ids,
        "slurm_records": _slurm_records(run),
        "evaluator_tasks": evaluator_tasks,
        "reference_tasks": reference_tasks,
        "completion": {
            "expected_evaluator_tasks": expected_evaluator_tasks,
            "completed_evaluator_tasks": evaluator_complete,
            "expected_reference_tasks": expected_reference_tasks,
            "completed_reference_tasks": reference_complete,
            "join_complete": (run / "evaluators" / "production_join_summary.json").is_file(),
            "reference_join_complete": (run / "references" / "reference_run_summary.json").is_file(),
            "aggregation_complete": (run / "report" / "aggregation_summary.json").is_file(),
            "final_validation_complete": (run / "validation" / "production_validation_report.json").is_file(),
        },
    }
    atomic_write_json(run / "production_status.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = build_status(config_path=args.config, run_dir=args.run_dir)
    print(json.dumps(result["completion"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
