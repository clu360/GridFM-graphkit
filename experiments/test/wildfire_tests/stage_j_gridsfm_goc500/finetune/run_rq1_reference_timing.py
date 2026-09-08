"""Run and validate persistent-process Reference A timing for RQ1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd


WILDFIRE_ROOT = Path(__file__).resolve().parents[2]
if str(WILDFIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_ROOT))

from stage_j_gridsfm_goc500.finetune.rq1_common import (  # noqa: E402
    SUCCESS,
    load_config,
    read_csv,
    read_json,
    resolve_repo_path,
    write_json,
)


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["unique_id", "decision_sha256", "offline_branch_ids", "alpha_csv"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _existing_objectives(config, unique_rows, provenance_rows) -> dict[str, float]:
    publication = pd.read_parquet(resolve_repo_path(config["ft7_warm_start_parquet"]))
    cold = publication.loc[publication["warm_start_type"].eq("cold_start")]
    objective_by_provenance = {
        f"{row['comparison_variant']}:{row['setting_code']}": float(row["objective"])
        for _, row in cold.iterrows()
    }
    provenance = {row["provenance_id"]: row for row in provenance_rows}
    output = {}
    for row in unique_rows:
        first = row["provenance_ids"].split(";")[0]
        if first not in provenance or first not in objective_by_provenance:
            raise RuntimeError(f"missing existing Reference A objective for {first}")
        output[row["unique_id"]] = objective_by_provenance[first]
    return output


def run(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    output = Path(config["working_root"]).resolve()
    preflight = read_json(output / "RQ1_PREFLIGHT.json")
    if preflight["status"] != "RQ1_P0_PREFLIGHT_PASS":
        raise RuntimeError("RQ1 preflight has not passed")
    unique_rows = read_csv(output / "rq1_unique_instances.csv")
    provenance_rows = read_csv(output / "rq1_fixed_provenance.csv")
    manifest = output / "reference_a" / "rq1_reference_manifest.tsv"
    _write_tsv(manifest, unique_rows)
    raw_output = output / "reference_a" / "rq1_reference_a_repetitions.csv"
    julia_status = output / "reference_a" / "RQ1_REFERENCE_A_JULIA_STATUS.json"
    script = Path(__file__).resolve().with_name("rq1_reference_a_timing.jl")
    command = [
        str(Path(config["julia_exe"]).resolve()),
        str(script),
        str(Path(config["powermodels_case_path"]).resolve()),
        str(manifest),
        str(raw_output),
        str(config["refa_warmup_evaluations"]),
        str(config["refa_repetitions"]),
        str(julia_status),
    ]
    env = dict(os.environ)
    env["JULIA_DEPOT_PATH"] = str(Path(config["julia_depot_path"]).resolve())
    started = time.time()
    process = subprocess.run(
        command, env=env, text=True, capture_output=True,
        timeout=int(config.get("reference_timeout_seconds", 7200)),
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"persistent Reference A runner failed ({process.returncode}):\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    rows = read_csv(raw_output)
    expected = int(config["expected_unique_decisions"]) * int(config["refa_repetitions"])
    existing = _existing_objectives(config, unique_rows, provenance_rows)
    by_unique: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_unique.setdefault(row["unique_id"], []).append(row)
    max_spread = max(
        max(float(row["objective"]) for row in group)
        - min(float(row["objective"]) for row in group)
        for group in by_unique.values()
    )
    max_existing_delta = max(
        abs(float(row["objective"]) - existing[row["unique_id"]]) for row in rows
    )
    gates = {
        "row_count": len(rows) == expected,
        "unique_coverage": set(by_unique) == {row["unique_id"] for row in unique_rows},
        "all_solved": all(row["status"] in SUCCESS for row in rows),
        "all_state_finite": all(row["state_finite"].lower() == "true" for row in rows),
        "all_alpha_complete": all(int(row["missing_alpha_load_count"]) == 0 for row in rows),
        "standard_reference_a_initialization": all(
            row["start_source"] == "explicit_generic_V1_theta0_Pg_midpoint_Qg0" for row in rows
        ),
        "positive_ipopt_time": all(
            math.isfinite(float(row["ipopt_solve_time_seconds"]))
            and float(row["ipopt_solve_time_seconds"]) > 0 for row in rows
        ),
        "positive_evaluator_time": all(
            math.isfinite(float(row["evaluator_total_seconds"]))
            and float(row["evaluator_total_seconds"]) > 0 for row in rows
        ),
        "objective_repeatability": max_spread <= 1e-3,
        "existing_reference_a_parity": max_existing_delta <= 1e-3,
        "persistent_process": bool(read_json(julia_status)["persistent_process"]),
        "base_case_parsed_once": bool(read_json(julia_status)["base_case_parsed_once"]),
    }
    status = "RQ1_P2_REFERENCE_A_TIMING_PASS" if all(gates.values()) else "RQ1_P2_REFERENCE_A_TIMING_FAIL"
    result = {
        "status": status,
        "gates": gates,
        "rows": len(rows),
        "unique_decisions": len(by_unique),
        "max_within_instance_objective_spread": max_spread,
        "max_existing_reference_a_objective_delta": max_existing_delta,
        "elapsed_seconds": time.time() - started,
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
        "primary_timing_boundary": "fixed candidate received to usable electrical state returned",
        "initialization": "existing Reference A default: V=1, theta=0, Pg midpoint, Qg=0",
        "reference_interpretation": "locally solved fixed-decision AC OPF under declared solver tolerances",
    }
    write_json(output / "reference_a" / "RQ1_REFERENCE_A_TIMING_VALIDATION.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.config)
    return 0 if result["status"] == "RQ1_P2_REFERENCE_A_TIMING_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
