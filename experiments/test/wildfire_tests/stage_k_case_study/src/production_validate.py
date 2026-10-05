"""Final scientific-accounting validation for a Stage K production package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .io_utils import atomic_write_json
from .production_join import EVALUATORS
from .schemas import ELIGIBLE_STATUSES, classify_solver_status


def validate_production(*, config_path: str | Path, run_dir: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    run = Path(run_dir)
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    manifest = json.loads((run / "prepared" / "input_manifest.json").read_text(encoding="utf-8"))
    checks["prepared_config_hash"] = manifest.get("config_sha256") == config["config_sha256"]
    checks["stored_baseline_authoritative"] = (
        manifest.get("baseline_authority") == "stored_scenario16_branch_loading"
        and manifest.get("baseline_replaced_by_new_ac_solve") is False
    )
    checks["r_base_frozen"] = abs(float(manifest.get("r_base", 0.0)) - 32.975425699285616) <= 1e-10
    checks["input_hashes_complete"] = set(manifest.get("input_sha256", {})) == {
        "case", "loads", "coordinates", "environment", "config"
    }
    expected_per_lambda = 1 + int(config["search"]["k1_count"]) + int(config["search"]["k2_max_unique"])
    expected_total = expected_per_lambda * len(config["search"]["lambda_r"])

    for evaluator in EVALUATORS:
        candidates = pd.read_parquet(run / "evaluators" / f"{evaluator}_candidate_results.parquet")
        finalists = pd.read_parquet(run / "evaluators" / f"{evaluator}_finalists.parquet")
        checks[f"{evaluator}_candidate_total"] = len(candidates) == expected_total
        checks[f"{evaluator}_no_duplicate_keys"] = not candidates.duplicated(["lambda_r", "topology_key"]).any()
        checks[f"{evaluator}_five_finalists"] = len(finalists) == len(config["search"]["lambda_r"])
        for lambda_r in map(float, config["search"]["lambda_r"]):
            subset = candidates.loc[candidates["lambda_r"].astype(float).eq(lambda_r)]
            counts = subset.groupby("k").size().to_dict()
            checks[f"{evaluator}_{lambda_r:.6g}_accounting"] = counts == {
                0: 1, 1: int(config["search"]["k1_count"]), 2: int(config["search"]["k2_max_unique"])
            }
        if evaluator == "gridsfm":
            recomputed = candidates["j_trade"].astype(float) + float(config["objective"]["rho_phys"]) * candidates["pac_total"].astype(float)
            checks["gridsfm_j_total_formula"] = bool(np.allclose(candidates["j_total"], recomputed, atol=1e-10, rtol=0.0))
            checks["gridsfm_search_uses_j_total"] = bool(np.allclose(candidates["search_objective"], candidates["j_total"], atol=1e-12, rtol=0.0))

    reference_a = pd.read_parquet(run / "references" / "reference_a.parquet")
    reference_b = pd.read_parquet(run / "references" / "reference_b.parquet")
    discrepancies = pd.read_parquet(run / "references" / "reference_discrepancies.parquet")
    expected_refs = len(EVALUATORS) * len(config["search"]["lambda_r"])
    checks["reference_a_complete"] = len(reference_a) == expected_refs and not reference_a.duplicated(["evaluator", "lambda_r"]).any()
    checks["reference_b_complete"] = len(reference_b) == expected_refs and not reference_b.duplicated(["evaluator", "lambda_r"]).any()
    checks["reference_discrepancies_complete"] = len(discrepancies) == expected_refs
    checks["reference_a_status_explicit"] = reference_a["status"].notna().all()
    checks["reference_b_status_explicit"] = reference_b[["b1_status", "b2_status"]].notna().all().all()
    b1_failures = [value for value in reference_b["b1_status"].astype(str) if classify_solver_status(value) not in ELIGIBLE_STATUSES]
    b2_failures = [value for value in reference_b["b2_status"].astype(str) if classify_solver_status(value) not in ELIGIBLE_STATUSES]
    checks["reference_b1_all_eligible"] = not b1_failures
    b2_valid = reference_b["b2_solver_eligible"].astype(bool)
    checks["reference_b2_policy"] = bool(
        reference_b.loc[b2_valid, "diagnostic_state_source"].eq("b2_economic_tiebreak").all()
        and reference_b.loc[b2_valid, "economic_cost_tiebreak"].notna().all()
        and reference_b.loc[~b2_valid, "diagnostic_state_source"].eq("b1_maximum_service_fallback").all()
        and reference_b.loc[~b2_valid, "economic_cost_tiebreak"].isna().all()
    )
    if b2_failures:
        warnings.append(f"{len(b2_failures)} Reference B2 tie-break solve(s) were not solver-eligible")
    checks["aggregation_complete"] = (run / "report" / "aggregation_summary.json").is_file()
    checks["environment_manifest_complete"] = (run / "preflight" / "gpu" / "gridsfm_environment_manifest.json").is_file()
    checks["production_status_present"] = (run / "production_status.json").is_file()
    # Pandas reductions return numpy.bool_, which json.dumps cannot serialize.
    checks = {key: bool(value) for key, value in checks.items()}
    failed = sorted(key for key, passed in checks.items() if not passed)
    status = "FAIL" if failed else "PASS_WITH_WARNINGS" if warnings else "PASS"
    result = {"status": status, "checks": checks, "failed_checks": failed, "warnings": warnings}
    atomic_write_json(run / "validation" / "production_validation_report.json", result)
    if status in {"PASS", "PASS_WITH_WARNINGS"}:
        atomic_write_json(run / "RUN_COMPLETE.json", {
            "status": "complete", "validation_status": status,
            "note": "Gate 5 production package is scientifically accounted and validated.",
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = validate_production(config_path=args.config, run_dir=args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
