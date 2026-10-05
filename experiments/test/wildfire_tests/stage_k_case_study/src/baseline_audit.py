"""Authoritative stored-baseline versus newly solved intact AC consistency audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .config import load_config
from .identity import build_identity
from .io_utils import atomic_write_json


ACCEPTED_AC_TERMINATION_STATUSES = {
    "OPTIMAL",
    "LOCALLY_SOLVED",
    "ALMOST_OPTIMAL",
    "ALMOST_LOCALLY_SOLVED",
}


def deployment_audit_checks(
    termination_status: str,
    branch_state: pd.DataFrame,
    expected_powermodels_branch_ids: set[int],
) -> dict[str, bool]:
    observed_ids = pd.to_numeric(
        branch_state.get("powermodels_branch_id", pd.Series(dtype=float)), errors="coerce"
    )
    observed_loading = pd.to_numeric(
        branch_state.get("physical_loading", pd.Series(dtype=float)), errors="coerce"
    )
    observed_id_set = set(observed_ids.dropna().astype(int))
    return {
        "solver_status_ok": str(termination_status).strip().upper()
        in ACCEPTED_AC_TERMINATION_STATUSES,
        "branch_output_complete": (
            len(branch_state) == len(expected_powermodels_branch_ids)
            and not observed_ids.isna().any()
            and observed_ids.is_unique
            and observed_id_set == expected_powermodels_branch_ids
        ),
        "branch_output_finite": (
            len(observed_loading) == len(expected_powermodels_branch_ids)
            and bool(np.isfinite(observed_loading.to_numpy(dtype=float)).all())
        ),
    }


def run_audit(config_path: str | Path, output_dir: str | Path, *, julia: str = "julia") -> dict[str, object]:
    config = load_config(config_path)
    identity = build_identity()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parent / "native_opf"
    p_env_csv = output / "p_env_powermodels.csv"
    with p_env_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["powermodels_branch_id", "p_env"])
        for row in identity.branches.loc[identity.branches["is_switchable_transmission"]].itertuples(index=False):
            writer.writerow([int(row.powermodels_branch_id), float(row.p_env)])
    command = [
        julia, f"--project={project}", str(project / "stage_k_native_opf.jl"),
        "baseline_audit", str(Path(__file__).resolve().parents[1] / config["paths"]["case_file"]),
        "", "", str(p_env_csv), "-", "0.8", "1.0", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    summary_path = output / "baseline_audit_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"intact AC audit failed with exit code {completed.returncode}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    state = pd.read_csv(summary["branch_state_csv"])
    expected_pm_ids = set(identity.branches["powermodels_branch_id"].astype(int))
    checks = deployment_audit_checks(summary["termination_status"], state, expected_pm_ids)
    observed = state.assign(canonical_branch_id=state["powermodels_branch_id"].astype(int) - 1)[
        ["canonical_branch_id", "physical_loading"]
    ]
    authoritative = identity.branches[["canonical_branch_id", "baseline_loading", "p_env"]]
    comparison = authoritative.merge(observed, on="canonical_branch_id", how="left", validate="one_to_one")
    comparison["loading_error"] = comparison["physical_loading"] - comparison["baseline_loading"]
    comparison["absolute_loading_error"] = comparison["loading_error"].abs()
    comparison["weighted_risk_difference"] = comparison["p_env"] * (
        comparison["physical_loading"] ** 2 - comparison["baseline_loading"] ** 2
    )
    comparison.to_parquet(output / "baseline_consistency_by_branch.parquet", index=False)
    tolerance = float(config["solvers"]["baseline_audit_loading_atol"])
    max_error = float(comparison["absolute_loading_error"].max())
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "status_basis": "accepted_solver_status_and_complete_finite_intact_branch_output",
        "baseline_authority": "stored_scenario16_branch_loading",
        "baseline_replaced": False,
        "termination_status": summary["termination_status"],
        **checks,
        "expected_branch_count": len(expected_pm_ids),
        "observed_branch_count": len(state),
        "loading_comparison_role": "diagnostic_only",
        "loading_comparison_gate": False,
        "loading_atol": tolerance,
        "loading_within_atol": bool(np.isfinite(max_error) and max_error <= tolerance),
        "maximum_absolute_loading_error": max_error,
        "weighted_risk_difference": float(comparison["weighted_risk_difference"].sum()),
        "largest_discrepant_branch_ids": comparison.nlargest(10, "absolute_loading_error")["canonical_branch_id"].astype(int).tolist(),
    }
    atomic_write_json(output / "baseline_consistency_audit.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    report = run_audit(args.config, args.output_dir, julia=args.julia)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
