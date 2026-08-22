"""Prepare Stage J GOC-500 baseline/proxy/scenario input artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):  # pragma: no cover - direct script path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from stage_j_gridsfm_goc500.goc500_adapter import build_goc500_identity, require_valid_risk_ratings
    from stage_j_gridsfm_goc500.metrics import UnitCompatibilityReport, require_unit_compatibility
    from stage_j_gridsfm_goc500.scenario_builder import (
        connectivity_service_impact_proxy,
        construct_stage_j_s1_s3_scenarios,
        load_baseline_loading_csv,
        write_stage_j_scenario_artifacts,
    )
else:
    from .goc500_adapter import build_goc500_identity, require_valid_risk_ratings
    from .metrics import UnitCompatibilityReport, require_unit_compatibility
    from .scenario_builder import (
        connectivity_service_impact_proxy,
        construct_stage_j_s1_s3_scenarios,
        load_baseline_loading_csv,
        write_stage_j_scenario_artifacts,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _read_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def _load_baseline_rows(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            branch_id = int(row["branch_id"])
            rows[branch_id] = {
                "f_bus": float(row["f_bus"]),
                "t_bus": float(row["t_bus"]),
                "rate_a": float(row["rate_a"]),
                "baseline_loading": float(row["baseline_loading"]),
            }
    return rows


def _write_branch_identity(path: Path, identity) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "canonical_branch_id",
                "original_case_branch_id",
                "edge_family",
                "family_index",
                "from_bus_id",
                "to_bus_id",
                "from_bus_index",
                "to_bus_index",
                "orientation_sign",
                "rate_a",
                "is_risk",
                "is_candidate",
            ],
        )
        writer.writeheader()
        for branch in identity.branches:
            writer.writerow(
                {
                    "canonical_branch_id": branch.canonical_branch_id,
                    "original_case_branch_id": branch.original_case_branch_id,
                    "edge_family": branch.edge_family,
                    "family_index": branch.family_index,
                    "from_bus_id": branch.from_bus_id,
                    "to_bus_id": branch.to_bus_id,
                    "from_bus_index": branch.from_bus_index,
                    "to_bus_index": branch.to_bus_index,
                    "orientation_sign": branch.orientation_sign,
                    "rate_a": branch.rate_a,
                    "is_risk": branch.is_risk,
                    "is_candidate": branch.is_candidate,
                }
            )


def prepare_inputs(*, raw_case_path: Path, baseline_loading_csv: Path, out_dir: Path) -> dict[str, Any]:
    raw_case = _read_json(raw_case_path)
    identity = build_goc500_identity(raw_case)
    require_valid_risk_ratings(identity)
    candidate_ids = [branch.canonical_branch_id for branch in identity.branches if branch.is_risk]
    identity = build_goc500_identity(raw_case, candidate_branch_ids=candidate_ids)

    baseline_loading = load_baseline_loading_csv(baseline_loading_csv)
    baseline_rows = _load_baseline_rows(baseline_loading_csv)

    identity_ids = {branch.canonical_branch_id for branch in identity.branches}
    baseline_ids = set(baseline_rows)
    missing_in_baseline = sorted(identity_ids.difference(baseline_ids))
    missing_in_identity = sorted(baseline_ids.difference(identity_ids))
    if missing_in_baseline or missing_in_identity:
        raise ValueError(
            "branch identity and exact AC baseline branch IDs differ: "
            f"missing_in_baseline={missing_in_baseline[:10]}, missing_in_identity={missing_in_identity[:10]}"
        )

    rate_deltas = []
    endpoint_mismatches = []
    for branch in identity.branches:
        row = baseline_rows[branch.canonical_branch_id]
        rate_deltas.append(abs(float(branch.rate_a) - float(row["rate_a"])))
        if int(row["f_bus"]) != branch.from_bus_id or int(row["t_bus"]) != branch.to_bus_id:
            endpoint_mismatches.append(branch.canonical_branch_id)
    max_rate_delta = max(rate_deltas) if rate_deltas else 0.0

    unit_report = UnitCompatibilityReport(
        base_mva=100.0,
        flow_units="per_unit_on_pglib_baseMVA_inferred_from_official_GOC500_sample",
        rating_units="per_unit_on_pglib_baseMVA_from_PowerModels_parse_file",
        compatible=(max_rate_delta <= 1e-8 and not endpoint_mismatches),
        notes=(
            "GridSFM raw sample does not expose baseMVA directly; compatibility is inferred "
            "from exact branch ID/rateA/endpoint parity with pglib_opf_case500_goc parsed by PowerModels."
        ),
    )
    require_unit_compatibility(unit_report)

    c_by_line = connectivity_service_impact_proxy(identity, candidate_ids)
    scenarios, score_rows = construct_stage_j_s1_s3_scenarios(
        identity,
        baseline_loading=baseline_loading,
        c_by_line=c_by_line,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_paths = write_stage_j_scenario_artifacts(scenarios, score_rows, out_dir=out_dir)
    branch_identity_path = out_dir / "stage_j_branch_identity.csv"
    unit_report_path = out_dir / "stage_j_unit_compatibility_report.json"
    manifest_path = out_dir / "stage_j_input_manifest.json"
    _write_branch_identity(branch_identity_path, identity)

    unit_payload = {
        "base_mva": unit_report.base_mva,
        "flow_units": unit_report.flow_units,
        "rating_units": unit_report.rating_units,
        "compatible": unit_report.compatible,
        "notes": unit_report.notes,
        "max_rate_a_abs_delta": max_rate_delta,
        "endpoint_mismatch_branch_ids": endpoint_mismatches,
    }
    with unit_report_path.open("w") as fh:
        json.dump(unit_payload, fh, indent=2, sort_keys=True)

    artifact_paths = {
        **scenario_paths,
        "branch_identity": str(branch_identity_path),
        "unit_compatibility_report": str(unit_report_path),
    }
    manifest = {
        "status": "ok",
        "raw_case_path": str(raw_case_path),
        "raw_case_sha256": _sha256(raw_case_path),
        "baseline_loading_csv": str(baseline_loading_csv),
        "baseline_loading_sha256": _sha256(baseline_loading_csv),
        "output_dir": str(out_dir),
        "artifact_paths": artifact_paths,
        "branch_count": len(identity.branches),
        "risk_branch_count": sum(1 for branch in identity.branches if branch.is_risk),
        "candidate_branch_count": sum(1 for branch in identity.branches if branch.is_candidate),
        "load_count": len(identity.loads),
        "max_baseline_loading": max(float(value) for value in baseline_loading.values()),
        "num_baseline_loading_gt_1": sum(1 for value in baseline_loading.values() if float(value) > 1.0),
        "c_l_backend": "connectivity_source_less_single_outage",
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "target_branch_ids": {scenario.scenario_id: list(scenario.target_branch_ids) for scenario in scenarios},
        "unit_report": unit_payload,
    }
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-case", type=Path, required=True)
    parser.add_argument("--baseline-loading-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_inputs(
        raw_case_path=args.raw_case,
        baseline_loading_csv=args.baseline_loading_csv,
        out_dir=args.out_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
