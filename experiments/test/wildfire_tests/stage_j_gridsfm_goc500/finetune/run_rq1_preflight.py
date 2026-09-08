"""Freeze the canonical decision bank for the RQ1 frozen-M0 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

WILDFIRE_ROOT = Path(__file__).resolve().parents[2]
if str(WILDFIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_ROOT))

from stage_j_gridsfm_goc500.goc500_adapter import (
    build_goc500_identity,
    source_less_load_ids,
)
from stage_j_gridsfm_goc500.load_service import compute_alpha_effective
from stage_j_gridsfm_goc500.finetune.rq1_common import (
    FAMILY_SPECS,
    M0_SHA256,
    alpha_effective_path,
    canonical_decision_components,
    canonical_selected_alpha,
    decision_hashes,
    finalist_entry,
    line_ids,
    load_config,
    package_roots,
    read_alpha_effective,
    read_json,
    resolve_repo_path,
    settings,
    sha256_file,
    sha256_value,
    write_csv,
    write_json,
)


def _write_alpha(path: Path, values: dict[int, float]) -> None:
    write_csv(path, [{"load_id": key, "alpha": values[key]} for key in sorted(values)])


def _provenance_lookup(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["warm_start_type"].eq("cold_start")].copy()
    if len(frame) != 75:
        raise RuntimeError(f"expected 75 cold provenance rows, observed {len(frame)}")
    return {
        (str(row["comparison_variant"]), str(row["setting_code"])): row.to_dict()
        for _, row in frame.iterrows()
    }


def run(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    output = Path(config["working_root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_case_path = Path(config["raw_case_path"]).resolve()
    raw_case = read_json(raw_case_path)
    identity = build_goc500_identity(raw_case)
    branch_ids = [row.canonical_branch_id for row in identity.branches]
    load_ids = [row.canonical_load_id for row in identity.loads]
    pd_pre = {row.canonical_load_id: row.pd_pre for row in identity.loads}
    qd_pre = {row.canonical_load_id: row.qd_pre for row in identity.loads}

    provenance_path = resolve_repo_path(config["ft7_warm_start_parquet"])
    published = _provenance_lookup(provenance_path)
    roots = package_roots(config)
    rows: list[dict[str, object]] = []
    unique_payloads: dict[str, dict[str, object]] = {}

    for family_id, spec in FAMILY_SPECS.items():
        for setting in settings(roots[spec["package"]]):
            entry = finalist_entry(setting, family_id)
            best = entry["best_topology"]
            offline = line_ids(best.get("topology_id"))
            source_less = source_less_load_ids(identity, offline)
            selected = json.loads(canonical_selected_alpha(best.get("best_alpha_selected")))
            alpha_requested = {load_id: float(selected.get(str(load_id), 1.0)) for load_id in load_ids}
            recomputed = compute_alpha_effective(load_ids, alpha_requested, source_less)
            alpha_path = alpha_effective_path(setting, family_id)
            saved = read_alpha_effective(alpha_path)
            if set(saved) != set(load_ids):
                raise RuntimeError(f"incomplete alpha vector: {alpha_path}")
            max_alpha_error = max(abs(saved[key] - recomputed[key]) for key in load_ids)
            if max_alpha_error > 1e-12:
                raise RuntimeError(
                    f"saved/recomputed alpha-effective mismatch {max_alpha_error}: {alpha_path}"
                )
            pd = {load_id: saved[load_id] * pd_pre[load_id] for load_id in load_ids}
            qd = {load_id: saved[load_id] * qd_pre[load_id] for load_id in load_ids}
            components = canonical_decision_components(
                branch_ids=branch_ids,
                offline_branch_ids=offline,
                load_ids=load_ids,
                alpha_effective=saved,
                pd_by_load=pd,
                qd_by_load=qd,
            )
            hashes = decision_hashes(components)
            pub = published[(family_id, setting.name)]
            if str(pub.get("finalist_topology_id") or "") != str(best.get("topology_id") or ""):
                raise RuntimeError(f"published topology mismatch: {family_id}/{setting.name}")
            # The FT7 table used a direct SHA over canonical JSON text.
            import hashlib

            selected_text_hash = hashlib.sha256(
                canonical_selected_alpha(best.get("best_alpha_selected")).encode("utf-8")
            ).hexdigest().upper()
            if selected_text_hash != str(pub["finalist_alpha_sha256"]):
                raise RuntimeError(f"published selected-alpha mismatch: {family_id}/{setting.name}")

            provenance_id = f"{family_id}:{setting.name}"
            row = {
                "provenance_id": provenance_id,
                "family_id": family_id,
                "finalist_family": spec["label"],
                "setting_code": setting.name,
                "scenario_id": best["scenario_id"],
                "lambda_r": float(best["lambda_r"]),
                "topology_id": str(best.get("topology_id") or ""),
                "offline_branch_ids": ";".join(str(value) for value in offline),
                "num_shutoffs": len(offline),
                "source_less_load_ids": ";".join(str(value) for value in source_less),
                "alpha_effective_source": str(alpha_path),
                "selected_alpha_sha256": selected_text_hash,
                "max_saved_recomputed_alpha_error": max_alpha_error,
                **hashes,
            }
            rows.append(row)
            unique_payloads.setdefault(
                hashes["decision_sha256"],
                {
                    "decision_sha256": hashes["decision_sha256"],
                    "component_hashes": hashes,
                    "offline_branch_ids": offline,
                    "source_less_load_ids": list(source_less),
                    "alpha_effective": {str(key): saved[key] for key in sorted(saved)},
                    "pd": {str(key): pd[key] for key in sorted(pd)},
                    "qd": {str(key): qd[key] for key in sorted(qd)},
                    "provenance_ids": [],
                },
            )["provenance_ids"].append(provenance_id)

    if len(rows) != int(config["expected_provenance_rows"]):
        raise RuntimeError(f"expected 75 provenance rows, observed {len(rows)}")
    if len(unique_payloads) != int(config["expected_unique_decisions"]):
        raise RuntimeError(f"expected 54 unique decisions, observed {len(unique_payloads)}")

    unique_rows: list[dict[str, object]] = []
    unique_dir = output / "inputs" / "unique"
    for index, decision_sha in enumerate(sorted(unique_payloads), start=1):
        payload = unique_payloads[decision_sha]
        unique_id = f"RQ1U{index:03d}"
        payload["unique_id"] = unique_id
        alpha_path = unique_dir / unique_id / "alpha_effective.csv"
        _write_alpha(alpha_path, {int(k): float(v) for k, v in payload["alpha_effective"].items()})
        payload["alpha_csv"] = str(alpha_path)
        for row in rows:
            if row["decision_sha256"] == decision_sha:
                row["unique_id"] = unique_id
        unique_rows.append(
            {
                "unique_id": unique_id,
                "decision_sha256": decision_sha,
                "z_sha256": payload["component_hashes"]["z_sha256"],
                "alpha_effective_sha256": payload["component_hashes"]["alpha_effective_sha256"],
                "pd_sha256": payload["component_hashes"]["pd_sha256"],
                "qd_sha256": payload["component_hashes"]["qd_sha256"],
                "offline_branch_ids": ";".join(str(value) for value in payload["offline_branch_ids"]),
                "source_less_load_ids": ";".join(str(value) for value in payload["source_less_load_ids"]),
                "provenance_count": len(payload["provenance_ids"]),
                "provenance_ids": ";".join(payload["provenance_ids"]),
                "alpha_csv": str(alpha_path),
            }
        )

    write_csv(output / "rq1_fixed_provenance.csv", rows)
    write_csv(output / "rq1_unique_instances.csv", unique_rows)
    write_json(output / "rq1_unique_instances.json", {"instances": list(unique_payloads.values())})

    checkpoint = Path(config["m0_checkpoint"]).resolve()
    gates = {
        "provenance_rows_75": len(rows) == 75,
        "unique_decisions_54": len(unique_payloads) == 54,
        "all_component_hashes_present": all(
            all(row.get(name) for name in ("z_sha256", "alpha_effective_sha256", "pd_sha256", "qd_sha256"))
            for row in rows
        ),
        "saved_alpha_matches_recomputed": all(
            float(row["max_saved_recomputed_alpha_error"]) <= 1e-12 for row in rows
        ),
        "m0_checkpoint_hash": sha256_file(checkpoint) == M0_SHA256,
        "raw_case_hash": sha256_file(raw_case_path) == str(config["expected_raw_case_sha256"]),
        "ft7_source_unchanged": sha256_file(provenance_path) == str(config["expected_ft7_warm_start_sha256"]),
    }
    status = "RQ1_P0_PREFLIGHT_PASS" if all(gates.values()) else "RQ1_P0_PREFLIGHT_FAIL"
    result = {
        "status": status,
        "gates": gates,
        "provenance_rows": len(rows),
        "unique_decisions": len(unique_payloads),
        "duplicate_provenance_rows": len(rows) - len(unique_payloads),
        "max_provenance_multiplicity": max(len(value["provenance_ids"]) for value in unique_payloads.values()),
        "canonical_identity": "SHA256(z, alpha_effective, scaled_Pd, scaled_Qd) with exact float.hex encoding",
        "m0_checkpoint": str(checkpoint),
        "m0_checkpoint_sha256": sha256_file(checkpoint),
        "raw_case_path": str(raw_case_path),
        "raw_case_sha256": sha256_file(raw_case_path),
        "ft7_warm_start_parquet": str(provenance_path),
        "ft7_warm_start_sha256": sha256_file(provenance_path),
    }
    write_json(output / "RQ1_PREFLIGHT.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.config)
    return 0 if result["status"] == "RQ1_P0_PREFLIGHT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
