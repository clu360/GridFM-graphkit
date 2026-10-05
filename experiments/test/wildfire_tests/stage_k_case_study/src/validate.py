"""Fail-closed Stage K package and run validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .io_utils import atomic_write_json


def validate_prepared(config_path: str | Path, prepared_dir: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    prepared = Path(prepared_dir)
    manifest = json.loads((prepared / "input_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "manifest_pass": manifest.get("status") == "PASS",
        "config_hash_match": manifest.get("config_sha256") == config["config_sha256"],
        "bus_count": len(pd.read_parquet(prepared / "canonical_bus.parquet")) == 2751,
        "scenario16_load_count": len(pd.read_parquet(prepared / "canonical_load.parquet")) == 1125,
        "physical_branch_count": len(pd.read_parquet(prepared / "canonical_branch.parquet")) == 5344,
        "l_trans_count": manifest["counts"]["l_trans"] == 3993,
        "fixed_transformer_count": manifest["counts"]["transformer_or_other_fixed"] == 1351,
        "stored_baseline_authoritative": manifest["baseline_authority"] == "stored_scenario16_branch_loading",
        "baseline_not_replaced": manifest["baseline_replaced_by_new_ac_solve"] is False,
        "r_base_positive": float(manifest["r_base"]) > 0.0,
        "k1_count_per_lambda": bool(
            (pd.read_parquet(prepared / "shared_k1.parquet").groupby("lambda_r").size() == int(config["search"]["k1_count"])).all()
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {"status": status, "checks": checks, "gate3_required": [
        "intact PowerModels AC baseline consistency audit",
        "Julia/PowerModels/Ipopt native evaluator probe",
        "released GridSFM checkpoint inference on the approved CUDA device",
        "live Phoenix GNR/GPU selector and account validation",
    ]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = validate_prepared(args.config, args.prepared_dir)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
