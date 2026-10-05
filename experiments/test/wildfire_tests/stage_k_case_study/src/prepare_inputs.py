"""Prepare immutable Stage K canonical inputs and shared exact-K1 pools."""

from __future__ import annotations

import argparse
from inspect import getsource
import json
from pathlib import Path

import pandas as pd

from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.scenario_builder import (
    connectivity_service_impact_proxy,
)

from .config import load_config
from .identity import build_identity
from .io_utils import atomic_write_json, sha256_file
from .objectives import compute_r_base
from .topology_candidates import exact_k_candidates


def prepare(config_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = build_identity(scenario=config["snapshot"]["electrical_scenario"])
    branch = identity.branches
    l_trans = identity.l_trans
    baseline = dict(zip(branch["canonical_branch_id"].astype(int), branch["baseline_loading"].astype(float), strict=True))
    p_env = dict(zip(branch["canonical_branch_id"].astype(int), branch["p_env"].astype(float), strict=True))
    r_base = compute_r_base(l_trans, p_env, baseline)

    # Exact reuse of the Stage J source-less single-outage implementation.
    c_by_line = connectivity_service_impact_proxy(identity.stage_j_identity, l_trans)
    proxy = pd.DataFrame(
        {
            "canonical_branch_id": list(l_trans),
            "p_env": [p_env[i] for i in l_trans],
            "baseline_loading": [baseline[i] for i in l_trans],
            "weight": [p_env[i] * baseline[i] ** 2 for i in l_trans],
            "c_l": [c_by_line[i] for i in l_trans],
        }
    )
    weights = dict(zip(proxy["canonical_branch_id"], proxy["weight"], strict=True))

    identity.buses.to_parquet(output / "canonical_bus.parquet", index=False)
    identity.loads.to_parquet(output / "canonical_load.parquet", index=False)
    identity.generators.to_parquet(output / "canonical_generator.parquet", index=False)
    identity.branches.to_parquet(output / "canonical_branch.parquet", index=False)
    identity.branch_model_mapping.to_parquet(output / "branch_model_mapping.parquet", index=False)
    proxy.to_parquet(output / "proxy_components.parquet", index=False)

    pool_rows = []
    for lambda_r in config["search"]["lambda_r"]:
        pool = exact_k_candidates(
            line_ids=l_trans,
            weights=weights,
            c_by_line=c_by_line,
            lambda_r=float(lambda_r),
            k=1,
            count=int(config["search"]["k1_count"]),
        )
        for item in pool:
            row = item.as_dict()
            row["lambda_r"] = float(lambda_r)
            row["offline_branch_ids"] = ";".join(map(str, item.offline_branch_ids))
            pool_rows.append(row)
    pd.DataFrame(pool_rows).to_parquet(output / "shared_k1.parquet", index=False)

    package = Path(__file__).resolve().parents[1]
    input_files = {
        "case": package / config["paths"]["case_file"],
        "loads": package / config["paths"]["load_file"],
        "coordinates": package / config["paths"]["coordinate_file"],
        "environment": package / config["paths"]["environment_file"],
        "config": Path(config_path),
    }
    manifest = {
        "status": "PASS",
        "config_sha256": config["config_sha256"],
        "input_sha256": {name: sha256_file(path) for name, path in input_files.items()},
        "counts": {
            "buses": len(identity.buses),
            "loads_scenario16": len(identity.loads),
            "generators": len(identity.generators),
            "online_generators": int((identity.generators["GEN_STATUS"] > 0).sum()),
            "physical_branches": len(identity.branches),
            "l_trans": len(identity.l_trans),
            "transformer_or_other_fixed": len(identity.l_fixed),
        },
        "r_base": r_base,
        "baseline_authority": "stored_scenario16_branch_loading",
        "baseline_replaced_by_new_ac_solve": False,
        "c_l_implementation": (
            "experiments.test.wildfire_tests.stage_j_gridsfm_goc500.scenario_builder."
            "connectivity_service_impact_proxy"
        ),
        "c_l_source_sha256": __import__("hashlib").sha256(
            getsource(connectivity_service_impact_proxy).encode("utf-8")
        ).hexdigest(),
        "proxy_backend": "deterministic_exact_k_separable_ranking",
        "proxy_equivalence": (
            "J_proxy=lambda+sum_open((1-lambda)*c_l-lambda*w_l/R_base); "
            "exact-K ranking is additive and equivalent to repeated MILP no-good enumeration"
        ),
        "artifacts": sorted(path.name for path in output.iterdir()),
    }
    atomic_write_json(output / "input_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
