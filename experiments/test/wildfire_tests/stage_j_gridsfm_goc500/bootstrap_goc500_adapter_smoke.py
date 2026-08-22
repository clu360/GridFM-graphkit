"""J2/J3 smoke: mutate official case500_goc raw JSON and run GridSFM inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WILDFIRE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(WILDFIRE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_TESTS_ROOT))

from stage_j_gridsfm_goc500.goc500_adapter import (
    build_goc500_identity,
    mutate_raw_case_for_candidate,
    require_valid_risk_ratings,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gridsfm-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--offline-branch-id", type=int, default=None)
    parser.add_argument("--uniform-alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    from gridsfm import load_model, predict

    gridsfm_root = Path(args.gridsfm_root).expanduser().resolve()
    model_root = gridsfm_root / "model"
    sample = model_root / "samples" / "case500_goc.pyg.json"
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else model_root / "checkpoints" / "gridsfm_open_v1.1.pt"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path.home() / ".gridfm_stage_j" / "cache" / "mutated_cases"
    output_dir.mkdir(parents=True, exist_ok=True)

    with sample.open("r", encoding="utf-8") as handle:
        raw_case = json.load(handle)

    identity = build_goc500_identity(raw_case)
    require_valid_risk_ratings(identity)
    offline_branch_id = args.offline_branch_id
    if offline_branch_id is None:
        offline_branch_id = next(branch.canonical_branch_id for branch in identity.branches if branch.edge_family == "ac_line")

    alpha_requested = {load.canonical_load_id: args.uniform_alpha for load in identity.loads}
    mutated, breakdown, integrity = mutate_raw_case_for_candidate(
        raw_case,
        identity,
        offline_branch_ids=[offline_branch_id],
        alpha_requested=alpha_requested,
    )
    integrity.require_ok()

    mutated_path = output_dir / f"case500_goc_stagej_n1_{offline_branch_id}_alpha_{args.uniform_alpha:g}.pyg.json"
    with mutated_path.open("w", encoding="utf-8") as handle:
        json.dump(mutated, handle)

    model = load_model(str(checkpoint), device="cpu")
    out = predict(model, str(mutated_path))

    report = {
        "status": "PASS",
        "raw_mutation_before_official_preprocessing": True,
        "source_sample": str(sample),
        "mutated_path": str(mutated_path),
        "offline_branch_id": offline_branch_id,
        "num_branches_before": len(identity.branches),
        "num_ac_lines_after": len(mutated["grid"]["edges"]["ac_line"]["senders"]),
        "num_transformers_after": len(mutated["grid"]["edges"]["transformer"]["senders"]),
        "num_source_less_loads": len(breakdown.source_less_load_ids),
        "l_shed_total": breakdown.l_shed_total,
        "l_shed_control": breakdown.l_shed_control,
        "l_shed_island": breakdown.l_shed_island,
        "d_input": integrity.d_input,
        "feas": float(out["feas"]),
        "V_shape": tuple(out["V"].shape),
        "Pg_shape": tuple(out["Pg"].shape),
        "Pij_shape": tuple(out["Pij"].shape),
        "flow_edge_types": out["flow_edge_types"],
        "flow_edge_counts": out["flow_edge_counts"],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
