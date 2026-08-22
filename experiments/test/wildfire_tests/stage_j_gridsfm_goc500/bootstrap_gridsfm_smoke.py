"""Repeatable J0.5 GridSFM checkpoint/data/preprocessing/inference smoke.

Run this with the isolated GridSFM environment, for example:

    C:/Users/Caleb Lu/.gridfm_stage_j/envs/gridsfm/Scripts/python.exe \
      experiments/test/wildfire_tests/stage_j_gridsfm_goc500/bootstrap_gridsfm_smoke.py \
      --gridsfm-root "C:/Users/Caleb Lu/.gridfm_stage_j/repos/GridSFM"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gridsfm-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sample", default=None)
    args = parser.parse_args()

    from gridsfm import load_model, load_pyg_json, predict, prepare_for_inference
    import torch

    gridsfm_root = Path(args.gridsfm_root).expanduser().resolve()
    model_root = gridsfm_root / "model"
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else model_root / "checkpoints" / "gridsfm_open_v1.1.pt"
    sample = Path(args.sample).expanduser().resolve() if args.sample else model_root / "samples" / "case500_goc.pyg.json"

    data = load_pyg_json(sample)
    prepared = prepare_for_inference(data)
    model = load_model(str(checkpoint), device="cpu")
    out = predict(model, str(sample))

    report = {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "sample": str(sample),
        "sample_sha256": _sha256(sample),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "raw_bus_x_shape": tuple(load_pyg_json(sample)["bus"].x.shape),
        "prepared_bus_x_shape": tuple(prepared["bus"].x.shape),
        "prepared_node_types": list(prepared.node_types),
        "V_shape": tuple(out["V"].shape),
        "theta_shape": tuple(out["theta"].shape),
        "Pg_shape": tuple(out["Pg"].shape),
        "Qg_shape": tuple(out["Qg"].shape),
        "Pij_shape": tuple(out["Pij"].shape),
        "flow_edge_types": out["flow_edge_types"],
        "flow_edge_counts": out["flow_edge_counts"],
        "feas": float(out["feas"]),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
