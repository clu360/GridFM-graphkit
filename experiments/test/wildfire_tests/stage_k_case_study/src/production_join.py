"""Validate and join evaluator/lambda production chunks, then seal finalists."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .io_utils import atomic_write_json


EVALUATORS = ("gridsfm", "dc", "ac")


def lambda_slug(value: float) -> str:
    return f"lambda_{float(value):.6g}"


def _input_identity(manifest: dict[str, object]) -> str:
    return sha256(json.dumps(manifest["input_sha256"], sort_keys=True).encode("utf-8")).hexdigest()


def _validate_chunk(
    *, evaluator: str, lambda_r: float, chunk: Path, config: dict[str, object],
    prepared: Path, expected_input_sha: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    summary_path = chunk / f"{evaluator}_run_summary.json"
    result_path = chunk / f"{evaluator}_candidate_results.parquet"
    k2_path = chunk / f"{evaluator}_k2_candidates.parquet"
    finalist_path = chunk / f"{evaluator}_finalists.parquet"
    for path in (summary_path, result_path, k2_path, finalist_path):
        if not path.is_file():
            raise FileNotFoundError(f"production chunk artifact missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_summary = {
        "status": "PASS",
        "evaluator": evaluator,
        "config_sha256": config["config_sha256"],
        "input_sha256": expected_input_sha,
        "lambda_values": [float(lambda_r)],
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"chunk summary mismatch for {evaluator}/{lambda_r}: {key}")

    rows = pd.read_parquet(result_path)
    k2 = pd.read_parquet(k2_path)
    finalists = pd.read_parquet(finalist_path)
    if len(finalists) != 1:
        raise ValueError(f"expected one sealed chunk finalist for {evaluator}/{lambda_r}")
    if len(rows) != int(summary["attempted_states"]):
        raise ValueError(f"candidate accounting mismatch for {evaluator}/{lambda_r}")
    if rows.duplicated(["lambda_r", "topology_key"]).any():
        raise ValueError(f"duplicate candidate topology for {evaluator}/{lambda_r}")
    if set(rows["evaluator"]) != {evaluator} or set(rows["lambda_r"].astype(float)) != {float(lambda_r)}:
        raise ValueError(f"chunk identity mismatch for {evaluator}/{lambda_r}")
    counts = rows.groupby("k").size().to_dict()
    expected_k1 = int(config["search"]["k1_count"])
    expected_k2 = int(config["search"]["k2_max_unique"])
    if counts.get(0) != 1 or counts.get(1) != expected_k1 or counts.get(2) != expected_k2:
        raise ValueError(f"unexpected K accounting for {evaluator}/{lambda_r}: {counts}")
    expected_k1_keys = set(
        pd.read_parquet(prepared / "shared_k1.parquet")
        .loc[lambda frame: frame["lambda_r"].astype(float).eq(float(lambda_r)), "topology_key"]
        .astype(str)
    )
    observed_k1_keys = set(rows.loc[rows["k"].eq(1), "topology_key"].astype(str))
    if observed_k1_keys != expected_k1_keys:
        raise ValueError(f"shared K1 identity mismatch for {evaluator}/{lambda_r}")
    if len(k2) != expected_k2 or set(k2["topology_key"].astype(str)) != set(
        rows.loc[rows["k"].eq(2), "topology_key"].astype(str)
    ):
        raise ValueError(f"K2 reconstruction mismatch for {evaluator}/{lambda_r}")

    eligible = rows.loc[rows["eligible"] & rows["search_objective"].notna()].copy()
    if eligible.empty:
        raise ValueError(f"no eligible finalist for {evaluator}/{lambda_r}")
    expected_finalist = eligible.sort_values(["search_objective", "k", "topology_key"]).iloc[0]
    if str(finalists.iloc[0]["topology_key"]) != str(expected_finalist["topology_key"]):
        raise ValueError(f"chunk finalist mismatch for {evaluator}/{lambda_r}")

    if evaluator == "gridsfm":
        required = ["j_trade", "pac_operational", "pac_ac", "pac_total", "j_total"]
        if rows[required].isna().any().any():
            raise ValueError(f"GridSFM objective component missing for lambda {lambda_r}")
        recomputed = rows["j_trade"].astype(float) + float(config["objective"]["rho_phys"]) * rows["pac_total"].astype(float)
        if not np.allclose(rows["j_total"], recomputed, rtol=0.0, atol=1e-10):
            raise ValueError(f"GridSFM J_total formula mismatch for lambda {lambda_r}")
        if not np.allclose(rows["search_objective"], rows["j_total"], rtol=0.0, atol=1e-12):
            raise ValueError(f"GridSFM ranking did not use J_total for lambda {lambda_r}")
        selected_parents = set(
            rows.loc[rows["k"].eq(1) & rows["eligible"]]
            .sort_values(["search_objective", "topology_key"])
            .head(int(config["search"]["parent_count"]))["topology_key"].astype(str)
        )
        if set(k2["parent_topology_key"].astype(str)) != selected_parents:
            raise ValueError(f"GridSFM K2 parents were not selected by J_total for lambda {lambda_r}")

    alpha_path = chunk / "gridsfm_alpha_evaluations.parquet"
    alpha = pd.read_parquet(alpha_path) if evaluator == "gridsfm" and alpha_path.is_file() else None
    return rows, k2, alpha


def join_chunks(
    *, config_path: str | Path, prepared_dir: str | Path, chunk_root: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    config = load_config(config_path)
    prepared = Path(prepared_dir)
    chunks = Path(chunk_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((prepared / "input_manifest.json").read_text(encoding="utf-8"))
    if manifest["config_sha256"] != config["config_sha256"]:
        raise ValueError("prepared/config hash mismatch")
    input_sha = _input_identity(manifest)
    joined_summary: dict[str, object] = {"status": "PASS", "chunks": {}, "config_sha256": config["config_sha256"], "input_sha256": input_sha}

    for evaluator in EVALUATORS:
        frames: list[pd.DataFrame] = []
        k2_frames: list[pd.DataFrame] = []
        alpha_frames: list[pd.DataFrame] = []
        for lambda_r in map(float, config["search"]["lambda_r"]):
            chunk = chunks / evaluator / lambda_slug(lambda_r)
            rows, k2, alpha = _validate_chunk(
                evaluator=evaluator, lambda_r=lambda_r, chunk=chunk, config=config,
                prepared=prepared, expected_input_sha=input_sha,
            )
            frames.append(rows)
            k2_frames.append(k2)
            if alpha is not None:
                alpha_frames.append(alpha)
            joined_summary["chunks"][f"{evaluator}:{lambda_r:.6g}"] = {
                "status": "PASS", "attempted": len(rows), "eligible": int(rows["eligible"].sum()),
                "failed": int((~rows["eligible"]).sum()), "k0": int(rows["k"].eq(0).sum()),
                "k1": int(rows["k"].eq(1).sum()), "k2": int(rows["k"].eq(2).sum()),
            }
        joined = pd.concat(frames, ignore_index=True)
        if joined.duplicated(["lambda_r", "topology_key"]).any():
            raise ValueError(f"duplicate primary key after joining {evaluator}")
        joined.to_parquet(output / f"{evaluator}_candidate_results.parquet", index=False)
        pd.concat(k2_frames, ignore_index=True).to_parquet(output / f"{evaluator}_k2_candidates.parquet", index=False)
        if alpha_frames:
            pd.concat(alpha_frames, ignore_index=True).to_parquet(output / "gridsfm_alpha_evaluations.parquet", index=False)
        finalists = (
            joined.loc[joined["eligible"] & joined["search_objective"].notna()]
            .sort_values(["lambda_r", "search_objective", "k", "topology_key"])
            .groupby("lambda_r", as_index=False, sort=True).head(1)
        )
        if len(finalists) != len(config["search"]["lambda_r"]):
            raise ValueError(f"finalist sealing incomplete for {evaluator}")
        finalists.to_parquet(output / f"{evaluator}_finalists.parquet", index=False)
        atomic_write_json(output / f"{evaluator}_run_summary.json", {
            "status": "PASS", "evaluator": evaluator, "attempted_states": len(joined),
            "eligible_states": int(joined["eligible"].sum()), "finalist_count": len(finalists),
            "config_sha256": config["config_sha256"], "input_sha256": input_sha,
            "lambda_values": [float(value) for value in config["search"]["lambda_r"]],
        })
    atomic_write_json(output / "production_join_summary.json", joined_summary)
    return joined_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--chunk-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = join_chunks(
        config_path=args.config, prepared_dir=args.prepared_dir,
        chunk_root=args.chunk_root, output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
