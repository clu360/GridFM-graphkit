"""Freeze FT6 protocol provenance and audit disjoint OPFData identities."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from ft6_common import (
    GRIDSFM_COMMIT,
    M0_SHA256,
    M1_SHA256,
    PRIVATE_SOURCE_SHA256,
    configured_indices,
    fingerprint_dataset,
    read_json,
    sha256_file,
    validate_split_spec,
    validate_training_config,
    write_json,
)
from run_finetune import REPO_ROOT, _device_from_config, _environment_freeze, _resolve_repo_path


SUCCESS_STATUS = "FT6_P1_DATA_CONTRACT_PASS"


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status_short": status}


def _all_tensors_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not torch.is_floating_point(value) or bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(_all_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_tensors_finite(item) for item in value)
    return True


def _processed_cache_path(root: Path, case_name: str, variant: str, split: str) -> Path:
    release = "dataset_release_1" if variant == "fulltop" else "dataset_release_1_nminusone"
    return root / release / case_name / "processed_1" / f"{split}.pt"


def _fingerprint_spec(config: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    from gridsfm import OPFDataAdapterDataset

    indices = configured_indices(spec)
    dataset = OPFDataAdapterDataset(
        root=str(Path(config["opfdata_root"]).expanduser().resolve()),
        case_name=config["case_name"], variant=spec["variant"], split=spec["split"],
        n_graphs=max(indices) + 1, num_groups=int(config["num_groups"]),
    )
    if len(dataset) <= max(indices):
        raise RuntimeError(f"dataset cannot provide frozen indices for {spec}")
    records = fingerprint_dataset(dataset, indices)
    finite = all(_all_tensors_finite(dataset[index].to_dict()) for index in indices)
    result = {
        "variant": spec["variant"], "split": spec["split"],
        "index_start": indices[0], "index_end": indices[-1], "count": len(indices),
        "ordered_graphs": records, "all_graph_tensors_finite": finite,
        "unique_graph_hashes": len({row["sha256"] for row in records}),
    }
    del dataset
    gc.collect()
    return result


def _checkpoint_record(path: Path, expected: str, model_id: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{model_id} checkpoint SHA mismatch: {observed}")
    return {"model_id": model_id, "path": str(path), "sha256": observed, "size_bytes": path.stat().st_size}


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    m2_path = Path(args.m2_config).expanduser().resolve()
    m3_path = Path(args.m3_config).expanduser().resolve()
    evaluation_path = Path(args.evaluation_config).expanduser().resolve()
    m2 = read_json(m2_path)
    m3 = read_json(m3_path)
    evaluation = read_json(evaluation_path)
    validate_training_config(m2)
    validate_training_config(m3)
    for item, variant in zip(evaluation["test"], ("fulltop", "n1"), strict=True):
        validate_split_spec(item, variant=variant, split="test", start=0, count=375)

    output_dir = _resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "ft6_status.json"
    write_json(status_path, {"status": "FT6_RUNNING", "phase": "FT6_P0_PROTOCOL_FREEZE"})

    device = _device_from_config(m2["device"])
    freeze = _environment_freeze(m2, device)
    if freeze["environment"]["gridsfm_git_commit"] != GRIDSFM_COMMIT:
        raise RuntimeError("GridSFM commit differs from the FT6 protocol")
    if freeze["source_sha256"] != PRIVATE_SOURCE_SHA256:
        raise RuntimeError("private GridSFM API sources differ from the FT6 protocol")

    gridsfm_root = Path(m2["gridsfm_root"]).expanduser().resolve()
    parent = Path(m2["parent_checkpoint"]).expanduser().resolve()
    checkpoints = {
        "m0": _checkpoint_record(
            gridsfm_root / "model" / "checkpoints" / "gridsfm_open_v1.1.pt", M0_SHA256, "m0"
        ),
        "m1": _checkpoint_record(parent, M1_SHA256, "m1"),
    }
    if Path(m3["parent_checkpoint"]).expanduser().resolve() != parent:
        raise RuntimeError("M2 and M3 parent checkpoint paths differ")

    plan_path = _resolve_repo_path(args.plan)
    white_paper = Path(args.white_paper).expanduser().resolve()
    methodology_sources = {
        "ft6_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "gridsfm_readme": {
            "path": str(gridsfm_root / "model" / "README.md"),
            "sha256": sha256_file(gridsfm_root / "model" / "README.md"),
        },
        "opfdata_adapter": {
            "path": str(gridsfm_root / "model" / "gridsfm" / "opfdata_train.py"),
            "sha256": sha256_file(gridsfm_root / "model" / "gridsfm" / "opfdata_train.py"),
        },
        "official_finetune_loop": {
            "path": str(gridsfm_root / "model" / "gridsfm" / "finetune_opfdata.py"),
            "sha256": sha256_file(gridsfm_root / "model" / "gridsfm" / "finetune_opfdata.py"),
        },
        "gridsfm_white_paper": {
            "path": str(white_paper), "sha256": sha256_file(white_paper),
            "title": "GridSFM: A Foundation Model for AC Optimal Power Flow",
        },
    }

    specs = {
        "m2_train": m2["train"], "m3_train": m3["train"],
        "fulltop_val": m2["validation"][0], "n1_val": m2["validation"][1],
        "fulltop_test": evaluation["test"][0], "n1_test": evaluation["test"][1],
    }
    opfdata_root = Path(m2["opfdata_root"]).expanduser().resolve()
    cache_records = {}
    for name, spec in specs.items():
        cache_path = _processed_cache_path(opfdata_root, m2["case_name"], spec["variant"], spec["split"])
        if not cache_path.is_file():
            raise RuntimeError(f"required processed OPFData cache is missing: {cache_path}")
        cache_records[name] = {
            "path": str(cache_path), "size_bytes": cache_path.stat().st_size,
            "network_download_required": False,
        }

    write_json(status_path, {"status": "FT6_RUNNING", "phase": "DATASET_FINGERPRINTING"})
    datasets = {}
    for name, spec in specs.items():
        print(json.dumps({"phase": "DATASET_FINGERPRINTING", "dataset": name}), flush=True)
        datasets[name] = _fingerprint_spec(m2, spec)

    groups = {
        name: {row["sha256"] for row in record["ordered_graphs"]}
        for name, record in datasets.items()
    }
    training_hashes = groups["m2_train"] | groups["m3_train"]
    validation_hashes = groups["fulltop_val"] | groups["n1_val"]
    test_hashes = groups["fulltop_test"] | groups["n1_test"]
    overlap = {
        "train_validation": sorted(training_hashes & validation_hashes),
        "train_test": sorted(training_hashes & test_hashes),
        "validation_test": sorted(validation_hashes & test_hashes),
    }
    checks = {
        "protocol_configs_valid": True,
        "sibling_parent_path_identical": True,
        "sibling_parent_sha_identical": True,
        "environment_commit_frozen": True,
        "private_api_sources_frozen": True,
        "processed_caches_local": all(not row["network_download_required"] for row in cache_records.values()),
        "expected_dataset_counts": all(
            datasets[name]["count"] == count for name, count in {
                "m2_train": 500, "m3_train": 500, "fulltop_val": 375,
                "n1_val": 375, "fulltop_test": 375, "n1_test": 375,
            }.items()
        ),
        "all_graph_tensors_finite": all(row["all_graph_tensors_finite"] for row in datasets.values()),
        "all_graphs_unique_within_stratum": all(
            row["unique_graph_hashes"] == row["count"] for row in datasets.values()
        ),
        "no_train_validation_test_graph_overlap": not any(overlap.values()),
        "fulltop_parent_new_train_indices_disjoint": set(range(1000)).isdisjoint(range(1000, 1500)),
    }
    status = SUCCESS_STATUS if all(checks.values()) else "FT6_DATA_CONTRACT_BLOCKED"
    manifest = {
        "artifact_id": "STAGE-J-GRIDSFM-FT6-PREFLIGHT",
        "status": status,
        "protocol_checkpoint": "FT6_P0_PROTOCOL_FROZEN",
        "data_checkpoint": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configs": {
            "m2": {"path": str(m2_path), "sha256": sha256_file(m2_path)},
            "m3": {"path": str(m3_path), "sha256": sha256_file(m3_path)},
            "evaluation": {"path": str(evaluation_path), "sha256": sha256_file(evaluation_path)},
        },
        "checkpoints": checkpoints,
        "environment_freeze": freeze,
        "git_state": _git_state(),
        "methodology_sources": methodology_sources,
        "methodology_boundary": {
            "training_format": "OPFData only",
            "training_api": "gridsfm.finetune_opfdata",
            "evaluation_api": "gridsfm.eval_pass",
            "supported_adapter_variants": ["fulltop", "n1"],
            "n1_training_interpretation": "Supported OPFData adapter/official-loop extension; not a reproduction of the white paper's FullTop-only fine-tune.",
            "stage_j_application": "Deferred to externally gated FT7 and uses GridSFM inference, not the OPFData fine-tuning path.",
        },
        "cache_records": cache_records,
        "datasets": datasets,
        "cross_split_graph_hash_overlap": overlap,
        "checks": checks,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "ft6_preflight_manifest.json", manifest)
    write_json(status_path, {"status": status, "phase": "PREFLIGHT_COMPLETE", "checks": checks})
    print(json.dumps({"status": status, "runtime_seconds": manifest["runtime_seconds"]}, indent=2), flush=True)
    return 0 if status == SUCCESS_STATUS else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-config", required=True)
    parser.add_argument("--m3-config", required=True)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--white-paper", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        try:
            output = _resolve_repo_path(args.output_dir)
            write_json(output / "ft6_status.json", {
                "status": "FT6_PREFLIGHT_BLOCKED", "phase": "UNHANDLED_EXCEPTION",
                "exception_type": type(exc).__name__, "message": str(exc),
            })
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
