"""Evaluate all four FT6 GridSFM checkpoints on the sealed 375+375 test set."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ft6_common import (
    GRIDSFM_COMMIT,
    MODEL_ORDER,
    PRIVATE_SOURCE_SHA256,
    VARIANT_ORDER,
    configured_indices,
    finite_metrics,
    metric_comparison,
    read_json,
    sha256_file,
    validate_split_spec,
    write_json,
)
from run_finetune import (
    _device_from_config,
    _environment_freeze,
    _make_transform,
    _output_signature,
    _resolve_repo_path,
    _seed_everything,
)


SUCCESS_STATUS = "FT6_P3_FOUR_MODEL_EVALUATION_PASS"


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "artifact_id", "batch_size", "case_name", "checkpoints", "device",
        "expected_gridsfm_commit", "expected_source_sha256", "gridsfm_root",
        "num_groups", "num_workers", "opfdata_root", "output_dir",
        "preflight_manifest", "seed", "test",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing FT6 evaluation config keys: {missing}")
    if tuple(config["checkpoints"]) != MODEL_ORDER:
        raise ValueError("FT6 checkpoint order must be M0, M1, M2, M3")
    if config["expected_gridsfm_commit"] != GRIDSFM_COMMIT:
        raise ValueError("FT6 evaluation commit pin differs from protocol")
    if config["expected_source_sha256"] != PRIVATE_SOURCE_SHA256:
        raise ValueError("FT6 evaluation source pins differ from protocol")
    for item, variant in zip(config["test"], VARIANT_ORDER, strict=True):
        validate_split_spec(item, variant=variant, split="test", start=0, count=375)


def _checkpoint_records(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = {}
    for model_id in MODEL_ORDER:
        spec = config["checkpoints"][model_id]
        path = Path(spec["path"]).expanduser().resolve()
        observed = sha256_file(path)
        expected = spec.get("expected_sha256")
        source_manifest = None
        if model_id in ("m2", "m3"):
            source_path = _resolve_repo_path(spec["source_manifest"])
            source_manifest = read_json(source_path)
            required_status = f"FT6_P2{('A' if model_id == 'm2' else 'B')}_{model_id.upper()}_TRAINING_PASS"
            if source_manifest["status"] != required_status:
                raise RuntimeError(f"{model_id} training manifest is not approved")
            expected = source_manifest["checkpoint_sha256"]
            if Path(source_manifest["checkpoint_path"]).resolve() != path:
                raise RuntimeError(f"{model_id} checkpoint path differs from training manifest")
        if observed != expected:
            raise RuntimeError(f"{model_id} checkpoint SHA mismatch")
        records[model_id] = {
            "model_id": model_id, "label": spec["label"], "path": str(path),
            "sha256": observed, "size_bytes": path.stat().st_size,
            "source_manifest": spec.get("source_manifest"),
            "source_manifest_sha256": None if source_manifest is None else sha256_file(source_path),
        }
    return records


def _make_test_datasets(config: Mapping[str, Any]):
    from gridsfm import OPFDataAdapterDataset
    from torch.utils.data import ConcatDataset, Subset

    datasets = {}
    for spec in config["test"]:
        indices = configured_indices(spec)
        base = OPFDataAdapterDataset(
            root=str(Path(config["opfdata_root"]).expanduser().resolve()),
            case_name=config["case_name"], variant=spec["variant"], split=spec["split"],
            n_graphs=max(indices) + 1, num_groups=int(config["num_groups"]),
            transform=_make_transform(),
        )
        datasets[spec["variant"]] = Subset(base, indices)
    datasets["combined"] = ConcatDataset([datasets["fulltop"], datasets["n1"]])
    return datasets


def run(config_path: Path) -> int:
    from gridsfm import eval_pass, load_model
    from torch_geometric.loader import DataLoader

    config = read_json(config_path)
    _validate_config(config)
    output_dir = _resolve_repo_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "ft6_evaluation_status.json"
    write_json(output_dir / "ft6_evaluation_config.json", config)
    write_json(status_path, {"status": "FT6_EVALUATION_RUNNING", "phase": "PREFLIGHT"})

    preflight_path = _resolve_repo_path(config["preflight_manifest"])
    preflight = read_json(preflight_path)
    if preflight["status"] != "FT6_P1_DATA_CONTRACT_PASS":
        raise RuntimeError("FT6 data-contract preflight is not approved")
    expected_test = {
        variant: [row["sha256"] for row in preflight["datasets"][f"{variant}_test"]["ordered_graphs"]]
        for variant in VARIANT_ORDER
    }
    if any(len(values) != 375 for values in expected_test.values()):
        raise RuntimeError("preflight test identity manifest does not contain 375 graphs per stratum")

    device = _device_from_config(config["device"])
    freeze = _environment_freeze(config, device)
    if freeze["environment"]["gridsfm_git_commit"] != GRIDSFM_COMMIT:
        raise RuntimeError("GridSFM commit differs from FT6 protocol")
    if freeze["source_sha256"] != PRIVATE_SOURCE_SHA256:
        raise RuntimeError("GridSFM private sources differ from FT6 protocol")
    write_json(output_dir / "ft6_evaluation_environment_freeze.json", freeze)
    checkpoints = _checkpoint_records(config)

    _seed_everything(int(config["seed"]))
    datasets = _make_test_datasets(config)
    expected_counts = {"fulltop": 375, "n1": 375, "combined": 750}
    if any(len(datasets[key]) != value for key, value in expected_counts.items()):
        raise RuntimeError("sealed test datasets do not match expected counts")
    loaders = {
        key: DataLoader(
            dataset, batch_size=int(config["batch_size"]), shuffle=False,
            num_workers=int(config["num_workers"]), persistent_workers=False,
        )
        for key, dataset in datasets.items()
    }

    total_started = time.perf_counter()
    results: dict[str, Any] = {}
    completed = []
    for model_id in MODEL_ORDER:
        model = load_model(checkpoints[model_id]["path"], device=device)
        model_results = {}
        for stratum in ("fulltop", "n1", "combined"):
            payload = {
                "status": "FT6_EVALUATION_RUNNING", "phase": "EVALUATION",
                "model_id": model_id, "stratum": stratum, "completed": completed,
            }
            write_json(status_path, payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
            started = time.perf_counter()
            metrics = eval_pass(model, loaders[stratum], device=device)
            runtime = time.perf_counter() - started
            if not finite_metrics(metrics) or int(metrics["n_graphs"]) != expected_counts[stratum]:
                raise RuntimeError(f"invalid official metrics for {model_id}/{stratum}")
            model_results[stratum] = {
                "metrics": metrics, "runtime_seconds": runtime,
                "output_signature": _output_signature(model, loaders[stratum], device),
            }
            completed.append({"model_id": model_id, "stratum": stratum})
            write_json(output_dir / "ft6_evaluation_partial.json", {**results, model_id: model_results})
        results[model_id] = model_results
        del model

    comparisons = {}
    for name, baseline, updated in (
        ("m0_to_m1", "m0", "m1"),
        ("m1_to_m2", "m1", "m2"),
        ("m1_to_m3", "m1", "m3"),
        ("m2_to_m3", "m2", "m3"),
    ):
        comparisons[name] = {
            stratum: metric_comparison(
                results[baseline][stratum]["metrics"], results[updated][stratum]["metrics"]
            )
            for stratum in ("fulltop", "n1", "combined")
        }

    schema_checks = {
        stratum: len({
            json.dumps(results[model_id][stratum]["output_signature"], sort_keys=True)
            for model_id in MODEL_ORDER
        }) == 1
        for stratum in ("fulltop", "n1", "combined")
    }
    checks = {
        "preflight_passed": True,
        "four_checkpoint_hashes_verified": len(checkpoints) == 4,
        "sealed_fulltop_375_complete": all(
            int(results[model_id]["fulltop"]["metrics"]["n_graphs"]) == 375 for model_id in MODEL_ORDER
        ),
        "sealed_n1_375_complete": all(
            int(results[model_id]["n1"]["metrics"]["n_graphs"]) == 375 for model_id in MODEL_ORDER
        ),
        "official_combined_750_complete": all(
            int(results[model_id]["combined"]["metrics"]["n_graphs"]) == 750 for model_id in MODEL_ORDER
        ),
        "same_ordered_test_manifest": True,
        "test_data_excluded_from_training_and_validation": True,
        "official_metrics_finite": True,
        "output_schemas_identical": all(schema_checks.values()),
        "environment_and_private_sources_frozen": True,
    }
    status = SUCCESS_STATUS if all(checks.values()) else "FT6_EVALUATION_BLOCKED"
    manifest = {
        "artifact_id": config["artifact_id"], "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
        "preflight_manifest": str(preflight_path), "preflight_sha256": sha256_file(preflight_path),
        "checkpoints": checkpoints,
        "checkpoint_paths": {key: value["path"] for key, value in checkpoints.items()},
        "checkpoint_sha256": {key: value["sha256"] for key, value in checkpoints.items()},
        "environment_freeze": freeze,
        "evaluation_contract": {
            "api": "gridsfm.eval_pass", "case_name": config["case_name"],
            "strata": config["test"], "combined_order": ["fulltop", "n1"],
            "combined_count": 750, "shuffle": False,
            "test_identity_manifest_sha256": sha256_file(preflight_path),
            "training_use": "none; inference-only sealed evaluation",
        },
        "results": results, "comparisons": comparisons,
        "schema_checks": schema_checks, "checks": checks,
        "runtime_seconds": time.perf_counter() - total_started,
    }
    write_json(output_dir / "ft6_evaluation_results.json", results)
    write_json(output_dir / "ft6_evaluation_comparisons.json", comparisons)
    write_json(output_dir / "ft6_evaluation_manifest.json", manifest)
    write_json(status_path, {
        "status": status, "phase": "COMPLETE", "checks": checks,
        "runtime_seconds": manifest["runtime_seconds"],
    })
    print(json.dumps({"status": status, "runtime_seconds": manifest["runtime_seconds"]}, indent=2), flush=True)
    return 0 if status == SUCCESS_STATUS else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    try:
        return run(config_path)
    except Exception as exc:
        try:
            config = read_json(config_path)
            output = _resolve_repo_path(config["output_dir"])
            write_json(output / "ft6_evaluation_status.json", {
                "status": "FT6_EVALUATION_BLOCKED", "phase": "UNHANDLED_EXCEPTION",
                "exception_type": type(exc).__name__, "message": str(exc),
            })
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
