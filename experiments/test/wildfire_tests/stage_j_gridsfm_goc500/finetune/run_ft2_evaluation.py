"""Evaluate frozen and FT1 GridSFM checkpoints on held-out FT2 test sets."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from run_finetune import (
    EXPECTED_BASE_SHA256,
    EXPECTED_GRIDSFM_COMMIT,
    REPO_ROOT,
    _device_from_config,
    _environment_freeze,
    _make_transform,
    _output_signature,
    _read_json,
    _resolve_repo_path,
    _seed_everything,
    _sha256,
    _write_json,
)

SUCCESS_STATUS = "FT2_EVALUATED_READY_FOR_REVIEW"
BLOCKED_STATUS = "FT2_BLOCKED"
DEVIATION_STATUS = "FT2_IMPLEMENTATION_DEVIATION_REQUIRES_REVIEW"
METRIC_KEYS = (
    "loss", "cost_mape", "pg_mae", "qg_mae", "V_mae", "theta_mae",
    "brP_mae", "brQ_mae", "kcl_P_resid", "kcl_Q_resid",
    "thermal_max_loading", "thermal_frac_overload", "feas_acc", "n_graphs",
)


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "artifact_id", "base_checkpoint", "batch_size", "case_name", "device",
        "expected_base_sha256", "expected_gridsfm_commit", "expected_source_sha256",
        "fine_tuned_checkpoint", "ft1_manifest", "gridsfm_root", "n_graphs",
        "num_groups", "num_workers", "opfdata_root", "output_dir", "seed",
        "split", "variants",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing required FT2 config keys: {missing}")
    if config["case_name"] != "pglib_opf_case500_goc":
        raise ValueError("FT2 is frozen to pglib_opf_case500_goc")
    if config["split"] != "test" or int(config["n_graphs"]) != 750:
        raise ValueError("FT2 requires exactly 750 held-out test graphs per variant")
    if list(config["variants"]) != ["fulltop", "n1"]:
        raise ValueError("FT2 variants must be ordered as FullTop then N-1")
    if int(config["num_groups"]) != 1:
        raise ValueError("FT2 requires exactly one 15,000-graph OPFData group")


def _finite_metrics(metrics: Mapping[str, Any]) -> bool:
    return set(metrics) == set(METRIC_KEYS) and all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in metrics.values()
    )


def _comparison(frozen: Mapping[str, Any], fine_tuned: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in METRIC_KEYS:
        baseline = float(frozen[key])
        updated = float(fine_tuned[key])
        result[key] = {
            "frozen": frozen[key],
            "fine_tuned": fine_tuned[key],
            "delta_fine_tuned_minus_frozen": updated - baseline,
            "ratio_fine_tuned_over_frozen": updated / baseline if baseline != 0.0 else None,
        }
    return result


def _checkpoint_records(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    base = Path(config["base_checkpoint"]).expanduser().resolve()
    fine_tuned = Path(config["fine_tuned_checkpoint"]).expanduser().resolve()
    ft1_manifest_path = _resolve_repo_path(config["ft1_manifest"])
    ft1_manifest = _read_json(ft1_manifest_path)
    records = {
        "frozen": {"path": str(base), "sha256": _sha256(base), "size_bytes": base.stat().st_size},
        "fine_tuned": {
            "path": str(fine_tuned), "sha256": _sha256(fine_tuned),
            "size_bytes": fine_tuned.stat().st_size,
        },
    }
    if records["frozen"]["sha256"] != config["expected_base_sha256"]:
        raise RuntimeError("released checkpoint SHA-256 does not match the FT2 config")
    if ft1_manifest["status"] != "FT1_TRAINED_READY_FOR_FT2":
        raise RuntimeError(f"FT1 manifest is not approved for FT2: {ft1_manifest['status']}")
    if ft1_manifest["checkpoint_path"] != records["fine_tuned"]["path"]:
        raise RuntimeError("FT1 checkpoint path differs between manifest and FT2 config")
    if ft1_manifest["checkpoint_sha256"] != records["fine_tuned"]["sha256"]:
        raise RuntimeError("FT1 checkpoint SHA-256 differs from its manifest")
    records["fine_tuned"]["source_manifest"] = str(ft1_manifest_path)
    return records


def _write_status(path: Path, phase: str, **details: Any) -> None:
    payload = {"status": "FT2_RUNNING", "phase": phase, **details}
    _write_json(path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def run(config_path: Path) -> int:
    from gridsfm import OPFDataAdapterDataset, eval_pass, load_model
    from torch_geometric.loader import DataLoader

    config = _read_json(config_path)
    _validate_config(config)
    output_dir = _resolve_repo_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "ft2_status.json"
    _write_json(output_dir / "ft2_config.json", config)
    _write_status(status_path, "PREFLIGHT")

    device = _device_from_config(config["device"])
    freeze = _environment_freeze(config, device)
    environment = freeze["environment"]
    if environment["gridsfm_git_commit"] != config["expected_gridsfm_commit"]:
        raise RuntimeError("GridSFM commit differs from the pinned FT2 commit")
    if environment["gridsfm_git_commit"] != EXPECTED_GRIDSFM_COMMIT:
        raise RuntimeError("GridSFM commit differs from the Stage J implementation pin")
    if freeze["source_sha256"] != config["expected_source_sha256"]:
        raise RuntimeError("private GridSFM API source hashes differ from the FT2 pin")
    if config["expected_base_sha256"] != EXPECTED_BASE_SHA256:
        raise RuntimeError("FT2 base-checkpoint pin differs from the Stage J pin")
    checkpoints = _checkpoint_records(config)
    freeze_path = output_dir / "ft2_environment_freeze.json"
    _write_json(freeze_path, freeze)

    _seed_everything(int(config["seed"]))
    results: dict[str, Any] = {}
    completed: list[dict[str, str]] = []
    total_started = time.perf_counter()
    for variant in config["variants"]:
        _write_status(status_path, "DATASET_PREPARATION", variant=variant, completed=completed)
        prepare_started = time.perf_counter()
        dataset = OPFDataAdapterDataset(
            root=str(Path(config["opfdata_root"]).expanduser().resolve()),
            case_name=config["case_name"], variant=variant, split=config["split"],
            n_graphs=int(config["n_graphs"]), num_groups=int(config["num_groups"]),
            transform=_make_transform(),
        )
        if len(dataset) != int(config["n_graphs"]):
            raise RuntimeError(f"{variant} test split exposed {len(dataset)} graphs, expected 750")
        preparation_runtime = time.perf_counter() - prepare_started
        variant_results: dict[str, Any] = {
            "dataset": {
                "case_name": config["case_name"], "variant": variant,
                "split": config["split"], "indices": [0, int(config["n_graphs"]) - 1],
                "n_graphs": len(dataset), "num_groups": int(config["num_groups"]),
                "preparation_runtime_seconds": preparation_runtime,
            }
        }
        loader = DataLoader(
            dataset, batch_size=int(config["batch_size"]), shuffle=False,
            num_workers=int(config["num_workers"]), persistent_workers=False,
        )
        for checkpoint_label in ("frozen", "fine_tuned"):
            _write_status(
                status_path, "EVALUATION", variant=variant,
                checkpoint=checkpoint_label, completed=completed,
            )
            model = load_model(checkpoints[checkpoint_label]["path"], device=device)
            started = time.perf_counter()
            metrics = eval_pass(model, loader, device=device)
            runtime = time.perf_counter() - started
            signature = _output_signature(model, loader, device)
            if not _finite_metrics(metrics):
                raise RuntimeError(f"non-finite or incomplete metrics for {variant}/{checkpoint_label}")
            if int(metrics["n_graphs"]) != int(config["n_graphs"]):
                raise RuntimeError(f"{variant}/{checkpoint_label} did not evaluate all 750 graphs")
            variant_results[checkpoint_label] = {
                "checkpoint": checkpoints[checkpoint_label], "metrics": metrics,
                "output_signature": signature, "runtime_seconds": runtime,
            }
            completed.append({"variant": variant, "checkpoint": checkpoint_label})
            _write_json(output_dir / "ft2_partial_results.json", {**results, variant: variant_results})
            del model
        if variant_results["frozen"]["output_signature"] != variant_results["fine_tuned"]["output_signature"]:
            raise RuntimeError(f"output schema differs between checkpoints on {variant}")
        variant_results["comparison"] = _comparison(
            variant_results["frozen"]["metrics"], variant_results["fine_tuned"]["metrics"]
        )
        results[variant] = variant_results

    finished_utc = datetime.now(timezone.utc).isoformat()
    manifest = {
        "artifact_id": config["artifact_id"], "status": SUCCESS_STATUS,
        "created_utc": finished_utc, "checkpoints": checkpoints,
        "checkpoint_paths": {key: value["path"] for key, value in checkpoints.items()},
        "checkpoint_sha256": {key: value["sha256"] for key, value in checkpoints.items()},
        "environment": environment, "environment_freeze_artifact": str(freeze_path),
        "evaluation_contract": {
            "case_name": config["case_name"], "variants": config["variants"],
            "split": "test", "indices": [0, 749], "n_graphs_per_variant": 750,
            "num_groups": 1, "shuffle": False,
            "training_use": "none; inference-only held-out evaluation",
            "official_evaluator": "gridsfm.eval_pass",
        },
        "results": results,
        "runtime_seconds": time.perf_counter() - total_started,
        "pass_checks": {
            "environment_frozen": True, "checkpoint_hashes_verified": True,
            "fulltop_test_complete": results["fulltop"]["frozen"]["metrics"]["n_graphs"] == 750
                and results["fulltop"]["fine_tuned"]["metrics"]["n_graphs"] == 750,
            "n1_test_complete": results["n1"]["frozen"]["metrics"]["n_graphs"] == 750
                and results["n1"]["fine_tuned"]["metrics"]["n_graphs"] == 750,
            "same_ordered_test_cases": True, "test_data_excluded_from_training": True,
            "official_metrics_finite": True, "output_schemas_identical": True,
        },
    }
    _write_json(output_dir / "ft2_manifest.json", manifest)
    _write_json(output_dir / "ft2_results.json", results)
    _write_json(status_path, {
        "status": SUCCESS_STATUS, "phase": "COMPLETE",
        "checkpoint_paths": manifest["checkpoint_paths"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "runtime_seconds": manifest["runtime_seconds"],
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    try:
        return run(config_path)
    except Exception as exc:
        try:
            config = _read_json(config_path)
            output_dir = _resolve_repo_path(config["output_dir"])
            status = DEVIATION_STATUS if "differs" in str(exc) else BLOCKED_STATUS
            _write_json(output_dir / "ft2_status.json", {
                "status": status, "phase": "UNHANDLED_EXCEPTION",
                "exception_type": type(exc).__name__, "message": str(exc),
                "process_id": os.getpid(),
            })
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
