"""Train one FT6 sibling continuation through official GridSFM APIs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ft6_common import (
    GRIDSFM_COMMIT,
    M1_SHA256,
    configured_indices,
    finite_metrics,
    read_json,
    sha256_file,
    validate_training_config,
    write_csv,
    write_json,
)
from run_finetune import (
    REPO_ROOT,
    _device_from_config,
    _environment_freeze,
    _export_checkpoint,
    _make_transform,
    _output_signature,
    _parameter_change,
    _resolve_repo_path,
    _seed_everything,
)


def _make_subset(config: Mapping[str, Any], spec: Mapping[str, Any], *, transform=None):
    from gridsfm import OPFDataAdapterDataset
    from torch.utils.data import Subset

    indices = configured_indices(spec)
    dataset = OPFDataAdapterDataset(
        root=str(Path(config["opfdata_root"]).expanduser().resolve()),
        case_name=config["case_name"], variant=spec["variant"], split=spec["split"],
        n_graphs=max(indices) + 1, num_groups=int(config["num_groups"]), transform=transform,
    )
    if len(dataset) <= max(indices):
        raise RuntimeError(f"dataset cannot provide configured indices: {spec}")
    return Subset(dataset, indices)


def _make_validation_loaders(config: Mapping[str, Any]):
    from torch_geometric.loader import DataLoader

    common = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "persistent_workers": False,
    }
    validation_loaders = {}
    for spec in config["validation"]:
        dataset = _make_subset(config, spec, transform=_make_transform())
        validation_loaders[spec["variant"]] = DataLoader(dataset, shuffle=False, **common)
    return validation_loaders


def _make_loaders(config: Mapping[str, Any]):
    import torch
    from gridsfm import SyntheticMixedDataset
    from torch_geometric.loader import DataLoader

    train_base = _make_subset(config, config["train"])
    train = SyntheticMixedDataset(
        train_base, infeas_prob=float(config["infeas_prob"]),
        seed=int(config["seed"]), transform=_make_transform(),
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    common = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "persistent_workers": False,
    }
    train_loader = DataLoader(train, shuffle=True, generator=generator, **common)
    return train, train_loader, _make_validation_loaders(config)


def _evaluate_strata(model, loaders: Mapping[str, Any], device) -> dict[str, Any]:
    from gridsfm import eval_pass

    result = {}
    for variant, loader in loaders.items():
        started = time.perf_counter()
        metrics = eval_pass(model, loader, device=device)
        if not finite_metrics(metrics) or int(metrics["n_graphs"]) != 375:
            raise RuntimeError(f"invalid official validation metrics for {variant}")
        result[variant] = {
            "metrics": metrics,
            "runtime_seconds": time.perf_counter() - started,
            "output_signature": _output_signature(model, loader, device),
        }
    return result


def _fresh_reload(config_path: Path, checkpoint: Path, output: Path) -> int:
    from gridsfm import load_model

    config = read_json(config_path)
    validate_training_config(config)
    device = _device_from_config(config["device"])
    _seed_everything(int(config["seed"]))
    validation_loaders = _make_validation_loaders(config)
    model = load_model(str(checkpoint), device=device)
    evaluation = _evaluate_strata(model, validation_loaders, device)
    write_json(output, {
        "status": "PASS", "process_id": os.getpid(),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "validation": evaluation,
    })
    return 0


def run(config_path: Path) -> int:
    import torch
    from gridsfm import finetune_opfdata, load_model

    config = read_json(config_path)
    validate_training_config(config)
    model_id = config["model_id"]
    output_dir = _resolve_repo_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"{model_id}_status.json"
    write_json(output_dir / f"{model_id}_config.json", config)
    write_json(status_path, {"status": "FT6_TRAINING_RUNNING", "phase": "PREFLIGHT", "model_id": model_id})

    preflight_path = _resolve_repo_path(config["preflight_manifest"])
    preflight = read_json(preflight_path)
    if preflight["status"] != "FT6_P1_DATA_CONTRACT_PASS":
        raise RuntimeError(f"FT6 preflight is not approved: {preflight['status']}")
    parent = Path(config["parent_checkpoint"]).expanduser().resolve()
    parent_hash = sha256_file(parent)
    if parent_hash != M1_SHA256 or parent_hash != config["expected_parent_sha256"]:
        raise RuntimeError("FT6 continuation parent checkpoint SHA mismatch")
    checkpoint = Path(config["checkpoint_dir"]).expanduser().resolve() / config["checkpoint_name"]
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite existing FT6 checkpoint: {checkpoint}")

    device = _device_from_config(config["device"])
    freeze = _environment_freeze(config, device)
    if freeze["environment"]["gridsfm_git_commit"] != GRIDSFM_COMMIT:
        raise RuntimeError("GridSFM commit differs from FT6 protocol")
    if freeze["source_sha256"] != config["expected_source_sha256"]:
        raise RuntimeError("GridSFM private-source hashes differ from FT6 protocol")
    write_json(output_dir / f"{model_id}_environment_freeze.json", freeze)

    _seed_everything(int(config["seed"]))
    train_dataset, train_loader, validation_loaders = _make_loaders(config)
    model = load_model(str(parent), device=device)
    parent_validation = _evaluate_strata(model, validation_loaders, device)
    write_json(output_dir / f"{model_id}_parent_validation.json", parent_validation)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    live_log: list[dict[str, Any]] = []
    training_started = time.perf_counter()

    def record_epoch(entry: dict[str, Any]) -> None:
        validation_started = time.perf_counter()
        validation = _evaluate_strata(model, validation_loaders, device)
        for variant, record in validation.items():
            for key, value in record["metrics"].items():
                entry[f"val_{variant}_{key}"] = value
            entry[f"val_{variant}_runtime_seconds"] = record["runtime_seconds"]
        entry["validation_runtime_seconds"] = time.perf_counter() - validation_started
        live_log.append(dict(entry))
        write_json(output_dir / f"{model_id}_training_log.json", live_log)
        write_csv(output_dir / f"{model_id}_training_log.csv", live_log)
        elapsed = time.perf_counter() - training_started
        estimate = elapsed / len(live_log) * int(config["epochs"])
        payload = {
            "status": "FT6_TRAINING_RUNNING", "phase": "TRAINING",
            "model_id": model_id, "epochs_completed": len(live_log),
            "epochs_total": int(config["epochs"]), "elapsed_seconds": elapsed,
            "estimated_total_seconds": estimate, "latest_epoch": dict(entry),
        }
        write_json(status_path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)

    log = finetune_opfdata(
        model, train_loader, val_loader=None, epochs=int(config["epochs"]),
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
        on_epoch_end=record_epoch,
    )
    training_runtime = time.perf_counter() - training_started
    write_json(output_dir / f"{model_id}_training_log.json", log)
    write_csv(output_dir / f"{model_id}_training_log.csv", log)

    parameter_change = _parameter_change(before, model.state_dict())
    write_json(output_dir / f"{model_id}_parameter_change.json", parameter_change)
    provenance = {
        "artifact_id": config["artifact_id"], "model_id": model_id,
        "model_variant": config["model_variant"],
        "parent_checkpoint_path": str(parent), "parent_checkpoint_sha256": parent_hash,
        "optimizer_state_continued": False,
        "optimizer_reset": "official finetune_opfdata created a fresh AdamW optimizer",
        "gridsfm_git_commit": freeze["environment"]["gridsfm_git_commit"],
        "case_name": config["case_name"], "training_spec": config["train"],
        "epochs": config["epochs"], "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"], "seed": config["seed"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _export_checkpoint(model, parent, checkpoint, provenance)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fresh_path = output_dir / f"{model_id}_fresh_reload.json"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--fresh-reload",
         "--config", str(config_path), "--checkpoint", str(checkpoint),
         "--fresh-output", str(fresh_path)],
        cwd=REPO_ROOT, check=True,
    )
    fresh = read_json(fresh_path)
    checkpoint_hash = sha256_file(checkpoint)
    finite_log = all(
        math.isfinite(float(row["train_loss"])) and int(row["n_train_iters"]) > 0
        and int(row["n_train_skipped"]) == 0 for row in log
    )
    validation_complete = len(log) == int(config["epochs"]) and all(
        int(row.get("val_fulltop_n_graphs", 0)) == 375
        and int(row.get("val_n1_n_graphs", 0)) == 375 for row in log
    )
    schema_match = all(
        parent_validation[variant]["output_signature"] == fresh["validation"][variant]["output_signature"]
        for variant in ("fulltop", "n1")
    )
    checks = {
        "preflight_passed": True,
        "parent_m1_path_and_sha_verified": True,
        "fresh_optimizer_recorded": True,
        "official_opfdata_training_path": True,
        "train_graph_count_500": len(train_dataset) == 500,
        "finite_training_loss_and_no_skipped_batches": finite_log,
        "both_validation_strata_recorded_each_epoch": validation_complete,
        "weights_changed": parameter_change["at_least_one_tensor_changed"],
        "updated_parameters_finite": parameter_change["all_updated_parameters_finite"],
        "checkpoint_saved": checkpoint.is_file(),
        "fresh_process_reload_passed": fresh["status"] == "PASS" and fresh["process_id"] != os.getpid(),
        "fresh_validation_metrics_finite": all(
            finite_metrics(fresh["validation"][variant]["metrics"]) for variant in ("fulltop", "n1")
        ),
        "parent_child_output_schemas_identical": schema_match,
        "environment_and_private_sources_frozen": True,
    }
    success_status = f"FT6_P2{('A' if model_id == 'm2' else 'B')}_{model_id.upper()}_TRAINING_PASS"
    status = success_status if all(checks.values()) else "FT6_TRAINING_BLOCKED"
    manifest = {
        "artifact_id": config["artifact_id"], "status": status,
        "model_id": model_id, "model_variant": config["model_variant"],
        "parent_checkpoint": {"path": str(parent), "sha256": parent_hash},
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "fine_tuned_checkpoint": {
            "path": str(checkpoint), "sha256": checkpoint_hash,
            "size_bytes": checkpoint.stat().st_size,
        },
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
        "preflight_manifest": str(preflight_path), "preflight_sha256": sha256_file(preflight_path),
        "environment_freeze": freeze,
        "training": {
            "api": "gridsfm.finetune_opfdata", "optimizer": "fresh AdamW",
            "train_spec": config["train"], "validation_specs": config["validation"],
            "batch_size": config["batch_size"], "epochs": config["epochs"],
            "learning_rate": config["learning_rate"], "weight_decay": config["weight_decay"],
            "infeas_prob": config["infeas_prob"], "seed": config["seed"],
            "runtime_seconds": training_runtime,
        },
        "parent_validation": parent_validation,
        "fresh_reload_validation": fresh["validation"],
        "parameter_change": parameter_change,
        "per_epoch_metrics_artifacts": {
            "json": str(output_dir / f"{model_id}_training_log.json"),
            "csv": str(output_dir / f"{model_id}_training_log.csv"),
        },
        "checks": checks,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / f"{model_id}_manifest.json", manifest)
    write_json(status_path, {
        "status": status, "phase": "COMPLETE", "model_id": model_id,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "runtime_seconds": training_runtime, "checks": checks,
    })
    print(json.dumps({
        "status": status, "model_id": model_id, "checkpoint_sha256": checkpoint_hash,
        "runtime_seconds": training_runtime,
    }, indent=2), flush=True)
    return 0 if status == success_status else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fresh-reload", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--fresh-output")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    try:
        if args.fresh_reload:
            if not args.checkpoint or not args.fresh_output:
                parser.error("--fresh-reload requires --checkpoint and --fresh-output")
            return _fresh_reload(
                config_path, Path(args.checkpoint).expanduser().resolve(),
                Path(args.fresh_output).expanduser().resolve(),
            )
        return run(config_path)
    except Exception as exc:
        try:
            config = read_json(config_path)
            output = _resolve_repo_path(config["output_dir"])
            write_json(output / f"{config['model_id']}_status.json", {
                "status": "FT6_TRAINING_BLOCKED", "phase": "UNHANDLED_EXCEPTION",
                "exception_type": type(exc).__name__, "message": str(exc),
            })
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
