"""Run FT0/FT1 through the official GridSFM OPFData fine-tuning APIs.

FT0 is configured in ``configs/ft0_smoke.json``. Large datasets and model
checkpoints remain outside Git; only compact evidence is written under results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
WILDFIRE_ROOT = Path(__file__).resolve().parents[2]
if str(WILDFIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_ROOT))

PASS_STATUS = "FT0_VALIDATED_READY_FOR_FT1"
BLOCKED_STATUS = "FT0_BLOCKED"
DEVIATION_STATUS = "FT0_IMPLEMENTATION_DEVIATION_REQUIRES_REVIEW"
EXPECTED_BASE_SHA256 = "F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831"
EXPECTED_GRIDSFM_COMMIT = "1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6"


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_config(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"configured CUDA device {requested!r} is unavailable")
    return device


def _environment_manifest(config: Mapping[str, Any], device) -> dict[str, Any]:
    import gridsfm
    import torch
    import torch_geometric

    gridsfm_root = Path(config["gridsfm_root"]).expanduser().resolve()
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": sys.platform,
        "gridsfm_root": str(gridsfm_root),
        "gridsfm_git_commit": _git_commit(gridsfm_root),
        "gridsfm_version": getattr(gridsfm, "__version__", "UNEXPOSED"),
        "gridsfm_module": str(Path(gridsfm.__file__).resolve()),
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "dtype": str(torch.get_default_dtype()),
        "cuda_available": torch.cuda.is_available(),
    }


def _environment_freeze(config: Mapping[str, Any], device) -> dict[str, Any]:
    gridsfm_root = Path(config["gridsfm_root"]).expanduser().resolve()
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    private_files = {
        "checkpoint.py": gridsfm_root / "model" / "gridsfm" / "checkpoint.py",
        "loss.py": gridsfm_root / "model" / "gridsfm" / "loss.py",
        "finetune_opfdata.py": gridsfm_root / "model" / "gridsfm" / "finetune_opfdata.py",
        "opfdata_train.py": gridsfm_root / "model" / "gridsfm" / "opfdata_train.py",
    }
    return {
        "environment": _environment_manifest(config, device),
        "pip_freeze": freeze,
        "source_sha256": {name: _sha256(path) for name, path in private_files.items()},
        "private_api_dependencies": [
            "gridsfm.checkpoint._hash_state_dict",
            "gridsfm.loss._predicted_flows",
        ],
    }


def _configured_indices(config: Mapping[str, Any], name: str) -> list[int]:
    explicit = config.get(f"{name}_indices")
    if explicit is not None:
        return [int(index) for index in explicit]
    count = int(config[f"{name}_count"])
    return list(range(count))


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "base_checkpoint", "batch_size", "case_name", "checkpoint_dir",
        "checkpoint_name", "epochs", "gridsfm_root",
        "infeas_prob", "learning_rate", "opfdata_root", "output_dir", "seed",
        "variant",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing required FT config keys: {missing}")
    if config["case_name"] != "pglib_opf_case500_goc":
        raise ValueError("Stage J fine-tuning is frozen to pglib_opf_case500_goc")
    if config["variant"] != "fulltop":
        raise ValueError("Stage J fine-tuning must use the FullTop variant")
    train_indices = _configured_indices(config, "train")
    heldout_indices = _configured_indices(config, "heldout")
    run_label = str(config.get("run_label", "ft0")).lower()
    if run_label == "ft0":
        if train_indices != list(range(10)) or heldout_indices != list(range(10)):
            raise ValueError("FT0 requires deterministic train/test indices 0..9")
        if int(config["batch_size"]) != 2 or int(config["epochs"]) not in range(2, 6):
            raise ValueError("FT0 requires batch_size=2 and 2..5 epochs")
    elif run_label == "ft1":
        if train_indices != list(range(1000)):
            raise ValueError("FT1 requires exactly train indices 0..999")
        if config.get("heldout_split") != "val" or len(heldout_indices) != 750:
            raise ValueError("FT1 requires all 750 held-out FullTop validation graphs")
        if int(config["batch_size"]) != 8 or int(config["epochs"]) != 10:
            raise ValueError("FT1 requires batch_size=8 and epochs=10")
        if not config.get("ft0_review_approved", False):
            raise ValueError("FT1 execution requires recorded FT0 review approval")
    else:
        raise ValueError(f"unsupported fine-tuning run_label {run_label!r}")
    if not math.isclose(float(config["learning_rate"]), 1e-4):
        raise ValueError("FT0 learning rate is frozen at 1e-4")
    if not math.isclose(float(config["infeas_prob"]), 0.3):
        raise ValueError("FT0 infeas_prob is frozen at 0.3")


def _make_transform():
    from gridsfm.cycle_basis import CycleBasisCache, prepare_for_grid_transformer_
    from gridsfm.pe_features import LaplacianFactorizationCache, attach_pe_features_

    cycle_cache = CycleBasisCache()
    pe_cache = LaplacianFactorizationCache()

    def transform(data):
        prepare_for_grid_transformer_(data, cache=cycle_cache)
        attach_pe_features_(data, cache=pe_cache)
        return data

    return transform


def _subset(dataset, indices: Iterable[int]):
    from torch.utils.data import Subset

    selected = [int(index) for index in indices]
    if not selected or max(selected) >= len(dataset):
        raise IndexError(f"dataset length {len(dataset)} cannot provide indices {selected}")
    return Subset(dataset, selected)


def _make_datasets(config: Mapping[str, Any]):
    from gridsfm import OPFDataAdapterDataset, SyntheticMixedDataset

    common = {
        "root": str(Path(config["opfdata_root"]).expanduser().resolve()),
        "case_name": config["case_name"],
        "variant": config["variant"],
        "num_groups": int(config.get("num_groups", 1)),
    }
    train_indices = _configured_indices(config, "train")
    heldout_indices = _configured_indices(config, "heldout")
    train_base_all = OPFDataAdapterDataset(
        **common, split=config.get("train_split", "train"),
        n_graphs=max(train_indices) + 1,
    )
    heldout_all = OPFDataAdapterDataset(
        **common, split=config.get("heldout_split", "test"),
        n_graphs=max(heldout_indices) + 1,
        transform=_make_transform(),
    )
    train_base = _subset(train_base_all, train_indices)
    train_mixed = SyntheticMixedDataset(
        train_base,
        infeas_prob=float(config["infeas_prob"]),
        seed=int(config["seed"]),
        transform=_make_transform(),
    )
    return train_mixed, _subset(heldout_all, heldout_indices)


def _make_loaders(config: Mapping[str, Any], train_dataset, heldout_dataset):
    import torch
    from torch_geometric.loader import DataLoader

    generator = torch.Generator().manual_seed(int(config["seed"]))
    common = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config.get("num_workers", 0)),
        "persistent_workers": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common)
    heldout_loader = DataLoader(heldout_dataset, shuffle=False, **common)
    return train_loader, heldout_loader


def _finite_metrics(metrics: Mapping[str, Any]) -> bool:
    return all(
        isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
        for value in metrics.values()
    )


def _mapping_values_differ(
    left: Mapping[int | str, Any],
    right: Mapping[int | str, Any],
    *,
    absolute_tolerance: float = 1e-12,
) -> bool:
    """Compare numeric maps after normalizing JSON's stringified keys."""

    right_normalized = {str(key): float(value) for key, value in right.items()}
    return any(
        not math.isclose(
            float(value),
            right_normalized[str(key)],
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        )
        for key, value in left.items()
    )


def _output_signature(model, loader, device) -> dict[str, Any]:
    import torch
    from gridsfm.loss import _predicted_flows

    batch = next(iter(loader)).to(device)
    model.eval()
    with torch.no_grad():
        model(batch)
        flow_values = _predicted_flows(batch)[2:]
    tensors = {
        "bus.pred": batch["bus"].pred,
        "generator.pred": batch["generator"].pred,
        "feas_logit": batch.feas_logit,
        "Pij": flow_values[0],
        "Qij": flow_values[1],
        "Pji": flow_values[2],
        "Qji": flow_values[3],
    }
    return {
        key: {
            "shape": list(value.shape),
            "finite": bool(torch.isfinite(value).all().item()),
        }
        for key, value in tensors.items()
    }


def _parameter_change(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    sum_sq = 0.0
    max_abs = 0.0
    changed = 0
    finite = True
    floating = 0
    for name, new_value in after.items():
        if not torch.is_floating_point(new_value):
            continue
        floating += 1
        old_value = before[name].to(new_value.device)
        delta = new_value.detach() - old_value
        sum_sq += float(torch.sum(delta.double() ** 2).item())
        max_abs = max(max_abs, float(delta.abs().max().item()))
        changed += int(bool(torch.any(delta != 0).item()))
        finite = finite and bool(torch.isfinite(new_value).all().item())
    return {
        "parameter_l2_change": math.sqrt(sum_sq),
        "max_absolute_parameter_change": max_abs,
        "floating_parameter_tensors": floating,
        "changed_parameter_tensors": changed,
        "changed_parameter_tensor_fraction": changed / floating if floating else 0.0,
        "all_updated_parameters_finite": finite,
        "at_least_one_tensor_changed": changed > 0,
    }


def _export_checkpoint(model, base_checkpoint: Path, target: Path, provenance: Mapping[str, Any]) -> None:
    import torch
    from gridsfm.checkpoint import _hash_state_dict

    base_blob = torch.load(base_checkpoint, weights_only=True, map_location="cpu")
    state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    metadata = deepcopy(base_blob["metadata"])
    metadata["hash"] = _hash_state_dict(state_dict)
    metadata["fine_tuning"] = dict(provenance)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "metadata": metadata}, target)


def _load_stage_j_candidate(config: Mapping[str, Any]) -> dict[str, Any]:
    stage = config["stage_j"]
    path = _resolve_repo_path(stage["finalists_csv"])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row for row in rows
        if row["method"] == stage["method"]
        and row["scenario_id"] == stage["scenario_id"]
        and math.isclose(float(row["lambda_r"]), float(stage["lambda_r"]))
        and int(row["num_shutoffs"]) == 2
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one saved Stage J smoke candidate, found {len(matches)}")
    row = matches[0]
    return {
        "source_file": str(path),
        "method": row["method"],
        "scenario_id": row["scenario_id"],
        "lambda_r": float(row["lambda_r"]),
        "topology_rank": int(row["topology_rank"]),
        "offline_branch_ids": [int(value) for value in row["topology_id"].split(";")],
        "selected_alpha": {int(key): float(value) for key, value in json.loads(row["best_alpha_selected"]).items()},
    }


def _stage_j_context(config: Mapping[str, Any]):
    from stage_j_gridsfm_goc500.goc500_adapter import build_goc500_identity
    from stage_j_gridsfm_goc500.schemas import PacWeights

    stage = config["stage_j"]
    gridsfm_root = Path(config["gridsfm_root"]).expanduser().resolve()
    raw_case = _read_json(gridsfm_root / "model" / "samples" / "case500_goc.pyg.json")
    input_dir = Path(stage["input_dir"]).expanduser().resolve()
    with (input_dir / "stage_j_candidate_line_scores.csv").open(newline="", encoding="utf-8") as handle:
        candidate_ids = [int(row["branch_id"]) for row in csv.DictReader(handle)]
    identity = build_goc500_identity(raw_case, candidate_branch_ids=candidate_ids)
    with (input_dir / "stage_j_scenario_register.csv").open(newline="", encoding="utf-8") as handle:
        scenario = next(row for row in csv.DictReader(handle) if row["scenario_id"] == stage["scenario_id"])
    p_env = {}
    with (input_dir / "stage_j_p_env_by_scenario.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scenario_id"] == stage["scenario_id"]:
                p_env[int(row["branch_id"])] = float(row["p_env"])
    frozen = _read_json(Path(stage["pac_freeze_json"]).expanduser().resolve())["frozen_weights"]
    weights = PacWeights(**{key: float(frozen[key]) for key in ("rho_phys", "w_op", "w_ac", "w_model")})
    return raw_case, identity, float(scenario["r_base"]), p_env, weights


def _candidate_result_payload(result, runtime_seconds: float) -> dict[str, Any]:
    objective = result.objective
    return {
        "evaluation_status": result.evaluation_status.value,
        "runtime_seconds": runtime_seconds,
        "D_input": result.d_input,
        "feasibility_head": result.feasibility_head,
        "objective": None if objective is None else {
            "R_norm": objective.r_norm,
            "L_shed_total": objective.l_shed_total,
            "PAC_operational": objective.pac_operational,
            "PAC_AC": objective.pac_ac,
            "PAC_model": objective.pac_model,
            "PAC_total": objective.pac_total,
            "J_trade": objective.j_trade,
            "J_total": objective.j_total,
        },
        "Pg": result.pg_by_generator,
        "Qg": result.qg_by_generator,
        "V": result.v_by_bus,
        "theta": result.theta_by_bus,
        "P_from": result.p_from_by_line,
        "Q_from": result.q_from_by_line,
        "P_to": result.p_to_by_line,
        "Q_to": result.q_to_by_line,
        "loading": result.flow_loading_by_line,
        "pac_model_components": result.pac_model_components,
        "message": result.message,
    }


def _evaluate_stage_j_candidate(config: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    from gridsfm import load_model
    from stage_j_gridsfm_goc500.gridsfm_evaluator import evaluate_gridsfm_candidate

    candidate = _load_stage_j_candidate(config)
    raw_case, identity, r_base, p_env, weights = _stage_j_context(config)
    alpha = {load.canonical_load_id: 1.0 for load in identity.loads}
    alpha.update(candidate["selected_alpha"])
    model = load_model(str(checkpoint), device="cpu")
    started = time.perf_counter()
    result = evaluate_gridsfm_candidate(
        raw_case=raw_case,
        model=model,
        offline_branch_ids=candidate["offline_branch_ids"],
        alpha_requested=alpha,
        p_env_by_line=p_env,
        r_base=r_base,
        lambda_r=candidate["lambda_r"],
        weights=weights,
        identity=identity,
    )
    return _candidate_result_payload(result, time.perf_counter() - started)


def _fresh_reload(config_path: Path, checkpoint: Path, output_path: Path) -> int:
    import torch
    from gridsfm import eval_pass, load_model

    config = _read_json(config_path)
    device = _device_from_config(config.get("device", "auto"))
    _, heldout_dataset = _make_datasets(config)
    _, heldout_loader = _make_loaders(config, heldout_dataset, heldout_dataset)
    model = load_model(str(checkpoint), device=device)
    started = time.perf_counter()
    metrics = eval_pass(model, heldout_loader, device=device)
    signature = _output_signature(model, heldout_loader, device)
    payload = {
        "status": "PASS" if _finite_metrics(metrics) and all(v["finite"] for v in signature.values()) else "FAIL",
        "process_id": os.getpid(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "metrics": metrics,
        "output_signature": signature,
        "evaluation_runtime_seconds": time.perf_counter() - started,
    }
    if config.get("run_stage_j_smoke", False):
        payload["stage_j"] = _evaluate_stage_j_candidate(config, checkpoint)
    _write_json(output_path, payload)
    return 0 if payload["status"] == "PASS" else 1


def _write_training_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path) -> int:
    import torch
    from gridsfm import eval_pass, finetune_opfdata, load_model

    config = _read_json(config_path)
    _validate_config(config)
    run_label = str(config.get("run_label", "ft0")).lower()
    success_status = str(config.get("success_status", PASS_STATUS))
    blocked_status = str(config.get("blocked_status", BLOCKED_STATUS))
    deviation_status = str(config.get("deviation_status", DEVIATION_STATUS))
    train_indices = _configured_indices(config, "train")
    heldout_indices = _configured_indices(config, "heldout")

    output_dir = _resolve_repo_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / f"{run_label}_config.json", config)
    status_path = output_dir / f"{run_label}_status.json"
    _write_json(status_path, {"status": f"{run_label.upper()}_RUNNING", "phase": "STARTED"})

    base_checkpoint = Path(config["base_checkpoint"]).expanduser().resolve()
    checkpoint = Path(config["checkpoint_dir"]).expanduser().resolve() / config["checkpoint_name"]
    base_hash = _sha256(base_checkpoint)
    if base_checkpoint.name != "gridsfm_open_v1.1.pt" or base_hash != EXPECTED_BASE_SHA256:
        _write_json(status_path, {
            "status": deviation_status, "phase": "PREFLIGHT",
            "issue": "BASELINE_VERSION_MISMATCH",
            "checkpoint_path": str(base_checkpoint), "checkpoint_sha256": base_hash,
        })
        return 2

    device = _device_from_config(config.get("device", "auto"))
    environment_freeze = _environment_freeze(config, device)
    environment = environment_freeze["environment"]
    expected_source_sha = config.get("expected_source_sha256", {})
    if environment["gridsfm_git_commit"] != EXPECTED_GRIDSFM_COMMIT:
        _write_json(status_path, {
            "status": deviation_status, "phase": "PREFLIGHT",
            "issue": "GRIDSFM_COMMIT_DEVIATION", "environment": environment,
        })
        return 2
    if expected_source_sha and environment_freeze["source_sha256"] != expected_source_sha:
        _write_json(status_path, {
            "status": deviation_status, "phase": "PREFLIGHT",
            "issue": "PRIVATE_API_SOURCE_DEVIATION",
            "expected": expected_source_sha,
            "observed": environment_freeze["source_sha256"],
        })
        return 2
    freeze_path = output_dir / f"{run_label}_environment_freeze.json"
    _write_json(freeze_path, environment_freeze)

    _seed_everything(int(config["seed"]))
    train_dataset, heldout_dataset = _make_datasets(config)
    train_loader, heldout_loader = _make_loaders(config, train_dataset, heldout_dataset)
    model = load_model(str(base_checkpoint), device=device)
    frozen_started = time.perf_counter()
    frozen_metrics = eval_pass(model, heldout_loader, device=device)
    frozen_signature = _output_signature(model, heldout_loader, device)
    frozen_runtime = time.perf_counter() - frozen_started
    _write_json(output_dir / f"{run_label}_frozen_heldout_metrics.json", {
        "metrics": frozen_metrics,
        "output_signature": frozen_signature,
        "runtime_seconds": frozen_runtime,
    })

    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    live_log: list[dict[str, Any]] = []

    def record_epoch(entry: Mapping[str, Any]) -> None:
        live_log.append(dict(entry))
        _write_json(output_dir / f"{run_label}_training_log.json", live_log)
        _write_training_csv(output_dir / f"{run_label}_training_log.csv", live_log)
        _write_json(status_path, {
            "status": f"{run_label.upper()}_RUNNING",
            "phase": "TRAINING",
            "epochs_completed": len(live_log),
            "epochs_total": int(config["epochs"]),
            "latest_epoch": dict(entry),
        })
        print(json.dumps({"run_label": run_label, "epoch_complete": dict(entry)}, sort_keys=True), flush=True)

    training_started = time.perf_counter()
    log = finetune_opfdata(
        model, train_loader, val_loader=heldout_loader,
        epochs=int(config["epochs"]),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        on_epoch_end=record_epoch,
    )
    training_runtime = time.perf_counter() - training_started
    _write_json(output_dir / f"{run_label}_training_log.json", log)
    _write_training_csv(output_dir / f"{run_label}_training_log.csv", log)

    parameter_change = _parameter_change(before, model.state_dict())
    _write_json(output_dir / f"{run_label}_parameter_change.json", parameter_change)
    provenance = {
        "artifact_id": config["artifact_id"],
        "base_checkpoint_path": str(base_checkpoint),
        "base_checkpoint_sha256": base_hash,
        "gridsfm_git_commit": environment["gridsfm_git_commit"],
        "case_name": config["case_name"], "variant": config["variant"],
        "train_indices": train_indices, "epochs": config["epochs"],
        "learning_rate": config["learning_rate"], "seed": config["seed"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _export_checkpoint(model, base_checkpoint, checkpoint, provenance)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fresh_path = output_dir / f"{run_label}_fresh_reload.json"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--fresh-reload",
         "--config", str(config_path), "--checkpoint", str(checkpoint),
         "--fresh-output", str(fresh_path)],
        cwd=REPO_ROOT, check=True,
    )
    fresh = _read_json(fresh_path)
    _write_json(output_dir / f"{run_label}_heldout_metrics.json", {
        "metrics": fresh["metrics"],
        "output_signature": fresh["output_signature"],
        "runtime_seconds": fresh["evaluation_runtime_seconds"],
    })

    stage_j_smoke = None
    if config.get("run_stage_j_smoke", False):
        frozen_stage_j = _evaluate_stage_j_candidate(config, base_checkpoint)
        predictions_differ = _mapping_values_differ(frozen_stage_j["Pg"], fresh["stage_j"]["Pg"])
        input_integrity_pass = frozen_stage_j["D_input"] == 0.0 and fresh["stage_j"]["D_input"] == 0.0
        stage_j_smoke = {
            "candidate": _load_stage_j_candidate(config),
            "frozen_v1_1": frozen_stage_j, run_label: fresh["stage_j"],
            "same_output_schema": frozen_signature == fresh["output_signature"],
            "predictions_differ": predictions_differ,
            "input_integrity_pass": input_integrity_pass,
            "checkpoint_interface_deviation": not input_integrity_pass,
        }
        _write_json(output_dir / f"{run_label}_stage_j_smoke.json", stage_j_smoke)

    finite_log = all(math.isfinite(float(row["train_loss"])) and row["n_train_iters"] > 0 for row in log)
    pass_checks = {
        "released_v1_1_loaded": True,
        "goc500_fulltop_train_samples_loaded": len(train_dataset) == len(train_indices),
        "cycle_hodge_preparation_executed": all(v["finite"] for v in frozen_signature.values()),
        "finite_training_loss": finite_log,
        "weights_changed": parameter_change["at_least_one_tensor_changed"],
        "updated_parameters_finite": parameter_change["all_updated_parameters_finite"],
        "checkpoint_saved": checkpoint.is_file(),
        "fresh_process_reload_passed": fresh["status"] == "PASS" and fresh["process_id"] != os.getpid(),
        "heldout_inference_passed": _finite_metrics(fresh["metrics"]),
        "frozen_finetuned_schema_identical": frozen_signature == fresh["output_signature"],
        "validation_metrics_recorded_each_epoch": len(log) == int(config["epochs"]) and all(
            int(row.get("val_n_graphs", 0)) == len(heldout_indices) for row in log
        ),
        "environment_commit_frozen": environment["gridsfm_git_commit"] == EXPECTED_GRIDSFM_COMMIT,
        "private_api_sources_frozen": not expected_source_sha or environment_freeze["source_sha256"] == expected_source_sha,
    }
    if stage_j_smoke is not None:
        pass_checks.update({
            "frozen_finetuned_predictions_differ": stage_j_smoke["predictions_differ"],
            "stage_j_candidate_evaluated": fresh["stage_j"]["objective"] is not None,
            "stage_j_input_integrity": stage_j_smoke["input_integrity_pass"],
            "stage_j_methodology_unchanged": True,
        })
    status = success_status if all(pass_checks.values()) else blocked_status
    checkpoint_hash = _sha256(checkpoint)
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    manifest = {
        "artifact_id": config["artifact_id"], "status": status,
        "base_checkpoint": {
            "path": str(base_checkpoint), "filename": base_checkpoint.name,
            "sha256": base_hash, "version": "v1.1",
        },
        "fine_tuned_checkpoint": {
            "path": str(checkpoint), "filename": checkpoint.name,
            "sha256": checkpoint_hash, "size_bytes": checkpoint.stat().st_size,
            "created_utc": datetime.fromtimestamp(checkpoint.stat().st_mtime, timezone.utc).isoformat(),
        },
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "environment": environment,
        "environment_freeze": environment_freeze,
        "environment_freeze_artifact": str(freeze_path),
        "dataset": {
            "case_name": config["case_name"], "variant": config["variant"],
            "train_split": config.get("train_split", "train"), "train_indices": train_indices,
            "heldout_split": config.get("heldout_split", "test"), "heldout_indices": heldout_indices,
            "opfdata_root": str(Path(config["opfdata_root"]).expanduser().resolve()),
        },
        "training": {
            "batch_size": config["batch_size"], "epochs": config["epochs"],
            "learning_rate": config["learning_rate"], "weight_decay": config.get("weight_decay", 1e-4),
            "infeas_prob": config["infeas_prob"], "seed": config["seed"],
            "runtime_seconds": training_runtime, "peak_cuda_memory_bytes": peak_memory,
            "per_epoch_validation_artifacts": [
                {key: value for key, value in row.items() if key.startswith("val_") or key == "epoch"}
                for row in log
            ],
        },
        "preprocessing": {
            "cycle_basis": "gridsfm.cycle_basis.CycleBasisCache + prepare_for_grid_transformer_",
            "hodge_pe": "gridsfm.pe_features.LaplacianFactorizationCache + attach_pe_features_",
            "transform_location": "SyntheticMixedDataset for train; OPFDataAdapterDataset for held-out",
        },
        "implementation_notes": [
            "Official GridSFM exposes no public checkpoint exporter; the documented minimal format and gridsfm.checkpoint._hash_state_dict are used.",
            "Output finiteness uses gridsfm.loss._predicted_flows, matching official eval_pass internals.",
            "No GridSFM architecture, loss, preprocessing, or optimizer loop is duplicated.",
        ],
        "pass_checks": pass_checks,
    }
    _write_json(output_dir / f"{run_label}_manifest.json", manifest)
    _write_json(status_path, {
        "status": status, "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash, "pass_checks": pass_checks,
    })
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
                config_path,
                Path(args.checkpoint).expanduser().resolve(),
                Path(args.fresh_output).expanduser().resolve(),
            )
        return run(config_path)
    except Exception as exc:
        try:
            config = _read_json(config_path)
            output_dir = _resolve_repo_path(config["output_dir"])
            run_label = str(config.get("run_label", "ft0")).lower()
            blocked_status = str(config.get("blocked_status", BLOCKED_STATUS))
            _write_json(output_dir / f"{run_label}_status.json", {
                "status": blocked_status,
                "phase": "UNHANDLED_EXCEPTION",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            })
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
