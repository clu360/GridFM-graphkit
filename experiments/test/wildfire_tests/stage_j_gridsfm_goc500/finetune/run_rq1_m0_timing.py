"""Benchmark frozen GridSFM M0 on the canonical RQ1 decision bank."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np


WILDFIRE_ROOT = Path(__file__).resolve().parents[2]
if str(WILDFIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_ROOT))

from stage_j_gridsfm_goc500.goc500_adapter import (  # noqa: E402
    build_goc500_identity,
    mutate_raw_case_for_candidate,
)
from stage_j_gridsfm_goc500.gridsfm_evaluator import (  # noqa: E402
    evaluate_gridsfm_candidate,
)
from stage_j_gridsfm_goc500.metrics import compute_ac_loading_two_ended  # noqa: E402
from stage_j_gridsfm_goc500.schemas import PacWeights  # noqa: E402
from stage_j_gridsfm_goc500.finetune.rq1_common import (  # noqa: E402
    M0_SHA256,
    load_config,
    read_csv,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _sync(torch, device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _write_candidate(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)


def _extract_usable_state(torch, out, active_identity) -> dict[str, object]:
    from gridsfm.schema import AC_LINE_KEY, TRANSFORMER_KEY

    bus_pred = out["bus"].pred
    gen_pred = out["generator"].pred
    flow_tensors = []
    for key in (AC_LINE_KEY, TRANSFORMER_KEY):
        if key in out.edge_types and hasattr(out[key], "edge_flow_pred"):
            flow_tensors.append(out[key].edge_flow_pred)
    flows = torch.cat(flow_tensors, dim=0) if flow_tensors else torch.zeros(
        0, 4, device=bus_pred.device, dtype=bus_pred.dtype
    )
    bus = bus_pred.detach().cpu().numpy()
    gen = gen_pred.detach().cpu().numpy()
    edge = flows.detach().cpu().numpy()
    if not (np.isfinite(bus).all() and np.isfinite(gen).all() and np.isfinite(edge).all()):
        raise RuntimeError("M0 returned nonfinite electrical-state values")
    branches = list(active_identity.branches)
    if len(edge) != len(branches):
        raise RuntimeError(f"flow/branch count mismatch: {len(edge)} != {len(branches)}")
    line_ids = [row.canonical_branch_id for row in branches]
    p_from = {line_id: float(edge[index, 0]) for index, line_id in enumerate(line_ids)}
    q_from = {line_id: float(edge[index, 1]) for index, line_id in enumerate(line_ids)}
    p_to = {line_id: float(edge[index, 2]) for index, line_id in enumerate(line_ids)}
    q_to = {line_id: float(edge[index, 3]) for index, line_id in enumerate(line_ids)}
    rate_a = {row.canonical_branch_id: row.rate_a for row in branches}
    loading = compute_ac_loading_two_ended(p_from, q_from, p_to, q_to, rate_a, line_ids)
    return {
        "bus_ids": list(active_identity.bus_ids),
        "vm": bus[:, 1].astype(float).tolist(),
        "va": bus[:, 0].astype(float).tolist(),
        "pg": gen[:, 0].astype(float).tolist(),
        "qg": gen[:, 1].astype(float).tolist(),
        "branch_ids": line_ids,
        "p_from": edge[:, 0].astype(float).tolist(),
        "q_from": edge[:, 1].astype(float).tolist(),
        "p_to": edge[:, 2].astype(float).tolist(),
        "q_to": edge[:, 3].astype(float).tolist(),
        "loading": [loading[line_id] for line_id in line_ids],
        "feasibility_head": float(torch.sigmoid(out.feas_logit).detach().cpu().item()),
        "bus_count": len(bus),
        "generator_count": len(gen),
        "branch_count": len(edge),
    }


def _prepare_candidate(raw_case, identity, instance, path, device):
    from gridsfm import load_pyg_json, prepare_for_inference

    alpha = {int(key): float(value) for key, value in instance["alpha_effective"].items()}
    mutated, breakdown, integrity = mutate_raw_case_for_candidate(
        raw_case,
        identity,
        offline_branch_ids=instance["offline_branch_ids"],
        alpha_requested=alpha,
    )
    integrity.require_ok()
    _write_candidate(path, mutated)
    data = prepare_for_inference(load_pyg_json(path)).to(device)
    return mutated, breakdown, data


def _one_evaluator(torch, model, raw_case, identity, instance, work_dir, device):
    work_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = work_dir / "candidate.pyg.json"
    total_start = time.perf_counter_ns()
    mutated, breakdown, data = _prepare_candidate(
        raw_case, identity, instance, candidate_path, device
    )
    prepare_end = time.perf_counter_ns()
    _sync(torch, device)
    forward_start = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(data)
    _sync(torch, device)
    forward_end = time.perf_counter_ns()
    state = _extract_usable_state(torch, out, build_goc500_identity(mutated))
    total_end = time.perf_counter_ns()
    return {
        "prepare_seconds": (prepare_end - total_start) / 1e9,
        "forward_seconds": (forward_end - forward_start) / 1e9,
        "postprocess_seconds": (total_end - forward_end) / 1e9,
        "evaluator_total_seconds": (total_end - total_start) / 1e9,
        "state": state,
        "breakdown": breakdown,
        "prepared": data,
    }


def _core_times(torch, model, data, device, config) -> list[float]:
    for _ in range(3):
        with torch.inference_mode():
            model(data)
    _sync(torch, device)
    times: list[float] = []
    total = 0.0
    minimum = int(config["m0_core_min_repetitions"])
    maximum = int(config["m0_core_max_repetitions"])
    target = float(config["m0_core_min_total_seconds"])
    while len(times) < maximum and (len(times) < minimum or total < target):
        _sync(torch, device)
        started = time.perf_counter_ns()
        with torch.inference_mode():
            model(data)
        _sync(torch, device)
        elapsed = (time.perf_counter_ns() - started) / 1e9
        times.append(elapsed)
        total += elapsed
    return times


def _scenario_context(config: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, dict[int, float]]]:
    input_dir = Path(config["input_dir"]).resolve()
    register = read_csv(input_dir / "stage_j_scenario_register.csv")
    r_base = {row["scenario_id"]: float(row["r_base"]) for row in register}
    p_env: dict[str, dict[int, float]] = {}
    for row in read_csv(input_dir / "stage_j_p_env_by_scenario.csv"):
        p_env.setdefault(row["scenario_id"], {})[int(row["branch_id"])] = float(row["p_env"])
    return r_base, p_env


def run(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    gridsfm_root = Path(config["gridsfm_root"]).resolve()
    model_root = gridsfm_root / "model"
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))
    import torch
    from gridsfm import load_model

    output = Path(config["working_root"]).resolve()
    preflight = read_json(output / "RQ1_PREFLIGHT.json")
    if preflight["status"] != "RQ1_P0_PREFLIGHT_PASS":
        raise RuntimeError("RQ1 preflight has not passed")
    if _git_commit(gridsfm_root) != config["expected_gridsfm_commit"]:
        raise RuntimeError("GridSFM commit does not match the frozen RQ1 contract")
    checkpoint = Path(config["m0_checkpoint"]).resolve()
    if sha256_file(checkpoint) != M0_SHA256:
        raise RuntimeError("M0 checkpoint hash mismatch")

    random.seed(int(config["bootstrap_seed"]))
    np.random.seed(int(config["bootstrap_seed"]))
    torch.manual_seed(int(config["bootstrap_seed"]))
    device = torch.device("cpu")
    model = load_model(str(checkpoint), device=str(device))
    model.eval()
    raw_case = read_json(config["raw_case_path"])
    identity = build_goc500_identity(raw_case)
    instances = read_json(output / "rq1_unique_instances.json")["instances"]
    instances = sorted(instances, key=lambda row: row["unique_id"])
    provenance = read_csv(output / "rq1_fixed_provenance.csv")
    provenance_by_id = {row["provenance_id"]: row for row in provenance}
    r_base, p_env = _scenario_context(config)
    frozen = read_json(config["pac_freeze_json"])["frozen_weights"]
    weights = PacWeights(**{key: float(frozen[key]) for key in ("rho_phys", "w_op", "w_ac", "w_model")})

    scratch = output / "m0" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    for index in range(int(config["m0_warmup_evaluations"])):
        instance = instances[index % len(instances)]
        _one_evaluator(
            torch, model, raw_case, identity, instance,
            scratch / f"warmup_{index:02d}", device,
        )

    evaluator_rows: list[dict[str, object]] = []
    core_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    states_dir = output / "m0" / "states"
    repetitions = int(config["m0_evaluator_repetitions"])
    for repetition in range(repetitions):
        ordered = instances[repetition:] + instances[:repetition]
        for position, instance in enumerate(ordered, start=1):
            unique_id = instance["unique_id"]
            work_dir = scratch / unique_id
            work_dir.mkdir(parents=True, exist_ok=True)
            measured = _one_evaluator(
                torch, model, raw_case, identity, instance, work_dir, device
            )
            evaluator_rows.append(
                {
                    "unique_id": unique_id,
                    "decision_sha256": instance["decision_sha256"],
                    "repetition": repetition + 1,
                    "execution_position": position,
                    "prepare_seconds": measured["prepare_seconds"],
                    "forward_seconds": measured["forward_seconds"],
                    "postprocess_seconds": measured["postprocess_seconds"],
                    "evaluator_total_seconds": measured["evaluator_total_seconds"],
                    "timing_boundary": "fixed_candidate_received_to_usable_electrical_state_returned",
                    "downstream_scoring_included": False,
                    "publication_io_included": False,
                }
            )
            if repetition == 0:
                state = measured["state"]
                write_json(states_dir / f"{unique_id}.json", state)
                for core_rep, elapsed in enumerate(
                    _core_times(torch, model, measured["prepared"], device, config), start=1
                ):
                    core_rows.append(
                        {
                            "unique_id": unique_id,
                            "decision_sha256": instance["decision_sha256"],
                            "repetition": core_rep,
                            "forward_seconds": elapsed,
                            "timing_boundary": "prepared_single_graph_model_forward_only",
                            "batch_size": 1,
                        }
                    )
                first_provenance = provenance_by_id[instance["provenance_ids"][0]]
                scenario_id = first_provenance["scenario_id"]
                alpha = {int(key): float(value) for key, value in instance["alpha_effective"].items()}
                full = evaluate_gridsfm_candidate(
                    raw_case=raw_case,
                    identity=identity,
                    model=model,
                    offline_branch_ids=instance["offline_branch_ids"],
                    alpha_requested=alpha,
                    p_env_by_line=p_env[scenario_id],
                    r_base=r_base[scenario_id],
                    lambda_r=float(first_provenance["lambda_r"]),
                    weights=weights,
                    work_dir=scratch / unique_id / "diagnostic",
                )
                if full.objective is None:
                    raise RuntimeError(f"M0 diagnostic failed for {unique_id}: {full.message}")
                diagnostic_rows.append(
                    {
                        "unique_id": unique_id,
                        "decision_sha256": instance["decision_sha256"],
                        "evaluation_status": full.evaluation_status.value,
                        "pac_total": full.objective.pac_total,
                        "pac_operational": full.objective.pac_operational,
                        "pac_ac": full.objective.pac_ac,
                        "pac_model": full.objective.pac_model,
                        "max_loading": max(full.flow_loading_by_line.values()),
                        "num_loading_gt_1": sum(value > 1.0 for value in full.flow_loading_by_line.values()),
                        "feasibility_head": full.feasibility_head,
                        "d_input": full.d_input,
                        "l_shed_total": full.load_shedding.l_shed_total,
                        "state_path": str(states_dir / f"{unique_id}.json"),
                    }
                )

    write_csv(output / "m0" / "rq1_m0_evaluator_repetitions.csv", evaluator_rows)
    write_csv(output / "m0" / "rq1_m0_core_repetitions.csv", core_rows)
    write_csv(output / "m0" / "rq1_m0_diagnostics.csv", diagnostic_rows)
    gates = {
        "unique_evaluator_coverage": len({row["unique_id"] for row in evaluator_rows}) == 54,
        "evaluator_repetition_count": len(evaluator_rows) == 54 * repetitions,
        "unique_core_coverage": len({row["unique_id"] for row in core_rows}) == 54,
        "diagnostic_rows": len(diagnostic_rows) == 54,
        "finite_positive_evaluator_times": all(
            math.isfinite(float(row["evaluator_total_seconds"]))
            and float(row["evaluator_total_seconds"]) > 0
            for row in evaluator_rows
        ),
        "finite_positive_core_times": all(
            math.isfinite(float(row["forward_seconds"])) and float(row["forward_seconds"]) > 0
            for row in core_rows
        ),
        "all_expected_state_families": all(
            read_json(row["state_path"])["bus_count"]
            == len(raw_case["grid"]["nodes"]["bus"])
            and read_json(row["state_path"])["generator_count"]
            == len(raw_case["grid"]["nodes"]["generator"])
            and all(
                key in read_json(row["state_path"])
                for key in ("vm", "va", "pg", "qg", "p_from", "q_from", "p_to", "q_to")
            )
            for row in diagnostic_rows
        ),
        "m0_checkpoint_hash": sha256_file(checkpoint) == M0_SHA256,
    }
    status = "RQ1_P1_M0_TIMING_PASS" if all(gates.values()) else "RQ1_P1_M0_TIMING_FAIL"
    result = {
        "status": status,
        "gates": gates,
        "evaluator_rows": len(evaluator_rows),
        "core_rows": len(core_rows),
        "diagnostic_rows": len(diagnostic_rows),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "m0_checkpoint_sha256": sha256_file(checkpoint),
        "gridsfm_commit": _git_commit(gridsfm_root),
        "primary_timing_boundary": "fixed candidate received to usable electrical state returned",
        "primary_exclusions": ["downstream wildfire scoring", "publication I/O", "model loading", "environment startup"],
    }
    write_json(output / "m0" / "RQ1_M0_TIMING_VALIDATION.json", result)
    print(json.dumps(result, indent=2))
    return result


def validate_existing(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    output = Path(config["working_root"]).resolve()
    evaluator_rows = read_csv(output / "m0" / "rq1_m0_evaluator_repetitions.csv")
    core_rows = read_csv(output / "m0" / "rq1_m0_core_repetitions.csv")
    diagnostic_rows = read_csv(output / "m0" / "rq1_m0_diagnostics.csv")
    instances = read_json(output / "rq1_unique_instances.json")["instances"]
    instance_by_id = {row["unique_id"]: row for row in instances}
    raw_case = read_json(config["raw_case_path"])
    expected_bus = len(raw_case["grid"]["nodes"]["bus"])
    expected_gen = len(raw_case["grid"]["nodes"]["generator"])
    expected_branch = sum(
        len(raw_case["grid"]["edges"][family]["senders"])
        for family in ("ac_line", "transformer")
    )
    states = {row["unique_id"]: read_json(row["state_path"]) for row in diagnostic_rows}
    required = {"vm", "va", "pg", "qg", "p_from", "q_from", "p_to", "q_to", "loading"}
    repetitions = int(config["m0_evaluator_repetitions"])
    gates = {
        "unique_evaluator_coverage": len({row["unique_id"] for row in evaluator_rows}) == 54,
        "evaluator_repetition_count": len(evaluator_rows) == 54 * repetitions,
        "unique_core_coverage": len({row["unique_id"] for row in core_rows}) == 54,
        "diagnostic_rows": len(diagnostic_rows) == 54,
        "finite_positive_evaluator_times": all(
            math.isfinite(float(row["evaluator_total_seconds"]))
            and float(row["evaluator_total_seconds"]) > 0 for row in evaluator_rows
        ),
        "finite_positive_core_times": all(
            math.isfinite(float(row["forward_seconds"]))
            and float(row["forward_seconds"]) > 0 for row in core_rows
        ),
        "all_expected_state_families": all(
            required.issubset(state)
            and state["bus_count"] == expected_bus
            and state["generator_count"] == expected_gen
            and state["branch_count"]
            == expected_branch - len(instance_by_id[unique_id]["offline_branch_ids"])
            for unique_id, state in states.items()
        ),
        "m0_checkpoint_hash": sha256_file(config["m0_checkpoint"]) == M0_SHA256,
    }
    previous_path = output / "m0" / "RQ1_M0_TIMING_VALIDATION.json"
    previous = read_json(previous_path) if previous_path.is_file() else {}
    status = "RQ1_P1_M0_TIMING_PASS" if all(gates.values()) else "RQ1_P1_M0_TIMING_FAIL"
    result = {
        **{key: value for key, value in previous.items() if key not in {"status", "gates"}},
        "status": status,
        "gates": gates,
        "validation_rerun_only": True,
        "expected_grid_counts": {
            "buses": expected_bus,
            "gridsfm_generators": expected_gen,
            "base_branches": expected_branch,
        },
    }
    write_json(previous_path, result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = validate_existing(args.config) if args.validate_only else run(args.config)
    return 0 if result["status"] == "RQ1_P1_M0_TIMING_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
