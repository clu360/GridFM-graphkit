"""Guided-GridSFM fixed-(z, alpha) candidate evaluation."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .goc500_adapter import (
    LOAD_PD_IDX,
    LOAD_QD_IDX,
    build_goc500_identity,
    mutate_raw_case_for_candidate,
)
from .load_service import LoadSheddingBreakdown
from .metrics import compute_ac_loading_two_ended, compute_j_total_sfm
from .schemas import EvaluationStatus, PacWeights, StageJObjective


@dataclass(frozen=True)
class GridSFMCandidateResult:
    """Saved GridSFM candidate result fields required by Stage J."""

    evaluation_status: EvaluationStatus
    objective: StageJObjective | None = None
    flow_loading_by_line: Mapping[int, float] = field(default_factory=dict)
    p_from_by_line: Mapping[int, float] = field(default_factory=dict)
    q_from_by_line: Mapping[int, float] = field(default_factory=dict)
    p_to_by_line: Mapping[int, float] = field(default_factory=dict)
    q_to_by_line: Mapping[int, float] = field(default_factory=dict)
    pg_by_generator: Mapping[int, float] = field(default_factory=dict)
    qg_by_generator: Mapping[int, float] = field(default_factory=dict)
    v_by_bus: Mapping[int, float] = field(default_factory=dict)
    theta_by_bus: Mapping[int, float] = field(default_factory=dict)
    load_shedding: LoadSheddingBreakdown | None = None
    d_input: float | None = None
    feasibility_head: float | None = None
    pac_model_components: Mapping[str, float | bool | str] = field(default_factory=dict)
    message: str = ""
    pac_notes: str = ""


def evaluate_gridsfm_candidate(
    *,
    raw_case: Mapping[str, Any],
    model,
    offline_branch_ids: Iterable[int],
    alpha_requested: Mapping[int, float],
    p_env_by_line: Mapping[int, float],
    r_base: float,
    lambda_r: float,
    weights: PacWeights,
    work_dir: str | Path | None = None,
    identity=None,
) -> GridSFMCandidateResult:
    """Mutate raw case, run official GridSFM prediction, and compute Stage J merit."""

    weights.validate()
    identity = identity or build_goc500_identity(raw_case)
    try:
        mutated, breakdown, integrity = mutate_raw_case_for_candidate(
            raw_case,
            identity,
            offline_branch_ids=offline_branch_ids,
            alpha_requested=alpha_requested,
        )
        integrity.require_ok()
    except Exception as exc:
        return GridSFMCandidateResult(
            evaluation_status=EvaluationStatus.INPUT_INTEGRITY_FAILURE,
            d_input=1.0,
            message=str(exc),
        )

    try:
        from gridsfm import predict
    except Exception as exc:  # pragma: no cover - external env dependent
        return GridSFMCandidateResult(
            evaluation_status=EvaluationStatus.GRIDSFM_INFERENCE_FAILURE,
            d_input=integrity.d_input,
            message=f"gridsfm import failed: {exc}",
        )

    try:
        if work_dir is None:
            with tempfile.TemporaryDirectory() as tmp:
                out = _predict_from_mutated_path(model, mutated, Path(tmp) / "candidate.pyg.json")
        else:
            target_dir = Path(work_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            out = _predict_from_mutated_path(model, mutated, target_dir / "candidate.pyg.json")
    except Exception as exc:  # pragma: no cover - external env dependent
        return GridSFMCandidateResult(
            evaluation_status=EvaluationStatus.GRIDSFM_INFERENCE_FAILURE,
            d_input=integrity.d_input,
            load_shedding=breakdown,
            message=str(exc),
        )

    active_identity = build_goc500_identity(mutated)
    active_branches = list(active_identity.branches)
    try:
        p_from, q_from, p_to, q_to = _map_flow_outputs_to_lines(out, active_branches)
        ratea = {branch.canonical_branch_id: branch.rate_a for branch in active_branches}
        line_ids = [branch.canonical_branch_id for branch in active_branches]
        loading = compute_ac_loading_two_ended(p_from, q_from, p_to, q_to, ratea, line_ids)
        r_raw = sum(float(p_env_by_line.get(line_id, 0.0)) * float(loading[line_id]) ** 2 for line_id in line_ids)
        if r_base <= 0.0 or not np.isfinite(r_base):
            raise ValueError(f"r_base must be positive and finite, got {r_base}")
        r_norm = float(r_raw / r_base)
        pac_operational = _pac_operational(raw_case=mutated, identity=active_identity, output=out, loading=loading)
        pac_ac = _pac_ac_from_feasibility_head(out)
        pac_model, pac_model_components, pac_model_notes = _pac_model_consistency(
            raw_case=mutated,
            identity=active_identity,
            output=out,
            load_shedding=breakdown,
        )
        objective = compute_j_total_sfm(
            lambda_r=lambda_r,
            r_norm=r_norm,
            l_shed_total=breakdown.l_shed_total,
            pac_operational=pac_operational,
            pac_ac=pac_ac,
            pac_model=pac_model,
            weights=weights,
        )
    except Exception as exc:
        return GridSFMCandidateResult(
            evaluation_status=EvaluationStatus.METHODOLOGY_FAILURE,
            d_input=integrity.d_input,
            load_shedding=breakdown,
            message=str(exc),
        )

    status = EvaluationStatus.MODEL_OUTPUT_PENALIZED if (objective.pac_total or 0.0) > 1e-12 else EvaluationStatus.OK
    return GridSFMCandidateResult(
        evaluation_status=status,
        objective=objective,
        flow_loading_by_line=loading,
        p_from_by_line=p_from,
        q_from_by_line=q_from,
        p_to_by_line=p_to,
        q_to_by_line=q_to,
        pg_by_generator={idx: float(value) for idx, value in enumerate(_as_float_array(out["Pg"]))},
        qg_by_generator={idx: float(value) for idx, value in enumerate(_as_float_array(out["Qg"]))} if "Qg" in out else {},
        v_by_bus={int(bus): float(value) for bus, value in zip(active_identity.bus_ids, _as_float_array(out["V"]))},
        theta_by_bus={int(bus): float(value) for bus, value in zip(active_identity.bus_ids, _as_float_array(out["theta"]))},
        load_shedding=breakdown,
        d_input=integrity.d_input,
        feasibility_head=float(out["feas"]),
        pac_model_components=pac_model_components,
        message="GridSFM candidate evaluated through official preprocessing/predict path.",
        pac_notes=pac_model_notes,
    )


def _predict_from_mutated_path(model, mutated: Mapping[str, Any], path: Path):
    from gridsfm import predict

    with path.open("w", encoding="utf-8") as fh:
        json.dump(mutated, fh)
    return predict(model, str(path))


def _as_float_array(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def _map_flow_outputs_to_lines(out: Mapping[str, Any], active_branches) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    p_ij = _as_float_array(out["Pij"])
    q_ij = _as_float_array(out["Qij"])
    p_ji = _as_float_array(out["Pji"])
    q_ji = _as_float_array(out["Qji"])
    n = len(active_branches)
    if not (len(p_ij) == len(q_ij) == len(p_ji) == len(q_ji) == n):
        raise ValueError(f"GridSFM flow output length mismatch: predicted={len(p_ij)}, active_branches={n}")
    line_ids = [branch.canonical_branch_id for branch in active_branches]
    return (
        {int(line_id): float(value) for line_id, value in zip(line_ids, p_ij)},
        {int(line_id): float(value) for line_id, value in zip(line_ids, q_ij)},
        {int(line_id): float(value) for line_id, value in zip(line_ids, p_ji)},
        {int(line_id): float(value) for line_id, value in zip(line_ids, q_ji)},
    )


def _pac_operational(*, raw_case: Mapping[str, Any], identity, output: Mapping[str, Any], loading: Mapping[int, float]) -> float:
    thermal = np.mean([max(0.0, float(value) - 1.0) ** 2 for value in loading.values()]) if loading else 0.0

    gen_rows = raw_case["grid"]["nodes"].get("generator", [])
    pg = _as_float_array(output["Pg"])
    gen_viol = []
    for idx, row in enumerate(gen_rows):
        pmin = float(row[2])
        pmax = float(row[3])
        scale = max(pmax - pmin, 1e-9)
        value = float(pg[idx])
        gen_viol.append(max(0.0, (value - pmax) / scale) ** 2 + max(0.0, (pmin - value) / scale) ** 2)
    gen_p = float(np.mean(gen_viol)) if gen_viol else 0.0

    bus_rows = raw_case["grid"]["nodes"].get("bus", [])
    voltage = _as_float_array(output["V"])
    v_viol = []
    for idx, row in enumerate(bus_rows):
        vmin = float(row[2])
        vmax = float(row[3])
        value = float(voltage[idx])
        v_viol.append(max(0.0, value - vmax) ** 2 + max(0.0, vmin - value) ** 2)
    voltage_bounds = float(np.mean(v_viol)) if v_viol else 0.0

    return float(thermal + gen_p + voltage_bounds)


def _pac_ac_from_feasibility_head(output: Mapping[str, Any]) -> float:
    feas = float(output["feas"])
    if not np.isfinite(feas):
        return float("inf")
    return max(0.0, 1.0 - feas) ** 2


def _normalized_mse(predicted: np.ndarray, commanded: np.ndarray, scale: np.ndarray) -> float:
    denom = np.maximum(np.abs(scale), 1e-9)
    residual = (np.asarray(predicted, dtype=float) - np.asarray(commanded, dtype=float)) / denom
    return float(np.mean(residual**2)) if residual.size else 0.0


def _pac_model_consistency(
    *,
    raw_case: Mapping[str, Any],
    identity,
    output: Mapping[str, Any],
    load_shedding: LoadSheddingBreakdown,
) -> tuple[float, dict[str, float | bool | str], str]:
    """Stage I-style command consistency guardrail for GridSFM outputs.

    Stage I could directly penalize GridFM-predicted `Pd/Qd` disagreement with
    commanded alpha. The current official GridSFM `predict()` API does not
    expose predicted load-demand channels, so demand-output consistency is
    recorded as unavailable unless future model outputs include `Pd` and `Qd`.
    The raw input command check remains hard-gated through `D_input`; this
    helper only contributes to the soft `PAC_model` term when comparable model
    output channels exist.
    """

    load_rows = raw_case["grid"]["nodes"].get("load", [])
    command_pd = np.asarray([float(load_rows[load.load_index][LOAD_PD_IDX]) for load in identity.loads], dtype=float)
    command_qd = np.asarray([float(load_rows[load.load_index][LOAD_QD_IDX]) for load in identity.loads], dtype=float)
    expected_pd = np.asarray(
        [float(load_shedding.alpha_effective[load.canonical_load_id]) * float(load.pd_pre) for load in identity.loads],
        dtype=float,
    )
    expected_qd = np.asarray(
        [float(load_shedding.alpha_effective[load.canonical_load_id]) * float(load.qd_pre) for load in identity.loads],
        dtype=float,
    )
    input_pd_mse = _normalized_mse(command_pd, expected_pd, np.asarray([load.pd_pre for load in identity.loads], dtype=float))
    input_qd_mse = _normalized_mse(command_qd, expected_qd, np.asarray([load.qd_pre for load in identity.loads], dtype=float))

    components: dict[str, float | bool | str] = {
        "input_load_command_mse": float(np.mean([input_pd_mse, input_qd_mse])),
        "predicted_load_command_available": False,
        "predicted_load_command_mse": 0.0,
        "source_less_load_count": float(len(load_shedding.source_less_load_ids)),
        "source_less_alpha_effective_max_abs": float(
            max(
                (
                    abs(float(load_shedding.alpha_effective[load_id]))
                    for load_id in load_shedding.source_less_load_ids
                ),
                default=0.0,
            )
        ),
        "model_consistency_basis": "input_command_guard_only_no_predicted_Pd_Qd",
    }

    if "Pd" not in output or "Qd" not in output:
        return (
            0.0,
            components,
            (
                "PAC_model load-command consistency follows the Stage I guardrail, "
                "but the current official GridSFM predict() API does not expose "
                "predicted Pd/Qd channels. Source-less alpha clamping is enforced "
                "before inference and input command consistency is hard-gated by D_input."
            ),
        )

    predicted_pd = _as_float_array(output["Pd"])
    predicted_qd = _as_float_array(output["Qd"])
    if len(predicted_pd) != len(identity.loads) or len(predicted_qd) != len(identity.loads):
        raise ValueError(
            "GridSFM predicted Pd/Qd length mismatch: "
            f"Pd={len(predicted_pd)}, Qd={len(predicted_qd)}, loads={len(identity.loads)}"
        )

    pd_scale = np.asarray([load.pd_pre for load in identity.loads], dtype=float)
    qd_scale = np.asarray([load.qd_pre for load in identity.loads], dtype=float)
    predicted_pd_mse = _normalized_mse(predicted_pd, expected_pd, pd_scale)
    predicted_qd_mse = _normalized_mse(predicted_qd, expected_qd, qd_scale)
    predicted_load_mse = float(np.mean([predicted_pd_mse, predicted_qd_mse]))
    components.update(
        {
            "predicted_load_command_available": True,
            "predicted_load_command_mse": predicted_load_mse,
            "predicted_pd_command_mse": predicted_pd_mse,
            "predicted_qd_command_mse": predicted_qd_mse,
            "model_consistency_basis": "predicted_Pd_Qd_vs_alpha_effective_command",
        }
    )
    return (
        predicted_load_mse,
        components,
        (
            "PAC_model uses Stage I-style load command consistency: predicted Pd/Qd "
            "are compared against source-less-clamped alpha_effective commands."
        ),
    )
