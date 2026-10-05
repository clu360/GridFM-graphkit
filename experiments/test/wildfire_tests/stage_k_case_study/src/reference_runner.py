"""Sealed-finalist Reference A/B launcher and discrepancy calculation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from .config import load_config
from .identity import build_identity, source_less_load_ids
from .io_utils import atomic_write_json
from .objectives import compute_j_trade, compute_l_shed, compute_r_norm, reference_deltas
from .schemas import ELIGIBLE_STATUSES, classify_solver_status


def _parse_ids(value: str) -> tuple[int, ...]:
    return tuple(sorted(int(item) for item in str(value).split(";") if item and item != "nan"))


def _write_two_column(path: Path, header: tuple[str, str], values: dict[int, float], *, offset: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for key, value in sorted(values.items()):
            writer.writerow([key + offset, value])


def _loading_from_branch_state(path: str | Path, l_trans: set[int]) -> dict[int, float]:
    state = pd.read_csv(path)
    return {
        int(row.powermodels_branch_id) - 1: float(row.physical_loading)
        for row in state.itertuples(index=False)
        if int(row.powermodels_branch_id) - 1 in l_trans
    }


def _reference_b_diagnostic_summary(
    summary_b1: dict[str, object], summary_b2: dict[str, object]
) -> tuple[dict[str, object], bool]:
    """Use B2 only when its economic tie-break solve is solver-eligible."""

    b1_valid = classify_solver_status(str(summary_b1.get("termination_status", ""))) in ELIGIBLE_STATUSES
    if not b1_valid:
        raise RuntimeError("Reference B1 maximum-service solve is not solver-eligible")
    b2_valid = classify_solver_status(str(summary_b2.get("termination_status", ""))) in ELIGIBLE_STATUSES
    return (summary_b2 if b2_valid else summary_b1), b2_valid


def _rmse(left: pd.Series, right: pd.Series) -> float:
    values = (left.astype(float) - right.astype(float)).to_numpy()
    return float(np.sqrt(np.mean(values**2))) if len(values) else float("nan")


def _state_discrepancy(evaluator: str, native_state_path: str, summary_a: dict[str, object]) -> dict[str, object]:
    reference_branch = pd.read_csv(summary_a["branch_state_csv"]).assign(
        canonical_branch_id=lambda frame: frame["powermodels_branch_id"].astype(int) - 1
    )
    reference_bus = pd.read_csv(summary_a["bus_state_csv"])
    reference_gen = pd.read_csv(summary_a["gen_state_csv"]).assign(
        canonical_generator_id=lambda frame: frame["gen_id"].astype(int) - 1
    )
    native_root = Path(native_state_path)
    if evaluator == "gridsfm":
        branch = pd.read_parquet(native_root / "branch_state.parquet").merge(
            reference_branch, on="canonical_branch_id", suffixes=("_native", "_ref"), validate="one_to_one"
        )
        bus = pd.read_parquet(native_root / "bus_state.parquet").merge(
            reference_bus, on="bus_id", suffixes=("_native", "_ref"), validate="one_to_one"
        )
        gen = pd.read_parquet(native_root / "gen_state.parquet").merge(
            reference_gen, on="canonical_generator_id", suffixes=("_native", "_ref"), validate="one_to_one"
        )
        return {
            "branch_p_from_rmse": _rmse(branch["p_from"], branch["pf"]),
            "branch_q_from_rmse": _rmse(branch["q_from"], branch["qf"]),
            "branch_p_to_rmse": _rmse(branch["p_to"], branch["pt"]),
            "branch_q_to_rmse": _rmse(branch["q_to"], branch["qt"]),
            "branch_loading_rmse": _rmse(branch["physical_loading_native"], branch["physical_loading_ref"]),
            "bus_vm_rmse": _rmse(bus["vm_native"], bus["vm_ref"]),
            "bus_va_rmse": _rmse(bus["va_native"], bus["va_ref"]),
            "gen_pg_rmse": _rmse(gen["pg_native"], gen["pg_ref"]),
            "gen_qg_rmse": _rmse(gen["qg_native"], gen["qg_ref"]),
            "reactive_voltage_available": True,
        }
    native_branch = pd.read_csv(native_root / f"native_{evaluator}_branch_state.csv").assign(
        canonical_branch_id=lambda frame: frame["powermodels_branch_id"].astype(int) - 1
    )
    native_bus = pd.read_csv(native_root / f"native_{evaluator}_bus_state.csv")
    native_gen = pd.read_csv(native_root / f"native_{evaluator}_gen_state.csv").assign(
        canonical_generator_id=lambda frame: frame["gen_id"].astype(int) - 1
    )
    branch = native_branch.merge(reference_branch, on="canonical_branch_id", suffixes=("_native", "_ref"), validate="one_to_one")
    bus = native_bus.merge(reference_bus, on="bus_id", suffixes=("_native", "_ref"), validate="one_to_one")
    gen = native_gen.merge(reference_gen, on="canonical_generator_id", suffixes=("_native", "_ref"), validate="one_to_one")
    values: dict[str, object] = {
        "branch_p_from_rmse": _rmse(branch["pf_native"], branch["pf_ref"]),
        "branch_p_to_rmse": _rmse(branch["pt_native"], branch["pt_ref"]),
        "branch_loading_rmse": _rmse(branch["physical_loading_native"], branch["physical_loading_ref"]),
        "bus_va_rmse": _rmse(bus["va_native"], bus["va_ref"]),
        "gen_pg_rmse": _rmse(gen["pg_native"], gen["pg_ref"]),
        "reactive_voltage_available": evaluator == "ac",
    }
    if evaluator == "ac":
        values.update({
            "branch_q_from_rmse": _rmse(branch["qf_native"], branch["qf_ref"]),
            "branch_q_to_rmse": _rmse(branch["qt_native"], branch["qt_ref"]),
            "bus_vm_rmse": _rmse(bus["vm_native"], bus["vm_ref"]),
            "gen_qg_rmse": _rmse(gen["qg_native"], gen["qg_ref"]),
        })
    else:
        values.update({
            "branch_q_from_rmse": None, "branch_q_to_rmse": None,
            "bus_vm_rmse": None, "gen_qg_rmse": None,
        })
    return values


def _operating_diagnostics(identity, summary_a: dict[str, object]) -> dict[str, object]:
    branch = pd.read_csv(summary_a["branch_state_csv"])
    bus = pd.read_csv(summary_a["bus_state_csv"]).merge(
        identity.buses[["bus_id", "VMIN", "VMAX"]], on="bus_id", validate="one_to_one"
    )
    gen = pd.read_csv(summary_a["gen_state_csv"]).assign(
        canonical_generator_id=lambda frame: frame["gen_id"].astype(int) - 1
    ).merge(
        identity.generators[["canonical_generator_id", "PMIN", "PMAX", "QMIN", "QMAX"]],
        on="canonical_generator_id", validate="one_to_one",
    )
    vm_violation = np.maximum(bus["VMIN"] - bus["vm"], 0.0) + np.maximum(bus["vm"] - bus["VMAX"], 0.0)
    pg_violation = np.maximum(gen["PMIN"] / 100.0 - gen["pg"], 0.0) + np.maximum(gen["pg"] - gen["PMAX"] / 100.0, 0.0)
    qg_violation = np.maximum(gen["QMIN"] / 100.0 - gen["qg"], 0.0) + np.maximum(gen["qg"] - gen["QMAX"] / 100.0, 0.0)
    return {
        "maximum_ac_loading": float(branch["physical_loading"].max()),
        "ac_loading_gt_1_count": int((branch["physical_loading"] > 1.0 + 1e-8).sum()),
        "maximum_voltage_violation": float(vm_violation.max()),
        "maximum_pg_violation_pu": float(pg_violation.max()),
        "maximum_qg_violation_pu": float(qg_violation.max()),
    }


def run_references(
    *, config_path: str | Path, prepared_dir: str | Path, evaluator_dir: str | Path,
    output_dir: str | Path, julia: str = "julia", evaluator_filter: str | None = None,
    lambda_values: list[float] | None = None,
) -> dict[str, object]:
    config = load_config(config_path)
    identity = build_identity()
    prepared = Path(prepared_dir)
    evaluator_dir = Path(evaluator_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    proxy = pd.read_parquet(prepared / "proxy_components.parquet")
    p_env = dict(zip(proxy["canonical_branch_id"].astype(int), proxy["p_env"].astype(float), strict=True))
    manifest = json.loads((prepared / "input_manifest.json").read_text(encoding="utf-8"))
    r_base = float(manifest["r_base"])
    pd_by_load = dict(zip(identity.loads["canonical_load_id"].astype(int), identity.loads["pd_requested_mw"].astype(float), strict=True))
    project = Path(__file__).resolve().parent / "native_opf"
    script = project / "stage_k_native_opf.jl"
    case_path = Path(__file__).resolve().parents[1] / config["paths"]["case_file"]

    finalists = []
    for evaluator in ("gridsfm", "dc", "ac"):
        path = evaluator_dir / f"{evaluator}_finalists.parquet"
        if not path.exists():
            raise FileNotFoundError(f"sealed finalist table missing: {path}")
        frame = pd.read_parquet(path)
        frame["evaluator"] = evaluator
        finalists.append(frame)
    finalist_table = pd.concat(finalists, ignore_index=True)
    if evaluator_filter is not None:
        finalist_table = finalist_table.loc[finalist_table["evaluator"].eq(evaluator_filter)]
    if lambda_values is not None:
        allowed = {float(value) for value in lambda_values}
        frozen = {float(value) for value in config["search"]["lambda_r"]}
        if not allowed.issubset(frozen):
            raise ValueError("requested reference lambda is not present in the frozen config")
        finalist_table = finalist_table.loc[finalist_table["lambda_r"].astype(float).isin(allowed)]
    if finalist_table.empty:
        raise ValueError("reference filter selected no sealed finalists")
    if finalist_table.duplicated(["evaluator", "lambda_r"]).any():
        raise ValueError("finalists are not unique by evaluator/lambda")

    rows_a, rows_b, deltas = [], [], []
    b2_failure_count = 0
    for finalist in finalist_table.itertuples(index=False):
        evaluator = str(finalist.evaluator)
        lambda_r = float(finalist.lambda_r)
        offline = _parse_ids(finalist.offline_branch_ids)
        alpha = {int(k): float(v) for k, v in json.loads(finalist.alpha_effective_json).items()}
        if set(alpha) != set(pd_by_load):
            raise ValueError(f"{evaluator} finalist alpha vector is incomplete")
        run_dir = output / evaluator / f"lambda_{lambda_r:.6g}"
        run_dir.mkdir(parents=True, exist_ok=True)
        p_env_csv = run_dir / "p_env_powermodels.csv"
        alpha_csv = run_dir / "alpha_effective_powermodels.csv"
        _write_two_column(p_env_csv, ("powermodels_branch_id", "p_env"), p_env, offset=1)
        _write_two_column(alpha_csv, ("load_id", "alpha_effective"), alpha, offset=1)
        source_less = source_less_load_ids(identity, offline)
        common = [
            julia, f"--project={project}", str(script),
            "reference_a", str(case_path), ";".join(str(i + 1) for i in offline),
            ";".join(str(i + 1) for i in source_less), str(p_env_csv), str(alpha_csv),
            str(lambda_r), str(r_base), str(run_dir),
            str(config["solvers"]["reference_b_service_tolerance_mw"]),
        ]
        started = time.perf_counter()
        completed_a = subprocess.run(common, capture_output=True, text=True, check=False)
        elapsed_a = time.perf_counter() - started
        (run_dir / "reference_a_stdout.log").write_text(completed_a.stdout, encoding="utf-8")
        (run_dir / "reference_a_stderr.log").write_text(completed_a.stderr, encoding="utf-8")
        summary_a_path = run_dir / "reference_a_summary.json"
        if not summary_a_path.exists():
            raise RuntimeError(f"Reference A failed for {evaluator}, lambda={lambda_r}")
        summary_a = json.loads(summary_a_path.read_text(encoding="utf-8"))
        loading_a = _loading_from_branch_state(summary_a["branch_state_csv"], set(identity.l_trans))
        r_a = compute_r_norm(identity.l_trans, offline, p_env, loading_a, r_base)
        l_shed = compute_l_shed(pd_by_load, alpha)
        j_a = compute_j_trade(lambda_r, r_a, l_shed)
        row_a = {
            "evaluator": evaluator, "lambda_r": lambda_r,
            "topology_key": finalist.topology_key,
            "status": classify_solver_status(summary_a.get("termination_status", "")),
            "economic_cost": summary_a.get("objective"), "r_norm_reference_a": r_a,
            "l_shed_reference_a": l_shed, "j_trade_reference_a": j_a,
            "elapsed_seconds": elapsed_a, "solver_seconds": summary_a.get("solver_time_seconds"),
            "state_path": str(run_dir),
        }
        row_a.update(_state_discrepancy(evaluator, str(finalist.state_path), summary_a))
        row_a.update(_operating_diagnostics(identity, summary_a))
        rows_a.append(row_a)

        command_b = common.copy()
        command_b[3] = "reference_b"
        command_b[8] = "-"
        started = time.perf_counter()
        completed_b = subprocess.run(command_b, capture_output=True, text=True, check=False)
        elapsed_b = time.perf_counter() - started
        (run_dir / "reference_b_stdout.log").write_text(completed_b.stdout, encoding="utf-8")
        (run_dir / "reference_b_stderr.log").write_text(completed_b.stderr, encoding="utf-8")
        summary_b_path = run_dir / "reference_b_summary.json"
        if not summary_b_path.exists():
            raise RuntimeError(f"Reference B failed for {evaluator}, lambda={lambda_r}")
        summary_b = json.loads(summary_b_path.read_text(encoding="utf-8"))
        summary_b1 = json.loads((run_dir / "reference_b_b1_summary.json").read_text(encoding="utf-8"))
        summary_b2 = json.loads((run_dir / "reference_b_b2_summary.json").read_text(encoding="utf-8"))
        diagnostic_summary, b2_valid = _reference_b_diagnostic_summary(summary_b1, summary_b2)
        b2_failure_count += int(not b2_valid)
        loading_b = _loading_from_branch_state(diagnostic_summary["branch_state_csv"], set(identity.l_trans))
        r_b = compute_r_norm(identity.l_trans, offline, p_env, loading_b, r_base)
        service_b = float(summary_b["maximum_served_pu"]) / sum(value / 100.0 for value in pd_by_load.values())
        selected_service = 1.0 - l_shed
        row_b = {
            "evaluator": evaluator, "lambda_r": lambda_r,
            "topology_key": finalist.topology_key, "b1_status": summary_b["b1_status"],
            "b2_status": summary_b["b2_status"], "maximum_service_fraction": service_b,
            "selected_service_fraction": selected_service,
            "delta_s_b": service_b - selected_service, "elapsed_seconds": elapsed_b,
            "economic_cost_tiebreak": summary_b2.get("objective") if b2_valid else None,
            "diagnostic_state_source": "b2_economic_tiebreak" if b2_valid else "b1_maximum_service_fallback",
            "b2_solver_eligible": b2_valid,
            "r_norm_reference_b": r_b,
            "maximum_ac_loading": max(loading_b.values()) if loading_b else None,
            "ac_loading_gt_1_count": sum(value > 1.0 + 1e-8 for value in loading_b.values()),
            "state_path": str(run_dir),
        }
        rows_b.append(row_b)
        delta = reference_deltas(
            native_r_norm=float(finalist.r_norm), reference_a_r_norm=r_a,
            native_j_trade=float(finalist.j_trade), reference_a_j_trade=j_a,
            selected_service=selected_service, reference_b_max_service=service_b,
        )
        deltas.append({"evaluator": evaluator, "lambda_r": lambda_r, "topology_key": finalist.topology_key, **delta})

    pd.DataFrame(rows_a).to_parquet(output / "reference_a.parquet", index=False)
    pd.DataFrame(rows_b).to_parquet(output / "reference_b.parquet", index=False)
    pd.DataFrame(deltas).to_parquet(output / "reference_discrepancies.parquet", index=False)
    summary = {
        "status": "PASS" if b2_failure_count == 0 else "PASS_WITH_WARNINGS",
        "reference_a_rows": len(rows_a),
        "reference_b_rows": len(rows_b),
        "reference_b2_failure_count": b2_failure_count,
        "config_sha256": config["config_sha256"],
        "evaluators": sorted(finalist_table["evaluator"].astype(str).unique()),
        "lambda_values": sorted(finalist_table["lambda_r"].astype(float).unique()),
    }
    atomic_write_json(output / "reference_run_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--evaluator-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--evaluator", choices=["gridsfm", "dc", "ac"])
    parser.add_argument("--lambda-r", type=float, action="append")
    args = parser.parse_args()
    print(json.dumps(run_references(
        config_path=args.config, prepared_dir=args.prepared_dir,
        evaluator_dir=args.evaluator_dir, output_dir=args.output_dir, julia=args.julia,
        evaluator_filter=args.evaluator, lambda_values=args.lambda_r,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
