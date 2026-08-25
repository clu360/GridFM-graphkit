"""Run J9/J10 exact AC reference smoke on existing guided J8 finalists."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

WILDFIRE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(WILDFIRE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_TESTS_ROOT))

from stage_j_gridsfm_goc500.dc_economic_recourse import solve_fixed_topology_economic_dc_opf
from stage_j_gridsfm_goc500.goc500_adapter import build_goc500_identity, source_less_load_ids
from stage_j_gridsfm_goc500.gridsfm_evaluator import evaluate_gridsfm_candidate
from stage_j_gridsfm_goc500.load_service import compute_alpha_effective, compute_load_shedding
from stage_j_gridsfm_goc500.metrics import compute_ac_loading_two_ended, compute_j_trade
from stage_j_gridsfm_goc500.model_selection import resolve_model_selection
from stage_j_gridsfm_goc500.scenario_builder import load_baseline_loading_csv
from stage_j_gridsfm_goc500.schemas import EvaluationStatus, PacWeights


# Keep Julia-produced artifacts below Windows' legacy 260-character path limit.
# Human-readable method names remain in all result tables and summaries.
METHOD_OUTPUT_STUB = {
    "Guided-DC": "gdc",
    "Guided-GridSFM": "gsfm",
}


def _read_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _line_ids(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        text = str(value or "").strip()
        if not text:
            return ()
        return tuple(sorted(int(part) for part in text.replace(",", ";").split(";") if part.strip()))
    return tuple(sorted(int(v) for v in value))


def _line_key(values: Iterable[int]) -> str:
    return ";".join(str(int(value)) for value in sorted(int(v) for v in values))


def _p_env(path: Path, scenario_id: str) -> dict[int, float]:
    out = {}
    for row in _read_csv_dicts(path):
        if row["scenario_id"] == scenario_id:
            out[int(row["branch_id"])] = float(row["p_env"])
    return out


def _scenario(path: Path, scenario_id: str) -> dict[str, str]:
    for row in _read_csv_dicts(path):
        if row["scenario_id"] == scenario_id:
            return row
    raise KeyError(f"scenario {scenario_id} not found in {path}")


def _alpha_from_best(identity, best_topology: Mapping[str, object]) -> dict[int, float]:
    alpha = {load.canonical_load_id: 1.0 for load in identity.loads}
    selected = json.loads(str(best_topology.get("best_alpha_selected") or "{}"))
    for load_id, value in selected.items():
        alpha[int(load_id)] = float(value)
    return alpha


def _load_finalist_inputs(root: Path, manifest_path: Path | None) -> list[dict[str, object]]:
    """Load explicit finalist entries, with the original J8 smoke as fallback."""

    if manifest_path is None:
        return [
            {
                "method": "Guided-DC",
                "backend": "dc",
                "artifact_stub": "gdc",
                "payload": _read_json(root / "gdc" / "j8_guided_dc_summary.json"),
            },
            {
                "method": "Guided-GridSFM",
                "backend": "gridsfm",
                "artifact_stub": "gsfm",
                "payload": _read_json(root / "gsfm" / "j8_guided_gridsfm_summary.json"),
            },
        ]

    manifest = _read_json(manifest_path)
    entries = manifest.get("finalists", manifest) if isinstance(manifest, dict) else manifest
    if not isinstance(entries, list) or not entries:
        raise ValueError("--finalists-manifest must contain a nonempty finalists list")
    loaded: list[dict[str, object]] = []
    seen_stubs: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each finalist manifest entry must be an object")
        method = str(entry["method"])
        backend = str(entry["backend"])
        if backend not in {"dc", "gridsfm"}:
            raise ValueError(f"unsupported finalist backend {backend!r} for {method}")
        artifact_stub = str(entry["artifact_stub"])
        if not artifact_stub.isascii() or not artifact_stub.replace("_", "").isalnum() or artifact_stub in seen_stubs:
            raise ValueError(f"artifact_stub must be unique ASCII alphanumeric/underscore text, got {artifact_stub!r}")
        best = entry.get("best_topology")
        if not isinstance(best, dict):
            raise ValueError(f"finalist {method!r} is missing best_topology")
        lambda_r = float(entry["lambda_r"])
        payload = {"lambda_r": lambda_r, "best_topology": best}
        loaded.append({
            "method": method,
            "backend": backend,
            "artifact_stub": artifact_stub,
            "payload": payload,
            "finalist_model_selection": entry.get(
                "model_selection", best.get("model_selection", "")
            ),
            "finalist_model_variant": entry.get(
                "model_variant", best.get("model_variant", "")
            ),
            "finalist_checkpoint_path": best.get("checkpoint_path", ""),
            "finalist_checkpoint_sha256": best.get("checkpoint_sha256", ""),
        })
        seen_stubs.add(artifact_stub)
    return loaded


def _write_alpha(
    path: Path,
    values: Mapping[int, float],
    column: str = "alpha",
    metadata: Mapping[str, object] | None = None,
) -> None:
    metadata = metadata or {}
    rows = [
        {"load_id": int(load_id), column: float(values[load_id]), **metadata}
        for load_id in sorted(values)
    ]
    _write_rows(path, rows)


def _generator_id_map(raw_case) -> list[int]:
    metadata = raw_case.get("metadata", {})
    values = metadata.get("gen_id_map")
    if values is None:
        return list(range(len(raw_case["grid"]["nodes"].get("generator", []))))
    return [int(value) for value in values]


def _write_gen_start(path: Path, pg_by_gen_id: Mapping[int, float], qg_by_gen_id: Mapping[int, float] | None = None) -> None:
    rows = []
    qg_by_gen_id = qg_by_gen_id or {}
    for gen_id in sorted(pg_by_gen_id):
        rows.append({"gen_id": gen_id, "pg": pg_by_gen_id[gen_id], "qg": qg_by_gen_id.get(gen_id, "")})
    _write_rows(path, rows)


def _write_bus_start(path: Path, theta_by_bus: Mapping[int, float], v_by_bus: Mapping[int, float] | None = None) -> None:
    rows = []
    v_by_bus = v_by_bus or {}
    for bus_id in sorted(theta_by_bus):
        rows.append({"bus_id": bus_id, "vm": v_by_bus.get(bus_id, ""), "va": theta_by_bus[bus_id]})
    _write_rows(path, rows)


def _load_float_table(path: Path, key: str) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for row in _read_csv_dicts(path):
        item = {}
        for col, value in row.items():
            if col == key:
                continue
            text = str(value or "").strip()
            if text:
                item[col] = float(text)
        out[int(row[key])] = item
    return out


def _run_julia(
    *,
    julia_exe: Path,
    julia_depot_path: Path,
    script: Path,
    mode: str,
    case_path: Path,
    alpha_csv: Path,
    offline_branch_ids: Iterable[int],
    source_less_load_ids_: Iterable[int],
    output_dir: Path,
    start_dir: Path | None = None,
    load_shed_limit: float | None = None,
    timeout_seconds: int = 900,
) -> tuple[bool, str, str, float]:
    cmd = [
        str(julia_exe),
        str(script),
        mode,
        str(case_path),
        str(alpha_csv),
        _line_key(offline_branch_ids),
        _line_key(source_less_load_ids_),
        str(output_dir),
        "" if start_dir is None else str(start_dir),
    ]
    if load_shed_limit is not None:
        cmd.append(str(load_shed_limit))
    env = dict(os.environ)
    env["JULIA_DEPOT_PATH"] = str(julia_depot_path)
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(Path.cwd()), env=env, text=True, capture_output=True, timeout=timeout_seconds)
    return proc.returncode == 0, proc.stdout, proc.stderr, time.time() - start


def _risk_from_ac_branch_csv(branch_csv: Path, p_env: Mapping[int, float], r_base: float) -> tuple[float | None, float | None, int | None]:
    if not branch_csv.exists():
        return None, None, None
    rows = _read_csv_dicts(branch_csv)
    r_raw = 0.0
    max_loading = 0.0
    num_loading_gt_1 = 0
    for row in rows:
        line_id = int(row["branch_id"])
        loading = float(row["ac_loading"])
        r_raw += float(p_env.get(line_id, 0.0)) * loading**2
        max_loading = max(max_loading, loading)
        num_loading_gt_1 += int(loading > 1.0)
    return r_raw / r_base if r_base > 0 else None, max_loading, num_loading_gt_1


def _reference_b_l_shed(load_csv: Path, identity) -> float | None:
    if not load_csv.exists():
        return None
    served = sum(float(row["pd"]) for row in _read_csv_dicts(load_csv))
    total = sum(float(load.pd_pre) for load in identity.loads)
    if total <= 0:
        return None
    return max(0.0, 1.0 - served / total)


def _generator_ranges(raw_case) -> tuple[dict[int, float], dict[int, float], list[dict[str, object]]]:
    gen_ids = _generator_id_map(raw_case)
    pg_scale = {}
    qg_scale = {}
    rows = []
    for idx, row in enumerate(raw_case["grid"]["nodes"].get("generator", [])):
        gen_id = gen_ids[idx]
        pmin, pmax = float(row[2]), float(row[3])
        qmax, qmin = float(row[4]), float(row[5])
        pg_scale[gen_id] = pmax - pmin
        qg_scale[gen_id] = qmax - qmin
        rows.append(
            {
                "generator_id": gen_id,
                "bus_id": int(raw_case["metadata"]["gen_bus_map"][idx]),
                "Pg_min_original": pmin,
                "Pg_max_original": pmax,
                "Qg_min_original": qmin,
                "Qg_max_original": qmax,
                "Pg_scale": pmax - pmin,
                "Qg_scale": qmax - qmin,
            }
        )
    return pg_scale, qg_scale, rows


def _bus_voltage_scales(raw_case) -> dict[int, float]:
    bus_ids = [int(v) for v in raw_case["metadata"]["bus_id_map"]]
    return {bus_id: float(row[3]) - float(row[2]) for bus_id, row in zip(bus_ids, raw_case["grid"]["nodes"]["bus"])}


def _ref_bus(raw_case) -> int:
    bus_ids = [int(v) for v in raw_case["metadata"]["bus_id_map"]]
    for bus_id, row in zip(bus_ids, raw_case["grid"]["nodes"]["bus"]):
        if int(row[1]) == 3:
            return bus_id
    return bus_ids[0]


def _family_metrics(native: Mapping[int, float], ac: Mapping[int, float], scale: Mapping[int, float] | None = None, *, epsilon: float = 1e-9) -> dict[str, float | int | None]:
    common = sorted(set(native).intersection(ac))
    if not common:
        return {"count": 0, "mae": None, "rmse": None, "max": None, "nmae": None, "nrmse": None, "nmax": None, "degenerate_count": 0}
    raw = np.asarray([float(native[k]) - float(ac[k]) for k in common], dtype=float)
    out: dict[str, float | int | None] = {
        "count": len(common),
        "mae": float(np.mean(np.abs(raw))),
        "rmse": float(np.sqrt(np.mean(raw**2))),
        "max": float(np.max(np.abs(raw))),
        "nmae": None,
        "nrmse": None,
        "nmax": None,
        "degenerate_count": 0,
    }
    if scale is not None:
        valid = [k for k in common if abs(float(scale.get(k, 0.0))) > epsilon]
        out["degenerate_count"] = len(common) - len(valid)
        if valid:
            norm = np.asarray([(float(native[k]) - float(ac[k])) / float(scale[k]) for k in valid], dtype=float)
            out.update({"nmae": float(np.mean(np.abs(norm))), "nrmse": float(np.sqrt(np.mean(norm**2))), "nmax": float(np.max(np.abs(norm)))})
    return out


def _theta_metrics(native: Mapping[int, float], ac: Mapping[int, float], ref_bus: int) -> dict[str, float | int | None]:
    if ref_bus not in native or ref_bus not in ac:
        return {"count": 0, "mae": None, "rmse": None, "max": None, "ref_bus": ref_bus}
    n_aligned = {k: float(v) - float(native[ref_bus]) for k, v in native.items()}
    a_aligned = {k: float(v) - float(ac[ref_bus]) for k, v in ac.items()}
    out = _family_metrics(n_aligned, a_aligned)
    out["ref_bus"] = ref_bus
    return out


def _collect_state_metrics(method: str, native_state: Mapping[str, Mapping[int, float]], ac_dir: Path, raw_case) -> list[dict[str, object]]:
    pg_scale, qg_scale, _ = _generator_ranges(raw_case)
    v_scale = _bus_voltage_scales(raw_case)
    ref_bus = _ref_bus(raw_case)
    ac_gen = _load_float_table(ac_dir / "reference_a_gen_dispatch.csv", "gen_id")
    ac_bus = _load_float_table(ac_dir / "reference_a_bus_state.csv", "bus_id")
    ac_branch = _load_float_table(ac_dir / "reference_a_branch_state.csv", "branch_id")
    rows = []

    def add(family: str, metrics: Mapping[str, object], units: str, normalization: str) -> None:
        row = {"method": method, "metric_family": family, "raw_units": units, "normalization": normalization}
        row.update(metrics)
        rows.append(row)

    add("Pg", _family_metrics(native_state.get("pg", {}), {k: v["pg"] for k, v in ac_gen.items()}, pg_scale), "p.u.", "Pgmax-Pgmin original")
    if native_state.get("qg"):
        add("Qg", _family_metrics(native_state.get("qg", {}), {k: v["qg"] for k, v in ac_gen.items()}, qg_scale), "p.u.", "Qgmax-Qgmin original")
    else:
        add("Qg", {"count": 0, "mae": None, "rmse": None, "max": None, "nmae": None, "nrmse": None, "nmax": None, "degenerate_count": 0}, "N/A", "N/A")
    if native_state.get("v"):
        add("V", _family_metrics(native_state.get("v", {}), {k: v["vm"] for k, v in ac_bus.items()}, v_scale), "p.u.", "Vmax-Vmin original")
    else:
        add("V", {"count": 0, "mae": None, "rmse": None, "max": None, "nmae": None, "nrmse": None, "nmax": None, "degenerate_count": 0}, "N/A", "N/A")
    add("theta", _theta_metrics(native_state.get("theta", {}), {k: v["va"] for k, v in ac_bus.items()}, ref_bus), "rad", "reference-bus aligned")

    rate_a = {k: v["rate_a"] for k, v in ac_branch.items()}
    for native_key, ac_key, label in (("pij", "pf", "Pij"), ("qij", "qf", "Qij"), ("pji", "pt", "Pji"), ("qji", "qt", "Qji")):
        native = native_state.get(native_key, {})
        if not native:
            add(label, {"count": 0, "mae": None, "rmse": None, "max": None, "nmae": None, "nrmse": None, "nmax": None, "degenerate_count": 0}, "N/A", "N/A")
        else:
            add(label, _family_metrics(native, {k: v[ac_key] for k, v in ac_branch.items()}, rate_a), "p.u.", "rateA")
    return rows


def _make_native_dc(raw_case, identity, offline, alpha_requested, gen_ids) -> tuple[dict[str, dict[int, float]], float | None, str]:
    result = solve_fixed_topology_economic_dc_opf(raw_case=raw_case, identity=identity, offline_branch_ids=offline, alpha_requested=alpha_requested)
    if result.evaluation_status is not EvaluationStatus.OK:
        return {}, None, result.message
    pg = {gen_ids[idx]: value for idx, value in result.pg_by_generator.items() if idx < len(gen_ids)}
    return (
        {
            "pg": pg,
            "theta": dict(result.theta_by_bus),
            "pij": dict(result.flow_by_line),
            "pji": {line_id: -value for line_id, value in result.flow_by_line.items()},
        },
        result.objective_cost,
        result.message,
    )


def _make_native_gridsfm(raw_case, identity, model, offline, alpha_requested, p_env, r_base, lambda_r, weights, work_dir, gen_ids):
    result = evaluate_gridsfm_candidate(
        raw_case=raw_case,
        identity=identity,
        model=model,
        offline_branch_ids=offline,
        alpha_requested=alpha_requested,
        p_env_by_line=p_env,
        r_base=r_base,
        lambda_r=lambda_r,
        weights=weights,
        work_dir=work_dir,
    )
    pg = {gen_ids[idx]: value for idx, value in result.pg_by_generator.items() if idx < len(gen_ids)}
    qg = {gen_ids[idx]: value for idx, value in result.qg_by_generator.items() if idx < len(gen_ids)}
    return (
        {
            "pg": pg,
            "qg": qg,
            "v": dict(result.v_by_bus),
            "theta": dict(result.theta_by_bus),
            "pij": dict(result.p_from_by_line),
            "qij": dict(result.q_from_by_line),
            "pji": dict(result.p_to_by_line),
            "qji": dict(result.q_to_by_line),
        },
        result,
    )


def _write_native_state(out_dir: Path, state: Mapping[str, Mapping[int, float]]) -> None:
    if state.get("pg"):
        _write_gen_start(out_dir / "gen_start.csv", state["pg"], state.get("qg", {}))
    if state.get("theta"):
        _write_bus_start(out_dir / "bus_start.csv", state["theta"], state.get("v", {}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--j8-root", default="experiments/test/wildfire_tests/goc_500_results/stage_j/j8s/s1_l08")
    parser.add_argument("--input-dir", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\inputs\case500_goc_e0")
    parser.add_argument("--gridsfm-root", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\repos\GridSFM")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument("--model-selection", choices=["frozen", "ft"], default="frozen")
    parser.add_argument("--case-path", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\repos\pglib-opf\pglib_opf_case500_goc.m")
    parser.add_argument("--julia-exe", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\tools\julia-1.10.11\bin\julia.exe")
    parser.add_argument("--julia-depot-path", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\julia_depot")
    parser.add_argument("--pac-freeze-json", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\pac_calibration\v001\PAC_WEIGHT_FREEZE.json")
    parser.add_argument("--scenario-id", default="J-S1")
    parser.add_argument("--output-dir", default="experiments/test/wildfire_tests/goc_500_results/stage_j/j9_j10_smoke/s1_l08")
    parser.add_argument(
        "--finalists-manifest",
        default=None,
        help="Optional JSON list of explicit finalists, including TH-GridSFM rows.",
    )
    parser.add_argument("--run-reference-b", action="store_true")
    parser.add_argument(
        "--warm-starts-only",
        action="store_true",
        help="Rebuild all warm-start variants from an existing successful Reference A result.",
    )
    parser.add_argument(
        "--full-gridsfm-only",
        action="store_true",
        help=(
            "Evaluate only the selected checkpoint's full Pg/Qg/V/theta start and write "
            "a checkpoint-specific additive table."
        ),
    )
    parser.add_argument(
        "--warm-start-id",
        choices=["frozen", "m0", "m1", "m2", "m3"],
        help=(
            "Checkpoint-specific identifier for additive full-start artifacts. "
            "Defaults to frozen or ft for backward compatibility."
        ),
    )
    parser.add_argument(
        "--method-filter",
        action="append",
        default=[],
        help="Restrict --full-gridsfm-only to an exact finalist method name; repeat as needed.",
    )
    parser.add_argument(
        "--reference-b-service-tolerance-pd-units",
        type=float,
        default=1e-5,
        help="Absolute B1-to-B2 permitted load-shed increase in exact-model Pd units.",
    )
    parser.add_argument(
        "--reference-b-service-verification-tolerance-pd-units",
        type=float,
        default=1e-5,
        help="Additional absolute feasibility tolerance when verifying the B2 service-lock constraint.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.reference_b_service_tolerance_pd_units <= 0.0 or args.reference_b_service_verification_tolerance_pd_units <= 0.0:
        parser.error("Reference B service tolerances must be positive")
    if args.warm_starts_only and args.run_reference_b:
        parser.error("--warm-starts-only cannot be combined with --run-reference-b")
    if args.full_gridsfm_only and (args.warm_starts_only or args.run_reference_b):
        parser.error("--full-gridsfm-only cannot be combined with other reference modes")
    if args.method_filter and not args.full_gridsfm_only:
        parser.error("--method-filter requires --full-gridsfm-only")
    if args.warm_start_id and not args.full_gridsfm_only:
        parser.error("--warm-start-id requires --full-gridsfm-only")

    root = Path(args.j8_root).resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_selection = resolve_model_selection(
        args.model_selection,
        gridsfm_root=Path(args.gridsfm_root).expanduser().resolve(),
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        expected_sha256=args.expected_checkpoint_sha256,
    )
    model_metadata = model_selection.as_dict()

    raw_case = _read_json(Path(args.gridsfm_root) / "model" / "samples" / "case500_goc.pyg.json")
    gridsfm_model_root = Path(args.gridsfm_root) / "model"
    if str(gridsfm_model_root) not in sys.path:
        sys.path.insert(0, str(gridsfm_model_root))
    candidate_ids = [int(row["branch_id"]) for row in _read_csv_dicts(input_dir / "stage_j_candidate_line_scores.csv")]
    identity = build_goc500_identity(raw_case, candidate_branch_ids=candidate_ids)
    gen_ids = _generator_id_map(raw_case)
    p_env = _p_env(input_dir / "stage_j_p_env_by_scenario.csv", args.scenario_id)
    scenario = _scenario(input_dir / "stage_j_scenario_register.csv", args.scenario_id)
    r_base = float(scenario["r_base"])
    baseline_loading = load_baseline_loading_csv(Path(_read_json(input_dir / "stage_j_input_manifest.json")["baseline_loading_csv"]))
    weights_json = _read_json(Path(args.pac_freeze_json).expanduser().resolve())["frozen_weights"]
    weights = PacWeights(
        rho_phys=float(weights_json["rho_phys"]),
        w_op=float(weights_json["w_op"]),
        w_ac=float(weights_json["w_ac"]),
        w_model=float(weights_json["w_model"]),
    )
    script = Path(__file__).with_name("stage_j_ac_reference.jl")
    pg_scale, qg_scale, generator_rows = _generator_ranges(raw_case)
    _write_rows(output_dir / "generator_identity_ranges.csv", generator_rows)

    from gridsfm import load_model

    model = load_model(str(model_selection.checkpoint_path), device="cpu")

    finalists_manifest = Path(args.finalists_manifest).expanduser().resolve() if args.finalists_manifest else None
    method_inputs = _load_finalist_inputs(root, finalists_manifest)
    if args.method_filter:
        requested_methods = set(args.method_filter)
        method_inputs = [row for row in method_inputs if row["method"] in requested_methods]
        observed_methods = {str(row["method"]) for row in method_inputs}
        missing_methods = requested_methods - observed_methods
        if missing_methods:
            raise ValueError(f"requested finalist methods are absent: {sorted(missing_methods)}")
    summary_rows: list[dict[str, object]] = []
    fidelity_rows: list[dict[str, object]] = []
    warm_rows: list[dict[str, object]] = []
    ref_b_rows: list[dict[str, object]] = []

    for finalist in method_inputs:
        method = str(finalist["method"])
        backend = str(finalist["backend"])
        payload = finalist["payload"]
        if not isinstance(payload, Mapping):
            raise TypeError(f"invalid finalist payload for {method}")
        best = payload["best_topology"]
        if not isinstance(best, Mapping):
            raise TypeError(f"invalid best_topology for {method}")
        method_stub = str(finalist["artifact_stub"])
        method_dir = output_dir / method_stub
        method_dir.mkdir(parents=True, exist_ok=True)
        offline = _line_ids(best["topology_id"])
        alpha_requested = _alpha_from_best(identity, best)
        source_less = source_less_load_ids(identity, offline)
        alpha_effective = compute_alpha_effective([load.canonical_load_id for load in identity.loads], alpha_requested, source_less)

        if args.full_gridsfm_only:
            warm_id = args.warm_start_id or (
                "frozen" if model_selection.model_selection == "frozen" else "ft"
            )
            warm_type = f"gridsfm_{warm_id}_full_warm"
            crossed_dir = method_dir / "crossed_warm" / str(model_selection.model_variant)
            crossed_dir.mkdir(parents=True, exist_ok=True)
            crossed_alpha = crossed_dir / "alpha_effective_full.csv"
            _write_alpha(crossed_alpha, alpha_effective, "alpha", model_metadata)

            reference_a_dir = method_dir / "ra"
            ref_a_summary_path = reference_a_dir / "reference_a_summary.json"
            ref_a_summary = _read_json(ref_a_summary_path) if ref_a_summary_path.is_file() else {}
            reference_a_ok = (
                str(ref_a_summary.get("termination_status")) in {"LOCALLY_SOLVED", "OPTIMAL"}
                and (reference_a_dir / "reference_a_gen_dispatch.csv").is_file()
                and (reference_a_dir / "reference_a_bus_state.csv").is_file()
            )
            if not reference_a_ok:
                raise RuntimeError(
                    f"full-GridSFM-only mode requires successful Reference A artifacts: {reference_a_dir}"
                )

            gridsfm_start_t0 = time.perf_counter()
            g_state, _ = _make_native_gridsfm(
                raw_case,
                identity,
                model,
                offline,
                alpha_requested,
                p_env,
                r_base,
                float(payload["lambda_r"]),
                weights,
                crossed_dir / "native_gridsfm_eval",
                gen_ids,
            )
            missing_full_families = [
                family for family in ("pg", "qg", "v", "theta") if not g_state.get(family)
            ]
            if missing_full_families:
                raise RuntimeError(
                    f"GridSFM full warm start is missing state families: {missing_full_families}"
                )
            start_dir = crossed_dir / "state"
            start_dir.mkdir(exist_ok=True)
            _write_native_state(start_dir, g_state)
            gridsfm_start_seconds = time.perf_counter() - gridsfm_start_t0

            solve_dir = crossed_dir / "solve"
            ok_w, out_w, err_w, wall_w = _run_julia(
                julia_exe=Path(args.julia_exe),
                julia_depot_path=Path(args.julia_depot_path),
                script=script,
                mode="reference_a",
                case_path=Path(args.case_path),
                alpha_csv=crossed_alpha,
                offline_branch_ids=offline,
                source_less_load_ids_=source_less,
                output_dir=solve_dir,
                start_dir=start_dir,
                timeout_seconds=args.timeout_seconds,
            )
            warm_summary_path = solve_dir / "reference_a_summary.json"
            warm_summary = _read_json(warm_summary_path) if warm_summary_path.is_file() else {}
            finalist_selection = str(finalist.get("finalist_model_selection") or "")
            if not finalist_selection:
                finalist_selection = "dc" if backend == "dc" else "frozen"
            finalist_variant = str(finalist.get("finalist_model_variant") or "")
            if not finalist_variant:
                finalist_variant = "guided_dc" if finalist_selection == "dc" else "released_v1_1"
            warm_rows.append(
                {
                    "method": method,
                    "warm_start_type": warm_type,
                    "same_reference_a_instance": True,
                    "status": warm_summary.get("termination_status", "missing"),
                    "solver_runtime_seconds": warm_summary.get("ipopt_solve_time_seconds"),
                    "wall_seconds": wall_w,
                    "start_construction_seconds": gridsfm_start_seconds,
                    "end_to_end_seconds": wall_w + gridsfm_start_seconds,
                    "iteration_count": warm_summary.get("iteration_count"),
                    "start_payload": "Pg,Qg,V,theta",
                    "objective": warm_summary.get("objective"),
                    "solver_start_source": warm_summary.get("start_source"),
                    "start_bus_count": warm_summary.get("start_bus_count"),
                    "start_gen_count": warm_summary.get("start_gen_count"),
                    "start_vm_min": warm_summary.get("start_vm_min"),
                    "start_vm_max": warm_summary.get("start_vm_max"),
                    "start_va_abs_max": warm_summary.get("start_va_abs_max"),
                    "start_pg_midpoint_max_abs_error": warm_summary.get(
                        "start_pg_midpoint_max_abs_error"
                    ),
                    "start_qg_abs_max": warm_summary.get("start_qg_abs_max"),
                    "finalist_model_selection": finalist_selection,
                    "finalist_model_variant": finalist_variant,
                    "finalist_checkpoint_path": finalist.get("finalist_checkpoint_path", ""),
                    "finalist_checkpoint_sha256": finalist.get(
                        "finalist_checkpoint_sha256", ""
                    ),
                    "warm_start_model_selection": model_selection.model_selection,
                    "warm_start_model_variant": model_selection.model_variant,
                    "warm_start_checkpoint_path": model_selection.checkpoint_path,
                    "warm_start_checkpoint_sha256": model_selection.checkpoint_sha256,
                    "stdout": out_w.strip(),
                    "stderr": err_w.strip(),
                    "subprocess_success": ok_w,
                }
            )
            continue

        alpha_requested_csv = method_dir / "alpha_requested_full.csv"
        alpha_effective_csv = method_dir / "alpha_effective_full.csv"
        _write_alpha(alpha_requested_csv, alpha_requested, "alpha_requested", model_metadata)
        _write_alpha(alpha_effective_csv, alpha_effective, "alpha", model_metadata)

        if backend == "dc":
            native_state, native_cost, native_message = _make_native_dc(raw_case, identity, offline, alpha_requested, gen_ids)
        else:
            native_state, gsfm_result = _make_native_gridsfm(
                raw_case,
                identity,
                model,
                offline,
                alpha_requested,
                p_env,
                r_base,
                float(payload["lambda_r"]),
                weights,
                method_dir / "native_gridsfm_eval",
                gen_ids,
            )
            native_cost = _cost_from_pg(raw_case, native_state.get("pg", {}))
            native_message = gsfm_result.message
        native_dir = method_dir / "native_state"
        native_dir.mkdir(exist_ok=True)
        _write_native_state(native_dir, native_state)

        reference_a_dir = method_dir / "ra"
        if args.warm_starts_only:
            ref_a_summary_path = reference_a_dir / "reference_a_summary.json"
            ref_a_summary = _read_json(ref_a_summary_path) if ref_a_summary_path.is_file() else {}
            ok = (
                str(ref_a_summary.get("termination_status")) in {"LOCALLY_SOLVED", "OPTIMAL"}
                and (reference_a_dir / "reference_a_gen_dispatch.csv").is_file()
                and (reference_a_dir / "reference_a_bus_state.csv").is_file()
            )
            if not ok:
                raise RuntimeError(f"warm-start-only mode requires successful Reference A artifacts: {reference_a_dir}")
            stdout, stderr, wall = "", "", 0.0
        else:
            ok, stdout, stderr, wall = _run_julia(
                julia_exe=Path(args.julia_exe),
                julia_depot_path=Path(args.julia_depot_path),
                script=script,
                mode="reference_a",
                case_path=Path(args.case_path),
                alpha_csv=alpha_effective_csv,
                offline_branch_ids=offline,
                source_less_load_ids_=source_less,
                output_dir=reference_a_dir,
                timeout_seconds=args.timeout_seconds,
            )
            ref_a_summary = _read_json(reference_a_dir / "reference_a_summary.json") if (reference_a_dir / "reference_a_summary.json").exists() else {}
        r_norm_ac, max_ac_loading, num_ac_over = _risk_from_ac_branch_csv(reference_a_dir / "reference_a_branch_state.csv", p_env, r_base)
        l_shed_native = float(best["l_shed_total"])
        j_true_a = compute_j_trade(float(payload["lambda_r"]), r_norm_ac, l_shed_native) if r_norm_ac is not None else None
        delta_r = float(best["r_norm"]) - r_norm_ac if r_norm_ac is not None else None
        delta_j = float(best["j_trade"]) - j_true_a if j_true_a is not None else None
        if not args.warm_starts_only:
            summary_rows.append(
                {
                "method": method,
                "scenario_id": args.scenario_id,
                "topology_id": _line_key(offline),
                "lambda_r": payload["lambda_r"],
                "reference_a_status": ref_a_summary.get("termination_status", "missing"),
                "reference_a_runtime_seconds": ref_a_summary.get("runtime_seconds"),
                "reference_a_wall_seconds": wall,
                "reference_a_objective": ref_a_summary.get("objective"),
                "native_r_norm": best["r_norm"],
                "reference_a_r_norm_ac": r_norm_ac,
                "delta_r_norm_native_minus_ac": delta_r,
                "native_l_shed_total": l_shed_native,
                "reference_a_l_shed_total": l_shed_native,
                "native_j_trade": best["j_trade"],
                "reference_a_j_true": j_true_a,
                "delta_j_true_native_minus_ac": delta_j,
                "lambda_times_delta_r": float(payload["lambda_r"]) * delta_r if delta_r is not None else None,
                "max_ac_loading": max_ac_loading,
                "num_ac_loading_gt_1": num_ac_over,
                "native_cost": native_cost,
                "native_message": native_message,
                "julia_stdout": stdout.strip(),
                "julia_stderr": stderr.strip(),
                }
            )
        if ok and str(ref_a_summary.get("termination_status")) in {"LOCALLY_SOLVED", "OPTIMAL"}:
            if not args.warm_starts_only:
                fidelity_rows.extend(_collect_state_metrics(method, native_state, reference_a_dir, raw_case))

            gt_start_dir = method_dir / "wgt"
            gt_start_dir.mkdir(exist_ok=True)
            shutil.copyfile(reference_a_dir / "reference_a_gen_dispatch.csv", gt_start_dir / "gen_start.csv")
            shutil.copyfile(reference_a_dir / "reference_a_bus_state.csv", gt_start_dir / "bus_start.csv")
            warm_specs = [("cold_start", "cold", None, 0.0), ("gt_warm", "gt", gt_start_dir, 0.0)]
            dc_start_t0 = time.perf_counter()
            dc_state, _, _ = _make_native_dc(raw_case, identity, offline, alpha_requested, gen_ids)
            (method_dir / "wdc").mkdir(exist_ok=True)
            _write_native_state(method_dir / "wdc", dc_state)
            dc_start_seconds = time.perf_counter() - dc_start_t0
            warm_specs.insert(1, ("dc_partial_warm", "dc", method_dir / "wdc", dc_start_seconds))

            gridsfm_start_t0 = time.perf_counter()
            g_state, _ = _make_native_gridsfm(
                raw_case,
                identity,
                model,
                offline,
                alpha_requested,
                p_env,
                r_base,
                float(payload["lambda_r"]),
                weights,
                method_dir / "wge",
                gen_ids,
            )
            gridsfm_partial_dir = method_dir / "wgs"
            gridsfm_partial_dir.mkdir(exist_ok=True)
            _write_gen_start(gridsfm_partial_dir / "gen_start.csv", g_state.get("pg", {}))
            _write_bus_start(gridsfm_partial_dir / "bus_start.csv", g_state.get("theta", {}))
            gridsfm_full_dir = method_dir / "wgf"
            gridsfm_full_dir.mkdir(exist_ok=True)
            missing_full_families = [
                family for family in ("pg", "qg", "v", "theta") if not g_state.get(family)
            ]
            if missing_full_families:
                raise RuntimeError(
                    f"GridSFM full warm start is missing state families: {missing_full_families}"
                )
            _write_native_state(gridsfm_full_dir, g_state)
            gridsfm_start_seconds = time.perf_counter() - gridsfm_start_t0
            warm_specs.insert(2, ("gridsfm_partial_warm", "gsfm", gridsfm_partial_dir, gridsfm_start_seconds))
            warm_specs.insert(3, ("gridsfm_full_warm", "gsfm_full", gridsfm_full_dir, gridsfm_start_seconds))
            for warm_type, warm_stub, start_dir, prep_time in warm_specs:
                wdir = method_dir / "ws" / warm_stub
                ok_w, out_w, err_w, wall_w = _run_julia(
                    julia_exe=Path(args.julia_exe),
                    julia_depot_path=Path(args.julia_depot_path),
                    script=script,
                    mode="reference_a",
                    case_path=Path(args.case_path),
                    alpha_csv=alpha_effective_csv,
                    offline_branch_ids=offline,
                    source_less_load_ids_=source_less,
                    output_dir=wdir,
                    start_dir=start_dir,
                    timeout_seconds=args.timeout_seconds,
                )
                warm_summary = _read_json(wdir / "reference_a_summary.json") if (wdir / "reference_a_summary.json").exists() else {}
                warm_rows.append(
                    {
                        "method": method,
                        "warm_start_type": warm_type,
                        "same_reference_a_instance": True,
                        "status": warm_summary.get("termination_status", "missing"),
                        "solver_runtime_seconds": warm_summary.get("ipopt_solve_time_seconds"),
                        "wall_seconds": wall_w,
                        "start_construction_seconds": prep_time,
                        "end_to_end_seconds": wall_w + prep_time,
                        "iteration_count": warm_summary.get("iteration_count"),
                        "start_payload": (
                            "Pg,theta"
                            if warm_type in {"dc_partial_warm", "gridsfm_partial_warm"}
                            else "Pg,Qg,V,theta"
                            if warm_type == "gridsfm_full_warm"
                            else "exact_Pg,Qg,V,theta"
                            if warm_type == "gt_warm"
                            else "V=1,theta=0,Pg=(Pmin+Pmax)/2,Qg=0"
                        ),
                        "objective": warm_summary.get("objective"),
                        "solver_start_source": warm_summary.get("start_source"),
                        "start_bus_count": warm_summary.get("start_bus_count"),
                        "start_gen_count": warm_summary.get("start_gen_count"),
                        "start_vm_min": warm_summary.get("start_vm_min"),
                        "start_vm_max": warm_summary.get("start_vm_max"),
                        "start_va_abs_max": warm_summary.get("start_va_abs_max"),
                        "start_pg_midpoint_max_abs_error": warm_summary.get(
                            "start_pg_midpoint_max_abs_error"
                        ),
                        "start_qg_abs_max": warm_summary.get("start_qg_abs_max"),
                        "stdout": out_w.strip(),
                        "stderr": err_w.strip(),
                    }
                )

        if args.run_reference_b and not args.warm_starts_only:
            b1_dir = method_dir / "rb1"
            ok_b1, out_b1, err_b1, wall_b1 = _run_julia(
                julia_exe=Path(args.julia_exe),
                julia_depot_path=Path(args.julia_depot_path),
                script=script,
                mode="reference_b_b1",
                case_path=Path(args.case_path),
                alpha_csv=alpha_effective_csv,
                offline_branch_ids=offline,
                source_less_load_ids_=source_less,
                output_dir=b1_dir,
                timeout_seconds=args.timeout_seconds,
            )
            b1_summary = _read_json(b1_dir / "reference_b_b1_summary.json") if (b1_dir / "reference_b_b1_summary.json").exists() else {}
            b1_l_shed = _reference_b_l_shed(b1_dir / "reference_b_b1_load_service.csv", identity)
            b1_r_norm, b1_max_loading, b1_num_over = _risk_from_ac_branch_csv(b1_dir / "reference_b_b1_branch_state.csv", p_env, r_base)
            total_pd = sum(float(load.pd_pre) for load in identity.loads)
            b1_service_target_pd_units = total_pd * (1.0 - b1_l_shed) if b1_l_shed is not None else None
            ref_b_rows.append(
                {
                    "method": method,
                    "reference_b_stage": "B1_load_delivery",
                    "status": b1_summary.get("termination_status", "missing"),
                    "solver_objective": b1_summary.get("objective"),
                    "b1_service_target_pd_units": b1_service_target_pd_units,
                    "b2_load_shed_tolerance_pd_units": None,
                    "runtime_seconds": b1_summary.get("runtime_seconds"),
                    "wall_seconds": wall_b1,
                    "l_shed_ac_mld": b1_l_shed,
                    "r_norm_ac_mld": b1_r_norm,
                    "j_trade_ac_mld": compute_j_trade(float(payload["lambda_r"]), b1_r_norm, b1_l_shed) if b1_r_norm is not None and b1_l_shed is not None else None,
                    "service_recovery": l_shed_native - b1_l_shed if b1_l_shed is not None else None,
                    "max_ac_loading": b1_max_loading,
                    "num_ac_loading_gt_1": b1_num_over,
                    "stdout": out_b1.strip(),
                    "stderr": err_b1.strip(),
                }
            )
            if ok_b1 and b1_l_shed is not None:
                b2_limit = b1_l_shed * total_pd + args.reference_b_service_tolerance_pd_units
                b2_dir = method_dir / "rb2"
                ok_b2, out_b2, err_b2, wall_b2 = _run_julia(
                    julia_exe=Path(args.julia_exe),
                    julia_depot_path=Path(args.julia_depot_path),
                    script=script,
                    mode="reference_b_b2",
                    case_path=Path(args.case_path),
                    alpha_csv=alpha_effective_csv,
                    offline_branch_ids=offline,
                    source_less_load_ids_=source_less,
                    output_dir=b2_dir,
                    load_shed_limit=b2_limit,
                    timeout_seconds=args.timeout_seconds,
                )
                b2_summary = _read_json(b2_dir / "reference_b_b2_summary.json") if (b2_dir / "reference_b_b2_summary.json").exists() else {}
                b2_l_shed = _reference_b_l_shed(b2_dir / "reference_b_b2_load_service.csv", identity)
                b2_r_norm, b2_max_loading, b2_num_over = _risk_from_ac_branch_csv(b2_dir / "reference_b_b2_branch_state.csv", p_env, r_base)
                b2_shed_pd_units = b2_l_shed * total_pd if b2_l_shed is not None else None
                b2_constraint_residual_pd_units = (
                    max(0.0, b2_shed_pd_units - b2_limit) if b2_shed_pd_units is not None else None
                )
                ref_b_rows.append(
                    {
                        "method": method,
                        "reference_b_stage": "B2_cost_tiebreak",
                        "status": b2_summary.get("termination_status", "missing"),
                        "solver_objective": b2_summary.get("objective"),
                        "b1_service_target_pd_units": b1_service_target_pd_units,
                        "b2_load_shed_tolerance_pd_units": b2_limit,
                        "service_lock_constraint_satisfied": (
                            b2_l_shed is not None
                            and b2_l_shed <= b1_l_shed + args.reference_b_service_tolerance_pd_units / total_pd
                        ),
                        "service_lock_constraint_residual_pd_units": b2_constraint_residual_pd_units,
                        "service_lock_satisfied_with_solver_tolerance": (
                            b2_constraint_residual_pd_units is not None
                            and b2_constraint_residual_pd_units <= args.reference_b_service_verification_tolerance_pd_units
                        ),
                        "runtime_seconds": b2_summary.get("runtime_seconds"),
                        "wall_seconds": wall_b2,
                        "l_shed_ac_mld": b2_l_shed,
                        "r_norm_ac_mld": b2_r_norm,
                        "j_trade_ac_mld": compute_j_trade(float(payload["lambda_r"]), b2_r_norm, b2_l_shed) if b2_r_norm is not None and b2_l_shed is not None else None,
                        "service_recovery": l_shed_native - b2_l_shed if b2_l_shed is not None else None,
                        "max_ac_loading": b2_max_loading,
                        "num_ac_loading_gt_1": b2_num_over,
                        "stdout": out_b2.strip(),
                        "stderr": err_b2.strip(),
                    }
                )

    for rows in (summary_rows, fidelity_rows, warm_rows, ref_b_rows):
        for row in rows:
            row.update(model_metadata)

    if args.full_gridsfm_only:
        selection = str(model_selection.model_selection)
        warm_id = args.warm_start_id or ("frozen" if selection == "frozen" else "ft")
        table_path = output_dir / f"reference_a_crossed_full_warm_{warm_id}.csv"
        _write_rows(table_path, warm_rows)
        successful = all(
            row.get("status") in {"LOCALLY_SOLVED", "OPTIMAL"}
            and row.get("subprocess_success") is True
            and row.get("start_payload") == "Pg,Qg,V,theta"
            for row in warm_rows
        )
        status = {
            "status": "PASS" if warm_rows and successful else "FAIL",
            **model_metadata,
            "scope": "checkpoint-specific full GridSFM warm starts",
            "j8_root": str(root),
            "output_dir": str(output_dir),
            "methods": [str(finalist["method"]) for finalist in method_inputs],
            "finalists_manifest": "" if finalists_manifest is None else str(finalists_manifest),
            "warm_start_rows": len(warm_rows),
            "warm_start_type": f"gridsfm_{warm_id}_full_warm",
            "same_reference_a_instance": True,
            "start_payload": "Pg,Qg,V,theta",
            "result_table": str(table_path),
        }
        _write_json(output_dir / f"crossed_full_warm_{warm_id}_summary.json", status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0 if status["status"] == "PASS" else 1

    previous_status_path = output_dir / "j9_j10_reference_smoke_summary.json"
    previous_status = _read_json(previous_status_path) if args.warm_starts_only and previous_status_path.is_file() else {}
    if not args.warm_starts_only:
        _write_rows(output_dir / "reference_a_summary_table.csv", summary_rows)
        _write_rows(output_dir / "reference_a_state_fidelity_metrics.csv", fidelity_rows)
    _write_rows(output_dir / "reference_a_warm_start_summary.csv", warm_rows)
    if not args.warm_starts_only:
        _write_rows(output_dir / "reference_b_summary_table.csv", ref_b_rows)
    status = {
        "status": "PASS_WITH_LIMITATIONS",
        **model_metadata,
        "scope": "explicit finalist manifest" if finalists_manifest else "guided finalists only",
        "j8_root": str(root),
        "output_dir": str(output_dir),
        "methods": [str(finalist["method"]) for finalist in method_inputs],
        "finalists_manifest": "" if finalists_manifest is None else str(finalists_manifest),
        "reference_a_rows": previous_status.get("reference_a_rows", len(summary_rows)),
        "state_fidelity_rows": previous_status.get("state_fidelity_rows", len(fidelity_rows)),
        "warm_start_rows": len(warm_rows),
        "reference_b_rows": previous_status.get("reference_b_rows", len(ref_b_rows)),
        "reference_b_requested": previous_status.get("reference_b_requested", bool(args.run_reference_b)),
        "warm_starts_only": bool(args.warm_starts_only),
        "warm_start_types": [
            "cold_start", "dc_partial_warm", "gridsfm_partial_warm",
            "gridsfm_full_warm", "gt_warm",
        ],
        "reference_b_service_tolerance_pd_units": args.reference_b_service_tolerance_pd_units,
        "reference_b_service_verification_tolerance_pd_units": args.reference_b_service_verification_tolerance_pd_units,
        "notes": [
            "Iteration count is recorded as N/A because the current PowerModels/IPOPT wrapper does not expose a reliable structured iteration field.",
            "Warm-start timings include solver-reported runtime, Julia subprocess wall time, and measured regenerated-start construction time; GridSFM construction is not yet split into preprocessing and inference subcomponents.",
            "Cold starts explicitly set V=1, theta=0, Pg=(Pmin+Pmax)/2, and Qg=0 before PowerModels model construction.",
        ],
    }
    _write_json(output_dir / "j9_j10_reference_smoke_summary.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _cost_from_pg(raw_case, pg_by_gen_id: Mapping[int, float]) -> float | None:
    if not pg_by_gen_id:
        return None
    total = 0.0
    gen_ids = _generator_id_map(raw_case)
    for idx, row in enumerate(raw_case["grid"]["nodes"].get("generator", [])):
        gen_id = gen_ids[idx]
        if gen_id not in pg_by_gen_id:
            continue
        pg = float(pg_by_gen_id[gen_id])
        total += float(row[8]) * pg * pg + float(row[9]) * pg + float(row[10])
    return total


if __name__ == "__main__":
    raise SystemExit(main())
