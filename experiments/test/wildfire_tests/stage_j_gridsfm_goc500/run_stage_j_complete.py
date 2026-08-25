"""Run the locked Stage J S1-S3 comparison and its post-hoc AC audits.

Large GridSFM mutated-graph artifacts remain in a short external cache. This
driver copies the lightweight result evidence into goc_500_results/stage_j.
It is resumable at every method-setting and reference-setting boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

WILDFIRE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(WILDFIRE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_TESTS_ROOT))

from stage_j_gridsfm_goc500.model_selection import resolve_model_selection, sha256_file


J8_RUNNER = Path(__file__).with_name("run_j8_budgeted_alpha_topology_smoke.py")
J9_RUNNER = Path(__file__).with_name("run_j9_j10_reference_smoke.py")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _windows_extended_path(path: Path) -> str:
    """Use Win32 extended paths for final evidence below a long OneDrive root."""

    resolved = str(path.resolve())
    return resolved if resolved.startswith("\\\\?\\") else "\\\\?\\" + resolved


def _lambda_code(value: float) -> str:
    return f"l{value:.1f}".replace(".", "p")


def _setting_code(scenario_id: str, lambda_r: float) -> str:
    return f"{scenario_id.lower().replace('j-', '')}_{_lambda_code(lambda_r)}"


def _summary_path(main_dir: Path, method: str) -> Path:
    return main_dir / method / f"j8_{method.replace('-', '_')}_summary.json"


def _j8_complete(
    main_dir: Path,
    method: str,
    *,
    expected_model_selection: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> bool:
    path = _summary_path(main_dir, method)
    if not path.exists():
        return False
    payload = _read_json(path)
    # PARTIAL_FAIL is accepted only for legacy pre-v001 J8 outputs where
    # valid finalists coexist with deliberately rejected DC-infeasible
    # candidates. New outputs use the explicit PASS_WITH... status.
    complete = payload.get("status") in {
        "PASS",
        "PASS_WITH_INFEASIBLE_CANDIDATES",
        "PARTIAL_FAIL",
    } and bool(payload.get("best_topology"))
    if expected_model_selection is not None:
        complete = complete and payload.get("model_selection") == expected_model_selection
    if expected_checkpoint_sha256 is not None:
        complete = complete and payload.get("checkpoint_sha256") == expected_checkpoint_sha256
    return complete


def _reference_complete(
    ref_dir: Path,
    expected_methods: int,
    *,
    expected_model_selection: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> bool:
    status_path = ref_dir / "j9_j10_reference_smoke_summary.json"
    if not status_path.exists():
        return False
    status = _read_json(status_path)
    complete = (
        status.get("reference_a_rows") == expected_methods
        and status.get("warm_start_rows") == 5 * expected_methods
        and status.get("reference_b_rows") == 2 * expected_methods
    )
    if expected_model_selection is not None:
        complete = complete and status.get("model_selection") == expected_model_selection
    if expected_checkpoint_sha256 is not None:
        complete = complete and status.get("checkpoint_sha256") == expected_checkpoint_sha256
    return complete


def _run(command: list[str], *, env: dict[str, str], cwd: Path, log_path: Path, timeout_seconds: int) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        proc = subprocess.run(command, cwd=str(cwd), env=env, text=True, capture_output=True, timeout=timeout_seconds)
        payload = {
            "command": command,
            "returncode": proc.returncode,
            "runtime_seconds": time.time() - started,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        _write_json(log_path, payload)
        return proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        _write_json(
            log_path,
            {
                "command": command,
                "returncode": "timeout",
                "runtime_seconds": time.time() - started,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            },
        )
        return False


def _copy_lightweight_setting(cache_setting: Path, final_setting: Path) -> list[dict[str, object]]:
    """Copy final evidence, intentionally excluding mutated GridSFM graph caches."""

    copied: list[dict[str, object]] = []
    allow_names = {
        "j8_guided_dc_summary.json",
        "j8_guided_gridsfm_summary.json",
        "j8_th_gridsfm_summary.json",
        "j8_guided_dc_topology_summary.csv",
        "j8_guided_gridsfm_topology_summary.csv",
        "j8_th_gridsfm_topology_summary.csv",
        "j8_guided_dc_alpha_trace.csv",
        "j8_guided_gridsfm_alpha_trace.csv",
        "j8_th_gridsfm_alpha_trace.csv",
        "j8_guided_dc_best_selected_alpha.csv",
        "j8_guided_gridsfm_best_selected_alpha.csv",
        "j8_th_gridsfm_best_selected_alpha.csv",
        "j8_topology_pool.csv",
        "j8_th_gridsfm_topology_pool.csv",
        "finalists.json",
        "j9_j10_reference_smoke_summary.json",
        "reference_a_summary_table.csv",
        "reference_a_state_fidelity_metrics.csv",
        "reference_a_warm_start_summary.csv",
        "reference_b_summary_table.csv",
        "generator_identity_ranges.csv",
        "alpha_requested_full.csv",
        "alpha_effective_full.csv",
    }
    for source in cache_setting.rglob("*"):
        # The shared topology pools are named tersely to keep external cache
        # paths below legacy Windows path limits, so preserve them by location.
        is_pool = source.parent.name == "pools" and source.suffix == ".csv"
        if not source.is_file() or (source.name not in allow_names and not is_pool):
            continue
        relative = source.relative_to(cache_setting)
        # Keep portable evidence paths below the Windows legacy path limit.
        # The cache retains the full hierarchy; the final evidence tree only
        # contains uniquely named summary artifacts.
        if relative.parts[0] == "main":
            short_method = {
                "guided-dc": "gdc",
                "guided-gridsfm": "gsfm",
                "th-gridsfm": "th",
            }.get(relative.parts[1], relative.parts[1])
            relative = Path("methods") / short_method / source.name
        elif relative.parts[0] == "pools":
            relative = Path("pools") / source.name
        elif relative.parts[0] == "refs":
            # Keep per-finalist alpha vectors distinct while keeping root
            # summary tables concise and portable.
            relative = (
                Path("ac_audits") / source.name
                if len(relative.parts) == 2
                else Path("ac_audits") / relative.parts[1] / source.name
            )
        destination = final_setting / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            # shutil.copy2 uses CopyFile2, which can reject valid OneDrive
            # paths above the legacy MAX_PATH boundary. Copyfile with the
            # extended Win32 form retains the content evidence we need here.
            shutil.copyfile(_windows_extended_path(source), _windows_extended_path(destination))
            shutil.copystat(_windows_extended_path(source), _windows_extended_path(destination))
        except OSError as exc:
            raise RuntimeError(
                "Lightweight evidence copy failed; "
                f"source={source!s}; destination={destination!s}; "
                f"source_exists={source.exists()}; destination_parent_exists={destination.parent.exists()}"
            ) from exc
        copied.append({"relative_path": str(relative), "size_bytes": source.stat().st_size, "source_cache_path": str(source)})
    return copied


def _build_finalists(
    main_dir: Path,
    setting_dir: Path,
    *,
    model_selection: str,
    model_variant: str,
    include_th: bool = True,
) -> Path:
    guided_gridsfm = _read_json(_summary_path(main_dir, "guided-gridsfm"))
    finalists: list[dict[str, object]] = [
        {
            "method": "Guided-GridSFM",
            "backend": "gridsfm",
            "artifact_stub": "gsfm",
            "lambda_r": guided_gridsfm["lambda_r"],
            "model_selection": model_selection,
            "model_variant": model_variant,
            "best_topology": guided_gridsfm["best_topology"],
        },
    ]
    if include_th:
        th_rows = _read_csv(main_dir / "th-gridsfm" / "j8_th_gridsfm_topology_summary.csv")
        for row in th_rows:
            if str(row.get("best_found", "")).lower() != "true":
                raise RuntimeError(f"TH finalist missing an eligible alpha result: {row}")
            top_k = int(row["num_shutoffs"])
            finalists.append(
                {
                    "method": f"TH-GridSFM-top{top_k}",
                    "backend": "gridsfm",
                    "artifact_stub": f"th{top_k}",
                    "lambda_r": float(row["lambda_r"]),
                    "model_selection": model_selection,
                    "model_variant": model_variant,
                    "best_topology": row,
                }
            )
    path = setting_dir / "finalists.json"
    _write_json(path, {
        "model_selection": model_selection,
        "model_variant": model_variant,
        "finalists": finalists,
    })
    return path


def _cached_setting_records(
    cache_root: Path,
    *,
    model_selection: str,
    checkpoint_sha256: str,
    include_th: bool = True,
) -> list[dict[str, object]]:
    """Reconstruct all completed-setting status from the external cache.

    A targeted reference rerun must never replace the main-run aggregates with
    only its requested subset.
    """

    records: list[dict[str, object]] = []
    for setting_dir in cache_root.iterdir():
        if not setting_dir.is_dir():
            continue
        main_dir = setting_dir / "main"
        guided_path = _summary_path(main_dir, "guided-gridsfm")
        if not guided_path.exists():
            continue
        guided = _read_json(guided_path)
        records.append(
            {
                "setting_code": setting_dir.name,
                "scenario_id": guided.get("scenario_id", ""),
                "lambda_r": guided.get("lambda_r", ""),
                "model_selection": model_selection,
                "model_variant": guided.get("model_variant", ""),
                "checkpoint_sha256": checkpoint_sha256,
                "guided_gridsfm_ok": _j8_complete(
                    main_dir, "guided-gridsfm",
                    expected_model_selection=model_selection,
                    expected_checkpoint_sha256=checkpoint_sha256,
                ),
                "th_gridsfm_ok": (
                    _j8_complete(
                        main_dir, "th-gridsfm",
                        expected_model_selection=model_selection,
                        expected_checkpoint_sha256=checkpoint_sha256,
                    ) if include_th else None
                ),
                "references_ok": _reference_complete(
                    setting_dir / "refs", expected_methods=3 if include_th else 1,
                    expected_model_selection=model_selection,
                    expected_checkpoint_sha256=checkpoint_sha256,
                ),
                "finalists_manifest": str(setting_dir / "finalists.json") if (setting_dir / "finalists.json").exists() else "",
                "copied_lightweight_artifact_count": "reconstructed_from_cache",
                "cache_setting_dir": str(setting_dir),
            }
        )
    return sorted(records, key=lambda row: (str(row["scenario_id"]), float(row["lambda_r"])))


def _collect_aggregate(
    cache_root: Path,
    final_root: Path,
    setting_records: Iterable[dict[str, object]],
    *,
    include_th: bool = True,
) -> None:
    topology_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    finalists: list[dict[str, object]] = []
    reference_rows: dict[str, list[dict[str, object]]] = {
        "reference_a_all.csv": [],
        "state_fidelity_all.csv": [],
        "warm_start_all.csv": [],
        "reference_b_all.csv": [],
    }
    for record in setting_records:
        setting_dir = cache_root / str(record["setting_code"])
        main_dir = setting_dir / "main"
        methods = ("guided-gridsfm", "th-gridsfm") if include_th else ("guided-gridsfm",)
        for method in methods:
            method_dir = main_dir / method
            summary_csv = method_dir / f"j8_{method.replace('-', '_')}_topology_summary.csv"
            trace_csv = method_dir / f"j8_{method.replace('-', '_')}_alpha_trace.csv"
            if summary_csv.exists():
                topology_rows.extend(_read_csv(summary_csv))
            if trace_csv.exists():
                trace_rows.extend(_read_csv(trace_csv))
        finalists_path = setting_dir / "finalists.json"
        finalist_context: dict[str, dict[str, object]] = {}
        if finalists_path.exists():
            for finalist in _read_json(finalists_path).get("finalists", []):
                best_topology = dict(finalist.get("best_topology", {}))
                finalist_context[str(finalist.get("method"))] = {
                    "setting_code": record["setting_code"],
                    "scenario_id": record["scenario_id"],
                    "lambda_r": record["lambda_r"],
                    "finalist_backend": finalist.get("backend"),
                    "model_selection": finalist.get("model_selection"),
                    "model_variant": finalist.get("model_variant"),
                    "finalist_topology_id": best_topology.get("topology_id", ""),
                    "finalist_topology_rank": best_topology.get("topology_rank", ""),
                    "finalist_num_shutoffs": best_topology.get("num_shutoffs", ""),
                }
                finalists.append(
                    {
                        "method": finalist.get("method"),
                        "backend": finalist.get("backend"),
                        "artifact_stub": finalist.get("artifact_stub"),
                        "lambda_r": finalist.get("lambda_r"),
                        "model_selection": finalist.get("model_selection"),
                        "model_variant": finalist.get("model_variant"),
                        **dict(finalist.get("best_topology", {})),
                    }
                )
        ref_dir = setting_dir / "refs"
        for filename, target in (
            ("reference_a_summary_table.csv", "reference_a_all.csv"),
            ("reference_a_state_fidelity_metrics.csv", "state_fidelity_all.csv"),
            ("reference_a_warm_start_summary.csv", "warm_start_all.csv"),
            ("reference_b_summary_table.csv", "reference_b_all.csv"),
        ):
            path = ref_dir / filename
            if path.exists():
                for row in _read_csv(path):
                    context = finalist_context.get(str(row.get("method")), {
                        "setting_code": record["setting_code"],
                        "scenario_id": record["scenario_id"],
                        "lambda_r": record["lambda_r"],
                    })
                    reference_rows[target].append({**context, **row})
    core = final_root / "core_results"
    _write_csv(core / "topology_objectives_all.csv", topology_rows)
    _write_csv(core / "candidate_evaluations_all.csv", trace_rows)
    _write_csv(core / "method_finalists_all.csv", finalists)
    for filename, rows in reference_rows.items():
        _write_csv(core / filename, rows)


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def _prepare_topology_pool(
    *,
    source_root: Path,
    setting_code: str,
    pool_name: str,
    destination: Path,
) -> dict[str, object]:
    source = source_root / "settings" / setting_code / "pools" / pool_name
    if not source.is_file():
        raise FileNotFoundError(f"frozen topology pool is missing: {source}")
    source_hash = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination_hash = sha256_file(destination)
        if destination_hash != source_hash:
            raise RuntimeError(
                f"cached topology pool differs from frozen source: {destination}"
            )
    else:
        shutil.copyfile(source, destination)
        destination_hash = sha256_file(destination)
    if destination_hash != source_hash:
        raise RuntimeError(f"topology pool copy verification failed: {destination}")
    return {
        "setting_code": setting_code,
        "pool_name": pool_name,
        "source_path": str(source.resolve()),
        "destination_path": str(destination.resolve()),
        "sha256": source_hash,
        "row_count": len(_read_csv(source)),
        "ordered_byte_identical": True,
    }


def _validate_ft_result_counts(
    final_root: Path,
    *,
    n_settings: int,
    continuous_eval_budget: int,
    topology_budget: int = 100,
    include_th: bool = True,
) -> dict[str, object]:
    core = final_root / "core_results"
    methods_per_setting = 3 if include_th else 1
    topologies_per_setting = topology_budget + 2 if include_th else topology_budget
    expected = {
        "topology_objectives_all.csv": topologies_per_setting * n_settings,
        "candidate_evaluations_all.csv": topologies_per_setting * continuous_eval_budget * n_settings,
        "method_finalists_all.csv": methods_per_setting * n_settings,
        "reference_a_all.csv": methods_per_setting * n_settings,
        "reference_b_all.csv": 2 * methods_per_setting * n_settings,
        "warm_start_all.csv": 5 * methods_per_setting * n_settings,
        "state_fidelity_all.csv": 8 * methods_per_setting * n_settings,
    }
    observed = {
        filename: len(_read_csv(core / filename)) if (core / filename).is_file() else -1
        for filename in expected
    }
    return {
        "status": "PASS" if observed == expected else "FAIL",
        "expected": expected,
        "observed": observed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="stage_j_complete_run_v001")
    parser.add_argument("--cache-root", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\complete_run_v001")
    parser.add_argument("--final-root", default="experiments/test/wildfire_tests/goc_500_results/stage_j/complete_run")
    parser.add_argument("--gridsfm-root", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\repos\GridSFM")
    parser.add_argument("--model-selection", choices=["frozen", "ft"], default="frozen")
    parser.add_argument(
        "--guided-only", action="store_true",
        help="Run only Guided-GridSFM; historical Guided+TH execution remains the default.",
    )
    parser.add_argument("--gridsfm-checkpoint", default=None)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument(
        "--pool-source-root",
        default=None,
        help="Completed frozen result package supplying ordered guided.csv and th.csv pools.",
    )
    parser.add_argument("--input-dir", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\inputs\case500_goc_e0")
    parser.add_argument("--baseline-loading-csv", default=None)
    parser.add_argument("--pac-freeze-json", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\pac_calibration\v001\PAC_WEIGHT_FREEZE.json")
    parser.add_argument("--case-path", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\repos\pglib-opf\pglib_opf_case500_goc.m")
    parser.add_argument("--julia-exe", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\tools\julia-1.10.11\bin\julia.exe")
    parser.add_argument("--julia-depot-path", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\julia_depot")
    parser.add_argument("--xdg-cache-home", default=r"C:\Users\Caleb Lu\.gridfm_stage_j\cache\xdg")
    parser.add_argument("--scenarios", nargs="+", default=["J-S1", "J-S2", "J-S3"])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.2, 0.5, 0.8, 1.0])
    parser.add_argument("--topology-budget", type=int, default=100)
    parser.add_argument("--continuous-eval-budget", type=int, default=20)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--j8-timeout-seconds", type=int, default=7200)
    parser.add_argument("--reference-timeout-seconds", type=int, default=900)
    parser.add_argument("--skip-references", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve()
    final_root = Path(args.final_root).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    gridsfm_root = Path(args.gridsfm_root).expanduser().resolve()
    model_selection = resolve_model_selection(
        args.model_selection,
        gridsfm_root=gridsfm_root,
        checkpoint=Path(args.gridsfm_checkpoint) if args.gridsfm_checkpoint else None,
        expected_sha256=args.expected_checkpoint_sha256,
    )
    pool_source_root = Path(args.pool_source_root).expanduser().resolve() if args.pool_source_root else None
    if pool_source_root is None:
        parser.error("--pool-source-root is required for paired Stage J execution")
    final_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    baseline_loading_csv = Path(args.baseline_loading_csv).expanduser().resolve() if args.baseline_loading_csv else Path(_read_json(input_dir / "stage_j_input_manifest.json")["baseline_loading_csv"]).resolve()

    config = {
        "run_id": args.run_id,
        **model_selection.as_dict(),
        "scenarios": args.scenarios,
        "lambda_r": args.lambdas,
        "lambda_r_proxy": "coupled_to_lambda_r",
        "k_constraint": "<= 2",
        "topology_budget": args.topology_budget,
        "continuous_eval_budget": args.continuous_eval_budget,
        "q": args.q,
        "methods": (
            ["Guided-GridSFM"] if args.guided_only
            else ["Guided-GridSFM", "TH-GridSFM-top1", "TH-GridSFM-top2"]
        ),
        "guided_only": args.guided_only,
        "comparison_evidence": {
            "dc_and_frozen_result_root": str(pool_source_root),
            "execution_policy": "existing evidence referenced; only selected GridSFM variant is executed",
        },
        "cache_root": str(cache_root),
        "final_root": str(final_root),
        "large_artifacts": "mutated GridSFM candidate graphs remain only under cache_root",
    }
    _write_json(final_root / "RUN_CONFIG.json", config)
    provenance = {
        **model_selection.as_dict(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "stage_j_git_commit": _git_commit(WILDFIRE_TESTS_ROOT.parents[2]),
        "gridsfm_git_commit": _git_commit(gridsfm_root),
        "gridsfm_root": str(gridsfm_root),
        "pool_source_root": str(pool_source_root),
        "pool_source_run_config_sha256": sha256_file(pool_source_root / "RUN_CONFIG.json"),
        "pac_freeze_path": str(Path(args.pac_freeze_json).expanduser().resolve()),
        "pac_freeze_sha256": sha256_file(Path(args.pac_freeze_json).expanduser().resolve()),
        "pip_freeze": subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=True,
            capture_output=True, text=True,
        ).stdout.splitlines(),
    }
    _write_json(final_root / "PROVENANCE_MANIFEST.json", provenance)
    env = dict(os.environ)
    env["XDG_CACHE_HOME"] = str(Path(args.xdg_cache_home).expanduser().resolve())
    setting_records: list[dict[str, object]] = []
    pool_records: list[dict[str, object]] = []

    for scenario_id in args.scenarios:
        for lambda_r in args.lambdas:
            code = _setting_code(scenario_id, lambda_r)
            setting_dir = cache_root / code
            main_dir = setting_dir / "main"
            pool_dir = setting_dir / "pools"
            logs = setting_dir / "logs"
            main_dir.mkdir(parents=True, exist_ok=True)
            pool_dir.mkdir(parents=True, exist_ok=True)
            pool_names = ("guided.csv",) if args.guided_only else ("guided.csv", "th.csv")
            for pool_name in pool_names:
                pool_records.append(
                    _prepare_topology_pool(
                        source_root=pool_source_root,
                        setting_code=code,
                        pool_name=pool_name,
                        destination=pool_dir / pool_name,
                    )
                )
            _write_json(final_root / "TOPOLOGY_POOL_MANIFEST.json", {
                "status": "PASS",
                "source_root": str(pool_source_root),
                "records": pool_records,
            })
            j8_ok: dict[str, bool] = {}
            execution_methods = (
                ("guided-gridsfm",) if args.guided_only
                else ("guided-gridsfm", "th-gridsfm")
            )
            for method in execution_methods:
                method_dir = main_dir / method
                pool_name = "th.csv" if method == "th-gridsfm" else "guided.csv"
                if _j8_complete(
                    main_dir, method,
                    expected_model_selection=model_selection.model_selection,
                    expected_checkpoint_sha256=model_selection.checkpoint_sha256,
                ) and not args.force:
                    j8_ok[method] = True
                    continue
                command = [
                    sys.executable,
                    str(J8_RUNNER),
                    "--method", method,
                    "--model-selection", model_selection.model_selection,
                    "--gridsfm-root", str(gridsfm_root),
                    "--checkpoint", str(model_selection.checkpoint_path),
                    "--expected-checkpoint-sha256", str(model_selection.checkpoint_sha256),
                    "--input-dir", str(input_dir),
                    "--baseline-loading-csv", str(baseline_loading_csv),
                    "--scenario-id", scenario_id,
                    "--lambda-r", str(lambda_r),
                    "--lambda-r-proxy", str(lambda_r),
                    "--k", "2",
                    "--topology-budget", str(args.topology_budget),
                    "--continuous-eval-budget", str(args.continuous_eval_budget),
                    "--q", str(args.q),
                    "--topology-pool-csv", str(pool_dir / pool_name),
                    "--xdg-cache-home", env["XDG_CACHE_HOME"],
                    "--output-dir", str(method_dir),
                ]
                if method != "guided-dc":
                    command.extend(["--pac-freeze-json", str(Path(args.pac_freeze_json).expanduser().resolve())])
                j8_ok[method] = _run(command, env=env, cwd=WILDFIRE_TESTS_ROOT.parent, log_path=logs / f"{method}.json", timeout_seconds=args.j8_timeout_seconds)
                j8_ok[method] = j8_ok[method] and _j8_complete(
                    main_dir, method,
                    expected_model_selection=model_selection.model_selection,
                    expected_checkpoint_sha256=model_selection.checkpoint_sha256,
                )

            finalists_manifest = None
            refs_ok = args.skip_references
            if all(j8_ok.values()):
                finalists_manifest = _build_finalists(
                    main_dir, setting_dir,
                    model_selection=model_selection.model_selection,
                    model_variant=model_selection.model_variant,
                    include_th=not args.guided_only,
                )
                ref_dir = setting_dir / "refs"
                if _reference_complete(
                    ref_dir, expected_methods=1 if args.guided_only else 3,
                    expected_model_selection=model_selection.model_selection,
                    expected_checkpoint_sha256=model_selection.checkpoint_sha256,
                ) and not args.force:
                    refs_ok = True
                elif not args.skip_references:
                    command = [
                        sys.executable,
                        str(J9_RUNNER),
                        "--j8-root", str(main_dir),
                        "--finalists-manifest", str(finalists_manifest),
                        "--input-dir", str(input_dir),
                        "--gridsfm-root", str(gridsfm_root),
                        "--model-selection", model_selection.model_selection,
                        "--checkpoint", str(model_selection.checkpoint_path),
                        "--expected-checkpoint-sha256", str(model_selection.checkpoint_sha256),
                        "--case-path", str(Path(args.case_path).expanduser().resolve()),
                        "--julia-exe", str(Path(args.julia_exe).expanduser().resolve()),
                        "--julia-depot-path", str(Path(args.julia_depot_path).expanduser().resolve()),
                        "--pac-freeze-json", str(Path(args.pac_freeze_json).expanduser().resolve()),
                        "--scenario-id", scenario_id,
                        "--run-reference-b",
                        "--timeout-seconds", str(args.reference_timeout_seconds),
                        "--output-dir", str(ref_dir),
                    ]
                    refs_ok = _run(command, env=env, cwd=WILDFIRE_TESTS_ROOT.parent, log_path=logs / "references.json", timeout_seconds=3 * 7 * args.reference_timeout_seconds)
                    refs_ok = refs_ok and _reference_complete(
                        ref_dir, expected_methods=1 if args.guided_only else 3,
                        expected_model_selection=model_selection.model_selection,
                        expected_checkpoint_sha256=model_selection.checkpoint_sha256,
                    )

            copied = _copy_lightweight_setting(setting_dir, final_root / "settings" / code)
            setting_records.append(
                {
                    "setting_code": code,
                    "scenario_id": scenario_id,
                    "lambda_r": lambda_r,
                    "model_selection": model_selection.model_selection,
                    "model_variant": model_selection.model_variant,
                    "checkpoint_sha256": model_selection.checkpoint_sha256,
                    "guided_gridsfm_ok": j8_ok.get("guided-gridsfm", False),
                    "th_gridsfm_ok": j8_ok.get("th-gridsfm") if not args.guided_only else None,
                    "references_ok": refs_ok,
                    "finalists_manifest": "" if finalists_manifest is None else str(finalists_manifest),
                    "copied_lightweight_artifact_count": len(copied),
                    "cache_setting_dir": str(setting_dir),
                }
            )
            _write_json(
                final_root / "RUN_STATUS.json",
                {
                    "status": "IN_PROGRESS", "config": config,
                    "settings": _cached_setting_records(
                        cache_root,
                        model_selection=model_selection.model_selection,
                        checkpoint_sha256=str(model_selection.checkpoint_sha256),
                        include_th=not args.guided_only,
                    ),
                },
            )
            print(json.dumps({
                "status": "SETTING_COMPLETE" if refs_ok else "SETTING_FAILED",
                "setting_code": code,
                "scenario_id": scenario_id,
                "lambda_r": lambda_r,
                "model_variant": model_selection.model_variant,
                "guided_gridsfm_ok": j8_ok.get("guided-gridsfm", False),
                "th_executed": not args.guided_only,
                "references_ok": refs_ok,
            }, sort_keys=True), flush=True)

    all_setting_records = _cached_setting_records(
        cache_root,
        model_selection=model_selection.model_selection,
        checkpoint_sha256=str(model_selection.checkpoint_sha256),
        include_th=not args.guided_only,
    )
    _collect_aggregate(
        cache_root, final_root, all_setting_records, include_th=not args.guided_only
    )
    _write_csv(final_root / "core_results" / "setting_status.csv", all_setting_records)
    complete = all(
        bool(record["guided_gridsfm_ok"])
        and (args.guided_only or bool(record["th_gridsfm_ok"]))
        and bool(record["references_ok"])
        for record in all_setting_records
    ) and len(all_setting_records) == len(args.scenarios) * len(args.lambdas)
    count_validation = _validate_ft_result_counts(
        final_root,
        n_settings=len(args.scenarios) * len(args.lambdas),
        continuous_eval_budget=args.continuous_eval_budget,
        topology_budget=args.topology_budget,
        include_th=not args.guided_only,
    )
    complete = complete and count_validation["status"] == "PASS"
    _write_json(final_root / "ARTIFACT_COUNT_VALIDATION.json", count_validation)
    _write_json(
        final_root / "RUN_STATUS.json",
        {
            "status": "COMPLETE" if complete else "PARTIAL_OR_FAILED",
            "config": config, "settings": all_setting_records,
            "artifact_count_validation": count_validation,
        },
    )
    print(json.dumps({"status": "COMPLETE" if complete else "PARTIAL_OR_FAILED", "settings": len(all_setting_records), "final_root": str(final_root)}, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
