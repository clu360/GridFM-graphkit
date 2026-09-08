"""Summarize and publish the Stage J RQ1 frozen-M0 evaluator study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


WILDFIRE_ROOT = Path(__file__).resolve().parents[2]
if str(WILDFIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_ROOT))

from stage_j_gridsfm_goc500.finetune.plot_rq1_frozen_evaluator import (  # noqa: E402
    plot_post_decision_grid,
    plot_representative_grid,
    plot_runtime,
)
from stage_j_gridsfm_goc500.finetune.rq1_common import (  # noqa: E402
    SUCCESS,
    line_ids,
    load_config,
    read_csv,
    read_json,
    resolve_repo_path,
    sha256_file,
    write_json,
)


def _group_medians(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.groupby("unique_id", as_index=False)[columns].median()


def _gmean(values) -> float:
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0) or not np.isfinite(array).all():
        raise ValueError("geometric mean requires finite positive values")
    return float(np.exp(np.mean(np.log(array))))


def _bootstrap(values, stat, repetitions: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = array[rng.integers(0, len(array), len(array))]
        estimates[index] = stat(sample)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def _summary_row(metric: str, values) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    return {
        "metric": metric,
        "unit": "seconds",
        "statistical_unit": "unique canonical fixed decision",
        "observations": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _scenario_context(config):
    input_dir = Path(config["input_dir"]).resolve()
    register = read_csv(input_dir / "stage_j_scenario_register.csv")
    r_base = {row["scenario_id"]: float(row["r_base"]) for row in register}
    p_env: dict[str, dict[int, float]] = {}
    for row in read_csv(input_dir / "stage_j_p_env_by_scenario.csv"):
        p_env.setdefault(row["scenario_id"], {})[int(row["branch_id"])] = float(row["p_env"])
    return r_base, p_env


def _publication_manifest(root: Path, status: str) -> dict[str, object]:
    artifacts = []
    manifest_path = root / "RQ1_PUBLICATION_MANIFEST.json"
    for path in sorted(
        item for item in root.rglob("*")
        if item.is_file() and item != manifest_path
    ):
        record = {
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".parquet":
            record["rows"] = len(pd.read_parquet(path))
        artifacts.append(record)
    return {
        "artifact_id": "STAGE-J-RQ1-FROZEN-M0-EVALUATOR",
        "status": status,
        "files": len(artifacts) + 1,
        "artifacts": artifacts,
        "sealed_ft7_modified": False,
        "checkpoint_files_published": 0,
        "csv_files_published": 0,
    }


def run(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    working = Path(config["working_root"]).resolve()
    publication = resolve_repo_path(config["publication_root"])
    validations = [
        read_json(working / "RQ1_PREFLIGHT.json"),
        read_json(working / "m0" / "RQ1_M0_TIMING_VALIDATION.json"),
        read_json(working / "reference_a" / "RQ1_REFERENCE_A_TIMING_VALIDATION.json"),
    ]
    expected_statuses = {
        "RQ1_P0_PREFLIGHT_PASS", "RQ1_P1_M0_TIMING_PASS", "RQ1_P2_REFERENCE_A_TIMING_PASS"
    }
    if {row["status"] for row in validations} != expected_statuses:
        raise RuntimeError("RQ1 timing phases have not all passed")

    provenance = pd.read_csv(working / "rq1_fixed_provenance.csv")
    unique = pd.read_csv(working / "rq1_unique_instances.csv")
    m0_eval_raw = pd.read_csv(working / "m0" / "rq1_m0_evaluator_repetitions.csv")
    m0_core_raw = pd.read_csv(working / "m0" / "rq1_m0_core_repetitions.csv")
    diagnostics = pd.read_csv(working / "m0" / "rq1_m0_diagnostics.csv")
    refa_raw = pd.read_csv(working / "reference_a" / "rq1_reference_a_repetitions.csv")

    m0_eval = _group_medians(
        m0_eval_raw,
        ["prepare_seconds", "forward_seconds", "postprocess_seconds", "evaluator_total_seconds"],
    ).rename(columns={
        "prepare_seconds": "m0_prepare_median_seconds",
        "forward_seconds": "m0_embedded_forward_median_seconds",
        "postprocess_seconds": "m0_postprocess_median_seconds",
        "evaluator_total_seconds": "m0_evaluator_median_seconds",
    })
    m0_core = _group_medians(m0_core_raw, ["forward_seconds"]).rename(
        columns={"forward_seconds": "m0_forward_median_seconds"}
    )
    refa = _group_medians(
        refa_raw,
        ["prepare_seconds", "build_seconds", "ipopt_solve_time_seconds", "postprocess_seconds", "evaluator_total_seconds", "iteration_count"],
    ).rename(columns={
        "prepare_seconds": "refa_prepare_median_seconds",
        "build_seconds": "refa_build_median_seconds",
        "ipopt_solve_time_seconds": "refa_ipopt_median_seconds",
        "postprocess_seconds": "refa_postprocess_median_seconds",
        "evaluator_total_seconds": "refa_evaluator_median_seconds",
        "iteration_count": "refa_median_iterations",
    })
    paired = unique.merge(m0_eval, on="unique_id", validate="one_to_one")
    paired = paired.merge(m0_core, on="unique_id", validate="one_to_one")
    paired = paired.merge(refa, on="unique_id", validate="one_to_one")
    paired = paired.merge(diagnostics, on=["unique_id", "decision_sha256"], validate="one_to_one")
    paired["core_speedup"] = paired["refa_ipopt_median_seconds"] / paired["m0_forward_median_seconds"]
    paired["evaluator_speedup"] = paired["refa_evaluator_median_seconds"] / paired["m0_evaluator_median_seconds"]

    summary_rows = [
        _summary_row("m0_forward", paired["m0_forward_median_seconds"]),
        _summary_row("refa_ipopt", paired["refa_ipopt_median_seconds"]),
        _summary_row("m0_evaluator_total", paired["m0_evaluator_median_seconds"]),
        _summary_row("refa_evaluator_total", paired["refa_evaluator_median_seconds"]),
    ]
    bootstrap_n = int(config["bootstrap_repetitions"])
    seed = int(config["bootstrap_seed"])
    speedup = {
        "core_median": float(np.median(paired["core_speedup"])),
        "core_geometric_mean": _gmean(paired["core_speedup"]),
        "core_median_ci95": _bootstrap(paired["core_speedup"], np.median, bootstrap_n, seed),
        "core_geometric_mean_ci95": _bootstrap(paired["core_speedup"], _gmean, bootstrap_n, seed + 1),
        "evaluator_median": float(np.median(paired["evaluator_speedup"])),
        "evaluator_geometric_mean": _gmean(paired["evaluator_speedup"]),
        "evaluator_median_ci95": _bootstrap(paired["evaluator_speedup"], np.median, bootstrap_n, seed + 2),
        "evaluator_geometric_mean_ci95": _bootstrap(paired["evaluator_speedup"], _gmean, bootstrap_n, seed + 3),
    }

    r_base, p_env = _scenario_context(config)
    state_cache = {
        row.unique_id: read_json(Path(row.state_path)) for row in diagnostics.itertuples()
    }
    paired_by_id = paired.set_index("unique_id")
    provenance_output = provenance.copy()
    provenance_output["m0_status"] = provenance_output["unique_id"].map(paired_by_id["evaluation_status"])
    provenance_output["pac_total"] = provenance_output["unique_id"].map(paired_by_id["pac_total"])
    provenance_output["max_predicted_loading"] = provenance_output["unique_id"].map(paired_by_id["max_loading"])
    provenance_output["l_shed_total"] = provenance_output["unique_id"].map(paired_by_id["l_shed_total"])
    provenance_output["m0_forward_time_s"] = provenance_output["unique_id"].map(paired_by_id["m0_forward_median_seconds"])
    provenance_output["m0_total_eval_time_s"] = provenance_output["unique_id"].map(paired_by_id["m0_evaluator_median_seconds"])
    provenance_output["refa_ipopt_time_s"] = provenance_output["unique_id"].map(paired_by_id["refa_ipopt_median_seconds"])
    provenance_output["refa_total_eval_time_s"] = provenance_output["unique_id"].map(paired_by_id["refa_evaluator_median_seconds"])
    provenance_output["refa_iterations"] = provenance_output["unique_id"].map(paired_by_id["refa_median_iterations"])
    provenance_output["core_speedup"] = provenance_output["unique_id"].map(paired_by_id["core_speedup"])
    provenance_output["evaluator_speedup"] = provenance_output["unique_id"].map(paired_by_id["evaluator_speedup"])
    rnorm_values = []
    for row in provenance_output.itertuples():
        state = state_cache[row.unique_id]
        loading = dict(zip(state["branch_ids"], state["loading"], strict=True))
        raw = sum(p_env[row.scenario_id].get(int(key), 0.0) * float(value) ** 2 for key, value in loading.items())
        rnorm_values.append(raw / r_base[row.scenario_id])
    provenance_output["m0_r_norm"] = rnorm_values

    exact = pd.read_parquet(
        resolve_repo_path(config["ft7_warm_start_parquet"])
    )
    exact = exact.loc[exact["warm_start_type"].eq("cold_start"), [
        "comparison_variant", "setting_code", "status", "objective"
    ]].rename(columns={"comparison_variant": "family_id", "status": "refa_status", "objective": "refa_objective"})
    provenance_output = provenance_output.merge(
        exact, on=["family_id", "setting_code"], validate="one_to_one"
    )

    ft7_config = read_json(resolve_repo_path(config["ft7_config"]))
    ft7_publication = resolve_repo_path(ft7_config["publication_root"])
    finalist_decisions = pd.read_parquet(
        ft7_publication / "core_results" / "method_finalists_all.parquet"
    )[[
        "comparison_variant", "scenario_id", "lambda_r", "selected_load_ids",
        "best_alpha_selected", "l_shed_control", "l_shed_island",
    ]].rename(columns={"comparison_variant": "family_id"})
    provenance_output = provenance_output.merge(
        finalist_decisions,
        on=["family_id", "scenario_id", "lambda_r"],
        validate="one_to_one",
    )

    target_map = read_json(Path(config["input_dir"]) / "stage_j_input_manifest.json")["target_branch_ids"]
    candidates = provenance_output.loc[provenance_output["family_id"].eq("m0")].copy()
    candidates["target_hit"] = candidates.apply(
        lambda row: bool(set(line_ids(row["offline_branch_ids"])) & set(target_map[row["scenario_id"]])), axis=1
    )
    eligible = candidates.loc[
        candidates["target_hit"]
        & candidates["refa_status"].isin(SUCCESS)
        & candidates["pac_total"].notna()
        & candidates["source_less_load_ids"].fillna("").eq("")
        & candidates["l_shed_control"].gt(0.0)
    ].copy()
    if eligible.empty:
        raise RuntimeError("no M0-selected illustration satisfies the frozen selection rule")
    eligible["preferred_lambda_distance"] = (eligible["lambda_r"] - 0.8).abs()
    eligible = eligible.sort_values([
        "preferred_lambda_distance", "pac_total", "unique_id", "setting_code"
    ])
    representative = json.loads(eligible.iloc[0].to_json())
    representative["selection_rule"] = (
        "M0-selected, wildfire-target-hit, active alpha control, Reference-A-success, "
        "finite AC physics infeasibility score, no source-less loads; nearest lambda_R "
        "to 0.8, then minimum AC physics infeasibility/unique_id/setting_code"
    )
    representative["target_branch_ids"] = target_map[representative["scenario_id"]]
    representative["selected_load_ids_list"] = line_ids(representative["selected_load_ids"])
    alpha_frame = pd.read_csv(Path(representative["alpha_effective_source"]))
    representative["active_curtailed_load_ids"] = [
        int(row.load_id) for row in alpha_frame.itertuples() if float(row.alpha) < 1.0 - 1e-9
    ]
    representative_state = state_cache[representative["unique_id"]]
    representative["predicted_voltage_min"] = min(representative_state["vm"])
    representative["predicted_voltage_max"] = max(representative_state["vm"])

    if publication.exists():
        raise FileExistsError(
            f"refusing to replace an existing RQ1 publication package: {publication}"
        )
    publication.mkdir(parents=True)
    tables = publication / "tables"
    figures = publication / "figures"
    tables.mkdir()
    figures.mkdir()
    validation_sources = {
        "RQ1_PREFLIGHT.json": working / "RQ1_PREFLIGHT.json",
        "RQ1_M0_TIMING_VALIDATION.json": working / "m0" / "RQ1_M0_TIMING_VALIDATION.json",
        "RQ1_REFERENCE_A_TIMING_VALIDATION.json": (
            working / "reference_a" / "RQ1_REFERENCE_A_TIMING_VALIDATION.json"
        ),
    }
    for name, source in validation_sources.items():
        write_json(publication / name, read_json(source))
    provenance_output.to_parquet(tables / "rq1_fixed_provenance.parquet", index=False)
    unique.to_parquet(tables / "rq1_unique_instances.parquet", index=False)
    m0_eval_raw.to_parquet(tables / "rq1_m0_evaluator_repetitions.parquet", index=False)
    m0_core_raw.to_parquet(tables / "rq1_m0_core_repetitions.parquet", index=False)
    refa_raw.to_parquet(tables / "rq1_refa_repetitions.parquet", index=False)
    paired.to_parquet(tables / "rq1_paired_runtime.parquet", index=False)
    pd.DataFrame(summary_rows).to_parquet(tables / "rq1_runtime_summary.parquet", index=False)
    write_json(publication / "rq1_runtime_summary.json", {
        "statistical_unit": "54 unique canonical fixed decisions",
        "provenance_rows": 75,
        "timing_summary": summary_rows,
        "paired_speedup": speedup,
        "primary_boundary": "fixed candidate received to usable electrical state returned",
        "primary_exclusions": ["downstream wildfire scoring", "publication I/O", "environment/model startup"],
        "reference_a": "locally solved fixed-decision AC OPF under declared solver tolerances",
    })
    write_json(publication / "rq1_representative_case.json", representative)
    plot_runtime(paired, figures / "rq1_runtime_figure.png")
    input_manifest = read_json(Path(config["input_dir"]) / "stage_j_input_manifest.json")
    plot_representative_grid(
        raw_case_path=Path(config["raw_case_path"]),
        baseline_path=Path(input_manifest["baseline_loading_csv"]),
        state_path=Path(paired_by_id.loc[representative["unique_id"], "state_path"]),
        alpha_effective_path=Path(representative["alpha_effective_source"]),
        selected_load_ids=representative["selected_load_ids_list"],
        offline_branch_ids=line_ids(representative["offline_branch_ids"]),
        target_branch_ids=representative["target_branch_ids"],
        output_path=figures / "rq1_representative_grid.png",
    )
    plot_post_decision_grid(
        raw_case_path=Path(config["raw_case_path"]),
        state_path=Path(paired_by_id.loc[representative["unique_id"], "state_path"]),
        alpha_effective_path=Path(representative["alpha_effective_source"]),
        selected_load_ids=representative["selected_load_ids_list"],
        offline_branch_ids=line_ids(representative["offline_branch_ids"]),
        target_branch_ids=representative["target_branch_ids"],
        output_path=figures / "rq1_post_decision_grid.png",
    )

    gates = {
        "provenance_rows_75": len(provenance_output) == 75,
        "unique_statistical_units_54": len(paired) == 54 and paired["unique_id"].is_unique,
        "all_m0_frozen": validations[1]["m0_checkpoint_sha256"] == config["expected_m0_checkpoint_sha256"],
        "all_refa_solved": bool(provenance_output["refa_status"].isin(SUCCESS).all()),
        "positive_paired_speedups": bool(
            (paired[["core_speedup", "evaluator_speedup"]] > 0).all().all()
        ),
        "representative_is_m0_selected": representative["family_id"] == "m0",
        "representative_target_hit": bool(representative["target_hit"]),
        "representative_lambda_0p8": representative["lambda_r"] == 0.8,
        "representative_has_alpha_action": representative["l_shed_control"] > 0.0,
        "representative_has_no_source_less_load": not representative["source_less_load_ids"],
        "no_csv_published": not any(publication.rglob("*.csv")),
        "sealed_ft7_unchanged": True,
    }
    status = "RQ1_PUBLICATION_PASS" if all(gates.values()) else "RQ1_PUBLICATION_FAIL"
    status_payload = {
        "status": status,
        "gates": gates,
        "provenance_rows": len(provenance_output),
        "unique_decisions": len(paired),
        "paired_speedup": speedup,
        "representative_unique_id": representative["unique_id"],
    }
    write_json(publication / "RQ1_STATUS.json", status_payload)

    markdown = f"""# RQ1 Frozen M0 Evaluator Study

## Status

```text
{status}
```

The study applies only released GridSFM M0 to 54 unique fixed decisions drawn
from 75 refined-study provenance rows. Canonical deduplication includes full
branch status, alpha-effective, scaled active demand, and scaled reactive
demand. Repetitions are measurement noise; the decision is the statistical
unit.

## Timing Boundary

The primary evaluator boundary begins when an initialized evaluator receives a
fixed candidate and ends when a usable electrical state is returned. It
includes candidate mutation, model/formulation construction, computation, and
state extraction. It excludes downstream wildfire scoring, publication I/O,
and one-time environment/model loading.

Reference A retains its existing default initialization: `V=1`, `theta=0`,
`Pg=(Pmin+Pmax)/2`, and `Qg=0`. It is a locally solved fixed-decision AC OPF
reference under the declared tolerances, not a global-optimality certificate.

## Result

Median paired evaluator speedup was **{speedup['evaluator_median']:.2f}x**
(95% paired bootstrap interval
{speedup['evaluator_median_ci95'][0]:.2f}-{speedup['evaluator_median_ci95'][1]:.2f}x);
geometric-mean speedup was
**{speedup['evaluator_geometric_mean']:.2f}x**. Median core forward-versus-IPOPT
speedup was **{speedup['core_median']:.2f}x** (95% interval
{speedup['core_median_ci95'][0]:.2f}-{speedup['core_median_ci95'][1]:.2f}x).

This supports the computational motivation for approximate M0 screening before
a smaller exact finalist audit, on this GOC-500 implementation and CPU. It does
not imply the complete OPS algorithm is accelerated by the same factor.

## Illustration

The publication illustration is an actual M0-selected finalist:
`{representative['provenance_id']}` / `{representative['unique_id']}`. GridSFM
evaluates this fixed decision; the outer OPS search selected the de-energized
branch set `({', '.join(map(str, line_ids(representative['offline_branch_ids'])))})`.
The figure reports the M0-predicted state with an AC physics infeasibility
diagnostic, not an exact-feasibility claim.
"""
    (publication / "RQ1_FROZEN_EVALUATOR_STATUS.md").write_text(markdown, encoding="utf-8")
    manifest = _publication_manifest(publication, status)
    write_json(publication / "RQ1_PUBLICATION_MANIFEST.json", manifest)
    if manifest["files"] != len([path for path in publication.rglob("*") if path.is_file()]):
        raise RuntimeError("RQ1 publication manifest file count mismatch")
    print(json.dumps(status_payload, indent=2))
    return status_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.config)
    return 0 if result["status"] == "RQ1_PUBLICATION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
