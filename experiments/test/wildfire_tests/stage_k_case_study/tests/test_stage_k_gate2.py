from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.scenario_builder import (
    connectivity_service_impact_proxy,
)
from experiments.test.wildfire_tests.stage_k_case_study.src.aggregate import (
    empirical_nondominated,
    full_run_workload_ratio,
)
from experiments.test.wildfire_tests.stage_k_case_study.src.alpha_search import budgeted_powell_search
from experiments.test.wildfire_tests.stage_k_case_study.src.baseline_audit import deployment_audit_checks
from experiments.test.wildfire_tests.stage_k_case_study.src.config import load_config
from experiments.test.wildfire_tests.stage_k_case_study.src.gridsfm_adapter import (
    build_raw_gridsfm_case,
    mutate_candidate,
)
from experiments.test.wildfire_tests.stage_k_case_study.src.gridsfm_environment import (
    contract_checks,
    load_gridsfm_contract,
)
from experiments.test.wildfire_tests.stage_k_case_study.src.evaluator_runner import _candidate_with_checkpoint
from experiments.test.wildfire_tests.stage_k_case_study.src.identity import build_identity
from experiments.test.wildfire_tests.stage_k_case_study.src.io_utils import (
    atomic_write_json,
    checkpoint_matches,
    require_new_run_dir,
)
from experiments.test.wildfire_tests.stage_k_case_study.src.objectives import (
    compute_j_trade,
    compute_r_base,
    compute_r_norm,
    reference_deltas,
)
from experiments.test.wildfire_tests.stage_k_case_study.src.reference_runner import (
    _reference_b_diagnostic_summary,
)
from experiments.test.wildfire_tests.stage_k_case_study.src.production_join import (
    _validate_chunk,
)
from experiments.test.wildfire_tests.stage_k_case_study.src.production_reference_join import (
    join_reference_tasks,
)
from experiments.test.wildfire_tests.stage_k_case_study.src.schemas import CandidateResult, classify_solver_status
from experiments.test.wildfire_tests.stage_k_case_study.src.service import select_topology_relative_loads
from experiments.test.wildfire_tests.stage_k_case_study.src.topology_candidates import (
    exact_k_candidates,
    k2_children,
    proxy_components,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def identity():
    return build_identity()


def test_smoke_and_full_config_contracts():
    smoke = load_config(ROOT / "config" / "smoke.yaml")
    full = load_config(ROOT / "config" / "full.yaml")
    assert smoke["search"] == {
        "lambda_r": [0.8], "k1_count": 10, "parent_count": 5,
        "k2_children_per_parent": 2, "k2_max_unique": 10,
        "gridsfm_q": 5, "gridsfm_alpha_budget": 5,
    }
    assert full["search"]["lambda_r"] == [0.0, 0.2, 0.5, 0.8, 1.0]
    assert full["search"]["k1_count"] == 50
    assert full["search"]["k2_max_unique"] == 250
    assert full["search"]["gridsfm_alpha_budget"] == 20


def test_canonical_texas2k_counts(identity):
    assert len(identity.buses) == 2751
    assert len(identity.loads) == 1125
    assert len(identity.generators) == 1099
    assert int((identity.generators["GEN_STATUS"] > 0).sum()) == 736
    assert len(identity.branches) == 5344
    assert len(identity.l_trans) == 3993
    assert len(identity.l_fixed) == 1351


def test_transformer_classification_reconciles(identity):
    branches = identity.branches
    fixed = branches.loc[~branches["is_switchable_transmission"]]
    assert len(fixed) == int((branches["TAP"].abs() > 1e-12).sum())
    assert len(fixed) == int(((branches["from_BASE_KV"] - branches["to_BASE_KV"]).abs() > 1e-9).sum())
    assert set(fixed["branch_family"]) == {"transformer"}


def test_scenario16_case_scaled_load_is_authoritative(identity):
    assert identity.loads["pd_requested_mw"].sum() == pytest.approx(92895.4705)
    assert identity.loads["pd_parquet_mw"].sum() == pytest.approx(88719.45)
    assert identity.loads["qd_requested_mvar"].sum() == pytest.approx(23007.429)


def test_environment_and_r_base(identity):
    branch = identity.branches
    assert set(branch["weather_timestamp"].astype(str)) == {"2023-06-23 16:00 CDT"}
    assert branch["weather_coverage_valid"].all()
    p_env = dict(zip(branch.canonical_branch_id, branch.p_env, strict=True))
    baseline = dict(zip(branch.canonical_branch_id, branch.baseline_loading, strict=True))
    assert compute_r_base(identity.l_trans, p_env, baseline) == pytest.approx(32.975425699285736)


def test_stage_j_c_l_implementation_is_reused(identity):
    sample = identity.l_trans[:8]
    observed = connectivity_service_impact_proxy(identity.stage_j_identity, sample)
    assert set(observed) == set(sample)
    assert all(0.0 <= value <= 1.0 for value in observed.values())


def test_proxy_objective_and_exact_k_ranking_equivalence():
    ids = (1, 2, 3, 4)
    weights = {1: 5.0, 2: 2.0, 3: 7.0, 4: 1.0}
    c = {1: 0.2, 2: 0.0, 3: 0.5, 4: 0.1}
    lam = 0.8
    expected = sorted(
        combinations(ids, 2),
        key=lambda combo: (proxy_components(combo, line_ids=ids, weights=weights, c_by_line=c, lambda_r=lam)[0], combo),
    )
    actual = exact_k_candidates(
        line_ids=ids, weights=weights, c_by_line=c, lambda_r=lam, k=2, count=6
    )
    assert [row.offline_branch_ids for row in actual] == expected


def test_k2_parent_fixation_deduplication_and_refill():
    ids = (1, 2, 3, 4, 5)
    weights = {i: float(i) for i in ids}
    c = {i: 0.0 for i in ids}
    rows = k2_children(
        parents=[(1, 5), (2, 4)], line_ids=ids, weights=weights, c_by_line=c,
        lambda_r=0.8, children_per_parent=2, max_unique=4,
    )
    assert len(rows) == 4
    assert len({row.topology_key for row in rows}) == 4
    assert all(len(row.offline_branch_ids) == 2 for row in rows)
    assert all(int(row.parent_topology_key) in row.offline_branch_ids for row in rows)


def test_topology_relative_load_selection_is_deterministic(identity):
    offline = identity.l_trans[:1]
    first = select_topology_relative_loads(identity, offline, q=5)
    second = select_topology_relative_loads(identity, offline, q=5)
    assert first == second
    assert len(first) == 5


def test_common_objective_and_reference_delta_signs():
    assert compute_j_trade(0.8, 0.5, 0.1) == pytest.approx(0.42)
    deltas = reference_deltas(
        native_r_norm=0.6, reference_a_r_norm=0.5,
        native_j_trade=0.52, reference_a_j_trade=0.44,
        selected_service=0.8, reference_b_max_service=0.9,
    )
    assert deltas == pytest.approx({"delta_r_a": 0.1, "delta_j_a": 0.08, "delta_s_b": 0.1})


def test_risk_uses_all_energized_lines():
    p_env = {1: 1.0, 2: 0.5, 3: 0.2}
    loading = {1: 0.5, 2: 0.4, 3: 0.3}
    r_base = sum(p_env[i] * loading[i] ** 2 for i in p_env)
    assert compute_r_norm(p_env, [2], p_env, loading, r_base) == pytest.approx(
        (p_env[1] * loading[1] ** 2 + p_env[3] * loading[3] ** 2) / r_base
    )


def test_gridsfm_raw_schema_and_mapping(identity):
    raw = build_raw_gridsfm_case(identity)
    assert len(raw["grid"]["nodes"]["bus"]) == 2751
    assert len(raw["grid"]["nodes"]["load"]) == 1125
    assert len(raw["grid"]["nodes"]["generator"]) == 736
    assert len(raw["grid"]["edges"]["ac_line"]["features"]) == 3993
    assert len(raw["grid"]["edges"]["transformer"]["features"]) == 1351
    assert len(raw["grid"]["edges"]["ac_line"]["features"][0]) == 9
    assert len(raw["grid"]["edges"]["transformer"]["features"][0]) == 11


def test_gridsfm_mutation_opens_only_l_trans_and_clamps(identity):
    raw = build_raw_gridsfm_case(identity)
    offline = identity.l_trans[:1]
    mutated, alpha, islanded = mutate_candidate(raw, identity, offline_branch_ids=offline, alpha_requested={})
    assert len(mutated["grid"]["edges"]["ac_line"]["features"]) == 3992
    assert len(mutated["grid"]["edges"]["transformer"]["features"]) == 1351
    assert all(alpha[load_id] == 0.0 for load_id in islanded)
    with pytest.raises(ValueError, match="non-L_trans"):
        mutate_candidate(raw, identity, offline_branch_ids=identity.l_fixed[:1], alpha_requested={})


def test_alpha_search_obeys_exact_budget():
    calls = []
    def evaluator(alpha):
        calls.append(alpha)
        return sum((value - 0.5) ** 2 for value in alpha.values())
    result = budgeted_powell_search(
        load_ids=[1, 2, 3], selected_load_ids=[1, 2], budget=5,
        evaluator=evaluator, objective_getter=float,
    )
    assert len(calls) == 5
    assert len(result.trace) == 5


def test_empirical_nondominance():
    frame = pd.DataFrame({"r_norm": [1.0, 0.8, 0.7, 0.9], "l_shed_total": [0.0, 0.1, 0.2, 0.3]})
    assert empirical_nondominated(frame).tolist() == [True, True, True, False]


def test_full_run_extrapolation_includes_gridsfm_alpha_budget_change():
    dc = full_run_workload_ratio("dc")
    gridsfm = full_run_workload_ratio("gridsfm")
    assert dc["topology_ratio_full_to_smoke"] == pytest.approx((301 * 5) / 21)
    assert dc["inner_evaluation_ratio_full_to_smoke"] == 1.0
    assert gridsfm["inner_evaluation_ratio_full_to_smoke"] == 4.0
    assert gridsfm["workload_ratio_full_to_smoke"] == pytest.approx(
        4 * dc["workload_ratio_full_to_smoke"]
    )


def test_reference_b_other_error_is_not_solver_eligible():
    assert classify_solver_status("OTHER_ERROR") not in {"optimal_success", "locally_solved"}


def test_production_chunk_validation_enforces_counts_hashes_and_j_total(tmp_path):
    config = load_config(ROOT / "config" / "full.yaml")
    prepared = tmp_path / "prepared"
    chunk = tmp_path / "chunk"
    prepared.mkdir()
    chunk.mkdir()
    k1_keys = [str(index) for index in range(50)]
    pd.DataFrame({"lambda_r": [0.0] * 50, "topology_key": k1_keys}).to_parquet(
        prepared / "shared_k1.parquet", index=False
    )
    rows = []
    for k, keys in (
        (0, ["intact"]),
        (1, k1_keys),
        (2, [f"{index};{1000 + index}" for index in range(250)]),
    ):
        for rank, key in enumerate(keys):
            j_trade = float(rank + k) / 1000.0
            pac_total = 0.5
            rows.append({
                "evaluator": "gridsfm", "lambda_r": 0.0, "k": k,
                "topology_key": key, "offline_branch_ids": "" if k == 0 else key,
                "eligible": True, "search_objective": j_trade + 2.0 * pac_total,
                "j_trade": j_trade, "pac_operational": 0.4, "pac_ac": 0.1,
                "pac_total": pac_total, "j_total": j_trade + 2.0 * pac_total,
            })
    candidates = pd.DataFrame(rows)
    candidates.to_parquet(chunk / "gridsfm_candidate_results.parquet", index=False)
    parents = set(
        candidates.loc[candidates.k.eq(1)].sort_values(["search_objective", "topology_key"])
        .head(5).topology_key.astype(str)
    )
    k2 = candidates.loc[candidates.k.eq(2), ["topology_key"]].copy()
    ordered_parents = sorted(parents)
    k2["parent_topology_key"] = [ordered_parents[index % 5] for index in range(250)]
    k2.to_parquet(chunk / "gridsfm_k2_candidates.parquet", index=False)
    candidates.sort_values(["search_objective", "k", "topology_key"]).head(1).to_parquet(
        chunk / "gridsfm_finalists.parquet", index=False
    )
    input_sha = "input-identity"
    atomic_write_json(chunk / "gridsfm_run_summary.json", {
        "status": "PASS", "evaluator": "gridsfm", "config_sha256": config["config_sha256"],
        "input_sha256": input_sha, "lambda_values": [0.0], "attempted_states": 301,
        "eligible_states": 301, "finalist_count": 1,
    })
    observed, observed_k2, _ = _validate_chunk(
        evaluator="gridsfm", lambda_r=0.0, chunk=chunk, config=config,
        prepared=prepared, expected_input_sha=input_sha,
    )
    assert len(observed) == 301 and len(observed_k2) == 250
    broken = candidates.copy()
    broken.loc[100, "search_objective"] += 1.0
    broken.to_parquet(chunk / "gridsfm_candidate_results.parquet", index=False)
    with pytest.raises(ValueError, match="J_total"):
        _validate_chunk(
            evaluator="gridsfm", lambda_r=0.0, chunk=chunk, config=config,
            prepared=prepared, expected_input_sha=input_sha,
        )


def test_production_reference_join_requires_all_fifteen_tasks(tmp_path):
    evaluator_dir = tmp_path / "evaluators"
    task_root = tmp_path / "tasks"
    output = tmp_path / "joined"
    evaluator_dir.mkdir()
    lambdas = [0.0, 0.2, 0.5, 0.8, 1.0]
    config = load_config(ROOT / "config" / "full.yaml")
    for evaluator in ("gridsfm", "dc", "ac"):
        pd.DataFrame({
            "evaluator": [evaluator] * 5, "lambda_r": lambdas,
            "topology_key": [f"{evaluator}-{value}" for value in lambdas],
        }).to_parquet(evaluator_dir / f"{evaluator}_finalists.parquet", index=False)
        for value in lambdas:
            task = task_root / evaluator / f"lambda_{value:.6g}"
            task.mkdir(parents=True)
            base = {"evaluator": evaluator, "lambda_r": value, "topology_key": f"{evaluator}-{value}"}
            pd.DataFrame([{**base, "status": "locally_solved"}]).to_parquet(task / "reference_a.parquet", index=False)
            pd.DataFrame([{
                **base, "b1_status": "LOCALLY_SOLVED", "b2_status": "LOCALLY_SOLVED",
                "b2_solver_eligible": True, "diagnostic_state_source": "b2_economic_tiebreak",
                "economic_cost_tiebreak": 1.0,
            }]).to_parquet(task / "reference_b.parquet", index=False)
            pd.DataFrame([{**base, "delta_r_a": 0.0, "delta_j_a": 0.0, "delta_s_b": 0.0}]).to_parquet(task / "reference_discrepancies.parquet", index=False)
            atomic_write_json(task / "reference_run_summary.json", {
                "status": "PASS", "config_sha256": config["config_sha256"],
                "evaluators": [evaluator], "lambda_values": [value],
                "reference_b2_failure_count": 0,
            })
    summary = join_reference_tasks(
        config_path=ROOT / "config" / "full.yaml", evaluator_dir=evaluator_dir,
        task_root=task_root, output_dir=output,
    )
    assert summary["status"] == "PASS"
    assert summary["reference_a_rows"] == summary["reference_b_rows"] == 15


def test_status_classification():
    assert classify_solver_status("LOCALLY_SOLVED") == "locally_solved"
    assert classify_solver_status("INFEASIBLE") == "infeasible"
    assert classify_solver_status("MAXIMUM_ITERATIONS") == "iteration_limit"


def test_atomic_checkpoint_and_no_overwrite(tmp_path):
    checkpoint = atomic_write_json(tmp_path / "checkpoint.json", {"status": "complete", "run_id": "r1"})
    assert checkpoint_matches(checkpoint, {"run_id": "r1"})
    run = require_new_run_dir(tmp_path / "run")
    atomic_write_json(run / "RUN_COMPLETE.json", {"status": "complete"})
    with pytest.raises(FileExistsError, match="immutable"):
        require_new_run_dir(run, resume=True)


def test_candidate_resume_requires_matching_hashes_and_service_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_K_RUN_ID", "resume-test")
    calls = []
    def callback():
        calls.append(1)
        return CandidateResult(
            evaluator="dc", lambda_r=0.8, k=0, topology_key="intact",
            offline_branch_ids="", status="optimal_success", eligible=True,
            search_objective=0.5, alpha_effective_json='{"0": 1.0}',
        )
    kwargs = dict(
        callback=callback, checkpoint_dir=tmp_path, evaluator="dc", lambda_r=0.8,
        offline=(), config_sha256="config", input_sha256="input",
    )
    first = _candidate_with_checkpoint(**kwargs)
    second = _candidate_with_checkpoint(**kwargs)
    assert first.search_objective == second.search_objective == 0.5
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="mismatched"):
        _candidate_with_checkpoint(**{**kwargs, "config_sha256": "different"})


def test_native_ac_source_contains_all_line_epigraph_and_physical_export():
    source = (ROOT / "src" / "native_opf" / "stage_k_native_opf.jl").read_text(encoding="utf-8")
    assert "for (branch_id, probability) in P_ENV" in source
    assert "epigraph[branch_id] >= (p[(branch_id,f,t)]^2 + q[(branch_id,f,t)]^2)" in source
    assert "physical_loading,risk_epigraph" in source


def test_generation_cost_diagnostic_handles_missing_cost_vector():
    source = (ROOT / "src" / "native_opf" / "stage_k_native_opf.jl").read_text(encoding="utf-8")
    assert "isempty(coefficients) && return nothing" in source


def test_native_dc_loading_uses_active_power_only():
    source = (ROOT / "src" / "native_opf" / "stage_k_native_opf.jl").read_text(encoding="utf-8")
    assert 'MODE == "native_dc" ? max(abs(pf), abs(pt))' in source


def test_reference_modes_and_no_warm_start():
    source = (ROOT / "src" / "native_opf" / "stage_k_native_opf.jl").read_text(encoding="utf-8")
    assert 'MODE == "reference_a"' in source
    assert 'MODE == "reference_b"' in source
    assert "warm_start" not in source.lower()


def test_reference_b_uses_b1_fallback_when_economic_tiebreak_fails():
    b1 = {"termination_status": "LOCALLY_SOLVED", "branch_state_csv": "b1.csv"}
    b2 = {"termination_status": "OTHER_ERROR", "branch_state_csv": "b2.csv"}
    selected, b2_valid = _reference_b_diagnostic_summary(b1, b2)
    assert selected is b1
    assert not b2_valid
    with pytest.raises(RuntimeError, match="B1 maximum-service"):
        _reference_b_diagnostic_summary({"termination_status": "OTHER_ERROR"}, b2)


def test_baseline_deployment_audit_gates_on_solver_and_branch_integrity_only():
    state = pd.DataFrame({
        "powermodels_branch_id": [1, 2, 3],
        "physical_loading": [0.2, 1.4, 0.8],
    })
    assert all(deployment_audit_checks("LOCALLY_SOLVED", state, {1, 2, 3}).values())
    assert not deployment_audit_checks("INFEASIBLE", state, {1, 2, 3})["solver_status_ok"]
    assert not deployment_audit_checks("LOCALLY_SOLVED", state.iloc[:2], {1, 2, 3})[
        "branch_output_complete"
    ]
    nonfinite = state.copy()
    nonfinite.loc[1, "physical_loading"] = np.nan
    assert not deployment_audit_checks("LOCALLY_SOLVED", nonfinite, {1, 2, 3})[
        "branch_output_finite"
    ]


def test_baseline_audit_preserves_discrepancy_as_diagnostic():
    source = (ROOT / "src" / "baseline_audit.py").read_text(encoding="utf-8")
    assert '"loading_comparison_role": "diagnostic_only"' in source
    assert '"loading_comparison_gate": False' in source
    assert '"baseline_replaced": False' in source
    assert '"status": "PASS" if all(checks.values()) else "FAIL"' in source


def test_slurm_package_is_dry_run_guarded_and_account_free():
    submit = (ROOT / "pace" / "submit_smoke.sh").read_text(encoding="utf-8")
    assert "--submit" in submit and "DRY RUN" in submit
    assert "STAGE_K_CPU_RESOURCE_ARGS" in submit
    for path in (ROOT / "pace").glob("*.sbatch"):
        text = path.read_text(encoding="utf-8")
        assert "#SBATCH --account" not in text
        if "--cpus-per-task=2" in text:
            assert "--cpus-per-task=2" in text
            assert "--mem=8G" in text
        else:
            assert "--cpus-per-task=8" in text
            assert "--mem=32G" in text or "--mem=8G" in text


def test_gridsfm_environment_contract_is_exact_and_job_is_offline():
    contract = load_gridsfm_contract()
    assert contract["source"] == {
        "git_commit": "1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6",
        "git_repository": "https://github.com/microsoft/GridSFM.git",
        "package_name": "gridsfm",
        "package_version": "1.1.0",
    }
    assert contract["checkpoint"] == {
        "filename": "gridsfm_open_v1.1.pt",
        "repository": "microsoft/GridSFM_Open",
        "revision": "1b41299b80252adf1869d5c3b479a4a402c52591",
        "sha256": "f8a4396122e603e8303afdebe3b093819c0f64dac0878394aed0bd63205fd831",
    }
    assert contract["runtime"]["python"] == "3.11.9"
    assert contract["runtime"]["packages"]["torch"] == "2.7.1+cu126"
    assert contract["runtime"]["packages"]["lightning"] == "2.6.6"
    assert contract["device"] == {
        "cuda_device": "cuda:0",
        "driver_requirement": "NVIDIA driver compatible with CUDA 12.6",
        "expected_cuda_runtime": "12.6",
        "expected_compute_capability": [8, 0],
        "expected_gpu_count": 1,
        "required_name_fragment": "A100",
    }
    setup = (ROOT / "environment" / "setup_gridsfm_a100.sh").read_text(encoding="utf-8")
    gpu_job = (ROOT / "pace" / "stage_k_gridsfm_a100.sbatch").read_text(encoding="utf-8")
    assert '"$STAGE_K_GRIDSFM_ROOT/model"' in setup
    assert "requirements.txt" not in setup
    assert "HF_HUB_OFFLINE=1" in gpu_job
    assert "STAGE_K_GRIDSFM_ENV_MANIFEST" in gpu_job
    evaluator = (ROOT / "src" / "gridsfm_evaluator.py").read_text(encoding="utf-8")
    assert "sys.path" not in evaluator


def test_gridsfm_contract_checks_reject_import_stub_and_wrong_gpu():
    contract = load_gridsfm_contract()
    manifest = {
        "source": {
            "git_repository": contract["source"]["git_repository"],
            "git_commit": contract["source"]["git_commit"],
            "git_status_porcelain": "",
            "package_version": contract["source"]["package_version"],
            "package_in_prefix": True,
        },
        "checkpoint": {**contract["checkpoint"]},
        "runtime": {
            "python": contract["runtime"]["python"],
            "packages": contract["runtime"]["packages"],
            "huggingface_hub_file": None,
            "huggingface_hub_in_prefix": False,
        },
        "device": {
            "requested_device": "cuda:0",
            "cuda_available": True,
            "driver_version": "570.00",
            "cuda_runtime": "12.6",
            "gpu_count": 1,
            "name": "NVIDIA H100 80GB HBM3",
            "capability": [9, 0],
        },
    }
    checks = contract_checks(manifest, contract)
    assert not checks["huggingface_hub_real_import"]
    assert not checks["huggingface_hub_from_environment"]
    assert not checks["required_gpu_visible"]


def test_smoke_v100_contract_changes_only_device_contract():
    production = load_gridsfm_contract()
    smoke = load_gridsfm_contract(ROOT / "environment" / "GRIDSFM_SMOKE_V100_CONTRACT.json")
    assert smoke["source"] == production["source"]
    assert smoke["checkpoint"] == production["checkpoint"]
    assert smoke["runtime"] == production["runtime"]
    assert smoke["device"]["required_name_fragment"] == "V100"
    assert smoke["device"]["expected_compute_capability"] == [7, 0]
