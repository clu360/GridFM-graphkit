import numpy as np
import pytest


def test_case30_metadata_maps_active_physical_edges():
    pytest.importorskip("torch")
    from experiments.test.wildfire_tests.gridfm_support.branch_metadata import physical_line_ids
    from experiments.test.wildfire_tests.gridfm_support.pipeline_utils import load_single_test_scenario

    context = load_single_test_scenario()
    scenario = context.scenario
    physical_ids = physical_line_ids(scenario)

    assert len(physical_ids) > 0
    assert int(np.sum(scenario.is_self_loop)) == scenario.num_buses
    assert all(str(scenario.branch_mapping_status[line_id]) == "mapped" for line_id in physical_ids)
    assert all(np.isfinite(float(scenario.rate_a[line_id])) for line_id in physical_ids)
    assert not any(bool(scenario.is_self_loop[line_id]) for line_id in physical_ids)


def test_corrected_loading_uses_physical_branch_pairs_and_excludes_self_loops():
    pytest.importorskip("torch")
    from experiments.test.wildfire_tests.gridfm_support.overload_penalty import OverloadPenaltyEvaluator
    from experiments.test.wildfire_tests.gridfm_support.pipeline_utils import load_single_test_scenario

    scenario = load_single_test_scenario().scenario
    evaluator = OverloadPenaltyEvaluator(scenario)
    loading = evaluator.compute_loading(scenario.Vm_base, scenario.Va_base)

    assert loading.shape[0] == scenario.edge_index.shape[1]
    assert np.allclose(loading[scenario.is_self_loop], 0.0)
    for directed_ids in scenario.physical_branch_directed_line_ids.values():
        directed_loading = loading[np.asarray(directed_ids, dtype=int)]
        assert np.allclose(directed_loading, directed_loading[0])


def test_stage_g_corrected_gnn_baseline_is_physically_plausible():
    pytest.importorskip("torch")
    from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import _build_model_context

    context = _build_model_context("gnn", 0.30)
    state = context["baseline_state"]
    vm = np.asarray(state["Vm"], dtype=float)
    loading = np.asarray(state["loading_ratio"], dtype=float)

    assert float(np.nanmin(vm)) > 0.0
    assert float(np.nanmax(vm)) < 2.0
    assert float(loading[23]) < 10.0
    assert not bool(state["impossible_voltage"])


def test_stage_g_audit_frame_schema_and_outlier_columns():
    pytest.importorskip("torch")
    from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_loading_audit import (
        REQUIRED_COLUMNS,
        build_loading_ranking_frame,
    )

    frame = build_loading_ranking_frame("gnn", 0.30)

    assert list(frame.columns) == REQUIRED_COLUMNS
    assert len(frame) > 0
    assert not frame["is_self_loop"].any()
    assert set(frame["mapping_status"]) == {"mapped"}
    assert "outlier_flag" in frame
    assert frame["outlier_flag"].any()
    assert frame["rank"].tolist() == list(range(1, len(frame) + 1))


def test_matpower_flow_comparison_subset_confirms_prediction_drives_outlier():
    pytest.importorskip("torch")
    from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_loading_audit import (
        build_matpower_comparison_frame,
    )

    frame = build_matpower_comparison_frame("gnn", 0.30, top_n=3)

    assert len(frame) == 3
    assert (frame["matpower_ac_base_loading"] < 1.0).all()
    assert (frame["matpower_ac_pred_loading"] > 2.0).any()
    assert (frame["ratio_ybus_to_matpower_pred"] > 0.1).all()
    assert (frame["ratio_ybus_to_matpower_pred"] < 10.0).all()


def test_stage_g_probability_suppression_preserves_explicit_targets():
    from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import (
        SCENARIOS,
        p_env_for_scenario,
    )

    s1 = p_env_for_scenario(
        SCENARIOS["S1"],
        110,
        suppressed_value=0.005,
        extra_suppressed_line_ids=[23, 26],
    )
    s2 = p_env_for_scenario(
        SCENARIOS["S2"],
        110,
        suppressed_value=0.005,
        extra_suppressed_line_ids=[23, 26],
    )

    assert s1[23] == pytest.approx(0.005)
    assert s1[26] == pytest.approx(0.005)
    assert s1[27] == pytest.approx(1.0)
    assert s2[23] == pytest.approx(1.0)
    assert s2[26] == pytest.approx(0.005)


def test_stage_g_rank_calibrated_p_env_suppresses_only_non_target_outliers():
    from dataclasses import replace

    from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import SCENARIOS
    from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_physics_sensitivity import (
        calibrate_p_env_for_scenario,
    )

    scenario = replace(SCENARIOS["S2"], target_high_risk_line_ids=(2,), expected_target_set=(2,))
    loading = np.ones(6)
    impact = {0: 100.0, 1: 10.0, 2: 3.0, 3: 1.0, 4: 1.0, 5: 1.0}
    p_env, diagnostics = calibrate_p_env_for_scenario(
        scenario,
        6,
        physical_ids=[0, 1, 2, 3, 4, 5],
        baseline_loading=loading,
        impact_by_line=impact,
        outlier_line_ids=[0, 1, 2],
    )

    assert p_env[0] < 0.05
    assert p_env[1] == pytest.approx(0.05)
    assert p_env[2] == pytest.approx(1.0)
    assert p_env[3] == pytest.approx(0.05)
    assert set(diagnostics.columns).issuperset(
        {"scenario_id", "line_id", "old_p_env", "calibrated_p_env", "old_rank", "new_rank"}
    )


def test_stage_g_rank_calibrated_p_env_preserves_outlier_target():
    from dataclasses import replace

    from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import SCENARIOS
    from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_physics_sensitivity import (
        calibrate_p_env_for_scenario,
    )

    scenario = replace(SCENARIOS["S1"], target_high_risk_line_ids=(0,), expected_target_set=(0,))
    loading = np.ones(3)
    impact = {0: 100.0, 1: 10000.0, 2: 1.0}
    p_env, _ = calibrate_p_env_for_scenario(
        scenario,
        3,
        physical_ids=[0, 1, 2],
        baseline_loading=loading,
        impact_by_line=impact,
        outlier_line_ids=[0, 1],
    )

    assert p_env[0] == pytest.approx(1.0)
    assert p_env[1] < 0.05
    assert p_env[2] == pytest.approx(0.05)


def test_stage_g_rho_rescoring_changes_only_objective_terms():
    from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_physics_sensitivity import (
        rescore_topology_pool,
    )

    pool = pytest.importorskip("pandas").DataFrame(
        [
            {
                "topology_pool_id": 1,
                "scenario_id": "S1",
                "stage": "stage_d_k2_exhaustive",
                "lambda_R": 0.5,
                "R_norm": 0.2,
                "L_shed": 0.4,
                "PAC_total": 0.3,
                "status": "ok",
            }
        ]
    )
    rescored = rescore_topology_pool(pool, [0.0, 10.0])

    assert rescored["R_norm"].nunique() == 1
    assert rescored["L_shed"].nunique() == 1
    assert rescored["PAC_total"].nunique() == 1
    assert rescored.loc[rescored["rho_phys"].eq(0.0), "J_true"].iloc[0] == pytest.approx(0.3)
    assert rescored.loc[rescored["rho_phys"].eq(10.0), "J_true"].iloc[0] == pytest.approx(3.3)


def test_stage_g_methodology_checker_fails_if_rho_changes_intrinsic_metrics():
    pd = pytest.importorskip("pandas")
    from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_physics_sensitivity import (
        methodology_fidelity_checks,
    )

    class Scenario:
        is_self_loop = np.array([False, False])
        canonical_line_id = np.array([0, 1])

    calibration = pd.DataFrame(
        [
            {
                "scenario_id": "S1",
                "line_id": 0,
                "is_target": True,
                "is_gridfm_heavy_loading_outlier": False,
                "calibrated_p_env": 1.0,
            },
            {
                "scenario_id": "S1",
                "line_id": 1,
                "is_target": False,
                "is_gridfm_heavy_loading_outlier": False,
                "calibrated_p_env": 0.05,
            },
        ]
    )
    metric_pool = pd.DataFrame(
        [
            {
                "topology_pool_id": 1,
                "continuous_recourse_optimized": False,
                "fixed_control_u_base": True,
            }
        ]
    )
    rescored = pd.DataFrame(
        [
            {
                "topology_pool_id": 1,
                "rho_phys": 0.0,
                "R_norm": 1.0,
                "L_shed": 0.0,
                "PAC_total": 1.0,
                "R_raw": 1.0,
                "R_base_s": 1.0,
                "p_env_calibration_mode": "rank",
                "lambda_R": 0.5,
                "lambda_L": 0.5,
                "stage": "stage_d_k2_exhaustive",
                "shutoff_line_ids": "",
                "J_true": 0.5,
                "J_no_phys": 0.5,
                "risk_contribution": 0.5,
                "load_contribution": 0.0,
                "physics_contribution": 0.0,
            },
            {
                "topology_pool_id": 1,
                "rho_phys": 10.0,
                "R_norm": 2.0,
                "L_shed": 0.0,
                "PAC_total": 1.0,
                "R_raw": 1.0,
                "R_base_s": 1.0,
                "p_env_calibration_mode": "rank",
                "lambda_R": 0.5,
                "lambda_L": 0.5,
                "stage": "stage_d_k2_exhaustive",
                "shutoff_line_ids": "",
                "J_true": 11.0,
                "J_no_phys": 1.0,
                "risk_contribution": 1.0,
                "load_contribution": 0.0,
                "physics_contribution": 10.0,
            },
        ]
    )

    checks = methodology_fidelity_checks(
        metric_pool,
        rescored,
        calibration,
        Scenario(),
        physical_ids=[0, 1],
        rho_values=[0.0, 10.0],
        candidate_line_ids=[0, 1],
    )

    row = checks[checks["check_name"].eq("rho_does_not_change_intrinsic_metrics")].iloc[0]
    assert not bool(row["passed"])


def test_stage_g_scenario_baseline_p_env_keeps_targets_high_and_non_targets_low():
    from dataclasses import replace

    from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import SCENARIOS
    from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_scenario_baseline_physics_sensitivity import (
        P_ENV_MODE,
        p_env_for_scenario_baseline_revision,
    )

    scenario = replace(SCENARIOS["S1"], target_high_risk_line_ids=(1, 3), expected_target_set=(1, 3))
    p_env, table = p_env_for_scenario_baseline_revision(
        scenario,
        num_lines=5,
        physical_ids=[0, 1, 2, 3, 4],
    )

    assert p_env[1] == pytest.approx(1.0)
    assert p_env[3] == pytest.approx(1.0)
    assert p_env[0] == pytest.approx(0.05)
    assert p_env[2] == pytest.approx(0.05)
    assert p_env[4] == pytest.approx(0.05)
    assert table["p_env_mode"].eq(P_ENV_MODE).all()


def test_stage_g_scenario_baseline_methodology_checker_fails_if_wrong_loading_source():
    pd = pytest.importorskip("pandas")
    from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_scenario_baseline_physics_sensitivity import (
        BASELINE_LOADING_SOURCE,
        P_ENV_MODE,
        POST_TOPOLOGY_EVALUATION_SOURCE,
        scenario_baseline_methodology_fidelity_checks,
    )

    class Scenario:
        is_self_loop = np.zeros(24, dtype=bool)
        canonical_line_id = np.arange(24)

    ranking = pd.DataFrame(
        [
            {"canonical_line_id": 0, "loading_ratio": 0.2},
            {"canonical_line_id": 1, "loading_ratio": 0.3},
            {"canonical_line_id": 23, "loading_ratio": 0.5243973363957892},
        ]
    )
    p_env = pd.DataFrame(
        [
            {"scenario_id": "S1", "line_id": 0, "is_target": True, "p_env": 1.0, "p_env_mode": P_ENV_MODE},
            {"scenario_id": "S1", "line_id": 1, "is_target": False, "p_env": 0.05, "p_env_mode": P_ENV_MODE},
            {"scenario_id": "S1", "line_id": 23, "is_target": False, "p_env": 0.05, "p_env_mode": P_ENV_MODE},
        ]
    )
    metric_pool = pd.DataFrame(
        [
            {
                "topology_pool_id": 1,
                "scenario_id": "S1",
                "continuous_recourse_optimized": False,
                "fixed_control_u_base": True,
                "baseline_loading_source": "gridfm_inferred_baseline",
                "R_base_s_source": BASELINE_LOADING_SOURCE,
                "proxy_loading_source": BASELINE_LOADING_SOURCE,
                "post_topology_evaluation_source": POST_TOPOLOGY_EVALUATION_SOURCE,
                "p_env_mode": P_ENV_MODE,
                "R_base_s": 0.202,
            }
        ]
    )
    rescored = pd.DataFrame(
        [
            {
                "topology_pool_id": 1,
                "rho_phys": 0.0,
                "R_raw": 0.1,
                "R_norm": 0.5,
                "L_shed": 0.0,
                "PAC_total": 1.0,
                "max_loading_ratio": 0.5,
                "gridfm_status": "ok",
                "shutoff_line_ids": "",
                "p_env_mode": P_ENV_MODE,
                "lambda_R": 0.5,
                "lambda_L": 0.5,
                "stage": "stage_d_k2_exhaustive",
                "J_true": 0.25,
                "J_no_phys": 0.25,
                "risk_contribution": 0.25,
                "load_contribution": 0.0,
                "physics_contribution": 0.0,
            },
            {
                "topology_pool_id": 1,
                "rho_phys": 0.1,
                "R_raw": 0.1,
                "R_norm": 0.5,
                "L_shed": 0.0,
                "PAC_total": 1.0,
                "max_loading_ratio": 0.5,
                "gridfm_status": "ok",
                "shutoff_line_ids": "",
                "p_env_mode": P_ENV_MODE,
                "lambda_R": 0.5,
                "lambda_L": 0.5,
                "stage": "stage_d_k2_exhaustive",
                "J_true": 0.35,
                "J_no_phys": 0.25,
                "risk_contribution": 0.25,
                "load_contribution": 0.0,
                "physics_contribution": 0.1,
            },
        ]
    )

    checks = scenario_baseline_methodology_fidelity_checks(
        metric_pool,
        rescored,
        p_env,
        ranking,
        Scenario(),
        physical_ids=[0, 1, 23],
        rho_values=[0.0, 0.1],
        candidate_line_ids=[0, 1],
    )

    row = checks[checks["check_name"].eq("baseline_loading_source_recorded")].iloc[0]
    assert not bool(row["passed"])


def test_stage_g_scenario_baseline_methodology_checker_fails_if_p_env_suppressed():
    pd = pytest.importorskip("pandas")
    from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_scenario_baseline_physics_sensitivity import (
        BASELINE_LOADING_SOURCE,
        P_ENV_MODE,
        POST_TOPOLOGY_EVALUATION_SOURCE,
        scenario_baseline_methodology_fidelity_checks,
    )

    class Scenario:
        is_self_loop = np.zeros(24, dtype=bool)
        canonical_line_id = np.arange(24)

    ranking = pd.DataFrame(
        [
            {"canonical_line_id": 0, "loading_ratio": 0.2},
            {"canonical_line_id": 1, "loading_ratio": 0.3},
            {"canonical_line_id": 23, "loading_ratio": 0.5243973363957892},
        ]
    )
    p_env = pd.DataFrame(
        [
            {"scenario_id": "S1", "line_id": 0, "is_target": True, "p_env": 1.0, "p_env_mode": P_ENV_MODE},
            {"scenario_id": "S1", "line_id": 1, "is_target": False, "p_env": 0.005, "p_env_mode": P_ENV_MODE},
            {"scenario_id": "S1", "line_id": 23, "is_target": False, "p_env": 0.05, "p_env_mode": P_ENV_MODE},
        ]
    )
    metric_pool = pd.DataFrame(
        [
            {
                "topology_pool_id": 1,
                "scenario_id": "S1",
                "continuous_recourse_optimized": False,
                "fixed_control_u_base": True,
                "baseline_loading_source": BASELINE_LOADING_SOURCE,
                "R_base_s_source": BASELINE_LOADING_SOURCE,
                "proxy_loading_source": BASELINE_LOADING_SOURCE,
                "post_topology_evaluation_source": POST_TOPOLOGY_EVALUATION_SOURCE,
                "p_env_mode": P_ENV_MODE,
                "R_base_s": 0.157,
            }
        ]
    )
    rescored = pd.DataFrame(
        [
            {
                "topology_pool_id": 1,
                "rho_phys": 0.0,
                "R_raw": 0.1,
                "R_norm": 0.5,
                "L_shed": 0.0,
                "PAC_total": 1.0,
                "max_loading_ratio": 0.5,
                "gridfm_status": "ok",
                "shutoff_line_ids": "",
                "p_env_mode": P_ENV_MODE,
                "lambda_R": 0.5,
                "lambda_L": 0.5,
                "stage": "stage_d_k2_exhaustive",
                "J_true": 0.25,
                "J_no_phys": 0.25,
                "risk_contribution": 0.25,
                "load_contribution": 0.0,
                "physics_contribution": 0.0,
            }
        ]
    )

    checks = scenario_baseline_methodology_fidelity_checks(
        metric_pool,
        rescored,
        p_env,
        ranking,
        Scenario(),
        physical_ids=[0, 1, 23],
        rho_values=[0.0],
        candidate_line_ids=[0, 1],
    )

    row = checks[checks["check_name"].eq("non_target_lines_have_low_p_env")].iloc[0]
    assert not bool(row["passed"])
