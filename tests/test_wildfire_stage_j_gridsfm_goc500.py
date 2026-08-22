from __future__ import annotations

import math
import importlib.util
import sys
import types

import pytest

from experiments.test.wildfire_tests.stage_j_gridsfm_goc500 import (
    AlphaEvaluation,
    AlphaSearchConfig,
    ACReferenceResult,
    EvaluationStatus,
    InputIntegrityReport,
    PacWeights,
    ScreenedScipyAlphaConfig,
    UnitCompatibilityReport,
    assert_raw_gridsfm_case,
    build_goc500_identity,
    compute_ac_loading_two_ended,
    compute_alpha_effective,
    compute_j_total_sfm,
    compute_j_trade,
    compute_load_shedding,
    compute_pac_total,
    compute_r_base,
    mutate_raw_case_for_candidate,
    optimize_full_alpha,
    optimize_screened_scipy_alpha,
    require_valid_risk_ratings,
    source_less_load_ids,
)
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.dc_economic_recourse import (
    FixedTopologyDCEconomicRequest,
    FixedTopologyDCEconomicResult,
    solve_fixed_topology_economic_dc_opf,
)
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.gridsfm_evaluator import evaluate_gridsfm_candidate
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.metrics import require_unit_compatibility
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.outer_proxy import solve_proxy_topology_pool
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.run_j8_budgeted_alpha_topology_smoke import (
    CandidateEvaluation,
    build_th_gridsfm_topology_pool,
    run_budgeted_alpha_search,
    select_budgeted_alpha_loads,
)
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.scenario_builder import (
    connectivity_service_impact_proxy,
    construct_stage_j_s1_s3_scenarios,
    has_alternate_path_after_outage,
)


def test_full_per_load_alpha_and_source_less_override():
    load_ids = [10, 20, 30]
    effective = compute_alpha_effective(load_ids, {10: 1.0, 20: 0.7, 30: 0.4}, [20])
    assert effective == {10: 1.0, 20: 0.0, 30: 0.4}

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_alpha_effective(load_ids, {10: 1.1, 20: 0.7, 30: 0.4}, [])


def test_load_shedding_decomposition_handles_source_less_generators():
    source_less = (bus for bus in [20])
    result = compute_load_shedding(
        [10, 20, 30],
        {10: 50.0, 20: 30.0, 30: 20.0},
        {10: 1.0, 20: 0.9, 30: 0.5},
        source_less,
    )

    assert result.alpha_requested[20] == 0.9
    assert result.alpha_effective[20] == 0.0
    assert result.l_shed_island == pytest.approx(0.30)
    assert result.l_shed_control == pytest.approx(0.10)
    assert result.l_shed_total == pytest.approx(0.40)


def test_two_ended_ac_loading_and_ratea_guard():
    loading = compute_ac_loading_two_ended(
        p_from={1: 3.0},
        q_from={1: 4.0},
        p_to={1: 6.0},
        q_to={1: 8.0},
        ratea_by_line={1: 20.0},
        line_ids=[1],
    )
    assert loading[1] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="rateA"):
        compute_ac_loading_two_ended({1: 1.0}, {1: 0.0}, {1: 1.0}, {1: 0.0}, {1: 0.0}, [1])


def test_r_base_requires_positive_denominator():
    assert compute_r_base({1: 0.4, 2: 0.1}, {1: 2.0, 2: 1.0}, [1, 2]) == pytest.approx(1.7)
    with pytest.raises(ValueError, match="R_base"):
        compute_r_base({1: 0.0}, {1: 0.0}, [1])


def test_grid_sfm_trade_and_total_objective_are_separate():
    weights = PacWeights(rho_phys=2.0, w_op=0.5, w_ac=0.25, w_model=0.25)
    result = compute_j_total_sfm(
        lambda_r=0.8,
        r_norm=0.5,
        l_shed_total=0.1,
        pac_operational=0.2,
        pac_ac=0.4,
        pac_model=0.8,
        weights=weights,
    )
    assert result.j_trade == pytest.approx(compute_j_trade(0.8, 0.5, 0.1))
    assert result.pac_total == pytest.approx(0.4)
    assert result.j_total == pytest.approx(result.j_trade + 2.0 * 0.4)

    with pytest.raises(ValueError, match="finite"):
        compute_pac_total(0.0, 0.0, 0.0, PacWeights(rho_phys=math.nan, w_op=1.0, w_ac=1.0, w_model=1.0))


def test_unit_gate_is_hard():
    require_unit_compatibility(UnitCompatibilityReport(100.0, "per_unit_on_baseMVA", "per_unit_on_baseMVA", True))
    with pytest.raises(ValueError, match="unit compatibility failed"):
        require_unit_compatibility(UnitCompatibilityReport(100.0, "MW", "MVA", False, "ambiguous base"))


def test_input_integrity_failure_is_not_soft_penalty():
    report = InputIntegrityReport({"topology": True, "Pd": False, "rateA": True}, notes="Pd drift")
    assert report.d_input == pytest.approx(1.0 / 3.0)
    assert report.evaluation_status is EvaluationStatus.INPUT_INTEGRITY_FAILURE
    with pytest.raises(ValueError, match="input-integrity failure"):
        report.require_ok()


def test_ac_reference_infeasible_uses_na_distance():
    result = ACReferenceResult.infeasible("fixed_z_alpha_economic_ac_opf_reference", "ipopt infeasible")
    assert result.evaluation_status is EvaluationStatus.AC_REFERENCE_INFEASIBLE
    assert result.d_state_to_ac is None


def test_guided_dc_contract_rejects_bad_candidate():
    req = FixedTopologyDCEconomicRequest(
        topology_z={1: 1, 2: 0},
        alpha_requested={10: 1.0, 20: 0.0},
        electrical_scenario_id="e0",
        wildfire_scenario_id="S1",
    )
    req.validate_no_binary_inner_decisions()

    infeasible = FixedTopologyDCEconomicResult.infeasible("fixed topology has no feasible DC dispatch")
    assert infeasible.evaluation_status is EvaluationStatus.DC_INFEASIBLE
    assert math.isinf(infeasible.rejection_merit)

    with pytest.raises(TypeError, match="missing required keyword"):
        solve_fixed_topology_economic_dc_opf()


def _tiny_gridsfm_raw_case():
    return {
        "metadata": {
            "bus_id_map": [101, 102, 103],
            "gen_bus_map": [101],
            "load_id_map": [201, 202],
            "load_bus_map": [102, 103],
            "ac_line_branch_ids": [301, 302],
            "transformer_branch_ids": [401],
        },
        "grid": {
            "nodes": {
                "bus": [[138, 3, 0.9, 1.1], [138, 1, 0.9, 1.1], [138, 1, 0.9, 1.1]],
                "generator": [[100, 0, 0, 2, 0, -1, 1, 1, 0, 1, 0]],
                "load": [[1.0, 0.2], [2.0, 0.4]],
                "shunt": [],
            },
            "edges": {
                "ac_line": {
                    "senders": [0, 1],
                    "receivers": [1, 2],
                    "features": [
                        [-0.5, 0.5, 0, 0, 0.01, 0.1, 3.0, 3.0, 3.0],
                        [-0.5, 0.5, 0, 0, 0.01, 0.1, 4.0, 4.0, 4.0],
                    ],
                },
                "transformer": {
                    "senders": [0],
                    "receivers": [2],
                    "features": [[-0.5, 0.5, 0.01, 0.1, 5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]],
                },
                "load_link": {"senders": [0, 1], "receivers": [1, 2]},
                "generator_link": {"senders": [0], "receivers": [0]},
                "shunt_link": {"senders": [], "receivers": []},
            },
        },
        "solution": {
            "edges": {
                "ac_line": [[0.1, 0.0, -0.1, 0.0], [0.2, 0.0, -0.2, 0.0]],
                "transformer": [[0.3, 0.0, -0.3, 0.0]],
            }
        },
    }


def test_goc500_identity_and_risk_rating_gate():
    identity = build_goc500_identity(_tiny_gridsfm_raw_case(), candidate_branch_ids=[301])
    assert len(identity.branches) == 3
    assert identity.branch_by_id[301].edge_family == "ac_line"
    assert identity.branch_by_id[301].original_case_branch_id == 301
    assert identity.branch_by_id[301].orientation_sign == 1
    assert identity.branch_by_id[301].family_index == 0
    assert identity.branch_by_id[301].from_bus_id == 101
    assert identity.branch_by_id[301].to_bus_id == 102
    assert identity.branch_by_id[301].is_risk is True
    assert identity.branch_by_id[401].is_risk is False
    assert identity.branch_by_id[301].is_candidate is True
    assert identity.load_by_id[202].pd_pre == pytest.approx(2.0)
    require_valid_risk_ratings(identity)

    bad = _tiny_gridsfm_raw_case()
    bad["grid"]["edges"]["ac_line"]["features"][0][6] = 0.0
    with pytest.raises(ValueError, match="invalid rateA"):
        require_valid_risk_ratings(build_goc500_identity(bad))


def test_raw_case_gate_rejects_prepared_artifacts():
    raw = _tiny_gridsfm_raw_case()
    assert assert_raw_gridsfm_case(raw).evaluation_status is EvaluationStatus.OK

    prepared_like = _tiny_gridsfm_raw_case()
    prepared_like["grid"]["nodes"]["branch_ac"] = [[1.0]]
    prepared_like["grid"]["edges"]["endpoint_of"] = {"senders": [0], "receivers": [0]}
    report = assert_raw_gridsfm_case(prepared_like)
    assert report.evaluation_status is EvaluationStatus.INPUT_INTEGRITY_FAILURE
    with pytest.raises(ValueError, match="input-integrity failure"):
        build_goc500_identity(prepared_like)


def test_source_less_load_detection_uses_generator_components():
    identity = build_goc500_identity(_tiny_gridsfm_raw_case())
    assert source_less_load_ids(identity, []) == ()
    assert source_less_load_ids(identity, [401, 301]) == (201, 202)


def test_raw_candidate_mutation_removes_edges_and_preserves_commands():
    raw = _tiny_gridsfm_raw_case()
    identity = build_goc500_identity(raw)
    mutated, breakdown, report = mutate_raw_case_for_candidate(
        raw,
        identity,
        offline_branch_ids=[301],
        alpha_requested={201: 0.8, 202: 0.5},
    )

    assert report.evaluation_status is EvaluationStatus.OK
    assert report.d_input == pytest.approx(0.0)
    assert mutated["metadata"]["stage_j_mutation"]["mutation_level"] == "raw_pyg_json_before_prepare_for_inference"
    assert mutated["metadata"]["stage_j_mutation"]["official_preprocessing_required_next"] is True
    assert mutated["metadata"]["stage_j_mutation"]["offline_branch_ids"] == [301]
    assert set(mutated["metadata"]["stage_j_mutation"]["alpha_requested"]) == {"201", "202"}
    assert set(mutated["metadata"]["stage_j_mutation"]["alpha_effective"]) == {"201", "202"}
    assert mutated["metadata"]["ac_line_branch_ids"] == [302]
    assert mutated["grid"]["edges"]["ac_line"]["senders"] == [1]
    assert mutated["solution"]["edges"]["ac_line"] == [[0.2, 0.0, -0.2, 0.0]]
    assert mutated["grid"]["nodes"]["load"][0] == pytest.approx([0.8, 0.16])
    assert mutated["grid"]["nodes"]["load"][1] == pytest.approx([1.0, 0.2])
    assert breakdown.l_shed_total == pytest.approx(0.4)


def test_raw_candidate_mutation_requires_full_per_load_alpha():
    raw = _tiny_gridsfm_raw_case()
    identity = build_goc500_identity(raw)
    with pytest.raises(KeyError, match="alpha_requested"):
        mutate_raw_case_for_candidate(
            raw,
            identity,
            offline_branch_ids=[],
            alpha_requested={201: 1.0},
        )


def test_guided_dc_fixed_topology_economic_recourse_solves_tiny_case():
    if importlib.util.find_spec("gurobipy") is None:
        pytest.skip("gurobipy is unavailable")

    raw = _tiny_gridsfm_raw_case()
    identity = build_goc500_identity(raw)
    try:
        result = solve_fixed_topology_economic_dc_opf(
            raw_case=raw,
            identity=identity,
            offline_branch_ids=[301],
            alpha_requested={201: 0.8, 202: 0.5},
        )
    except RuntimeError as exc:
        if "Gurobi environment is unavailable" in str(exc):
            pytest.skip(str(exc))
        raise

    assert result.evaluation_status is EvaluationStatus.OK
    assert result.objective_cost == pytest.approx(1.8)
    assert sum(result.pg_by_generator.values()) == pytest.approx(1.8)
    assert 301 not in result.flow_by_line
    assert set(result.flow_by_line) == {302, 401}


def test_connectivity_proxy_and_diagnostic_scenario_construction():
    raw = _tiny_gridsfm_raw_case()
    raw["grid"]["edges"]["transformer"]["senders"] = []
    raw["grid"]["edges"]["transformer"]["receivers"] = []
    raw["grid"]["edges"]["transformer"]["features"] = []
    raw["solution"]["edges"]["transformer"] = []
    raw["metadata"]["transformer_branch_ids"] = []
    identity = build_goc500_identity(raw, candidate_branch_ids=[301, 302])
    c_by_line = connectivity_service_impact_proxy(identity, [301, 302])

    assert c_by_line[301] == pytest.approx(1.0)
    assert c_by_line[302] == pytest.approx(2.0 / 3.0)
    assert has_alternate_path_after_outage(identity, 301) is False
    assert has_alternate_path_after_outage(identity, 302) is False

    scenarios, score_rows = construct_stage_j_s1_s3_scenarios(
        identity,
        baseline_loading={301: 0.8, 302: 0.4},
        c_by_line=c_by_line,
        high_p_env=1.0,
        low_p_env=0.01,
    )

    assert [scenario.scenario_id for scenario in scenarios] == ["J-S1", "J-S2", "J-S3"]
    assert set(scenarios[0].target_branch_ids).issubset({301, 302})
    assert set(scenarios[1].target_branch_ids).issubset({301, 302})
    assert scenarios[0].target_branch_ids != scenarios[1].target_branch_ids
    assert all(scenario.r_base > 0.0 for scenario in scenarios)
    assert {int(row["branch_id"]) for row in score_rows} == {301, 302}


def test_gridsfm_candidate_evaluator_maps_flows_and_keeps_objectives_separate(monkeypatch, tmp_path):
    raw = _tiny_gridsfm_raw_case()
    identity = build_goc500_identity(raw)

    def fake_predict(_model, _path):
        return {
            "Pij": [0.4, 0.1],
            "Qij": [0.3, 0.0],
            "Pji": [-0.2, -0.1],
            "Qji": [0.0, 0.0],
            "Pg": [1.5],
            "Qg": [0.2],
            "V": [1.0, 1.0, 1.0],
            "theta": [0.0, -0.01, -0.02],
            "feas": 0.9,
        }

    monkeypatch.setitem(sys.modules, "gridsfm", types.SimpleNamespace(predict=fake_predict))
    result = evaluate_gridsfm_candidate(
        raw_case=raw,
        identity=identity,
        model=object(),
        offline_branch_ids=[301],
        alpha_requested={201: 0.8, 202: 0.5},
        p_env_by_line={302: 1.0, 401: 0.0},
        r_base=1.0,
        lambda_r=0.5,
        weights=PacWeights(rho_phys=2.0, w_op=1.0, w_ac=1.0, w_model=1.0),
        work_dir=tmp_path,
    )

    assert result.evaluation_status is EvaluationStatus.MODEL_OUTPUT_PENALIZED
    assert result.flow_loading_by_line[302] == pytest.approx(0.125)
    assert result.qg_by_generator[0] == pytest.approx(0.2)
    assert result.objective.r_norm == pytest.approx(0.125**2)
    assert result.objective.l_shed_total == pytest.approx(0.4)
    assert result.objective.j_trade == pytest.approx(0.5 * 0.125**2 + 0.5 * 0.4)
    assert result.objective.pac_model == pytest.approx(0.0)
    assert result.pac_model_components["predicted_load_command_available"] is False
    assert result.pac_model_components["model_consistency_basis"] == "input_command_guard_only_no_predicted_Pd_Qd"
    assert result.objective.j_total > result.objective.j_trade
    assert result.d_input == pytest.approx(0.0)


def test_gridsfm_candidate_evaluator_penalizes_predicted_load_command_mismatch(monkeypatch, tmp_path):
    raw = _tiny_gridsfm_raw_case()
    identity = build_goc500_identity(raw)

    def fake_predict(_model, _path):
        return {
            "Pij": [0.4, 0.1, 0.0],
            "Qij": [0.3, 0.0, 0.0],
            "Pji": [-0.2, -0.1, 0.0],
            "Qji": [0.0, 0.0, 0.0],
            "Pg": [1.5],
            "Qg": [0.0],
            "V": [1.0, 1.0, 1.0],
            "theta": [0.0, -0.01, -0.02],
            "Pd": [0.8, 2.0],
            "Qd": [0.16, 0.4],
            "feas": 1.0,
        }

    monkeypatch.setitem(sys.modules, "gridsfm", types.SimpleNamespace(predict=fake_predict))
    result = evaluate_gridsfm_candidate(
        raw_case=raw,
        identity=identity,
        model=object(),
        offline_branch_ids=[],
        alpha_requested={201: 0.8, 202: 0.5},
        p_env_by_line={301: 0.0, 302: 1.0, 401: 0.0},
        r_base=1.0,
        lambda_r=0.5,
        weights=PacWeights(rho_phys=2.0, w_op=0.0, w_ac=0.0, w_model=1.0),
        work_dir=tmp_path,
    )

    assert result.objective.pac_model > 0.0
    assert result.objective.pac_total == pytest.approx(result.objective.pac_model)
    assert result.objective.j_total == pytest.approx(result.objective.j_trade + 2.0 * result.objective.pac_model)
    assert result.pac_model_components["predicted_load_command_available"] is True
    assert result.pac_model_components["model_consistency_basis"] == "predicted_Pd_Qd_vs_alpha_effective_command"


def test_outer_proxy_uses_budget_and_no_good_cuts():
    if importlib.util.find_spec("gurobipy") is None:
        pytest.skip("gurobipy is unavailable")
    try:
        pool = solve_proxy_topology_pool(
            candidate_branch_ids=[1, 2, 3],
            p_env_by_line={1: 1.0, 2: 0.5, 3: 0.1},
            baseline_loading={1: 1.0, 2: 1.0, 3: 1.0},
            c_by_line={1: 0.0, 2: 0.0, 3: 0.0},
            lambda_r_proxy=1.0,
            k=2,
            pool_size=3,
        )
    except RuntimeError as exc:
        if "gurobipy" in str(exc) or "Gurobi environment is unavailable" in str(exc):
            pytest.skip(str(exc))
        raise

    assert len(pool) == 3
    assert pool[0].shutoff_branch_ids == (1, 2)
    assert all(len(item.shutoff_branch_ids) <= 2 for item in pool)
    assert len({item.shutoff_branch_ids for item in pool}) == 3


def test_full_alpha_optimizer_uses_seeds_coordinate_sweeps_and_cache():
    load_ids = [101, 102]
    calls: list[tuple[tuple[int, float], ...]] = []

    def evaluator(alpha):
        calls.append(tuple(sorted((int(k), float(v)) for k, v in alpha.items())))
        r_norm = alpha[101] ** 2 + 0.1 * alpha[102] ** 2
        l_shed = (2.0 - alpha[101] - alpha[102]) / 2.0
        j_trade = 0.8 * r_norm + 0.2 * l_shed
        return AlphaEvaluation(
            evaluation_status=EvaluationStatus.OK.value,
            search_objective=j_trade,
            l_shed_total=l_shed,
            l_shed_control=l_shed,
            l_shed_island=0.0,
            r_norm=r_norm,
            j_trade=j_trade,
            runtime_seconds=0.0,
        )

    result = optimize_full_alpha(
        load_ids=load_ids,
        fixed_zero_load_ids=[],
        offline_branch_ids=[285, 473],
        backend="synthetic",
        topology_id="285;473",
        search_run_id="TEST-ALPHA",
        evaluator=evaluator,
        config=AlphaSearchConfig(
            seed_values=(1.0,),
            delta_schedule=(0.10,),
            b_alpha=20,
            epsilon_abs=1e-12,
            epsilon_rel=1e-12,
        ),
    )

    assert result.best_alpha[101] < 1.0
    assert result.best_alpha[102] == pytest.approx(1.0)
    assert result.completed_sweeps >= 1
    assert result.accepted_moves >= 1
    assert result.actual_evaluation_count == len(set(calls))
    assert {row["candidate_type"] for row in result.trace_rows} >= {"seed_all_1", "coordinate_down"}
    assert any(row["accepted"] for row in result.trace_rows)


def test_full_alpha_optimizer_skips_source_less_coordinates_and_stops_on_budget():
    evaluated: list[dict[int, float]] = []

    def evaluator(alpha):
        evaluated.append(dict(alpha))
        value = sum(alpha.values())
        return AlphaEvaluation(
            evaluation_status=EvaluationStatus.OK.value,
            search_objective=value,
            l_shed_total=0.0,
            r_norm=value,
            j_trade=value,
            runtime_seconds=0.0,
        )

    result = optimize_full_alpha(
        load_ids=[1, 2, 3],
        fixed_zero_load_ids=[2],
        offline_branch_ids=[99],
        backend="synthetic",
        topology_id="99",
        search_run_id="TEST-BUDGET",
        evaluator=evaluator,
        config=AlphaSearchConfig(seed_values=(1.0,), delta_schedule=(0.1, 0.05), b_alpha=3),
    )

    assert result.termination_reason == "budget_exhausted"
    assert result.actual_evaluation_count == 3
    assert all("2" not in str(row["changed_load_ids"]).split(";") for row in result.trace_rows if row["changed_load_ids"])


def test_screened_scipy_alpha_uses_coordinate_screen_then_local_subset():
    if importlib.util.find_spec("scipy") is None:
        pytest.skip("scipy is unavailable")
    load_ids = [101, 102, 103]

    def evaluator(alpha):
        target = {101: 0.55, 102: 0.95, 103: 1.0}
        r_norm = sum((float(alpha[k]) - target[k]) ** 2 for k in load_ids)
        l_shed = sum(1.0 - float(alpha[k]) for k in load_ids) / len(load_ids)
        objective = r_norm + 0.01 * l_shed
        return AlphaEvaluation(
            evaluation_status=EvaluationStatus.OK.value,
            search_objective=objective,
            l_shed_total=l_shed,
            l_shed_control=l_shed,
            l_shed_island=0.0,
            r_norm=r_norm,
            j_trade=objective,
            runtime_seconds=0.0,
        )

    result = optimize_screened_scipy_alpha(
        load_ids=load_ids,
        fixed_zero_load_ids=[],
        offline_branch_ids=[285, 473],
        backend="synthetic",
        topology_id="285;473",
        search_run_id="TEST-SCREENED-SCIPY",
        evaluator=evaluator,
        config=ScreenedScipyAlphaConfig(
            seed_values=(1.0,),
            screen_delta=0.10,
            q_values=(1, 2),
            b_alpha=80,
            scipy_maxfev_per_q=20,
            alpha_round_decimals=4,
            epsilon_abs=1e-12,
            epsilon_rel=1e-12,
        ),
    )

    assert result.best_alpha[101] < 0.95
    assert result.actual_evaluation_count <= 80
    assert result.screening_best_improved is True
    assert result.selected_loads_by_q[1] == (101,)
    assert {row["candidate_type"] for row in result.trace_rows} >= {"coordinate_screen_down", "scipy_local"}
    assert len(result.q_summary_rows) == 2
    assert any(row["accepted"] for row in result.trace_rows)


def test_j8_budgeted_load_selector_prefers_topology_nearby_loads():
    raw = _tiny_gridsfm_raw_case()
    identity = build_goc500_identity(raw)

    selected = select_budgeted_alpha_loads(identity, [301], q=1)
    assert selected == (201,)

    selected_without_screen = select_budgeted_alpha_loads(identity, [], q=1)
    assert selected_without_screen == (202,)


def test_j8_th_gridsfm_pool_selects_top1_and_top2_by_transparent_score():
    pool = build_th_gridsfm_topology_pool(
        candidate_branch_ids=[10, 20, 30],
        p_env_by_line={10: 1.0, 20: 0.5, 30: 0.1},
        baseline_loading={10: 0.2, 20: 0.6, 30: 0.9},
        k=2,
    )

    assert [item.rank for item in pool] == [1, 2]
    assert pool[0].shutoff_branch_ids == (20,)
    assert pool[1].shutoff_branch_ids == (20, 30)
    assert pool[0].proxy_objective is None
    assert pool[1].l_proxy is None


def test_j8_budgeted_alpha_search_enforces_total_call_budget():
    if importlib.util.find_spec("scipy") is None:
        pytest.skip("scipy is unavailable")
    calls = []

    def evaluator(alpha):
        calls.append(dict(alpha))
        objective = (float(alpha[101]) - 0.5) ** 2
        return CandidateEvaluation(
            evaluation_status=EvaluationStatus.OK.value,
            search_objective=objective,
            r_norm=objective,
            l_shed_total=1.0 - float(alpha[101]),
            j_trade=objective,
        )

    summary, trace, best_alpha = run_budgeted_alpha_search(
        load_ids=[101, 102],
        selected_load_ids=[101],
        offline_branch_ids=[473],
        method="synthetic",
        topology_rank=1,
        evaluator=evaluator,
        continuous_eval_budget=5,
    )

    assert summary["actual_evaluation_count"] <= 5
    assert len(trace) <= 5
    assert len(calls) <= 5
    assert summary["best_found"] is True
    assert best_alpha[101] < 1.0
