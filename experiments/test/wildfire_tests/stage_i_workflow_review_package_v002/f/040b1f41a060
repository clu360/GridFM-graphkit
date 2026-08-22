from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from experiments.test.wildfire_tests.shared.decision_vector import PgQgAlphaDecisionVector
from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_revised_continuous_implementation import (
    STAGE_D,
    STAGE_E_K2,
    STAGE_E_UNCONSTRAINED,
    _clamp_prediction,
    _controlled_state_audit,
    _expected_vs_observed_all_lambdas,
    _load_shedding_provenance,
    _normalize_stages,
    _physics_components,
    _run_signature,
    _wildfire_risk_provenance,
)


def _scenario():
    scenario = SimpleNamespace(
        num_buses=4,
        Pd_base=np.array([0.0, 10.0, 20.0, 30.0]),
        Qd_base=np.array([0.0, 5.0, 10.0, 15.0]),
        Pg_base=np.array([50.0, 0.0, 0.0, 20.0]),
        Qg_base=np.array([10.0, 0.0, 0.0, 4.0]),
        Vm_base=np.ones(4),
        Va_base=np.zeros(4),
        Pg_min=np.zeros(4),
        Pg_max=np.ones(4) * 100.0,
        PV_mask=np.array([True, False, False, True]),
        REF_mask=np.array([False, False, False, False]),
        mask=torch.zeros((4, 6), dtype=torch.bool),
        edge_index=np.array([[0, 1, 2], [1, 2, 3]]),
        sn_mva=100.0,
    )
    scenario.get_baseline_node_features = lambda: np.column_stack(
        [
            scenario.Pd_base,
            scenario.Qd_base,
            scenario.Pg_base,
            scenario.Qg_base,
            scenario.Vm_base,
            scenario.Va_base,
        ]
    )
    return scenario


def test_pg_qg_alpha_decision_vector_updates_controlled_features():
    scenario = _scenario()
    decision = PgQgAlphaDecisionVector(
        scenario,
        selected_generator_buses=[0, 3],
        selected_load_buses=[1, 2],
        delta_pg_bound_mw=5.0,
        delta_qg_bound_mvar=5.0,
    )
    u = decision.combine_decision_vector(
        delta_pg=np.array([1.5, -2.0]),
        delta_qg=np.array([3.0, -1.0]),
        alpha=np.array([0.8, 0.25]),
    )

    features = decision.u_to_node_features(u)

    assert features[0, 2] == 51.5
    assert features[3, 2] == 18.0
    assert features[0, 3] == 13.0
    assert features[3, 3] == 3.0
    assert features[1, 0] == 8.0
    assert features[1, 1] == 4.0
    assert features[2, 0] == 5.0
    assert features[2, 1] == 2.5


def test_controlled_feature_mask_marks_pg_qg_pd_qd_only():
    decision = PgQgAlphaDecisionVector(
        _scenario(),
        selected_generator_buses=[0, 3],
        selected_load_buses=[1, 2],
    )

    mask = decision.controlled_feature_mask(decision.u_base)

    assert mask[0, 2]
    assert mask[0, 3]
    assert mask[3, 2]
    assert mask[3, 3]
    assert mask[1, 0]
    assert mask[1, 1]
    assert mask[2, 0]
    assert mask[2, 1]
    assert not mask[:, 4].any()
    assert not mask[:, 5].any()


def test_clamp_prediction_overwrites_controlled_values_only():
    scenario = _scenario()
    decision = PgQgAlphaDecisionVector(scenario, [0], [1])
    u = decision.combine_decision_vector(
        delta_pg=np.array([2.0]),
        delta_qg=np.array([-3.0]),
        alpha=np.array([0.5]),
    )
    prediction = {
        "Pd": np.ones(4) * 99.0,
        "Qd": np.ones(4) * 88.0,
        "Pg": np.ones(4) * 77.0,
        "Qg": np.ones(4) * 66.0,
        "Vm": np.ones(4),
        "Va": np.zeros(4),
    }

    combined = _clamp_prediction(decision, u, prediction)

    assert combined["Pg"][0] == 52.0
    assert combined["Qg"][0] == 7.0
    assert combined["Pd"][1] == 5.0
    assert combined["Qd"][1] == 2.5
    assert combined["Pg"][3] == 77.0
    assert combined["Pd"][2] == 99.0


def test_controlled_state_audit_separates_raw_gridfm_from_objective_values():
    scenario = _scenario()
    decision = PgQgAlphaDecisionVector(scenario, [0], [1])
    u = decision.combine_decision_vector(
        delta_pg=np.array([2.0]),
        delta_qg=np.array([-3.0]),
        alpha=np.array([0.5]),
    )
    raw_prediction = {
        "Pd": np.ones(4) * 99.0,
        "Qd": np.ones(4) * 88.0,
        "Pg": np.ones(4) * 77.0,
        "Qg": np.ones(4) * 66.0,
        "Vm": np.ones(4),
        "Va": np.zeros(4),
    }
    combined = _clamp_prediction(decision, u, raw_prediction)

    frame = _controlled_state_audit(
        {"scenario": scenario, "decision_vector": decision},
        u,
        raw_prediction,
        combined,
        result_id=11,
    )

    assert frame["abs_command_vs_eval_error"].max() == 0.0
    assert (frame["commanded_value"] == frame["objective_evaluation_value"]).all()
    assert (frame["commanded_value"] == frame["gridfm_visible_input_value"]).all()
    assert (frame["commanded_value"] != frame["raw_gridfm_predicted_value"]).any()


def test_load_shedding_provenance_sums_to_effective_alpha_objective():
    scenario = _scenario()
    decision = PgQgAlphaDecisionVector(scenario, [0], [1, 2])
    u = decision.combine_decision_vector(
        delta_pg=np.array([0.0]),
        delta_qg=np.array([0.0]),
        alpha=np.array([0.5, 0.25]),
    )
    context = {"scenario": scenario, "decision_vector": decision}
    raw_prediction = {
        "Pd": np.array([0.0, 8.0, 14.0, 15.0]),
        "Qd": scenario.Qd_base.copy(),
        "Pg": scenario.Pg_base.copy(),
        "Qg": scenario.Qg_base.copy(),
        "Vm": scenario.Vm_base.copy(),
        "Va": scenario.Va_base.copy(),
    }

    frame, metrics = _load_shedding_provenance(context, u, source_less_buses=[], result_id=7, raw_prediction=raw_prediction)

    assert np.isclose(frame["load_shed_weighted"].sum(), metrics["L_shed"])
    assert np.isclose(metrics["L_shed_cmd"], (10.0 * 0.5 + 20.0 * 0.75) / 60.0)
    assert np.isclose(metrics["L_shed_gridfm_raw"], (10.0 * 0.2 + 20.0 * 0.3 + 30.0 * 0.5) / 60.0)
    assert np.isclose(metrics["L_shed_gridfm_effective"], metrics["L_shed_gridfm_raw"])
    assert np.isclose(metrics["L_shed_hybrid"], (10.0 * 0.5 + 20.0 * 0.75 + 30.0 * 0.5) / 60.0)
    assert metrics["L_shed"] == metrics["L_shed_hybrid"]
    assert frame.loc[frame["bus_id"] == 3, "gridfm_predicted_pd_used_for_l_shed"].iloc[0]
    assert np.isclose(frame.loc[frame["bus_id"] == 1, "alpha_commanded"].iloc[0], 0.5)
    assert np.isclose(frame.loc[frame["bus_id"] == 1, "alpha_effective"].iloc[0], 0.5)
    assert not frame["source_less_island_correction_applied"].any()


def test_load_shedding_provenance_counts_source_less_buses_as_fully_shed():
    scenario = _scenario()
    decision = PgQgAlphaDecisionVector(scenario, [0], [1, 2])
    u = decision.combine_decision_vector(
        delta_pg=np.array([0.0]),
        delta_qg=np.array([0.0]),
        alpha=np.array([0.4, 0.0]),
    )
    context = {"scenario": scenario, "decision_vector": decision}
    raw_prediction = {
        "Pd": np.array([0.0, 7.0, 11.0, 12.0]),
        "Qd": scenario.Qd_base.copy(),
        "Pg": scenario.Pg_base.copy(),
        "Qg": scenario.Qg_base.copy(),
        "Vm": scenario.Vm_base.copy(),
        "Va": scenario.Va_base.copy(),
    }

    frame, metrics = _load_shedding_provenance(context, u, source_less_buses=[2, 3], result_id=8, raw_prediction=raw_prediction)

    selected_island = frame.loc[frame["bus_id"] == 2].iloc[0]
    nonselected_island = frame.loc[frame["bus_id"] == 3].iloc[0]
    connected_selected = frame.loc[frame["bus_id"] == 1].iloc[0]
    assert np.isclose(selected_island["alpha_commanded"], 0.0)
    assert np.isclose(selected_island["alpha_effective"], 0.0)
    assert selected_island["alpha_source"] == "source_less_forced_unserved"
    assert np.isclose(nonselected_island["alpha_commanded"], 1.0)
    assert np.isclose(nonselected_island["alpha_effective"], 0.0)
    assert nonselected_island["alpha_source"] == "source_less_forced_unserved"
    assert np.isclose(connected_selected["alpha_effective"], 0.4)
    expected = (10.0 * 0.6 + 20.0 * 1.0 + 30.0 * 1.0) / 60.0
    assert np.isclose(metrics["L_shed"], expected)
    assert np.isclose(frame["load_shed_weighted"].sum(), metrics["L_shed"])


def test_physics_components_decomposes_pac_and_flags_branch_consistency_unavailable():
    scenario = _scenario()
    decision = PgQgAlphaDecisionVector(scenario, [0], [1])
    u = decision.combine_decision_vector(
        delta_pg=np.array([2.0]),
        delta_qg=np.array([0.0]),
        alpha=np.array([0.5]),
    )
    raw_prediction = {
        "Pd": np.array([0.0, 8.0, 20.0, 30.0]),
        "Qd": scenario.Qd_base.copy(),
        "Pg": np.array([150.0, 0.0, 0.0, 20.0]),
        "Qg": scenario.Qg_base.copy(),
        "Vm": np.array([1.0, 1.08, 1.0, 1.0]),
        "Va": np.zeros(4),
    }
    combined = _clamp_prediction(decision, u, raw_prediction)
    state = {"Vm": combined["Vm"], "Va": combined["Va"], "loading_ratio": np.array([1.2, 0.5, 0.7]), "_ac_balance_available": False}

    components = _physics_components(
        {"scenario": scenario, "decision_vector": decision},
        u,
        state,
        active_line_ids=[0, 1, 2],
        source_less_buses=[],
        raw_prediction=raw_prediction,
        combined_prediction=combined,
    )

    expected = (
        components["pac_operational_weight"] * components["PAC_operational"]
        + components["pac_ac_weight"] * components["PAC_AC"]
        + components["pac_model_consistency_weight"] * components["PAC_model_consistency"]
    )
    assert np.isclose(components["PAC_total"], expected)
    assert np.isnan(components["branch_flow_consistency"])
    assert not components["branch_flow_consistency_available"]
    assert components["branch_flow_consistency_weight"] == 0.0
    assert not components["p_balance_available"]
    assert not components["q_balance_available"]
    assert components["cmd_load"] > 0.0
    assert components["generator_limits_raw"] > 0.0


def test_wildfire_risk_provenance_excludes_impact_and_sums_to_raw_risk():
    scenario = _scenario()
    context = {"scenario": scenario}
    frame, total = _wildfire_risk_provenance(
        context,
        loading=np.array([0.2, 0.5, 0.8]),
        p_env_by_line={0: 1.0, 1: 0.5, 2: 0.25},
        z_by_line={0: 1, 1: 0, 2: 1},
        candidate_line_ids=[0, 1, 2],
        result_id=3,
    )

    expected = 1.0 * 0.2**2 + 0.0 + 0.25 * 0.8**2
    assert np.isclose(total, expected)
    assert np.isclose(frame["risk_raw_contribution"].sum(), expected)
    assert not frame["impact_used_in_true_risk"].astype(bool).any()


def test_normalize_stages_supports_stage_e_unconstrained_zoom():
    stages = _normalize_stages([STAGE_E_UNCONSTRAINED, STAGE_E_UNCONSTRAINED])

    assert stages == [STAGE_E_UNCONSTRAINED]
    assert STAGE_D not in stages
    assert STAGE_E_K2 not in stages


def test_run_signature_records_lambda_zoom_stage_filter():
    signature = _run_signature(
        model_type="gnn",
        scenario_ids=["S1", "S2", "S3", "S4", "S5"],
        lambda_values=[0.85, 0.90, 0.95],
        rho_values=[2.0],
        stages=[STAGE_E_UNCONSTRAINED],
        stage_d_limit=None,
        stage_e_budget=50,
        call_budget=100,
        delta_qg_bound_mvar=5.0,
    )

    assert signature["lambda_values"] == [0.85, 0.9, 0.95]
    assert np.allclose([1.0 - value for value in signature["lambda_values"]], [0.15, 0.10, 0.05])
    assert signature["rho_phys_values"] == [2.0]
    assert signature["stages"] == [STAGE_E_UNCONSTRAINED]
    assert signature["stage_e_budget"] == 50


def test_expected_vs_observed_keeps_lambda_zoom_values():
    best = pd.DataFrame(
        [
            {
                "scenario_id": "S1",
                "stage": STAGE_E_UNCONSTRAINED,
                "stage_label": "Stage E unconstrained",
                "lambda_R": 0.85,
                "lambda_L": 0.15,
                "rho_phys": 2.0,
                "shutoff_line_ids": "23,77",
                "R_norm": 1.5,
                "L_shed": 0.1,
                "PAC_total": 0.2,
                "J_true": 1.69,
            }
        ]
    )
    scenarios = [
        SimpleNamespace(
            scenario_id="S1",
            scenario_name="scenario one",
            expected_target_set=[23, 36],
        )
    ]

    observed = _expected_vs_observed_all_lambdas(best, scenarios)

    assert len(observed) == 1
    assert observed.loc[0, "lambda_R"] == 0.85
    assert observed.loc[0, "observed_target_subset"] == "23"
    assert observed.loc[0, "observed_non_target_lines"] == "77"
    assert observed.loc[0, "num_expected_target_lines"] == 2
    assert observed.loc[0, "num_observed_shutoff_lines"] == 2
    assert observed.loc[0, "num_observed_target_lines"] == 1
    assert observed.loc[0, "num_observed_non_target_lines"] == 1
    assert observed.loc[0, "target_recall"] == 0.5
    assert observed.loc[0, "target_precision"] == 0.5
    assert observed.loc[0, "target_overlap_fraction"] == 0.5
