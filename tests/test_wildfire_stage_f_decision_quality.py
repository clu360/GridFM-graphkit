from __future__ import annotations

import math

import numpy as np
import pandas as pd

from experiments.test.wildfire_tests.stage_d_deenergization.stage_d_deenergization import (
    enumerate_deenergization_subsets,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_decision_quality import (
    FIXED_T0P30_CANDIDATE_LINE_IDS,
    RISK_SCOPE,
    _gap_table,
    _risk_line_ids,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import (
    LOW_P_ENV_DEFAULT,
    get_scenarios,
    p_env_for_scenario,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_physics_decision_quality import (
    EXPECTED_COMPARISON_LAMBDAS,
    STAGES,
    _expected_vs_observed,
    _lambda_values,
    _stage_e_vs_stage_d_gap as _physics_stage_e_vs_stage_d_gap,
)


def test_stage_f_registry_has_five_scenarios():
    scenarios = get_scenarios()

    assert [scenario.scenario_id for scenario in scenarios] == ["S1", "S2", "S3", "S4", "S5"]


def test_stage_f_p_env_assigns_targets_high_and_others_low():
    scenario = get_scenarios(["S4"])[0]
    p_env = p_env_for_scenario(scenario, num_lines=110)

    assert p_env[77] == 1.0
    assert p_env[79] == 1.0
    assert p_env[23] == LOW_P_ENV_DEFAULT
    assert p_env[0] == LOW_P_ENV_DEFAULT


def test_stage_f_s3_is_not_s1_repeated():
    s1 = get_scenarios(["S1"])[0]
    s3 = get_scenarios(["S3"])[0]

    overlap = set(s1.target_high_risk_line_ids).intersection(s3.target_high_risk_line_ids)
    assert s3.suppressed_line_ids == (23,)
    assert len(overlap) < len(s1.target_high_risk_line_ids)
    assert {16, 18, 19, 22, 27}.issubset(set(s3.target_high_risk_line_ids))


def test_stage_f_k2_subset_count_for_current_candidate_set():
    candidate_line_ids = [2, 5, 6, 8, 10, 12, 16, 18, 19, 20, 22, 23, 25, 26, 27, 32, 33, 35, 36, 37, 38, 40, 47, 50, 51, 72, 74, 77, 79, 88, 91, 97, 101]

    assert list(FIXED_T0P30_CANDIDATE_LINE_IDS) == candidate_line_ids
    assert len(enumerate_deenergization_subsets(candidate_line_ids, max_deenergized_lines=2)) == 562


def test_stage_f_true_risk_scope_is_all_lines():
    assert RISK_SCOPE == "all_lines"
    assert _risk_line_ids(110) == list(range(110))


def test_stage_f_gap_table_uses_true_objective_and_relative_gap():
    d = pd.DataFrame(
        [
            {
                "model_type": "gnn",
                "scenario_id": "S1",
                "scenario_name": "S1_low_impact_high_risk",
                "lambda_case": "balanced",
                "lambda_R": 0.5,
                "lambda_L": 0.5,
                "J_true": 0.2,
                "R_norm": 0.3,
                "L_shed": 0.1,
                "shutoff_line_ids": "27",
            }
        ]
    )
    e = pd.DataFrame(
        [
            {
                "model_type": "gnn",
                "scenario_id": "S1",
                "scenario_name": "S1_low_impact_high_risk",
                "lambda_case": "balanced",
                "lambda_R": 0.5,
                "lambda_L": 0.5,
                "J_true": 0.25,
                "R_norm": 0.4,
                "L_shed": 0.1,
                "shutoff_line_ids": "32",
            }
        ]
    )

    gap = _gap_table(d, e)

    assert math.isclose(float(gap.loc[0, "gap_J"]), 0.05)
    assert math.isclose(float(gap.loc[0, "relative_gap_J"]), 0.25)


def test_physics_stage_f_lambda_sweep_and_expected_comparison_settings():
    assert _lambda_values(0.05) == [round(value, 2) for value in np.arange(0.0, 1.01, 0.05)]
    assert EXPECTED_COMPARISON_LAMBDAS == [1.0, 0.8, 0.5, 0.2, 0.0]
    assert STAGES == ["stage_d_k2_exhaustive", "stage_e_k2", "stage_e_unconstrained"]


def test_expected_vs_observed_records_target_subset_and_non_targets():
    scenario = get_scenarios(["S1"])[0]
    best = pd.DataFrame(
        [
            {
                "scenario_id": "S1",
                "scenario_name": scenario.scenario_name,
                "stage": "stage_d_k2_exhaustive",
                "stage_label": "Stage D exhaustive k<=2",
                "lambda_R": 0.8,
                "lambda_L": 0.2,
                "rho_phys": 100.0,
                "shutoff_line_ids": "27,23",
                "creates_source_less_island": False,
                "R_norm": 0.2,
                "L_shed": 0.1,
                "PAC_total": 0.01,
                "J_true": 1.18,
            }
        ]
    )

    comparison = _expected_vs_observed(best, [scenario]).iloc[0]

    assert comparison["observed_target_subset"] == "27"
    assert comparison["observed_non_target_lines"] == "23"
    assert comparison["expected_targets_not_selected"] == "32,36,101"


def test_physics_stage_e_gap_supports_constrained_and_unconstrained_methods():
    common = {
        "model_type": "gnn",
        "scenario_id": "S1",
        "scenario_name": "S1_low_impact_high_risk",
        "lambda_R": 0.8,
        "lambda_L": 0.2,
        "rho_phys": 100.0,
        "R_norm": 0.2,
        "L_shed": 0.1,
        "PAC_total": 0.01,
    }
    best = pd.DataFrame(
        [
            {**common, "stage": "stage_d_k2_exhaustive", "shutoff_line_ids": "27", "J_true": 1.2},
            {**common, "stage": "stage_e_k2", "shutoff_line_ids": "32", "J_true": 1.3},
            {**common, "stage": "stage_e_unconstrained", "shutoff_line_ids": "36", "J_true": 1.4},
        ]
    )

    gap = _physics_stage_e_vs_stage_d_gap(best)

    assert set(gap["stage_e_method"]) == {"stage_e_k2", "stage_e_unconstrained"}
    assert sorted(gap["gap_J_true"].round(10).tolist()) == [0.1, 0.2]
