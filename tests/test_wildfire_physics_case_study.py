from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.test.wildfire_tests.stage_e_gurobi_implementation.physics_infeasibility_evaluator import (
    DEFAULT_PHYSICS_WEIGHTS,
    recourse_bounds_with_island_limits,
    risk_only_stage_c_scores,
    source_less_island_buses,
    weighted_pac_total,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_physics_infeasibility_case_study import (
    TRADITIONAL_LAMBDA_R,
    USER_CASES,
    _lambda_settings,
    _pareto_frontier_points,
    _traditional_lambda_summary,
    _unique_candidate_points,
    nondominated_mask,
)


class _FakeScenario:
    def __init__(self):
        self.num_buses = 4
        self.edge_index = np.asarray([[0, 2], [1, 3]], dtype=int)
        self.Pg_base = np.asarray([10.0, 0.0, 0.0, 0.0])

    def get_pv_buses(self):
        return np.asarray([0], dtype=int)

    def get_ref_bus(self):
        return 0


class _FakeDecisionVector:
    def __init__(self):
        self.n_generators = 1
        self.selected_load_buses = np.asarray([1, 3], dtype=int)
        self.u_min = np.asarray([-5.0, 0.0, 0.0], dtype=float)
        self.u_max = np.asarray([5.0, 1.0, 1.0], dtype=float)


def test_risk_only_stage_c_scores_exclude_impact_and_consequence():
    scores = risk_only_stage_c_scores(
        candidate_line_ids=[0, 1],
        p_env_by_line={0: 0.5, 1: 1.0},
        baseline_loading=np.asarray([2.0, 3.0], dtype=float),
    )

    assert scores == {0: 2.0, 1: 9.0}


def test_source_less_island_loads_get_alpha_upper_zero():
    scenario = _FakeScenario()
    source_less = source_less_island_buses(scenario, removed_line_ids=[])
    lower, upper, forced = recourse_bounds_with_island_limits(_FakeDecisionVector(), source_less)

    assert source_less == [2, 3]
    assert forced == 1
    assert lower.tolist() == [-5.0, 0.0, 0.0]
    assert upper.tolist() == [5.0, 1.0, 0.0]


def test_weighted_pac_total_keeps_balance_diagnostics_zero_weight():
    components = {
        "voltage_limits": 1.0,
        "thermal_limits": 2.0,
        "generator_limits": 3.0,
        "island_source_feasibility": 4.0,
        "p_balance": 1000.0,
        "q_balance": 1000.0,
        "branch_flow_consistency": 1000.0,
    }

    assert weighted_pac_total(components, DEFAULT_PHYSICS_WEIGHTS) == 10.0


def test_lfg_user_label_maps_to_existing_largest_group_high_case():
    assert USER_CASES["lfg"] == "largest_group_high"


def test_nondominated_mask_minimizes_risk_and_load_only_by_default():
    frame = pd.DataFrame(
        {
            "R_norm": [1.0, 0.8, 0.8, 0.9, 0.7],
            "L_shed": [1.0, 0.9, 0.7, 0.8, 0.9],
            "PAC_total": [0.0, 0.0, 100.0, 0.0, 100.0],
        }
    )

    assert nondominated_mask(frame).tolist() == [False, False, True, False, True]


def test_lambda_step_generates_frontier_settings():
    settings = _lambda_settings(lambda_cases=None, lambda_step=0.5)

    assert settings == [
        {"lambda_case": "lr0p00", "lambda_R": 0.0, "lambda_L": 1.0},
        {"lambda_case": "lr0p50", "lambda_R": 0.5, "lambda_L": 0.5},
        {"lambda_case": "lr1p00", "lambda_R": 1.0, "lambda_L": 0.0},
    ]


def test_traditional_summary_keeps_standard_lambda_values():
    frame = pd.DataFrame(
        {
            "case_name": ["auto_env"] * 5,
            "rho_phys": [0.0] * 5,
            "lambda_R": [0.0, 0.2, 0.5, 0.8, 1.0],
            "stage": ["stage_d_k2_exhaustive"] * 5,
        }
    )

    summary = _traditional_lambda_summary(frame)

    assert summary["lambda_R"].tolist() == TRADITIONAL_LAMBDA_R


def test_pareto_frontier_is_computed_independently_for_each_stage():
    points = pd.DataFrame(
        {
            "case_name": ["auto_env"] * 4,
            "rho_phys": [0.0] * 4,
            "stage": ["stage_d_k2_exhaustive"] * 2 + ["stage_e_k2"] * 2,
            "R_norm": [0.2, 0.4, 0.3, 0.5],
            "L_shed": [0.4, 0.2, 0.5, 0.3],
        }
    )

    frontier = _pareto_frontier_points(points)

    assert frontier.groupby("stage").size().to_dict() == {
        "stage_d_k2_exhaustive": 2,
        "stage_e_k2": 2,
    }


def test_unique_candidate_points_pool_duplicate_topology_across_lambda_sweep():
    points = pd.DataFrame(
        {
            "case_name": ["auto_env", "auto_env"],
            "rho_phys": [0.0, 0.0],
            "stage": ["stage_d_k2_exhaustive", "stage_d_k2_exhaustive"],
            "line_id_key": ["23,27", "23,27"],
            "lambda_R": [0.2, 0.8],
            "lambda_L": [0.8, 0.2],
            "J_true": [0.3, 0.2],
        }
    )

    unique = _unique_candidate_points(points)

    assert len(unique) == 1
