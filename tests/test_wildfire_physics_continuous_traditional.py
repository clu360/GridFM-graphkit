from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_physics_continuous_traditional_lambdas import (
    STAGES,
    TRADITIONAL_LAMBDAS,
    _best_by_stage,
    _final_topology_summary,
    _parse_topology,
    _runtime_summary,
    _topology_annotation,
    _topology_key,
)


def test_traditional_continuous_study_uses_only_requested_lambdas():
    assert TRADITIONAL_LAMBDAS == [0.8, 0.5, 0.2]


def test_topology_round_trip_is_stable():
    assert _parse_topology("23,18,27") == [18, 23, 27]
    assert _topology_key(_parse_topology("23,18,27")) == "18,23,27"
    assert _parse_topology(np.nan) == []


def test_best_by_stage_selects_best_within_topology_result_across_topologies():
    rows = []
    for stage in STAGES:
        rows.extend(
            [
                {
                    "lambda_R": 0.8,
                    "rho_phys": 0.0,
                    "stage": stage,
                    "J_true": 0.4,
                    "topology_key": "1",
                },
                {
                    "lambda_R": 0.8,
                    "rho_phys": 0.0,
                    "stage": stage,
                    "J_true": 0.2,
                    "topology_key": "2",
                },
            ]
        )

    best = _best_by_stage(pd.DataFrame(rows))

    assert len(best) == len(STAGES)
    assert best["J_true"].eq(0.2).all()
    assert best["topology_key"].eq("2").all()


def test_final_topology_outputs_include_all_four_stages():
    best = pd.DataFrame(
        [
            {
                "stage": stage,
                "lambda_R": 0.8,
                "lambda_L": 0.2,
                "rho_phys": 0.0,
                "topology_key": topology,
            }
            for stage, topology in [
                ("stage_c_risk_only", "23,26,30,100"),
                ("stage_d_k2_exhaustive", "18,23"),
                ("stage_e_k2", "18,23"),
                ("stage_e_unconstrained", "18,23,26,30,100"),
            ]
        ]
    )

    summary = _final_topology_summary(best)
    assert summary.loc[0, "stage_c_risk_only_final_topology"] == "23,26,30,100"
    assert summary.loc[0, "stage_d_k2_exhaustive_final_topology"] == "18,23"
    assert summary.loc[0, "stage_e_k2_final_topology"] == "18,23"
    assert summary.loc[0, "stage_e_unconstrained_final_topology"] == "18,23,26,30,100"

    annotation = _topology_annotation(best)
    assert "Stage C: [23,26,30,100]" in annotation
    assert "Stage D: [18,23]" in annotation
    assert "Stage E constrained: [18,23]" in annotation
    assert "Stage E unconstrained: [18,23,26,30,100]" in annotation


def test_runtime_summary_records_calls_and_budget_rates():
    frame = pd.DataFrame(
        {
            "lambda_R": [0.8, 0.8],
            "rho_phys": [0.0, 0.0],
            "stage": ["stage_e_k2", "stage_e_k2"],
            "stage_label": ["Stage E constrained", "Stage E constrained"],
            "eval_id": [0, 1],
            "runtime_seconds": [1.0, 2.0],
            "gridfm_calls": [100, 80],
            "budget_exhausted": [True, False],
            "scipy_success": [False, True],
            "invalid_calls": [0, 1],
        }
    )

    summary = _runtime_summary(frame).iloc[0]

    assert summary["total_runtime_seconds"] == 3.0
    assert summary["total_gridfm_calls"] == 180
    assert summary["budget_exhaustion_rate"] == 0.5
    assert summary["scipy_convergence_rate"] == 0.5
