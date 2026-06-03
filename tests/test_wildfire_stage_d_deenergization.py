import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.test.wildfire_initial_tests.stage_d_deenergization import (
    LAMBDA_CASES,
    disconnected_buses_from_reference,
    enumerate_deenergization_subsets,
    expected_subset_count,
    line_risk_total,
    select_best_subset,
)


def test_z_zero_removes_line_risk_and_z_one_preserves_it():
    total, by_line = line_risk_total(
        loading_ratio=np.array([2.0, 3.0]),
        p_env_by_line={0: 1.0, 1: 2.0},
        impact=np.array([0.5, 0.25]),
        z_by_line={0: 0, 1: 1},
        candidate_line_ids=[0, 1],
    )

    assert by_line[0] == 0.0
    assert np.isclose(by_line[1], 2.0 * 9.0 * 0.25)
    assert np.isclose(total, by_line[1])


def test_candidate_subset_count_and_enumeration_are_from_candidate_set():
    subsets = enumerate_deenergization_subsets([7, 3, 5], max_deenergized_lines=2)

    assert expected_subset_count(3, 2) == 1 + 3 + 3
    assert subsets[0] == ()
    assert {(3,), (5,), (7,)}.issubset(set(subsets))
    assert {(3, 5), (3, 7), (5, 7)}.issubset(set(subsets))
    assert all(len(subset) <= 2 for subset in subsets)


def test_lambda_cases_are_exact_stage_d_tradeoffs():
    assert LAMBDA_CASES == {
        "risk_leaning": (0.8, 0.2),
        "balanced": (0.5, 0.5),
        "service_leaning": (0.2, 0.8),
    }


def test_best_subset_tiebreak_prefers_lower_l_norm_then_fewer_lines():
    evaluations = pd.DataFrame(
        [
            {"eval_id": 0, "deenergized_line_ids": [], "num_deenergized_lines": 0, "R_group": 1.0, "R_norm": 0.4, "L_norm": 0.3, "demand_weighted_load_shed": 0.3},
            {"eval_id": 1, "deenergized_line_ids": [2], "num_deenergized_lines": 1, "R_group": 1.0, "R_norm": 0.5, "L_norm": 0.2, "demand_weighted_load_shed": 0.2},
            {"eval_id": 2, "deenergized_line_ids": [1, 2], "num_deenergized_lines": 2, "R_group": 1.0, "R_norm": 0.5, "L_norm": 0.2, "demand_weighted_load_shed": 0.2},
        ]
    )

    best = select_best_subset(evaluations, "balanced", 0.5, 0.5)

    assert best.eval_id == 1
    assert best.deenergized_line_ids == [2]


def test_shared_evaluations_can_choose_different_best_by_lambda():
    evaluations = pd.DataFrame(
        [
            {"eval_id": 0, "deenergized_line_ids": [], "num_deenergized_lines": 0, "R_group": 10.0, "R_norm": 1.0, "L_norm": 0.0, "demand_weighted_load_shed": 0.0},
            {"eval_id": 1, "deenergized_line_ids": [3], "num_deenergized_lines": 1, "R_group": 1.0, "R_norm": 0.1, "L_norm": 0.9, "demand_weighted_load_shed": 0.9},
        ]
    )

    risk_best = select_best_subset(evaluations, "risk_leaning", 0.8, 0.2)
    service_best = select_best_subset(evaluations, "service_leaning", 0.2, 0.8)

    assert risk_best.eval_id == 1
    assert service_best.eval_id == 0


def test_disconnected_bus_diagnostics_from_reference_component():
    edge_index = np.array([[0, 1, 2], [1, 2, 3]])

    assert disconnected_buses_from_reference(edge_index, 4, [1]) == [2, 3]


def test_stage_d_runner_summary_rows_with_monkeypatched_runs(monkeypatch, tmp_path):
    from experiments.test.wildfire_initial_tests import run_stage_d_deenergization as runner

    def fake_root():
        return tmp_path / "stage_d_deenergization"

    def fake_context(model_type, grouping_top_fraction):
        return {"model": model_type, "grouping_top_fraction": grouping_top_fraction}

    def fake_write_lambda_run(
        model_context,
        case_name,
        env,
        output_root,
        grouping_top_fraction,
        max_deenergized_lines,
        lambda_case,
        lambda_R,
        lambda_L,
        candidate_line_ids,
        num_evaluated_subsets,
        evaluations,
        stage_c_eval,
        stage_c_line_ids,
        baseline_risk,
    ):
        run_dir = Path(output_root) / "fake_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "model_type": model_context["model"],
            "environmental_risk_case": case_name,
            "lambda_case": lambda_case,
            "lambda_R": lambda_R,
            "lambda_L": lambda_L,
            "grouping_top_fraction": grouping_top_fraction,
            "evaluation_mode": runner.EVALUATION_MODE,
            "max_deenergized_lines": max_deenergized_lines,
            "num_candidate_lines": len(candidate_line_ids),
            "num_evaluated_subsets": num_evaluated_subsets,
            "stage_c_comparison_available": bool(stage_c_eval),
            "status": "ok",
            "error": "",
            "run_dir": str(run_dir),
        }

    class FakeGroup:
        def __init__(self, line_ids):
            self.line_ids = line_ids

    class FakeWildfire:
        line_groups = [FakeGroup([0, 1]), FakeGroup([2])]

    monkeypatch.setattr(runner, "_stage_d_root", fake_root)
    monkeypatch.setattr(runner, "_build_model_context", fake_context)
    monkeypatch.setattr(runner, "_load_stage_c_summary", lambda: None)
    monkeypatch.setattr(runner, "apply_environmental_case", lambda wildfire, group_summary, case_name: type("Env", (), {"p_env_by_line": {0: 1.0, 1: 1.0, 2: 1.0}})())
    monkeypatch.setattr(runner, "line_risk_total", lambda *args, **kwargs: (1.0, {0: 1.0, 1: 0.5, 2: 0.25}))
    monkeypatch.setattr(runner, "_evaluate_all_subsets", lambda *args, **kwargs: pd.DataFrame([{"eval_id": 0, "status": "ok"}]))
    monkeypatch.setattr(runner, "_write_lambda_run", fake_write_lambda_run)
    fake_context_data = {
        "config": type("Cfg", (), {})(),
        "wildfire": FakeWildfire(),
        "fixed_impact": np.ones(3),
        "baseline_state": {"loading_ratio": np.ones(3)},
        "automatic_artifacts": {"group_summary": pd.DataFrame()},
    }
    monkeypatch.setattr(runner, "_build_model_context", lambda model_type, grouping_top_fraction: {**fake_context_data, "model": model_type})

    summary_csv = runner.run_stage_d_deenergization(
        grouping_top_fraction=0.30,
        models=["gps"],
        cases=["auto_env", "largest_group_high"],
        lambda_cases=["risk_leaning", "balanced", "service_leaning"],
    )
    frame = pd.read_csv(summary_csv)
    with open(tmp_path / "stage_d_deenergization" / "stage_d_deenergization_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert len(frame) == 6
    assert set(frame["lambda_case"]) == {"risk_leaning", "balanced", "service_leaning"}
    assert set(frame["num_candidate_lines"]) == {3}
    assert set(frame["num_evaluated_subsets"]) == {7}
    assert summary["num_successful_runs"] == 6
