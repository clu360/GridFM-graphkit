from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pandas as pd

from experiments.test.wildfire_tests.stage_j_gridsfm_goc500 import summarize_stage_j_complete as base
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.finetune import run_ft7_warm_starts as warm
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.finetune import run_ft7_iteration_timing as timing
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.finetune import summarize_ft7_refined_study as summary


def test_ft7_comparison_contract_is_dc_plus_four_gridsfm_models() -> None:
    assert list(summary.METHOD_SPECS) == ["dc", "m0", "m1", "m2", "m3"]
    assert len(summary.METHOD_ORDER) == 5
    assert all(not label.startswith("TH-") for label in summary.METHOD_ORDER)
    assert {spec["sha256"] for key, spec in summary.METHOD_SPECS.items() if key != "dc"} == {
        warm.MODEL_SPECS[key]["sha256"] for key in ("m0", "m1", "m2", "m3")
    }


def test_ft7_pareto_uses_all_lambdas_and_removes_dominated_points(monkeypatch) -> None:
    monkeypatch.setattr(base, "METHOD_ORDER", summary.METHOD_ORDER)
    rows = []
    for scenario in ("J-S1", "J-S2", "J-S3"):
        for method in summary.METHOD_ORDER:
            rows.extend([
                {"scenario_id": scenario, "method": method, "lambda_r": 0.0,
                 "topology_rank": 1, "candidate_index": 1, "evaluation_status": "ok",
                 "l_shed_total": 0.0, "r_norm": 2.0},
                {"scenario_id": scenario, "method": method, "lambda_r": 0.5,
                 "topology_rank": 1, "candidate_index": 2, "evaluation_status": "ok",
                 "l_shed_total": 1.0, "r_norm": 1.0},
                {"scenario_id": scenario, "method": method, "lambda_r": 1.0,
                 "topology_rank": 1, "candidate_index": 3, "evaluation_status": "ok",
                 "l_shed_total": 2.0, "r_norm": 2.0},
            ])
    front = base.build_evaluated_pareto_frontiers(pd.DataFrame(rows))
    base._validate_pareto_frontiers(pd.DataFrame(rows), front)
    assert len(front) == 30
    assert set(front["lambda_r"]) == {0.0, 0.5}
    assert front.groupby(["scenario_id", "method"])["evaluated_candidate_count"].first().eq(3).all()


def test_ft7_warm_start_validation_requires_complete_five_by_seven_matrix() -> None:
    rows = []
    for setting_index in range(15):
        for family in warm.FAMILY_SPECS:
            instance = f"{setting_index}-{family}"
            for start in warm.START_ORDER:
                row = {
                    "setting_code": f"s{setting_index}", "comparison_variant": family,
                    "warm_start_type": start, "reference_a_instance_id": instance,
                    "objective": "10.0", "status": "LOCALLY_SOLVED",
                    "same_reference_a_instance": "True", "runtime_metric": "ipopt_solve_time_seconds",
                    "solver_start_source": "", "start_vm_min": "", "start_vm_max": "",
                    "start_va_abs_max": "", "start_pg_midpoint_max_abs_error": "",
                    "start_qg_abs_max": "", "start_payload": "",
                    "warm_start_checkpoint_sha256": "",
                }
                if start == "cold_start":
                    row.update({
                        "solver_start_source": "explicit_generic_V1_theta0_Pg_midpoint_Qg0",
                        "start_vm_min": "1", "start_vm_max": "1", "start_va_abs_max": "0",
                        "start_pg_midpoint_max_abs_error": "0", "start_qg_abs_max": "0",
                    })
                if start.startswith("gridsfm_m"):
                    model_id = start.split("_")[1]
                    row["start_payload"] = "Pg,Qg,V,theta"
                    row["warm_start_checkpoint_sha256"] = warm.MODEL_SPECS[model_id]["sha256"]
                rows.append(row)
    validation = warm._validate(rows, 1e-3)
    assert validation["status"] == "FT7_P4_WARM_START_PASS"
    assert validation["rows"] == 525
    assert all(count == 15 for count in validation["cell_counts"].values())


def test_ft7_frozen_cross_command_filters_out_th(tmp_path: Path) -> None:
    setting = tmp_path / "s1_l0p0"
    setting.mkdir()
    (setting / "finalists.json").write_text(json.dumps({"finalists": [
        {"method": "Guided-DC", "best_topology": {"scenario_id": "J-S1"}},
        {"method": "Guided-GridSFM", "best_topology": {"scenario_id": "J-S1"}},
        {"method": "TH-GridSFM-top1", "best_topology": {"scenario_id": "J-S1"}},
    ]}), encoding="utf-8")
    args = Namespace(
        python=Path("python"), input_dir=tmp_path, gridsfm_root=tmp_path,
        timeout_seconds=900, checkpoints={"m2": tmp_path / "m2.pt"},
    )
    command = warm._command(args, setting, "frozen", "m2")
    filters = [command[index + 1] for index, value in enumerate(command) if value == "--method-filter"]
    assert filters == ["Guided-DC", "Guided-GridSFM"]
    assert all(not value.startswith("TH-") for value in filters)
    m0_command = warm._command(args, setting, "frozen", "m0")
    assert m0_command[m0_command.index("--warm-start-id") + 1] == "m0"


def test_ft7_iteration_rerun_rotates_all_start_positions() -> None:
    orders = [timing._ordered_starts(index) for index in range(7)]
    assert all(sorted(order) == sorted(warm.START_ORDER) for order in orders)
    for position in range(7):
        assert {order[position] for order in orders} == set(warm.START_ORDER)
