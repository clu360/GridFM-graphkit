import math

import numpy as np

from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_revised_continuous_implementation import (
    _make_continuous_context,
)
from experiments.test.wildfire_tests.stage_i_dc_comparison.dc_formulation import (
    STAGE_I_B,
    THETA_MAX,
    build_dc_network,
    common_operational_diagnostic,
)
from experiments.test.wildfire_tests.stage_i_dc_comparison.run_stage_h_dc_comparison import (
    methodology_fidelity_checks,
)


def test_dc_network_uses_matpower_rate_a_and_tap_shift_audit():
    context = _make_continuous_context("gnn", 5.0)
    network = build_dc_network(context["scenario"])

    assert network.branches
    assert all(branch.rate_a_mva > 0 for branch in network.branches)
    assert network.branch_audit["dc_branch_model_used"] in {"trivial_tap_shift", "matpower_tap_shift"}
    assert "num_nonunity_taps" in network.branch_audit
    assert "num_nonzero_phase_shifts" in network.branch_audit


def test_common_operational_diagnostic_zero_for_empty_feasible_values():
    context = _make_continuous_context("gnn", 5.0)
    scenario = context["scenario"]
    network = build_dc_network(scenario)
    flow = {branch.line_id: 0.0 for branch in network.branches}
    pg = {bus: float(scenario.Pg_min[bus]) for bus in network.generator_buses}
    service = {bus: 0.0 for bus in network.load_buses}

    diagnostic = common_operational_diagnostic(scenario, network, [], flow, pg, service, [])

    assert diagnostic["PAC_common_thermal_overlap"] == 0.0
    assert diagnostic["PAC_common_topology_offline_flow"] == 0.0
    assert diagnostic["PAC_common_load_service_bounds"] == 0.0


def test_methodology_checks_reject_more_than_two_shutoffs():
    import pandas as pd

    best = pd.DataFrame(
        [
            {
                "stage": STAGE_I_B,
                "stage_label": "Stage I-b DC MIQP K2",
                "num_shutoff_lines": 3,
                "max_abs_nodal_balance_residual": 0.0,
                "max_abs_angle_flow_residual": 0.0,
            }
        ]
    )
    checks = methodology_fidelity_checks(
        best,
        pd.DataFrame({"x": [1]}),
        pd.DataFrame({"x": [1]}),
        pd.DataFrame({"projection_status": ["attempted"], "projection_solution_id": ["a"]}),
        {"dc_branch_model_used": "trivial_tap_shift"},
        smoke=True,
    )

    row = checks[checks["check_name"].eq("all_shutoffs_leq_2")].iloc[0]
    assert not bool(row["passed"])

