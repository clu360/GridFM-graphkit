"""Wildfire predict-to-optimize research harness."""

from __future__ import annotations

import importlib

from experiments.test.wildfire_tests.shared.config import FirstPassConfig, load_first_pass_config
from experiments.test.wildfire_tests.shared.decision_vector import FirstPassDecisionVector
from experiments.test.wildfire_tests.shared.optimization_problem import FirstPassOptimizationProblem
from experiments.test.wildfire_tests.shared.wildfire_scenario import WildfireLineGroup, WildfireScenario

_MODULE_ALIASES = {
    "config": "experiments.test.wildfire_tests.shared.config",
    "decision_vector": "experiments.test.wildfire_tests.shared.decision_vector",
    "gridfm_runner": "experiments.test.wildfire_tests.shared.gridfm_runner",
    "objective": "experiments.test.wildfire_tests.shared.objective",
    "optimization_problem": "experiments.test.wildfire_tests.shared.optimization_problem",
    "plot_network_changes": "experiments.test.wildfire_tests.shared.plot_network_changes",
    "plot_optimization_behavior": "experiments.test.wildfire_tests.shared.plot_optimization_behavior",
    "reporting": "experiments.test.wildfire_tests.shared.reporting",
    "scenario": "experiments.test.wildfire_tests.shared.scenario",
    "state_extraction": "experiments.test.wildfire_tests.shared.state_extraction",
    "validation": "experiments.test.wildfire_tests.shared.validation",
    "wildfire_risk": "experiments.test.wildfire_tests.shared.wildfire_risk",
    "wildfire_scenario": "experiments.test.wildfire_tests.shared.wildfire_scenario",
    "wildfire_setup": "experiments.test.wildfire_tests.shared.wildfire_setup",
    "run_basic_case": "experiments.test.wildfire_tests.stage_a_first_pass.run_basic_case",
    "run_connected_corridor_tradeoffs": "experiments.test.wildfire_tests.stage_a_first_pass.run_connected_corridor_tradeoffs",
    "run_stability_sweep": "experiments.test.wildfire_tests.stage_a_first_pass.run_stability_sweep",
    "run_multi_group_threshold_sensitivity": "experiments.test.wildfire_tests.stage_b_multigroup.run_multi_group_threshold_sensitivity",
    "run_multistart_optimization": "experiments.test.wildfire_tests.stage_b_multigroup.run_multistart_optimization",
    "stage_c_psps": "experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps",
    "run_stage_c_psps_baseline": "experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline",
    "run_stage_d_deenergization": "experiments.test.wildfire_tests.stage_d_deenergization.run_stage_d_deenergization",
    "run_stage_e_gurobi_gridfm": "experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_gurobi_gridfm",
    "objective_analysis": "experiments.test.wildfire_tests.analysis.objective_analysis",
    "ac_opf_experiment": "experiments.test.wildfire_tests.analysis.ac_opf_experiment",
}


def __getattr__(name: str):
    if name in _MODULE_ALIASES:
        module = importlib.import_module(_MODULE_ALIASES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FirstPassConfig",
    "FirstPassDecisionVector",
    "FirstPassOptimizationProblem",
    "WildfireLineGroup",
    "WildfireScenario",
    "load_first_pass_config",
    *_MODULE_ALIASES.keys(),
]
