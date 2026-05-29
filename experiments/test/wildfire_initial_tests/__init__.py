"""
First-pass grouped wildfire predict-then-optimize experiment.

This package is additive to the existing `experiments.test` workflows. It keeps
the topology fixed, uses GridFM in memory as the surrogate solver, and evaluates
a reduced decision vector with selected generator redispatch and selected load
service fractions.
"""

from .config import FirstPassConfig, load_first_pass_config
from .decision_vector import FirstPassDecisionVector
from .optimization_problem import FirstPassOptimizationProblem
from .wildfire_scenario import WildfireLineGroup, WildfireScenario

__all__ = [
    "FirstPassConfig",
    "FirstPassDecisionVector",
    "FirstPassOptimizationProblem",
    "WildfireLineGroup",
    "WildfireScenario",
    "load_first_pass_config",
]
