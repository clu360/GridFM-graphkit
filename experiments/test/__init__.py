"""
Predict-then-optimize pipeline for GridFM power grid dispatch optimization.

This package provides a complete pipeline for optimizing power grid dispatch
using pretrained GridFM neural solvers (GNN_v0.1 or GPS_v0.2).

Main modules:
- scenario_data: Canonical scenario representation
- pv_dispatch: PV-bus Pg decision variable specification
- neural_solver: Neural solver wrapper for both models
- overload_penalty: Line loading and overload computation
- optimization: Dispatch optimization problem
- validation: Pipeline validation harness
"""

from .scenario_data import ScenarioData, extract_scenario_from_batch
from .pv_dispatch import PVDispatchDecisionSpec
from .neural_solver import NeuralSolverWrapper
from .overload_penalty import OverloadPenaltyEvaluator
from .optimization import DispatchOptimizationProblem
from .validation import PipelineValidationHarness

__all__ = [
    "ScenarioData",
    "extract_scenario_from_batch",
    "PVDispatchDecisionSpec",
    "NeuralSolverWrapper",
    "OverloadPenaltyEvaluator",
    "DispatchOptimizationProblem",
    "PipelineValidationHarness",
]
