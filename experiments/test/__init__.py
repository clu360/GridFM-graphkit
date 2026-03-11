"""
Wildfire-aware predict-then-optimize pipeline for GridFM dispatch experiments.

This package provides the scenario extraction, neural surrogate wrappers,
wildfire-risk evaluation, optimization, and validation utilities used by the
`experiments/test` IEEE-30 workflow.

Main modules:
- scenario_data: Canonical scenario representation
- pv_dispatch / extended_dispatch_spec: Decision variable specifications
- neural_solver: Neural solver wrapper for GNN and GPS models
- wildfire_penalty: Branch-loading wildfire-risk computation
- optimization: Wildfire-aware dispatch optimization problem
- validation: Pipeline validation harness
"""

from .scenario_data import ScenarioData, extract_scenario_from_batch
from .pv_dispatch import PVDispatchDecisionSpec
from .load_shedding_spec import LoadSheddingSpec
from .extended_dispatch_spec import ExtendedDispatchSpec
from .neural_solver import NeuralSolverWrapper
from .overload_penalty import OverloadPenaltyEvaluator
from .pipeline_utils import (
    TestScenarioContext,
    get_repo_root,
    load_first_test_batch,
    load_gnn_model,
    load_gps_model,
    load_single_test_scenario,
    load_test_config,
    load_test_datamodule,
)
from .wildfire_penalty import WildfirePenaltyEvaluator, compute_wildfire_penalty
from .optimization import DispatchOptimizationProblem
from .validation import PipelineValidationHarness

__all__ = [
    "ScenarioData",
    "extract_scenario_from_batch",
    "PVDispatchDecisionSpec",
    "LoadSheddingSpec",
    "ExtendedDispatchSpec",
    "NeuralSolverWrapper",
    "OverloadPenaltyEvaluator",
    "TestScenarioContext",
    "get_repo_root",
    "load_test_config",
    "load_test_datamodule",
    "load_first_test_batch",
    "load_single_test_scenario",
    "load_gnn_model",
    "load_gps_model",
    "WildfirePenaltyEvaluator",
    "compute_wildfire_penalty",
    "DispatchOptimizationProblem",
    "PipelineValidationHarness",
]
