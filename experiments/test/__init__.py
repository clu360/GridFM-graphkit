"""
Wildfire-aware IEEE-30 surrogate dispatch workflow.

`experiments/test` is the restored research sandbox around the legacy
homogeneous GridFM IEEE-30 data path. The current workflow:

- loads one IEEE-30 graph from `tests/config/gridFMv0.1_dummy.yaml`
- wraps pretrained GNN or GPS checkpoints as surrogate PF predictors
- optimizes PV redispatch and optional PQ load shedding
- scores candidate dispatches with a wildfire-risk term derived from
  predicted branch loading

`overload_penalty.py` remains available as a legacy postprocessing helper, but
the active objective in this package is wildfire-only.
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
    get_gnn_checkpoint_path,
    get_gps_checkpoint_path,
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
    "get_gnn_checkpoint_path",
    "get_gps_checkpoint_path",
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
