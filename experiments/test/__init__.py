"""Research support package for wildfire experiment workflows."""

from __future__ import annotations

import importlib
import sys


_SUPPORT_ALIASES = {
    "pipeline_utils": "experiments.test.wildfire_tests.gridfm_support.pipeline_utils",
    "neural_solver": "experiments.test.wildfire_tests.gridfm_support.neural_solver",
    "scenario_data": "experiments.test.wildfire_tests.gridfm_support.scenario_data",
    "overload_penalty": "experiments.test.wildfire_tests.gridfm_support.overload_penalty",
    "pv_dispatch": "experiments.test.wildfire_tests.gridfm_support.pv_dispatch",
}

for _name, _target in _SUPPORT_ALIASES.items():
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(_target)

__all__ = []
