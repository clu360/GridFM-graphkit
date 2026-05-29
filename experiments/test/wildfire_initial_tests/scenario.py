from __future__ import annotations

from experiments.test.pipeline_utils import load_single_test_scenario

from .config import FirstPassConfig


def load_first_pass_context(config: FirstPassConfig):
    return load_single_test_scenario(
        scenario_idx=config.scenario.scenario_idx,
        scenario_id=config.scenario.scenario_id,
        config_name=config.scenario.base_config_name,
    )
