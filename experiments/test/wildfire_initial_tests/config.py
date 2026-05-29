from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from experiments.test.pipeline_utils import get_repo_root


@dataclass
class ModelConfig:
    model_type: str = "gps"
    checkpoint_path: Optional[str] = None
    device: str = "cpu"


@dataclass
class ScenarioConfig:
    scenario_idx: int = 0
    scenario_id: str = "IEEE-30-wildfire-first-pass"
    base_config_name: str = "gridFMv0.1_dummy.yaml"


@dataclass
class DecisionConfig:
    selected_generator_buses: List[int] = field(default_factory=list)
    selected_load_buses: List[int] = field(default_factory=list)
    auto_select_generators: int = 3
    auto_select_loads: int = 5
    delta_pg_bound_mw: float = 5.0
    alpha_min: float = 0.00
    alpha_max: float = 1.00
    random_init_scale: float = 0.01


@dataclass
class WildfireConfig:
    selection_method: str = "top_loaded"
    selected_line_ids: List[int] = field(default_factory=list)
    num_high_risk_lines: int = 3
    high_hazard: float = 5.0
    default_hazard: float = 0.1
    default_impact: float = 1.0
    group_weight: float = 1.0
    hazard_multiplier: float = 1.0
    standard_rate_a_mva: float = 100.0


@dataclass
class ObjectiveConfig:
    lambda_R: float = 1.0
    lambda_L: float = 1.0
    normalize_terms: bool = True
    risk_normalizer: float = 0.0
    load_shedding_normalizer: float = 0.0
    invalid_prediction_penalty: float = 1.0e12
    strict: bool = False
    debug_save_evaluations: bool = False


@dataclass
class OptimizerConfig:
    method: str = "L-BFGS-B"
    maxiter: int = 80
    ftol: float = 1.0e-9
    gtol: float = 1.0e-6
    eps: float = 1.0e-4
    disp: bool = False


@dataclass
class OutputConfig:
    output_root: str = "experiments/test/wildfire_initial_tests/results"
    run_name: str = "basic_gps"


@dataclass
class SweepConfig:
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    model_types: List[str] = field(default_factory=lambda: ["gps", "gnn"])
    lambda_R_values: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    lambda_L_values: List[float] = field(default_factory=lambda: [0.1, 1.0])
    hazard_multipliers: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0, 10.0])
    initializations: List[str] = field(default_factory=lambda: ["baseline", "small_random"])


@dataclass
class FirstPassConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    wildfire: WildfireConfig = field(default_factory=WildfireConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    random_seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def output_root_path(self) -> Path:
        root = Path(self.output.output_root)
        if not root.is_absolute():
            root = get_repo_root() / root
        return root


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _from_nested_dict(data: Dict[str, Any]) -> FirstPassConfig:
    def filtered(cls, section: str) -> Dict[str, Any]:
        allowed = {item.name for item in fields(cls)}
        return {k: v for k, v in data.get(section, {}).items() if k in allowed}

    return FirstPassConfig(
        model=ModelConfig(**filtered(ModelConfig, "model")),
        scenario=ScenarioConfig(**filtered(ScenarioConfig, "scenario")),
        decision=DecisionConfig(**filtered(DecisionConfig, "decision")),
        wildfire=WildfireConfig(**filtered(WildfireConfig, "wildfire")),
        objective=ObjectiveConfig(**filtered(ObjectiveConfig, "objective")),
        optimizer=OptimizerConfig(**filtered(OptimizerConfig, "optimizer")),
        output=OutputConfig(**filtered(OutputConfig, "output")),
        sweep=SweepConfig(**filtered(SweepConfig, "sweep")),
        random_seed=int(data.get("random_seed", 0)),
    )


def load_first_pass_config(path: str | Path) -> FirstPassConfig:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    base = FirstPassConfig().to_dict()
    return _from_nested_dict(_deep_update(base, loaded))


def write_config_copy(config: FirstPassConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
