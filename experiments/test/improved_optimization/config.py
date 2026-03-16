"""
Central configuration for the Improved Optimization experiment suite.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from experiments.test.pipeline_utils import get_repo_root


@dataclass
class WildfireConfig:
    wildfire_threshold: float = 0.8
    threshold_buffer: float = 0.05
    top_n_risky_lines: int = 12
    top_percentile: float = 0.85
    softplus_alpha: float = 20.0
    use_softplus: bool = True
    standard_rate_a_mva: float = 100.0
    risk_weight_mode: str = "loading"


@dataclass
class ObjectiveConfig:
    lambda_g: float = 1.0
    lambda_s: float = 50.0
    lambda_w: float = 10.0
    shed_cost_default: float = 1.0
    generator_cost_default: float = 1.0


@dataclass
class SelectionConfig:
    top_k_generators: int = 5
    top_m_loads: int = 5
    epsilon_fd: float = 1e-3
    min_generator_headroom: float = 1e-12
    min_load_mw: float = 1e-3
    max_shed_fraction: float = 0.5


@dataclass
class OptimizationConfig:
    method: str = "L-BFGS-B"
    maxiter: int = 100
    ftol: float = 1e-9
    gtol: float = 1e-6
    eps: float = 1e-4
    disp: bool = False


@dataclass
class ParetoConfig:
    lambda_s_values: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])
    lambda_w: float = 1.0
    lambda_g: float = 1.0
    max_runs: int = 5


@dataclass
class SensitivityConfig:
    lambda_g_values: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    lambda_s_values: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 50.0])
    lambda_w_values: List[float] = field(default_factory=lambda: [1.0, 5.0, 10.0])
    wildfire_threshold_values: List[float] = field(default_factory=lambda: [0.75, 0.8, 0.85])
    threshold_buffer_values: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.1])
    softplus_alpha_values: List[float] = field(default_factory=lambda: [5.0, 20.0, 50.0])
    top_k_values: List[int] = field(default_factory=lambda: [3, 5, 7])
    top_m_values: List[int] = field(default_factory=lambda: [3, 5, 7])
    local_sweep_points: int = 21
    local_sweep_scale: float = 0.25


@dataclass
class ExperimentConfig:
    repo_root: Path
    package_root: Path
    results_root: Path
    scenario_idx: int = 0
    scenario_id: str = "IEEE-30-test"
    config_name: str = "gridFMv0.1_dummy.yaml"
    device: str = "cpu"
    model_name: str = "gnn"
    wildfire: WildfireConfig = field(default_factory=WildfireConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    pareto: ParetoConfig = field(default_factory=ParetoConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)

    def to_metadata(self) -> Dict:
        data = asdict(self)
        data["repo_root"] = str(self.repo_root)
        data["package_root"] = str(self.package_root)
        data["results_root"] = str(self.results_root)
        return data


def build_default_experiment_config(model_name: str = "gnn", device: str = "cpu") -> ExperimentConfig:
    repo_root = get_repo_root()
    package_root = repo_root / "experiments" / "test" / "improved_optimization"
    results_root = package_root / "results"
    return ExperimentConfig(
        repo_root=repo_root,
        package_root=package_root,
        results_root=results_root,
        device=device,
        model_name=model_name.lower(),
    )
