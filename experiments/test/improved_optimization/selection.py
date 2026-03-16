"""
Selection logic for targeted generator redispatch and targeted load shedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from experiments.test.neural_solver import NeuralSolverWrapper
from experiments.test.scenario_data import ScenarioData
from .config import SelectionConfig, WildfireConfig
from .wildfire_metrics import compute_line_loading_ratios, compute_line_weights, summarize_top_risky_lines


class TargetedDispatchDecisionSpec:
    """
    Compact decision specification for selected PV redispatch and selected PQ shedding.

    Decision vector ordering:
    [delta_pg_selected, shed_selected]
    """

    def __init__(
        self,
        scenario: ScenarioData,
        selected_generator_buses: Sequence[int],
        selected_load_buses: Sequence[int],
        max_shed_fraction: float = 0.5,
        generator_weights: np.ndarray | None = None,
        shed_weights: np.ndarray | None = None,
    ):
        self.scenario = scenario
        self.selected_generator_buses = np.asarray(selected_generator_buses, dtype=int)
        self.selected_load_buses = np.asarray(selected_load_buses, dtype=int)
        self.n_generators = int(len(self.selected_generator_buses))
        self.n_loads = int(len(self.selected_load_buses))
        self.n_total = self.n_generators + self.n_loads
        self.max_shed_fraction = float(max_shed_fraction)

        self.delta_pg_base = np.zeros(self.n_generators, dtype=float)
        self.shed_base = np.zeros(self.n_loads, dtype=float)
        self.u_base = np.zeros(self.n_total, dtype=float)

        self.delta_pg_min = scenario.Pg_min[self.selected_generator_buses] - scenario.Pg_base[self.selected_generator_buses]
        self.delta_pg_max = scenario.Pg_max[self.selected_generator_buses] - scenario.Pg_base[self.selected_generator_buses]
        self.shed_min = np.zeros(self.n_loads, dtype=float)
        self.shed_max = scenario.Pd_base[self.selected_load_buses] * self.max_shed_fraction

        self.u_min = np.hstack([self.delta_pg_min, self.shed_min])
        self.u_max = np.hstack([self.delta_pg_max, self.shed_max])

        self.generator_weights = np.ones(self.n_generators, dtype=float) if generator_weights is None else np.asarray(generator_weights, dtype=float)
        self.shed_weights = np.ones(self.n_loads, dtype=float) if shed_weights is None else np.asarray(shed_weights, dtype=float)

    def split_decision_vector(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta_pg = np.asarray(u[: self.n_generators], dtype=float)
        shed = np.asarray(u[self.n_generators :], dtype=float)
        return delta_pg, shed

    def combine_decision_vector(self, delta_pg: np.ndarray, shed: np.ndarray) -> np.ndarray:
        return np.hstack([delta_pg, shed]).astype(float)

    def u_to_node_features(self, u: np.ndarray) -> np.ndarray:
        from gridfm_graphkit.datasets.globals import PD, PG

        delta_pg, shed = self.split_decision_vector(u)
        node_features = self.scenario.get_baseline_node_features().copy()
        if self.n_generators:
            node_features[self.selected_generator_buses, PG] = self.scenario.Pg_base[self.selected_generator_buses] + delta_pg
        if self.n_loads:
            node_features[self.selected_load_buses, PD] = self.scenario.Pd_base[self.selected_load_buses] - shed
        return node_features

    def check_bounds(self, u: np.ndarray) -> tuple[bool, str]:
        u = np.asarray(u, dtype=float)
        violations_min = int(np.sum(u < self.u_min))
        violations_max = int(np.sum(u > self.u_max))
        if violations_min or violations_max:
            return False, f"Decision bound violations: {violations_min} lower, {violations_max} upper"
        return True, "All bounds satisfied"

    def decision_metadata(self) -> pd.DataFrame:
        rows = []
        for idx, bus in enumerate(self.selected_generator_buses):
            rows.append(
                {
                    "decision_type": "generator_redispatch",
                    "decision_index": idx,
                    "bus_idx": int(bus),
                    "lower_bound": float(self.delta_pg_min[idx]),
                    "upper_bound": float(self.delta_pg_max[idx]),
                    "baseline": 0.0,
                }
            )
        offset = self.n_generators
        for idx, bus in enumerate(self.selected_load_buses):
            rows.append(
                {
                    "decision_type": "load_shedding",
                    "decision_index": offset + idx,
                    "bus_idx": int(bus),
                    "lower_bound": float(self.shed_min[idx]),
                    "upper_bound": float(self.shed_max[idx]),
                    "baseline": 0.0,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class SelectionDiagnostics:
    generator_scores: pd.DataFrame
    load_scores: pd.DataFrame
    risky_lines: pd.DataFrame


def identify_wildfire_sensitive_lines(loading: np.ndarray, config: WildfireConfig) -> np.ndarray:
    threshold = config.wildfire_threshold - config.threshold_buffer
    mask = loading >= threshold
    if int(np.sum(mask)) < config.top_n_risky_lines:
        top_idx = np.argsort(loading)[::-1][: config.top_n_risky_lines]
        mask[top_idx] = True
    return mask


def _candidate_generator_buses(scenario: ScenarioData, selection_cfg: SelectionConfig) -> np.ndarray:
    pv_buses = scenario.get_pv_buses()
    headroom = scenario.Pg_max[pv_buses] - scenario.Pg_min[pv_buses]
    return pv_buses[headroom > selection_cfg.min_generator_headroom]


def _candidate_load_buses(scenario: ScenarioData, selection_cfg: SelectionConfig) -> np.ndarray:
    pq_buses = scenario.get_pq_buses()
    pd = scenario.Pd_base[pq_buses]
    return pq_buses[pd > selection_cfg.min_load_mw]


def _finite_difference_step(preferred_step: float, lower_bound: float, upper_bound: float) -> float:
    positive_room = max(0.0, upper_bound)
    negative_room = max(0.0, -lower_bound)
    max_room = max(positive_room, negative_room)
    if max_room <= 0.0:
        return 0.0
    step_mag = min(preferred_step, max_room)
    if positive_room >= step_mag and positive_room >= negative_room:
        return step_mag
    if negative_room >= step_mag:
        return -step_mag
    return step_mag if positive_room >= negative_room else -step_mag


def _predict_loading_with_controls(
    solver: NeuralSolverWrapper,
    spec: TargetedDispatchDecisionSpec,
    u: np.ndarray,
    wildfire_cfg: WildfireConfig,
) -> np.ndarray:
    pred = solver.predict_state(u)
    return compute_line_loading_ratios(
        solver.scenario,
        pred["Vm"],
        pred["Va"],
        standard_rate_a_mva=wildfire_cfg.standard_rate_a_mva,
    )


def compute_generator_relief_scores(
    scenario: ScenarioData,
    solver: NeuralSolverWrapper,
    selection_cfg: SelectionConfig,
    wildfire_cfg: WildfireConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_buses = _candidate_generator_buses(scenario, selection_cfg)
    screening_spec = TargetedDispatchDecisionSpec(
        scenario,
        selected_generator_buses=candidate_buses,
        selected_load_buses=[],
        max_shed_fraction=selection_cfg.max_shed_fraction,
    )
    screening_solver = NeuralSolverWrapper(
        solver.model,
        solver.model_type,
        scenario,
        screening_spec,
        device=solver.device,
    )
    baseline_loading = _predict_loading_with_controls(screening_solver, screening_spec, screening_spec.u_base, wildfire_cfg)
    risky_mask = identify_wildfire_sensitive_lines(baseline_loading, wildfire_cfg)
    line_weights = compute_line_weights(baseline_loading, wildfire_cfg)
    rows: List[Dict] = []
    eps = selection_cfg.epsilon_fd

    for local_idx, bus_idx in enumerate(candidate_buses):
        u = screening_spec.u_base.copy()
        step = _finite_difference_step(
            eps,
            float(screening_spec.delta_pg_min[local_idx]),
            float(screening_spec.delta_pg_max[local_idx]),
        )
        if abs(step) <= 0.0:
            continue
        u[local_idx] = step
        perturbed_loading = _predict_loading_with_controls(screening_solver, screening_spec, u, wildfire_cfg)
        sensitivity = (perturbed_loading - baseline_loading) / step
        relief = line_weights[risky_mask] * np.maximum(0.0, -sensitivity[risky_mask])
        rows.append(
            {
                "bus_idx": int(bus_idx),
                "baseline_pg": float(scenario.Pg_base[bus_idx]),
                "lower_delta": float(screening_spec.delta_pg_min[local_idx]),
                "upper_delta": float(screening_spec.delta_pg_max[local_idx]),
                "relief_score": float(np.sum(relief)),
                "max_single_line_relief": float(np.max(relief) if len(relief) else 0.0),
            }
        )

    score_df = pd.DataFrame(rows)
    if not score_df.empty:
        score_df = score_df.sort_values("relief_score", ascending=False).reset_index(drop=True)
    risky_df = summarize_top_risky_lines(
        baseline_loading,
        np.maximum(0.0, baseline_loading - wildfire_cfg.wildfire_threshold),
        line_weights,
        top_n=wildfire_cfg.top_n_risky_lines,
    )
    return score_df, risky_df


def compute_load_relief_scores(
    scenario: ScenarioData,
    solver: NeuralSolverWrapper,
    selection_cfg: SelectionConfig,
    wildfire_cfg: WildfireConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_buses = _candidate_load_buses(scenario, selection_cfg)
    screening_spec = TargetedDispatchDecisionSpec(
        scenario,
        selected_generator_buses=[],
        selected_load_buses=candidate_buses,
        max_shed_fraction=selection_cfg.max_shed_fraction,
    )
    screening_solver = NeuralSolverWrapper(
        solver.model,
        solver.model_type,
        scenario,
        screening_spec,
        device=solver.device,
    )
    baseline_loading = _predict_loading_with_controls(screening_solver, screening_spec, screening_spec.u_base, wildfire_cfg)
    risky_mask = identify_wildfire_sensitive_lines(baseline_loading, wildfire_cfg)
    line_weights = compute_line_weights(baseline_loading, wildfire_cfg)
    rows: List[Dict] = []
    eps = selection_cfg.epsilon_fd

    for local_idx, bus_idx in enumerate(candidate_buses):
        step = min(eps, float(screening_spec.shed_max[local_idx]))
        if step <= 0.0:
            continue
        u = screening_spec.u_base.copy()
        u[local_idx] = step
        perturbed_loading = _predict_loading_with_controls(screening_solver, screening_spec, u, wildfire_cfg)
        sensitivity = (perturbed_loading - baseline_loading) / step
        relief = line_weights[risky_mask] * np.maximum(0.0, -sensitivity[risky_mask])
        rows.append(
            {
                "bus_idx": int(bus_idx),
                "baseline_pd": float(scenario.Pd_base[bus_idx]),
                "upper_shed": float(screening_spec.shed_max[local_idx]),
                "relief_score": float(np.sum(relief)),
                "max_single_line_relief": float(np.max(relief) if len(relief) else 0.0),
            }
        )

    score_df = pd.DataFrame(rows)
    if not score_df.empty:
        score_df = score_df.sort_values("relief_score", ascending=False).reset_index(drop=True)
    risky_df = summarize_top_risky_lines(
        baseline_loading,
        np.maximum(0.0, baseline_loading - wildfire_cfg.wildfire_threshold),
        line_weights,
        top_n=wildfire_cfg.top_n_risky_lines,
    )
    return score_df, risky_df


def select_top_generators(generator_scores: pd.DataFrame, top_k: int) -> np.ndarray:
    if generator_scores.empty:
        return np.array([], dtype=int)
    return generator_scores.head(top_k)["bus_idx"].to_numpy(dtype=int)


def select_top_load_buses(load_scores: pd.DataFrame, top_m: int) -> np.ndarray:
    if load_scores.empty:
        return np.array([], dtype=int)
    return load_scores.head(top_m)["bus_idx"].to_numpy(dtype=int)


def build_selected_decision_spec(
    scenario: ScenarioData,
    generator_scores: pd.DataFrame,
    load_scores: pd.DataFrame,
    selection_cfg: SelectionConfig,
) -> TargetedDispatchDecisionSpec:
    selected_generators = select_top_generators(generator_scores, selection_cfg.top_k_generators)
    selected_loads = select_top_load_buses(load_scores, selection_cfg.top_m_loads)
    return TargetedDispatchDecisionSpec(
        scenario,
        selected_generator_buses=selected_generators,
        selected_load_buses=selected_loads,
        max_shed_fraction=selection_cfg.max_shed_fraction,
    )
