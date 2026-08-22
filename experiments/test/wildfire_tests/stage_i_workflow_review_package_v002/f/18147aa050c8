from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List


LOW_P_ENV_DEFAULT = 0.05
HIGH_P_ENV_DEFAULT = 1.0


@dataclass(frozen=True)
class DecisionQualityScenario:
    scenario_id: str
    scenario_name: str
    target_high_risk_line_ids: tuple[int, ...]
    suppressed_line_ids: tuple[int, ...]
    expected_target_set: tuple[int, ...]
    hypothesis_expected_behavior: str
    interpretation_focus: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["target_high_risk_line_ids"] = list(self.target_high_risk_line_ids)
        data["suppressed_line_ids"] = list(self.suppressed_line_ids)
        data["expected_target_set"] = list(self.expected_target_set)
        return data


SCENARIOS: Dict[str, DecisionQualityScenario] = {
    "S1": DecisionQualityScenario(
        scenario_id="S1",
        scenario_name="S1_low_impact_high_risk",
        target_high_risk_line_ids=(27, 32, 101, 36),
        suppressed_line_ids=(),
        expected_target_set=(27, 32, 101, 36),
        hypothesis_expected_behavior="Risk-focused and balanced weights should select one or two of the high-risk lower-impact target lines.",
        interpretation_focus="Clean sanity check for high-risk, lower-system-impact shutoff behavior.",
    ),
    "S2": DecisionQualityScenario(
        scenario_id="S2",
        scenario_name="S2_high_consequence_non_bridge",
        target_high_risk_line_ids=(23,),
        suppressed_line_ids=(),
        expected_target_set=(23,),
        hypothesis_expected_behavior="Risk-focused weights may select line 23; balanced and load-focused weights may avoid it because of load-service consequence.",
        interpretation_focus="Risk/load tradeoff on the dominant high-risk high-consequence non-bridge line.",
    ),
    "S3": DecisionQualityScenario(
        scenario_id="S3",
        scenario_name="S3_redundant_g1_corridor",
        target_high_risk_line_ids=(18, 27, 16, 19, 22),
        suppressed_line_ids=(23,),
        expected_target_set=(18, 27, 16, 19, 22),
        hypothesis_expected_behavior="Risk-focused and balanced weights should select one or two lines from the connected G1 corridor while line 23 is suppressed.",
        interpretation_focus="Connected/redundant G1 corridor behavior distinct from S1; target lines sit around buses 1, 4, 5, and 6.",
    ),
    "S4": DecisionQualityScenario(
        scenario_id="S4",
        scenario_name="S4_source_less_island_trap",
        target_high_risk_line_ids=(77, 79),
        suppressed_line_ids=(),
        expected_target_set=(77, 79),
        hypothesis_expected_behavior="Selecting both target lines creates the bus-20 source-less island trap; load-focused weights should be less willing to select both.",
        interpretation_focus="Preserve the fixed-control load-service proxy for the objective and save source-less island diagnostics for interpretation.",
    ),
    "S5": DecisionQualityScenario(
        scenario_id="S5",
        scenario_name="S5_local_vs_distributed",
        target_high_risk_line_ids=(27, 32, 36, 101, 47, 51, 77, 79, 88, 91),
        suppressed_line_ids=(23,),
        expected_target_set=(27, 32, 36, 101, 47, 51, 77, 79, 88, 91),
        hypothesis_expected_behavior="Risk-focused weights may favor local G1 lines; balanced/load-focused weights may choose distributed lower-service-impact alternatives.",
        interpretation_focus="Local cluster versus distributed moderate-risk candidate structure, with line 23 suppressed.",
    ),
}


def get_scenarios(ids: Iterable[str] | None = None) -> List[DecisionQualityScenario]:
    if ids is None:
        return [SCENARIOS[key] for key in sorted(SCENARIOS)]
    scenarios: List[DecisionQualityScenario] = []
    for scenario_id in ids:
        key = str(scenario_id)
        if key not in SCENARIOS:
            raise ValueError(f"Unknown Stage F scenario: {key}. Supported scenarios: {sorted(SCENARIOS)}")
        scenarios.append(SCENARIOS[key])
    return scenarios


def p_env_for_scenario(
    scenario: DecisionQualityScenario,
    num_lines: int,
    low_value: float = LOW_P_ENV_DEFAULT,
    high_value: float = HIGH_P_ENV_DEFAULT,
    suppressed_value: float | None = None,
    extra_suppressed_line_ids: Iterable[int] = (),
) -> Dict[int, float]:
    p_env = {line_id: float(low_value) for line_id in range(int(num_lines))}
    target_line_ids = {int(line_id) for line_id in scenario.target_high_risk_line_ids}
    suppressed_line_ids = {int(line_id) for line_id in scenario.suppressed_line_ids}
    suppressed_line_ids.update(int(line_id) for line_id in extra_suppressed_line_ids)
    suppression = float(low_value) if suppressed_value is None else float(suppressed_value)
    for line_id in suppressed_line_ids:
        line_id = int(line_id)
        if line_id < 0 or line_id >= int(num_lines):
            raise ValueError(f"Scenario {scenario.scenario_id} suppressed line_id={line_id} is outside 0..{int(num_lines)-1}.")
        if line_id not in target_line_ids:
            p_env[line_id] = suppression
    for line_id in target_line_ids:
        line_id = int(line_id)
        if line_id < 0 or line_id >= int(num_lines):
            raise ValueError(f"Scenario {scenario.scenario_id} target line_id={line_id} is outside 0..{int(num_lines)-1}.")
        p_env[line_id] = float(high_value)
    return p_env
