from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.gridfm_support.branch_metadata import (
    compute_matpower_ac_loading_by_line,
    physical_line_ids,
)
from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import make_run_dir, write_dataframe
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import (
    MODEL_CONFIGS,
    _build_model_context,
)


REQUIRED_COLUMNS = [
    "rank",
    "physical_branch_id",
    "canonical_line_id",
    "directed_line_ids",
    "from_bus",
    "to_bus",
    "rate_a_mva",
    "loading_ratio",
    "loading_squared",
    "p_env",
    "impact",
    "risk_score",
    "is_self_loop",
    "mapping_status",
    "outlier_flag",
    "outlier_reason",
]


def _edge_array(scenario) -> np.ndarray:
    return scenario.edge_index.cpu().numpy() if hasattr(scenario.edge_index, "cpu") else np.asarray(scenario.edge_index)


def _outlier_reason(
    loading: float,
    rate_a: float,
    mapping_status: str,
    is_self_loop: bool,
    vm_min: float,
    vm_max: float,
) -> tuple[bool, str]:
    reasons = []
    if is_self_loop:
        reasons.append("self_loop")
    if mapping_status != "mapped":
        reasons.append(f"mapping_{mapping_status}")
    if not np.isfinite(rate_a) or rate_a <= 0.0:
        reasons.append("missing_or_invalid_rate_a")
    if not np.isfinite(loading):
        reasons.append("nonfinite_loading")
    elif loading > 2.0:
        reasons.append("extreme_loading_gt_200pct")
    if vm_min < 0.0 or vm_max > 2.0:
        reasons.append("impossible_voltage_range")
    return bool(reasons), ";".join(reasons)


def build_loading_ranking_frame(model_type: str = "gnn", grouping_top_fraction: float = 0.30) -> pd.DataFrame:
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model_type={model_type}; expected one of {sorted(MODEL_CONFIGS)}.")
    context = _build_model_context(model_type, grouping_top_fraction)
    scenario = context["scenario"]
    state = context["baseline_state"]
    wildfire = context["wildfire"]
    consequence_df = context["consequence_df"]

    edge_array = _edge_array(scenario)
    loading = np.asarray(state["loading_ratio"], dtype=float)
    rates = np.asarray(scenario.rate_a, dtype=float)
    physical_ids = np.asarray(scenario.physical_branch_id, dtype=int)
    canonical_ids = np.asarray(scenario.canonical_line_id, dtype=int)
    is_self_loop = np.asarray(scenario.is_self_loop, dtype=bool)
    mapping_status = np.asarray(scenario.branch_mapping_status, dtype=object)
    p_env = wildfire.hazard_vector(len(loading))
    impact_by_line: Dict[int, float] = {
        int(row["line_id"]): float(row["I_l"] if "I_l" in row else row["c_l"])
        for _, row in consequence_df.iterrows()
    }

    rows = []
    vm = np.asarray(state["Vm"], dtype=float)
    vm_min = float(np.nanmin(vm))
    vm_max = float(np.nanmax(vm))
    for line_id in physical_line_ids(scenario):
        line_id = int(line_id)
        directed_ids = scenario.physical_branch_directed_line_ids.get(int(physical_ids[line_id]), [line_id])
        impact = float(impact_by_line.get(line_id, 1.0))
        risk = float(p_env[line_id] * loading[line_id] ** 2 * impact)
        outlier, reason = _outlier_reason(
            loading=float(loading[line_id]),
            rate_a=float(rates[line_id]),
            mapping_status=str(mapping_status[line_id]),
            is_self_loop=bool(is_self_loop[line_id]),
            vm_min=vm_min,
            vm_max=vm_max,
        )
        rows.append(
            {
                "physical_branch_id": int(physical_ids[line_id]),
                "canonical_line_id": line_id,
                "directed_line_ids": ",".join(str(int(item)) for item in directed_ids),
                "from_bus": int(edge_array[0, line_id]),
                "to_bus": int(edge_array[1, line_id]),
                "rate_a_mva": float(rates[line_id]),
                "loading_ratio": float(loading[line_id]),
                "loading_squared": float(loading[line_id] ** 2),
                "p_env": float(p_env[line_id]),
                "impact": impact,
                "risk_score": risk,
                "is_self_loop": bool(is_self_loop[line_id]),
                "mapping_status": str(mapping_status[line_id]),
                "outlier_flag": bool(outlier),
                "outlier_reason": reason,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["risk_score", "loading_ratio", "canonical_line_id"],
        ascending=[False, False, True],
    )
    frame.insert(0, "rank", np.arange(1, len(frame) + 1, dtype=int))
    return frame[REQUIRED_COLUMNS]


def build_matpower_comparison_frame(
    model_type: str = "gnn",
    grouping_top_fraction: float = 0.30,
    top_n: int = 10,
) -> pd.DataFrame:
    context = _build_model_context(model_type, grouping_top_fraction)
    scenario = context["scenario"]
    state = context["baseline_state"]
    prediction = context["baseline_prediction"]
    edge_array = _edge_array(scenario)
    ybus_loading = np.asarray(state["loading_ratio"], dtype=float)
    matpower_pred_loading = compute_matpower_ac_loading_by_line(
        scenario,
        prediction["Vm"],
        prediction["Va"],
        base_mva=float(scenario.sn_mva),
    )
    matpower_base_loading = compute_matpower_ac_loading_by_line(
        scenario,
        scenario.Vm_base,
        scenario.Va_base,
        base_mva=float(scenario.sn_mva),
    )
    physical_ids = np.asarray(scenario.physical_branch_id, dtype=int)
    rows = []
    for line_id in sorted(
        physical_line_ids(scenario),
        key=lambda item: (-float(ybus_loading[int(item)]), int(item)),
    )[: int(top_n)]:
        line_id = int(line_id)
        directed_ids = scenario.physical_branch_directed_line_ids.get(int(physical_ids[line_id]), [line_id])
        rows.append(
            {
                "canonical_line_id": int(line_id),
                "directed_line_ids": ",".join(str(int(item)) for item in directed_ids),
                "from_bus": int(edge_array[0, line_id]),
                "to_bus": int(edge_array[1, line_id]),
                "rate_a_mva": float(scenario.rate_a[line_id]),
                "stage_g_ybus_pred_loading": float(ybus_loading[line_id]),
                "matpower_ac_pred_loading": float(matpower_pred_loading[line_id]),
                "matpower_ac_base_loading": float(matpower_base_loading[line_id]),
                "abs_delta_pred": float(abs(ybus_loading[line_id] - matpower_pred_loading[line_id])),
                "ratio_ybus_to_matpower_pred": float(ybus_loading[line_id] / max(matpower_pred_loading[line_id], 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def run_audit(
    model_type: str = "gnn",
    grouping_top_fraction: float = 0.30,
    output_dir: Path | None = None,
) -> Path:
    frame = build_loading_ranking_frame(model_type=model_type, grouping_top_fraction=grouping_top_fraction)
    if output_dir is None:
        output_dir = make_run_dir(RESULTS_ROOT / "leq" / "stage_g" / "loading_ranking_audit", "run")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stage_g_loading_ranking_audit.csv"
    write_dataframe(output_path, frame)
    print(frame.to_csv(index=False))
    print(f"CSV written to: {output_path}")
    return output_path


def run_matpower_comparison(
    model_type: str = "gnn",
    grouping_top_fraction: float = 0.30,
    top_n: int = 10,
    output_dir: Path | None = None,
) -> Path:
    frame = build_matpower_comparison_frame(
        model_type=model_type,
        grouping_top_fraction=grouping_top_fraction,
        top_n=top_n,
    )
    if output_dir is None:
        output_dir = make_run_dir(RESULTS_ROOT / "leq" / "stage_g" / "matpower_flow_comparison", "run")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stage_g_matpower_flow_comparison.csv"
    write_dataframe(output_path, frame)
    print(frame.to_csv(index=False))
    print(f"CSV written to: {output_path}")
    return output_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage G corrected loading/risk ranking audit.")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="gnn")
    parser.add_argument("--grouping-top-fraction", type=float, default=0.30)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--matpower-comparison", action="store_true")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.matpower_comparison:
        run_matpower_comparison(
            model_type=args.model,
            grouping_top_fraction=float(args.grouping_top_fraction),
            top_n=int(args.top_n),
            output_dir=args.output_dir,
        )
        return 0
    run_audit(
        model_type=args.model,
        grouping_top_fraction=float(args.grouping_top_fraction),
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
