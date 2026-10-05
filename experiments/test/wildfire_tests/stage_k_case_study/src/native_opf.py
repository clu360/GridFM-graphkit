"""Python launcher and result parser for Stage K PowerModels evaluators."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import time
from typing import Iterable, Mapping

import pandas as pd

from .identity import StageKIdentity, source_less_load_ids
from .objectives import compute_j_trade, compute_l_shed, compute_r_norm
from .schemas import CandidateResult, ELIGIBLE_STATUSES, ResultStatus, classify_solver_status
from .topology_candidates import topology_key


class NativeOPFEvaluator:
    def __init__(
        self,
        *,
        mode: str,
        identity: StageKIdentity,
        case_path: str | Path,
        p_env: Mapping[int, float],
        r_base: float,
        lambda_r: float,
        output_root: str | Path,
        julia: str = "julia",
        project: str | Path | None = None,
        timeout_seconds: int = 3600,
    ) -> None:
        if mode not in {"native_dc", "native_ac"}:
            raise ValueError("mode must be native_dc or native_ac")
        self.mode = mode
        self.evaluator_name = "dc" if mode == "native_dc" else "ac"
        self.identity = identity
        self.case_path = Path(case_path).resolve()
        self.p_env = {int(k): float(v) for k, v in p_env.items()}
        self.r_base = float(r_base)
        self.lambda_r = float(lambda_r)
        self.output_root = Path(output_root)
        self.julia = julia
        self.project = Path(project) if project else Path(__file__).resolve().parent / "native_opf"
        self.script = self.project / "stage_k_native_opf.jl"
        self.timeout_seconds = int(timeout_seconds)
        self.branch_rate = dict(
            zip(identity.branches["canonical_branch_id"].astype(int), identity.branches["rate_a_mva"].astype(float), strict=True)
        )
        self.pd = dict(
            zip(identity.loads["canonical_load_id"].astype(int), identity.loads["pd_requested_mw"].astype(float), strict=True)
        )

    def _write_p_env(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["powermodels_branch_id", "p_env"])
            for branch_id in self.identity.l_trans:
                writer.writerow([branch_id + 1, self.p_env[branch_id]])

    def evaluate(self, offline_branch_ids: Iterable[int]) -> CandidateResult:
        offline = tuple(sorted(int(value) for value in offline_branch_ids))
        key = topology_key(offline)
        out = self.output_root / self.evaluator_name / f"lambda_{self.lambda_r:.6g}" / key.replace(";", "_")
        out.mkdir(parents=True, exist_ok=True)
        p_env_csv = out / "p_env_powermodels.csv"
        self._write_p_env(p_env_csv)
        islanded = source_less_load_ids(self.identity, offline)
        command = [
            self.julia,
            f"--project={self.project}",
            str(self.script),
            self.mode,
            str(self.case_path),
            ";".join(str(value + 1) for value in offline),
            ";".join(str(value + 1) for value in islanded),
            str(p_env_csv),
            "-",
            str(self.lambda_r),
            str(self.r_base),
            str(out),
        ]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            elapsed = time.perf_counter() - started
        except subprocess.TimeoutExpired as exc:
            return CandidateResult(
                evaluator=self.evaluator_name, lambda_r=self.lambda_r, k=len(offline), topology_key=key,
                offline_branch_ids=";".join(map(str, offline)), status=ResultStatus.TIME_LIMIT.value,
                elapsed_seconds=time.perf_counter() - started, message=str(exc),
            )
        except Exception as exc:
            return CandidateResult(
                evaluator=self.evaluator_name, lambda_r=self.lambda_r, k=len(offline), topology_key=key,
                offline_branch_ids=";".join(map(str, offline)), status=ResultStatus.EVALUATOR_EXCEPTION.value,
                elapsed_seconds=time.perf_counter() - started, message=str(exc),
            )
        (out / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (out / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        summary_path = out / f"{self.mode}_summary.json"
        if not summary_path.exists():
            return CandidateResult(
                evaluator=self.evaluator_name, lambda_r=self.lambda_r, k=len(offline), topology_key=key,
                offline_branch_ids=";".join(map(str, offline)), status=ResultStatus.EVALUATOR_EXCEPTION.value,
                elapsed_seconds=elapsed, message=f"Julia exit {completed.returncode}; missing summary",
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        status = classify_solver_status(str(summary.get("termination_status", "")))
        if status not in ELIGIBLE_STATUSES:
            return CandidateResult(
                evaluator=self.evaluator_name, lambda_r=self.lambda_r, k=len(offline), topology_key=key,
                offline_branch_ids=";".join(map(str, offline)), status=status, elapsed_seconds=elapsed,
                solver_seconds=summary.get("solver_time_seconds"), iterations=summary.get("iteration_count"),
                message=str(summary.get("termination_status", "")),
            )
        branch_state = pd.read_csv(summary["branch_state_csv"])
        if self.mode == "native_ac":
            expected = len(self.identity.l_trans) - len(offline)
            actual = int(summary.get("risk_epigraph_count", -1))
            declared = int(summary.get("risk_line_count_expected", -2))
            risk_rows = branch_state.loc[
                branch_state["powermodels_branch_id"].astype(int).sub(1).isin(self.identity.l_trans)
            ]
            epigraph_valid = (
                risk_rows["risk_epigraph"].notna().all()
                and bool((risk_rows["risk_epigraph"] + 1e-8 >= risk_rows["physical_loading"] ** 2).all())
            )
            if actual != expected or declared != expected or len(risk_rows) != expected or not epigraph_valid:
                return CandidateResult(
                    evaluator=self.evaluator_name, lambda_r=self.lambda_r, k=len(offline), topology_key=key,
                    offline_branch_ids=";".join(map(str, offline)),
                    status=ResultStatus.INPUT_MAPPING_FAILURE.value, elapsed_seconds=elapsed,
                    message=(
                        f"AC epigraph coverage failure: actual={actual}, declared={declared}, "
                        f"physical_rows={len(risk_rows)}, expected={expected}, bounds_valid={epigraph_valid}"
                    ), state_path=str(out),
                )
        loading = {}
        for row in branch_state.itertuples(index=False):
            canonical = int(row.powermodels_branch_id) - 1
            if canonical not in self.identity.l_trans:
                continue
            loading[canonical] = float(row.physical_loading)
        r_norm = compute_r_norm(self.identity.l_trans, offline, self.p_env, loading, self.r_base)
        load_state = pd.read_csv(summary["load_service_csv"])
        alpha = {int(row.load_id) - 1: float(row.service_fraction) for row in load_state.itertuples(index=False)}
        l_shed = compute_l_shed(self.pd, alpha)
        j_trade = compute_j_trade(self.lambda_r, r_norm, l_shed)
        return CandidateResult(
            evaluator=self.evaluator_name,
            lambda_r=self.lambda_r,
            k=len(offline),
            topology_key=key,
            offline_branch_ids=";".join(map(str, offline)),
            status=status,
            eligible=True,
            search_objective=j_trade,
            r_norm=r_norm,
            l_shed_total=l_shed,
            j_trade=j_trade,
            max_loading=max(loading.values()) if loading else None,
            loading_gt_1_count=sum(value > 1.0 for value in loading.values()),
            alpha_effective_json=json.dumps({str(k): v for k, v in alpha.items()}, sort_keys=True),
            elapsed_seconds=elapsed,
            solver_seconds=summary.get("solver_time_seconds"),
            iterations=summary.get("iteration_count"),
            message=str(summary.get("termination_status", "")),
            state_path=str(out),
            metadata={
                "economic_cost_diagnostic": summary.get("economic_cost_diagnostic"),
                "risk_epigraph_count": summary.get("risk_epigraph_count"),
                "risk_line_count_expected": summary.get("risk_line_count_expected"),
            },
        )
