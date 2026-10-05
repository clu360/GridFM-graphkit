"""Shared K0/K1 then evaluator-specific K2 Stage K trajectory."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import load_config
from .identity import build_identity
from .io_utils import atomic_write_json, checkpoint_matches
from .native_opf import NativeOPFEvaluator
from .gridsfm_evaluator import ReleasedGridSFMEvaluator
from .gridsfm_environment import (
    capture_gridsfm_environment,
    contract_checks,
    load_gridsfm_contract,
)
from .schemas import CandidateResult, ResultStatus
from .topology_candidates import k2_children, topology_key


def _parse_ids(value: str) -> tuple[int, ...]:
    if not value or value == "intact":
        return ()
    return tuple(sorted(int(item) for item in value.split(";") if item))


def select_finalist(rows: list[CandidateResult]) -> CandidateResult:
    eligible = [row for row in rows if row.eligible and row.search_objective is not None]
    if not eligible:
        raise RuntimeError("evaluator produced no eligible finalist")
    return min(eligible, key=lambda row: (float(row.search_objective), row.k, row.topology_key))


def _candidate_with_checkpoint(
    *, callback: Callable[[], CandidateResult], checkpoint_dir: Path,
    evaluator: str, lambda_r: float, offline: tuple[int, ...],
    config_sha256: str, input_sha256: str,
) -> CandidateResult:
    key = topology_key(offline)
    expected = {
        "run_id": os.environ.get("STAGE_K_RUN_ID", "gate2_local"),
        "config_sha256": config_sha256,
        "input_sha256": input_sha256,
        "evaluator": evaluator,
        "lambda_r": float(lambda_r),
        "k": len(offline),
        "topology_key": key,
    }
    path = checkpoint_dir / evaluator / f"lambda_{lambda_r:.6g}" / f"{key.replace(';', '_')}.json"
    if path.exists():
        if not checkpoint_matches(path, expected):
            raise RuntimeError(f"mismatched or incomplete candidate checkpoint: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("alpha_service_identity") != payload.get("result", {}).get("alpha_effective_json", ""):
            raise RuntimeError(f"candidate checkpoint service identity mismatch: {path}")
        return CandidateResult(**payload["result"])
    result = callback()
    payload = {
        **expected,
        "status": "complete",
        "alpha_service_identity": result.alpha_effective_json,
        "result": {**result.__dict__},
    }
    atomic_write_json(path, payload)
    return result


def run_native(
    *, evaluator_name: str, config_path: str | Path, prepared_dir: str | Path,
    output_dir: str | Path, julia: str = "julia", lambda_values: list[float] | None = None,
) -> dict[str, object]:
    config = load_config(config_path)
    identity = build_identity()
    prepared = Path(prepared_dir)
    proxy = pd.read_parquet(prepared / "proxy_components.parquet")
    k1_table = pd.read_parquet(prepared / "shared_k1.parquet")
    p_env = dict(zip(proxy["canonical_branch_id"].astype(int), proxy["p_env"].astype(float), strict=True))
    weights = dict(zip(proxy["canonical_branch_id"].astype(int), proxy["weight"].astype(float), strict=True))
    c_by_line = dict(zip(proxy["canonical_branch_id"].astype(int), proxy["c_l"].astype(float), strict=True))
    manifest = json.loads((prepared / "input_manifest.json").read_text(encoding="utf-8"))
    input_sha = sha256(json.dumps(manifest["input_sha256"], sort_keys=True).encode("utf-8")).hexdigest()
    output = Path(output_dir)
    if (output.parent / "RUN_COMPLETE.json").exists():
        raise FileExistsError(f"completed Stage K run is immutable: {output.parent}")
    all_rows: list[CandidateResult] = []
    k2_manifest_rows = []

    selected_lambdas = lambda_values or [float(value) for value in config["search"]["lambda_r"]]
    if any(value not in [float(item) for item in config["search"]["lambda_r"]] for value in selected_lambdas):
        raise ValueError("requested lambda is not present in the frozen config")
    for lambda_r in selected_lambdas:
        evaluator = NativeOPFEvaluator(
            mode=f"native_{evaluator_name}", identity=identity,
            case_path=Path(__file__).resolve().parents[1] / config["paths"]["case_file"],
            p_env=p_env, r_base=float(manifest["r_base"]), lambda_r=float(lambda_r),
            output_root=output, julia=julia,
            timeout_seconds=int(config["solvers"]["time_limit_seconds"]),
        )
        lambda_rows = [_candidate_with_checkpoint(
            callback=lambda: evaluator.evaluate(()), checkpoint_dir=output / "checkpoints",
            evaluator=evaluator_name, lambda_r=float(lambda_r), offline=(),
            config_sha256=config["config_sha256"], input_sha256=input_sha,
        )]
        k1_for_lambda = k1_table.loc[k1_table["lambda_r"] == float(lambda_r)]
        for row in k1_for_lambda.itertuples(index=False):
            ids = _parse_ids(row.offline_branch_ids)
            lambda_rows.append(_candidate_with_checkpoint(
                callback=lambda ids=ids: evaluator.evaluate(ids), checkpoint_dir=output / "checkpoints",
                evaluator=evaluator_name, lambda_r=float(lambda_r), offline=ids,
                config_sha256=config["config_sha256"], input_sha256=input_sha,
            ))
        parents = sorted(
            [row for row in lambda_rows if row.k == 1 and row.eligible],
            key=lambda row: (float(row.search_objective), row.topology_key),
        )[: int(config["search"]["parent_count"])]
        parent_pairs = [(rank, _parse_ids(row.offline_branch_ids)[0]) for rank, row in enumerate(parents, 1)]
        children = k2_children(
            parents=parent_pairs, line_ids=identity.l_trans, weights=weights, c_by_line=c_by_line,
            lambda_r=float(lambda_r), children_per_parent=int(config["search"]["k2_children_per_parent"]),
            max_unique=int(config["search"]["k2_max_unique"]),
        )
        for child in children:
            result = _candidate_with_checkpoint(
                callback=lambda child=child: evaluator.evaluate(child.offline_branch_ids),
                checkpoint_dir=output / "checkpoints", evaluator=evaluator_name,
                lambda_r=float(lambda_r), offline=child.offline_branch_ids,
                config_sha256=config["config_sha256"], input_sha256=input_sha,
            )
            lambda_rows.append(result)
            k2_manifest_rows.append({"evaluator": evaluator_name, "lambda_r": lambda_r, **child.as_dict()})
        all_rows.extend(lambda_rows)

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row.as_dict() for row in all_rows]).to_parquet(output / f"{evaluator_name}_candidate_results.parquet", index=False)
    pd.DataFrame(k2_manifest_rows).to_parquet(output / f"{evaluator_name}_k2_candidates.parquet", index=False)
    finalists = [select_finalist([row for row in all_rows if row.lambda_r == float(value)]) for value in selected_lambdas]
    pd.DataFrame([row.as_dict() for row in finalists]).to_parquet(output / f"{evaluator_name}_finalists.parquet", index=False)
    summary = {
        "status": "PASS",
        "evaluator": evaluator_name,
        "attempted_states": len(all_rows),
        "eligible_states": sum(row.eligible for row in all_rows),
        "finalist_count": len(finalists),
        "config_sha256": config["config_sha256"],
        "input_sha256": input_sha,
        "lambda_values": selected_lambdas,
    }
    atomic_write_json(output / f"{evaluator_name}_run_summary.json", summary)
    return summary


def run_gridsfm(
    *, config_path: str | Path, prepared_dir: str | Path, output_dir: str | Path,
    gridsfm_root: str | Path, checkpoint: str | Path, environment_manifest: str | Path,
    device: str = "cuda:0", environment_contract: str | Path | None = None,
    lambda_values: list[float] | None = None,
) -> dict[str, object]:
    contract = load_gridsfm_contract(environment_contract) if environment_contract else load_gridsfm_contract()
    frozen_environment = json.loads(Path(environment_manifest).read_text(encoding="utf-8"))
    environment_checks = contract_checks(frozen_environment, contract)
    if frozen_environment.get("status") != "PASS" or not all(environment_checks.values()):
        raise RuntimeError("GridSFM environment manifest is absent, failed, or differs from the frozen contract")
    _, current_checks = capture_gridsfm_environment(
        gridsfm_root, checkpoint, device=device,
        **({"contract_path": environment_contract} if environment_contract else {}),
    )
    if not all(current_checks.values()):
        failed = sorted(key for key, passed in current_checks.items() if not passed)
        raise RuntimeError(f"GridSFM job environment differs from the frozen contract: {failed}")
    config = load_config(config_path)
    identity = build_identity()
    prepared = Path(prepared_dir)
    proxy = pd.read_parquet(prepared / "proxy_components.parquet")
    k1_table = pd.read_parquet(prepared / "shared_k1.parquet")
    p_env = dict(zip(proxy["canonical_branch_id"].astype(int), proxy["p_env"].astype(float), strict=True))
    weights = dict(zip(proxy["canonical_branch_id"].astype(int), proxy["weight"].astype(float), strict=True))
    c_by_line = dict(zip(proxy["canonical_branch_id"].astype(int), proxy["c_l"].astype(float), strict=True))
    manifest = json.loads((prepared / "input_manifest.json").read_text(encoding="utf-8"))
    input_sha = sha256(json.dumps(manifest["input_sha256"], sort_keys=True).encode("utf-8")).hexdigest()
    output = Path(output_dir)
    if (output.parent / "RUN_COMPLETE.json").exists():
        raise FileExistsError(f"completed Stage K run is immutable: {output.parent}")
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[CandidateResult] = []
    alpha_rows = []
    k2_manifest_rows = []
    resident = None

    def evaluate_payload(evaluator, offline, lambda_r):
        key = topology_key(offline)
        try:
            payload = evaluator.evaluate(offline)
            result = payload["result"]
            if result is None or result.objective is None:
                return CandidateResult(
                    evaluator="gridsfm", lambda_r=lambda_r, k=len(offline), topology_key=key,
                    offline_branch_ids=";".join(map(str, offline)), status=ResultStatus.EVALUATOR_EXCEPTION.value,
                    elapsed_seconds=payload["elapsed_seconds"], message="no finite GridSFM alpha candidate",
                )
            objective = result.objective
            status = str(result.evaluation_status.value)
            state_dir = output / "gridsfm_states" / f"lambda_{lambda_r:.6g}" / key.replace(";", "_") / "best"
            state_dir.mkdir(parents=True, exist_ok=True)
            branch_ids = sorted(result.flow_loading_by_line)
            pd.DataFrame({
                "canonical_branch_id": branch_ids,
                "p_from": [result.p_from_by_line[i] for i in branch_ids],
                "q_from": [result.q_from_by_line[i] for i in branch_ids],
                "p_to": [result.p_to_by_line[i] for i in branch_ids],
                "q_to": [result.q_to_by_line[i] for i in branch_ids],
                "physical_loading": [result.flow_loading_by_line[i] for i in branch_ids],
            }).to_parquet(state_dir / "branch_state.parquet", index=False)
            bus_ids = sorted(result.v_by_bus)
            pd.DataFrame({
                "bus_id": bus_ids,
                "vm": [result.v_by_bus[i] for i in bus_ids],
                "va": [result.theta_by_bus[i] for i in bus_ids],
            }).to_parquet(state_dir / "bus_state.parquet", index=False)
            online_gen_ids = identity.generators.loc[
                identity.generators["GEN_STATUS"] > 0, "canonical_generator_id"
            ].astype(int).tolist()
            pd.DataFrame({
                "canonical_generator_id": online_gen_ids,
                "pg": [result.pg_by_generator[i] for i in range(len(online_gen_ids))],
                "qg": [result.qg_by_generator.get(i, np.nan) for i in range(len(online_gen_ids))],
            }).to_parquet(state_dir / "gen_state.parquet", index=False)
            pd.DataFrame({
                "canonical_load_id": list(range(len(payload["best_alpha"]))),
                "alpha_effective": [
                    result.load_shedding.alpha_effective[i]
                    if result.load_shedding is not None else payload["best_alpha"][i]
                    for i in range(len(payload["best_alpha"]))
                ],
            }).to_parquet(state_dir / "load_service.parquet", index=False)
            return CandidateResult(
                evaluator="gridsfm", lambda_r=lambda_r, k=len(offline), topology_key=key,
                offline_branch_ids=";".join(map(str, offline)), status=status, eligible=True,
                search_objective=objective.j_total, r_norm=objective.r_norm,
                l_shed_total=objective.l_shed_total, j_trade=objective.j_trade,
                pac_operational=objective.pac_operational, pac_ac=objective.pac_ac,
                pac_model=objective.pac_model, pac_total=objective.pac_total,
                j_total=objective.j_total,
                max_loading=max(result.flow_loading_by_line.values()) if result.flow_loading_by_line else None,
                loading_gt_1_count=sum(value > 1.0 for value in result.flow_loading_by_line.values()),
                selected_load_ids=";".join(map(str, payload["selected_load_ids"])),
                alpha_effective_json=json.dumps(
                    {
                        str(k): v
                        for k, v in (
                            result.load_shedding.alpha_effective.items()
                            if result.load_shedding is not None
                            else payload["best_alpha"].items()
                        )
                    },
                    sort_keys=True,
                ),
                elapsed_seconds=payload["elapsed_seconds"], message=result.message,
                state_path=str(state_dir),
                metadata={
                    "feasibility_head": result.feasibility_head,
                    "d_input": result.d_input,
                    "alpha_trace": list(payload["trace"]),
                },
            )
        except Exception as exc:
            return CandidateResult(
                evaluator="gridsfm", lambda_r=lambda_r, k=len(offline), topology_key=key,
                offline_branch_ids=";".join(map(str, offline)), status=ResultStatus.EVALUATOR_EXCEPTION.value,
                message=str(exc),
            )

    selected_lambdas = lambda_values or [float(value) for value in config["search"]["lambda_r"]]
    if any(value not in [float(item) for item in config["search"]["lambda_r"]] for value in selected_lambdas):
        raise ValueError("requested lambda is not present in the frozen config")
    for lambda_r in selected_lambdas:
        if resident is None:
            resident = ReleasedGridSFMEvaluator(
                identity=identity, gridsfm_root=gridsfm_root, checkpoint=checkpoint, device=device,
                q=int(config["search"]["gridsfm_q"]),
                alpha_budget=int(config["search"]["gridsfm_alpha_budget"]),
                p_env=p_env, r_base=float(manifest["r_base"]), lambda_r=float(lambda_r),
                work_dir=output / "gridsfm_states",
            )
        else:
            resident.lambda_r = float(lambda_r)
        lambda_rows = [_candidate_with_checkpoint(
            callback=lambda: evaluate_payload(resident, (), float(lambda_r)),
            checkpoint_dir=output / "checkpoints", evaluator="gridsfm",
            lambda_r=float(lambda_r), offline=(), config_sha256=config["config_sha256"],
            input_sha256=input_sha,
        )]
        for row in k1_table.loc[k1_table["lambda_r"] == float(lambda_r)].itertuples(index=False):
            ids = _parse_ids(row.offline_branch_ids)
            lambda_rows.append(_candidate_with_checkpoint(
                callback=lambda ids=ids: evaluate_payload(resident, ids, float(lambda_r)),
                checkpoint_dir=output / "checkpoints", evaluator="gridsfm",
                lambda_r=float(lambda_r), offline=ids, config_sha256=config["config_sha256"],
                input_sha256=input_sha,
            ))
        parents = sorted(
            [row for row in lambda_rows if row.k == 1 and row.eligible],
            key=lambda row: (float(row.search_objective), row.topology_key),
        )[: int(config["search"]["parent_count"])]
        parent_pairs = [(rank, _parse_ids(row.offline_branch_ids)[0]) for rank, row in enumerate(parents, 1)]
        children = k2_children(
            parents=parent_pairs, line_ids=identity.l_trans, weights=weights, c_by_line=c_by_line,
            lambda_r=float(lambda_r), children_per_parent=int(config["search"]["k2_children_per_parent"]),
            max_unique=int(config["search"]["k2_max_unique"]),
        )
        for child in children:
            lambda_rows.append(_candidate_with_checkpoint(
                callback=lambda child=child: evaluate_payload(resident, child.offline_branch_ids, float(lambda_r)),
                checkpoint_dir=output / "checkpoints", evaluator="gridsfm",
                lambda_r=float(lambda_r), offline=child.offline_branch_ids,
                config_sha256=config["config_sha256"], input_sha256=input_sha,
            ))
            k2_manifest_rows.append({"evaluator": "gridsfm", "lambda_r": lambda_r, **child.as_dict()})
        all_rows.extend(lambda_rows)

    for row in all_rows:
        for trace in row.metadata.get("alpha_trace", []):
            alpha_rows.append({
                "evaluator": "gridsfm", "lambda_r": row.lambda_r,
                "topology_key": row.topology_key, **trace,
            })
    pd.DataFrame([row.as_dict() for row in all_rows]).to_parquet(output / "gridsfm_candidate_results.parquet", index=False)
    pd.DataFrame(alpha_rows).to_parquet(output / "gridsfm_alpha_evaluations.parquet", index=False)
    pd.DataFrame(k2_manifest_rows).to_parquet(output / "gridsfm_k2_candidates.parquet", index=False)
    finalists = [select_finalist([row for row in all_rows if row.lambda_r == float(value)]) for value in selected_lambdas]
    pd.DataFrame([row.as_dict() for row in finalists]).to_parquet(output / "gridsfm_finalists.parquet", index=False)
    summary = {
        "status": "PASS", "evaluator": "gridsfm", "attempted_states": len(all_rows),
        "eligible_states": sum(row.eligible for row in all_rows), "finalist_count": len(finalists),
        "config_sha256": config["config_sha256"], "input_sha256": input_sha,
        "lambda_values": selected_lambdas, "checkpoint_resident_for_job": True,
    }
    atomic_write_json(output / "gridsfm_run_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", choices=["gridsfm", "dc", "ac"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--gridsfm-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--environment-manifest")
    parser.add_argument("--environment-contract")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lambda-r", type=float, action="append")
    args = parser.parse_args()
    if args.evaluator == "gridsfm":
        if not args.gridsfm_root or not args.checkpoint or not args.environment_manifest:
            parser.error("GridSFM requires --gridsfm-root, --checkpoint, and --environment-manifest")
        summary = run_gridsfm(
            config_path=args.config, prepared_dir=args.prepared_dir, output_dir=args.output_dir,
            gridsfm_root=args.gridsfm_root, checkpoint=args.checkpoint,
            environment_manifest=args.environment_manifest, device=args.device,
            environment_contract=args.environment_contract,
            lambda_values=args.lambda_r,
        )
    else:
        summary = run_native(
            evaluator_name=args.evaluator, config_path=args.config, prepared_dir=args.prepared_dir,
            output_dir=args.output_dir, julia=args.julia,
            lambda_values=args.lambda_r,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
