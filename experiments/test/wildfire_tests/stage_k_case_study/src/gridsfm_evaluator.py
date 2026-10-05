"""Released GridSFM Stage K evaluator with one resident checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Iterable

from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.goc500_adapter import build_goc500_identity
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.gridsfm_evaluator import evaluate_gridsfm_candidate
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.schemas import PacWeights

from .alpha_search import budgeted_powell_search
from .gridsfm_adapter import build_raw_gridsfm_case
from .identity import StageKIdentity, source_less_load_ids
from .service import select_topology_relative_loads


class ReleasedGridSFMEvaluator:
    def __init__(
        self,
        *,
        identity: StageKIdentity,
        gridsfm_root: str | Path,
        checkpoint: str | Path,
        device: str,
        q: int,
        alpha_budget: int,
        p_env: dict[int, float],
        r_base: float,
        lambda_r: float,
        work_dir: str | Path,
    ) -> None:
        root = Path(gridsfm_root).resolve()
        if not (root / "model" / "pyproject.toml").is_file():
            raise FileNotFoundError(f"official GridSFM model package not found under {root}")
        from gridsfm import load_model

        self.identity = identity
        self.raw_case = build_raw_gridsfm_case(identity)
        self.raw_identity = build_goc500_identity(self.raw_case, candidate_branch_ids=identity.l_trans)
        self.model = load_model(str(Path(checkpoint).resolve()), device=device)
        self.device = device
        self.q = int(q)
        self.alpha_budget = int(alpha_budget)
        self.p_env = p_env
        self.r_base = float(r_base)
        self.lambda_r = float(lambda_r)
        self.work_dir = Path(work_dir)
        self.weights = PacWeights(rho_phys=2.0, w_op=1.0, w_ac=1.0, w_model=0.0)

    def evaluate(self, offline_branch_ids: Iterable[int]):
        offline = tuple(sorted(int(value) for value in offline_branch_ids))
        islanded = source_less_load_ids(self.identity, offline)
        selected = select_topology_relative_loads(
            self.identity, offline, q=self.q, exclude_load_ids=islanded
        )
        load_ids = tuple(self.identity.loads["canonical_load_id"].astype(int))
        counter = {"value": 0}

        def callback(alpha):
            counter["value"] += 1
            return evaluate_gridsfm_candidate(
                raw_case=self.raw_case,
                identity=self.raw_identity,
                model=self.model,
                offline_branch_ids=offline,
                alpha_requested=alpha,
                p_env_by_line=self.p_env,
                r_base=self.r_base,
                lambda_r=self.lambda_r,
                weights=self.weights,
                work_dir=self.work_dir / ("intact" if not offline else "_".join(map(str, offline))) / f"alpha_{counter['value']:03d}",
            )

        started = time.perf_counter()
        search = budgeted_powell_search(
            load_ids=load_ids,
            selected_load_ids=selected,
            budget=self.alpha_budget,
            evaluator=callback,
            objective_getter=lambda result: None if result.objective is None else result.objective.j_total,
        )
        elapsed = time.perf_counter() - started
        result = search.best_result
        payload = {
            "selected_load_ids": selected,
            "source_less_load_ids": islanded,
            "best_alpha": search.best_alpha,
            "trace": search.trace,
            "elapsed_seconds": elapsed,
            "result": result,
        }
        return payload
