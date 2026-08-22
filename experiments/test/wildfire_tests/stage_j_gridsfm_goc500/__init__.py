"""Stage J GridSFM GOC-500 comparison scaffolding."""

from .schemas import (
    EvaluationStatus,
    PacWeights,
    StageJObjective,
    UnitCompatibilityReport,
)
from .contracts import ACReferenceResult, InputIntegrityReport
from .load_service import LoadSheddingBreakdown, compute_alpha_effective, compute_load_shedding
from .metrics import (
    compute_ac_loading_two_ended,
    compute_j_total_sfm,
    compute_j_trade,
    compute_pac_total,
    compute_r_base,
)
from .goc500_adapter import (
    BranchIdentity,
    GOC500Identity,
    LoadIdentity,
    assert_raw_gridsfm_case,
    build_goc500_identity,
    mutate_raw_case_for_candidate,
    require_valid_risk_ratings,
    source_less_load_ids,
)
from .gridsfm_evaluator import GridSFMCandidateResult, evaluate_gridsfm_candidate
from .outer_proxy import ProxyTopology, solve_proxy_topology_pool
from .alpha_optimizer import (
    AlphaEvaluation,
    AlphaSearchConfig,
    AlphaSearchResult,
    ScreenedScipyAlphaConfig,
    ScreenedScipyAlphaResult,
    alpha_hash,
    optimize_full_alpha,
    optimize_screened_scipy_alpha,
    topology_hash,
)

__all__ = [
    "AlphaEvaluation",
    "AlphaSearchConfig",
    "AlphaSearchResult",
    "ScreenedScipyAlphaConfig",
    "ScreenedScipyAlphaResult",
    "EvaluationStatus",
    "ACReferenceResult",
    "InputIntegrityReport",
    "LoadSheddingBreakdown",
    "PacWeights",
    "StageJObjective",
    "UnitCompatibilityReport",
    "BranchIdentity",
    "GOC500Identity",
    "LoadIdentity",
    "GridSFMCandidateResult",
    "ProxyTopology",
    "alpha_hash",
    "assert_raw_gridsfm_case",
    "build_goc500_identity",
    "compute_ac_loading_two_ended",
    "compute_alpha_effective",
    "compute_j_total_sfm",
    "compute_j_trade",
    "compute_load_shedding",
    "compute_pac_total",
    "compute_r_base",
    "evaluate_gridsfm_candidate",
    "optimize_full_alpha",
    "optimize_screened_scipy_alpha",
    "solve_proxy_topology_pool",
    "topology_hash",
    "mutate_raw_case_for_candidate",
    "require_valid_risk_ratings",
    "source_less_load_ids",
]
