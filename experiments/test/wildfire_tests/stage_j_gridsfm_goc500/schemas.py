"""Shared schemas and status labels for Stage J.

The classes here intentionally avoid importing GridSFM or solver packages.
They capture methodology contracts that can be tested before the external
GridSFM environment is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class EvaluationStatus(str, Enum):
    """Locked Stage J evaluation status labels."""

    OK = "ok"
    DC_INFEASIBLE = "dc_infeasible"
    GRIDSFM_PREPROCESSING_FAILURE = "gridsfm_preprocessing_failure"
    GRIDSFM_INFERENCE_FAILURE = "gridsfm_inference_failure"
    MODEL_OUTPUT_PENALIZED = "model_output_penalized"
    INPUT_INTEGRITY_FAILURE = "input_integrity_failure"
    AC_REFERENCE_INFEASIBLE = "ac_reference_infeasible"
    METHODOLOGY_FAILURE = "methodology_failure"


@dataclass(frozen=True)
class PacWeights:
    """Frozen weights for the GridSFM physics-aware merit term."""

    rho_phys: float
    w_op: float
    w_ac: float
    w_model: float

    def validate(self) -> None:
        values = {
            "rho_phys": self.rho_phys,
            "w_op": self.w_op,
            "w_ac": self.w_ac,
            "w_model": self.w_model,
        }
        for name, value in values.items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative, got {value}")


@dataclass(frozen=True)
class StageJObjective:
    """Objective components saved for each Stage J candidate."""

    r_norm: float
    l_shed_total: float
    j_trade: float
    pac_operational: float | None = None
    pac_ac: float | None = None
    pac_model: float | None = None
    pac_total: float | None = None
    j_total: float | None = None


@dataclass(frozen=True)
class UnitCompatibilityReport:
    """Hard gate for comparing predicted branch flows to thermal ratings."""

    base_mva: float
    flow_units: str
    rating_units: str
    compatible: bool
    notes: str = ""

    def require_compatible(self) -> None:
        if self.base_mva <= 0:
            raise ValueError(f"base_mva must be positive, got {self.base_mva}")
        if not self.compatible:
            raise ValueError(f"unit compatibility failed: {self.notes}")
