"""Data contracts for Stage J diagnostics and AC references."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import EvaluationStatus


@dataclass(frozen=True)
class InputIntegrityReport:
    """Hard GridSFM interface invariant report.

    A failed input-integrity report is a methodology failure, not a soft
    candidate penalty.
    """

    checks: dict[str, bool]
    notes: str = ""

    @property
    def d_input(self) -> float:
        if not self.checks:
            return 0.0
        failed = sum(1 for ok in self.checks.values() if not ok)
        return failed / len(self.checks)

    @property
    def evaluation_status(self) -> EvaluationStatus:
        return EvaluationStatus.OK if self.d_input == 0.0 else EvaluationStatus.INPUT_INTEGRITY_FAILURE

    def require_ok(self) -> None:
        if self.evaluation_status is not EvaluationStatus.OK:
            failed = [name for name, ok in self.checks.items() if not ok]
            raise ValueError(f"Stage J input-integrity failure: {failed}. {self.notes}")


@dataclass(frozen=True)
class ACReferenceResult:
    """Exact AC audit result contract for Stage J finalists."""

    ac_reference_type: str
    evaluation_status: EvaluationStatus
    objective_value: float | None = None
    d_state_to_ac: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def infeasible(ac_reference_type: str, message: str) -> "ACReferenceResult":
        return ACReferenceResult(
            ac_reference_type=ac_reference_type,
            evaluation_status=EvaluationStatus.AC_REFERENCE_INFEASIBLE,
            objective_value=None,
            d_state_to_ac=None,
            warnings=(message,),
        )
