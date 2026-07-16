from __future__ import annotations


CANONICAL_LAMBDA_CASES = {
    "risk_leaning": (0.9, 0.1),
    "balanced": (0.5, 0.5),
    "service_leaning": (0.1, 0.9),
}

LEGACY_NEAR_EXTREME_LAMBDA_CASES = {
    "risk": (0.999001, 0.000999),
    "balanced": (0.5, 0.5),
    "shed": (0.000999, 0.999001),
}

LEGACY_STAGE_D_LAMBDA_CASES = {
    "risk_leaning": (0.8, 0.2),
    "balanced": (0.5, 0.5),
    "service_leaning": (0.2, 0.8),
}

LAMBDA_FOLDERS = {
    "risk_leaning": "risk",
    "balanced": "bal",
    "service_leaning": "svc",
    "risk": "risk",
    "shed": "svc",
}

LAMBDA_DESCRIPTIONS = {
    "risk_leaning": "Canonical 0.9 weight on normalized wildfire risk.",
    "balanced": "Equal weights on normalized wildfire risk and load shedding.",
    "service_leaning": "Canonical 0.9 weight on normalized load-shedding penalty.",
    "risk": "Legacy near-one weight on normalized wildfire risk.",
    "shed": "Legacy near-one weight on normalized load-shedding penalty.",
}


def lambda_case_rows(cases: dict[str, tuple[float, float]]) -> dict[str, dict[str, float | str]]:
    rows: dict[str, dict[str, float | str]] = {}
    for name, (lambda_R, lambda_L) in cases.items():
        rows[name] = {
            "lambda_R": float(lambda_R),
            "lambda_L": float(lambda_L),
            "description": LAMBDA_DESCRIPTIONS.get(name, ""),
        }
    return rows
