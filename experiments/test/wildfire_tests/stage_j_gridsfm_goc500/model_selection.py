"""Canonical Stage J electrical model selection and checkpoint provenance."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


MODEL_SELECTIONS = ("dc", "frozen", "ft")
RELEASED_V1_1_SHA256 = "F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831"
FT_CHECKPOINT_VARIANTS = {
    "A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD": "fulltop_ft_n1000",
    "08EDA70270F787DB42C48B751BECC9DA2062B0482550A5C2A761B4EBA0C94FF6": "fulltop_ft_n1000_then_fulltop_n500",
    "4EC89D36DE80081BE2FC26C14A1BC5A5369D7423B557D71B9DD4A254462F302D": "fulltop_ft_n1000_then_n1_n500",
}


@dataclass(frozen=True)
class ModelSelection:
    model_selection: str
    model_variant: str
    checkpoint_path: str | None
    checkpoint_sha256: str | None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_model_selection(
    selection: str,
    *,
    gridsfm_root: Path,
    checkpoint: Path | None = None,
    expected_sha256: str | None = None,
) -> ModelSelection:
    selection = str(selection).lower()
    if selection not in MODEL_SELECTIONS:
        raise ValueError(f"model_selection must be one of {MODEL_SELECTIONS}, got {selection!r}")

    if selection == "dc":
        if checkpoint is not None or expected_sha256 is not None:
            raise ValueError("model_selection=dc does not accept checkpoint provenance")
        return ModelSelection("dc", "guided_dc", None, None)

    if selection == "frozen":
        default = gridsfm_root / "model" / "checkpoints" / "gridsfm_open_v1.1.pt"
        resolved = default.resolve() if checkpoint is None else checkpoint.expanduser().resolve()
        expected = RELEASED_V1_1_SHA256
        if expected_sha256 is not None and expected_sha256.upper() != expected:
            raise ValueError("frozen expected SHA-256 differs from released GridSFM v1.1")
        variant = "released_v1_1"
    else:
        if checkpoint is None or expected_sha256 is None:
            raise ValueError("model_selection=ft requires checkpoint and expected SHA-256")
        resolved = checkpoint.expanduser().resolve()
        expected = expected_sha256.upper()
        variant = FT_CHECKPOINT_VARIANTS.get(expected, "fulltop_ft_n1000")

    if not resolved.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {resolved}")
    observed = sha256_file(resolved)
    if observed != expected:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch for model_selection={selection}: "
            f"expected {expected}, observed {observed}"
        )
    return ModelSelection(selection, variant, str(resolved), observed)
