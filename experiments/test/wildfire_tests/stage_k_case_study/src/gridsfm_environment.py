"""Frozen GridSFM source, checkpoint, package, and CUDA device contract checks."""

from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

from .io_utils import sha256_file


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "environment" / "GRIDSFM_ENVIRONMENT_CONTRACT.json"


def load_gridsfm_contract(path: str | Path = CONTRACT_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def contract_checks(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, bool]:
    expected_packages = contract["runtime"]["packages"]
    observed_packages = manifest.get("runtime", {}).get("packages", {})
    return {
        "git_repository": manifest.get("source", {}).get("git_repository") == contract["source"]["git_repository"],
        "git_commit": manifest.get("source", {}).get("git_commit") == contract["source"]["git_commit"],
        "git_clean": manifest.get("source", {}).get("git_status_porcelain") == "",
        "gridsfm_package": manifest.get("source", {}).get("package_version") == contract["source"]["package_version"],
        "checkpoint_repository": manifest.get("checkpoint", {}).get("repository") == contract["checkpoint"]["repository"],
        "checkpoint_filename": manifest.get("checkpoint", {}).get("filename") == contract["checkpoint"]["filename"],
        "checkpoint_revision": manifest.get("checkpoint", {}).get("revision") == contract["checkpoint"]["revision"],
        "checkpoint_sha256": manifest.get("checkpoint", {}).get("sha256") == contract["checkpoint"]["sha256"],
        "python_version": manifest.get("runtime", {}).get("python") == contract["runtime"]["python"],
        **{
            f"package_{name}": observed_packages.get(name) == version
            for name, version in expected_packages.items()
        },
        "huggingface_hub_real_import": bool(manifest.get("runtime", {}).get("huggingface_hub_file")),
        "huggingface_hub_from_environment": manifest.get("runtime", {}).get("huggingface_hub_in_prefix") is True,
        "gridsfm_from_environment": manifest.get("source", {}).get("package_in_prefix") is True,
        "cuda_available": manifest.get("device", {}).get("cuda_available") is True,
        "nvidia_driver_visible": bool(manifest.get("device", {}).get("driver_version")),
        "one_gpu_visible": manifest.get("device", {}).get("gpu_count") == contract["device"]["expected_gpu_count"],
        "required_gpu_visible": contract["device"]["required_name_fragment"].lower()
        in str(manifest.get("device", {}).get("name", "")).lower(),
        "cuda_runtime": manifest.get("device", {}).get("cuda_runtime") == contract["device"]["expected_cuda_runtime"],
        "compute_capability": manifest.get("device", {}).get("capability")
        == contract["device"]["expected_compute_capability"],
        "cuda_device": manifest.get("device", {}).get("requested_device") == contract["device"]["cuda_device"],
    }


def capture_gridsfm_environment(
    gridsfm_root: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "cuda:0",
    contract_path: str | Path = CONTRACT_PATH,
) -> tuple[dict[str, Any], dict[str, bool]]:
    contract = load_gridsfm_contract(contract_path)
    root = Path(gridsfm_root).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_provenance_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".provenance.json"
    )
    checkpoint_provenance = (
        json.loads(checkpoint_provenance_path.read_text(encoding="utf-8"))
        if checkpoint_provenance_path.is_file()
        else {}
    )

    import huggingface_hub
    import numpy
    import scipy
    import torch
    import torch_geometric
    import gridsfm

    remote = _git(root, "remote", "get-url", "origin")
    gridsfm_file = Path(gridsfm.__file__).resolve()
    huggingface_hub_file = Path(huggingface_hub.__file__).resolve() if huggingface_hub.__file__ else None
    environment_prefix = Path(sys.prefix).resolve()
    driver_version = None
    if torch.cuda.is_available():
        driver_version = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    manifest: dict[str, Any] = {
        "contract_version": contract["contract_version"],
        "source": {
            "git_repository": remote,
            "git_commit": _git(root, "rev-parse", "HEAD"),
            "git_status_porcelain": _git(root, "status", "--porcelain"),
            "package_name": "gridsfm",
            "package_version": gridsfm.__version__,
            "package_file": str(gridsfm_file),
            "package_in_prefix": environment_prefix in gridsfm_file.parents,
        },
        "checkpoint": {
            "repository": checkpoint_provenance.get("repository"),
            "revision": checkpoint_provenance.get("revision"),
            "filename": checkpoint_path.name,
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
            "provenance_path": str(checkpoint_provenance_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(),
            "packages": {
                "torch": torch.__version__,
                "torch_geometric": metadata.version("torch_geometric"),
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "huggingface_hub": huggingface_hub.__version__,
                "lightning": metadata.version("lightning"),
            },
            "environment_prefix": str(environment_prefix),
            "huggingface_hub_file": str(huggingface_hub_file) if huggingface_hub_file else None,
            "huggingface_hub_in_prefix": bool(
                huggingface_hub_file and environment_prefix in huggingface_hub_file.parents
            ),
            "installed_distributions": {
                distribution.metadata["Name"]: distribution.version
                for distribution in sorted(
                    metadata.distributions(),
                    key=lambda item: (item.metadata.get("Name") or "").lower(),
                )
                if distribution.metadata.get("Name")
            },
        },
        "device": {
            "requested_device": device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "driver_version": driver_version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "execution": {
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        },
    }
    checks = contract_checks(manifest, contract)
    return manifest, checks
