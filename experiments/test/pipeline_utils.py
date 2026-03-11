"""
Shared setup helpers for the experiments/test wildfire-dispatch workflow.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml

from .scenario_data import ScenarioData, extract_scenario_from_batch
from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace, load_model


@dataclass
class TestScenarioContext:
    repo_root: Path
    config_dict: dict
    args: NestedNamespace
    datamodule: LitGridDataModule
    batch: object
    scenario: ScenarioData


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_gnn_checkpoint_path(repo_root: Optional[Path] = None) -> Path:
    repo_root = repo_root or get_repo_root()
    return repo_root / "examples" / "models" / "GridFM_v0_1.pth"


def get_gps_checkpoint_path(repo_root: Optional[Path] = None) -> Path:
    repo_root = repo_root or get_repo_root()
    return repo_root / "examples" / "models" / "GridFM_v0_2.pth"


def load_test_config(
    repo_root: Optional[Path] = None,
    config_name: str = "gridFMv0.1_dummy.yaml",
) -> Tuple[dict, NestedNamespace]:
    repo_root = repo_root or get_repo_root()
    config_path = repo_root / "tests" / "config" / config_name
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return config_dict, NestedNamespace(**config_dict)


def load_test_datamodule(
    args: NestedNamespace,
    repo_root: Optional[Path] = None,
) -> LitGridDataModule:
    repo_root = repo_root or get_repo_root()
    datamodule = LitGridDataModule(
        args,
        data_dir=str(repo_root / "tests" / "data"),
    )
    datamodule.setup(stage="test")
    return datamodule


def load_first_test_batch(datamodule: LitGridDataModule):
    return next(iter(datamodule.test_dataloader()[0]))


def load_single_test_scenario(
    scenario_idx: int = 0,
    scenario_id: str = "IEEE-30-test",
    config_name: str = "gridFMv0.1_dummy.yaml",
) -> TestScenarioContext:
    repo_root = get_repo_root()
    config_dict, args = load_test_config(repo_root, config_name=config_name)
    datamodule = load_test_datamodule(args, repo_root=repo_root)
    batch = load_first_test_batch(datamodule)
    scenario = extract_scenario_from_batch(
        batch,
        datamodule.node_normalizers[0],
        datamodule.edge_normalizers[0],
        scenario_idx=scenario_idx,
        scenario_id=scenario_id,
    )
    return TestScenarioContext(
        repo_root=repo_root,
        config_dict=config_dict,
        args=args,
        datamodule=datamodule,
        batch=batch,
        scenario=scenario,
    )


def load_checkpointed_model(
    args: NestedNamespace,
    checkpoint_path: Path,
    device: str = "cpu",
):
    model = load_model(args)
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def load_gnn_model(
    args: NestedNamespace,
    repo_root: Optional[Path] = None,
    device: str = "cpu",
):
    repo_root = repo_root or get_repo_root()
    return load_checkpointed_model(
        args,
        get_gnn_checkpoint_path(repo_root),
        device=device,
    )


def load_gps_model(
    config_dict: dict,
    repo_root: Optional[Path] = None,
    device: str = "cpu",
):
    repo_root = repo_root or get_repo_root()
    gps_config = dict(config_dict)
    gps_config["model"] = dict(config_dict["model"])
    gps_config["model"]["type"] = "GPSTransformer"
    args_gps = NestedNamespace(**gps_config)
    model = load_checkpointed_model(
        args_gps,
        get_gps_checkpoint_path(repo_root),
        device=device,
    )
    return model, args_gps
