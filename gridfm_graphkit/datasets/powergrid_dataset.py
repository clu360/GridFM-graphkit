from gridfm_graphkit.datasets.normalizers import Normalizer, BaseMVANormalizer
from gridfm_graphkit.datasets.transforms import (
    AddEdgeWeights,
    AddNormalizedRandomWalkPE,
)

import os.path as osp
import os
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
from tqdm import tqdm
from typing import Optional, Callable


class GridDatasetDisk(Dataset):
    def __init__(
        self,
        root: str,
        norm_method: str,
        node_normalizer: Normalizer,
        edge_normalizer: Normalizer,
        pe_dim: int,
        mask_dim: int = 6,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.norm_method = norm_method
        self.node_normalizer = node_normalizer
        self.edge_normalizer = edge_normalizer
        self.pe_dim = pe_dim
        self.mask_dim = mask_dim
        self.length = None

        super().__init__(root, transform, pre_transform, pre_filter)

        node_stats_path = osp.join(self.processed_dir, f"node_stats_{self.norm_method}.pt")
        edge_stats_path = osp.join(self.processed_dir, f"edge_stats_{self.norm_method}.pt")
        if osp.exists(node_stats_path) and osp.exists(edge_stats_path):
            self.node_stats = torch.load(node_stats_path, weights_only=False)
            self.edge_stats = torch.load(edge_stats_path, weights_only=False)
            self.node_normalizer.fit_from_dict(self.node_stats)
            self.edge_normalizer.fit_from_dict(self.edge_stats)

    @property
    def raw_file_names(self):
        return ["pf_node.csv", "pf_edge.csv"]

    @property
    def processed_done_file(self):
        return f"processed_{self.norm_method}_{self.mask_dim}_{self.pe_dim}.done"

    @property
    def processed_file_names(self):
        return [self.processed_done_file]

    def download(self):
        pass

    def process(self):
        node_df = pd.read_csv(osp.join(self.raw_dir, "pf_node.csv"))
        edge_df = pd.read_csv(osp.join(self.raw_dir, "pf_edge.csv"))
        scenarios = node_df["scenario"].unique()
        if not (scenarios == edge_df["scenario"].unique()).all():
            raise ValueError("Mismatch between node and edge scenario values.")

        cols_to_normalize = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]
        to_normalize = torch.tensor(node_df[cols_to_normalize].values, dtype=torch.float)
        self.node_stats = self.node_normalizer.fit(to_normalize)
        node_df[cols_to_normalize] = self.node_normalizer.transform(to_normalize).numpy()

        cols_to_normalize = ["G", "B"]
        to_normalize = torch.tensor(edge_df[cols_to_normalize].values, dtype=torch.float)
        if isinstance(self.node_normalizer, BaseMVANormalizer):
            self.edge_stats = self.edge_normalizer.fit(to_normalize, self.node_normalizer.baseMVA)
        else:
            self.edge_stats = self.edge_normalizer.fit(to_normalize)
        edge_df[cols_to_normalize] = self.edge_normalizer.transform(to_normalize).numpy()

        torch.save(self.node_stats, osp.join(self.processed_dir, f"node_stats_{self.norm_method}.pt"))
        torch.save(self.edge_stats, osp.join(self.processed_dir, f"edge_stats_{self.norm_method}.pt"))

        node_groups = node_df.groupby("scenario")
        edge_groups = edge_df.groupby("scenario")
        for scenario_idx in tqdm(scenarios):
            node_data = node_groups.get_group(scenario_idx)
            x = torch.tensor(
                node_data[["Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF"]].values,
                dtype=torch.float,
            )
            y = x[:, : self.mask_dim]

            edge_data = edge_groups.get_group(scenario_idx)
            edge_attr = torch.tensor(edge_data[["G", "B"]].values, dtype=torch.float)
            edge_index = torch.tensor(edge_data[["index1", "index2"]].values.T, dtype=torch.long)

            graph_data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                scenario_id=scenario_idx,
            )
            graph_data = AddEdgeWeights()(graph_data)
            graph_data = AddNormalizedRandomWalkPE(
                walk_length=self.pe_dim,
                attr_name="pe",
            )(graph_data)
            torch.save(
                graph_data,
                osp.join(
                    self.processed_dir,
                    f"data_{self.norm_method}_{self.mask_dim}_{self.pe_dim}_index_{scenario_idx}.pt",
                ),
            )
        with open(osp.join(self.processed_dir, self.processed_done_file), "w", encoding="utf-8") as f:
            f.write("done")

    def len(self):
        if self.length is None:
            files = [
                f
                for f in os.listdir(self.processed_dir)
                if f.startswith(f"data_{self.norm_method}_{self.mask_dim}_{self.pe_dim}_index_")
                and f.endswith(".pt")
            ]
            self.length = len(files)
        return self.length

    def get(self, idx):
        file_name = osp.join(
            self.processed_dir,
            f"data_{self.norm_method}_{self.mask_dim}_{self.pe_dim}_index_{idx}.pt",
        )
        if not osp.exists(file_name):
            raise IndexError(f"Data file {file_name} does not exist.")
        data = torch.load(file_name, weights_only=False)
        if self.transform:
            data = self.transform(data)
        return data

    def change_transform(self, new_transform):
        self.original_transform = self.transform
        self.transform = new_transform

    def reset_transform(self):
        if self.original_transform is None:
            raise ValueError("The original transform is None or change_transform was not called")
        self.transform = self.original_transform
