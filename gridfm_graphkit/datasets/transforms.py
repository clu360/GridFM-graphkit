from gridfm_graphkit.datasets.globals import PQ, PV, REF, PG, QG, VM, VA, G, B
from gridfm_graphkit.io.registries import MASKING_REGISTRY

import torch
from torch import Tensor
from torch_geometric.transforms import BaseTransform
from typing import Optional
import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.utils import (
    get_self_loop_attr,
    is_torch_sparse_tensor,
    to_edge_index,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)


class AddNormalizedRandomWalkPE(BaseTransform):
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = "random_walk_pe",
    ) -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        if data.edge_index is None:
            raise ValueError("Expected data.edge_index to be not None")
        row, col = data.edge_index
        n_nodes = data.num_nodes
        if n_nodes is None:
            raise ValueError("Expected data.num_nodes to be not None")

        if n_nodes <= 2000:
            adj = torch.zeros((n_nodes, n_nodes), device=row.device)
            adj[row, col] = data.edge_weight
            loop_index = torch.arange(n_nodes, device=row.device)
        elif torch_geometric.typing.WITH_WINDOWS:
            adj = to_torch_coo_tensor(
                data.edge_index,
                data.edge_weight,
                size=data.size(),
            )
        else:
            adj = to_torch_csr_tensor(
                data.edge_index,
                data.edge_weight,
                size=data.size(),
            )

        row_sums = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        adj = adj / row_sums

        def get_pe(out: Tensor) -> Tensor:
            if is_torch_sparse_tensor(out):
                return get_self_loop_attr(*to_edge_index(out), num_nodes=n_nodes)
            return out[loop_index, loop_index]

        out = adj
        pe_list = [get_pe(out)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_pe(out))

        data[self.attr_name] = torch.stack(pe_list, dim=-1)
        return data


class AddEdgeWeights(BaseTransform):
    def forward(self, data):
        if not hasattr(data, "edge_attr"):
            raise AttributeError("Data must have 'edge_attr'.")
        real = data.edge_attr[:, G]
        imag = data.edge_attr[:, B]
        data.edge_weight = torch.sqrt(real**2 + imag**2)
        return data


@MASKING_REGISTRY.register("none")
class AddIdentityMask(BaseTransform):
    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        if not hasattr(data, "y"):
            raise AttributeError("Data must have ground truth 'y'.")
        data.mask = torch.zeros_like(data.y, dtype=torch.bool)
        return data


@MASKING_REGISTRY.register("rnd")
class AddRandomMask(BaseTransform):
    def __init__(self, args):
        super().__init__()
        self.mask_dim = args.data.mask_dim
        self.mask_ratio = args.data.mask_ratio

    def forward(self, data):
        if not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x'.")
        data.mask = torch.rand(data.x.size(0), self.mask_dim) < self.mask_ratio
        return data


@MASKING_REGISTRY.register("pf")
class AddPFMask(BaseTransform):
    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        if not hasattr(data, "y") or not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x' and labels 'y'.")
        mask_pq = data.x[:, PQ] == 1
        mask_pv = data.x[:, PV] == 1
        mask_ref = data.x[:, REF] == 1
        mask = torch.zeros_like(data.y, dtype=torch.bool)
        mask[mask_pq, VM] = True
        mask[mask_pq, VA] = True
        mask[mask_pv, QG] = True
        mask[mask_pv, VA] = True
        mask[mask_ref, PG] = True
        mask[mask_ref, QG] = True
        data.mask = mask
        return data


@MASKING_REGISTRY.register("opf")
class AddOPFMask(BaseTransform):
    def __init__(self, args):
        super().__init__()

    def forward(self, data):
        if not hasattr(data, "y") or not hasattr(data, "x"):
            raise AttributeError("Data must have node features 'x' and labels 'y'.")
        mask_pq = data.x[:, PQ] == 1
        mask_pv = data.x[:, PV] == 1
        mask_ref = data.x[:, REF] == 1
        mask = torch.zeros_like(data.y, dtype=torch.bool)
        mask[mask_pq, PG] = True
        mask[mask_pq, QG] = True
        mask[mask_pq, VM] = True
        mask[mask_pq, VA] = True
        mask[mask_pv, QG] = True
        mask[mask_pv, VM] = True
        mask[mask_pv, VA] = True
        mask[mask_ref, PG] = True
        mask[mask_ref, QG] = True
        mask[mask_ref, VM] = True
        mask[mask_ref, VA] = True
        data.mask = mask
        return data
