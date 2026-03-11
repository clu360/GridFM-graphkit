from gridfm_graphkit.io.registries import MODELS_REGISTRY
from torch_geometric.nn import TransformerConv
from torch import nn
import torch


@MODELS_REGISTRY.register("GNN_TransformerConv")
class GNN_TransformerConv(nn.Module):
    """
    TransformerConv-based GNN used by the legacy GridFM v0.1 checkpoint.
    """

    def __init__(self, args):
        super().__init__()

        self.input_dim = args.model.input_dim
        self.hidden_dim = args.model.hidden_size
        self.output_dim = args.model.output_dim
        self.edge_dim = args.model.edge_dim
        self.num_layers = args.model.num_layers

        self.heads = getattr(args.model, "attention_head", 1)
        self.mask_dim = getattr(args.data, "mask_dim", 6)
        self.mask_value = getattr(args.data, "mask_value", -1.0)
        self.learn_mask = getattr(args.data, "learn_mask", False)

        self.layers = nn.ModuleList()
        current_dim = self.input_dim

        for _ in range(self.num_layers):
            self.layers.append(
                TransformerConv(
                    current_dim,
                    self.hidden_dim,
                    heads=self.heads,
                    edge_dim=self.edge_dim,
                    beta=False,
                ),
            )
            current_dim = self.hidden_dim * self.heads

        self.mlps = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        if self.learn_mask:
            self.mask_value = nn.Parameter(
                torch.randn(self.mask_dim) + self.mask_value,
                requires_grad=True,
            )
        else:
            self.mask_value = nn.Parameter(
                torch.zeros(self.mask_dim) + self.mask_value,
                requires_grad=False,
            )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = nn.LeakyReLU()(x)

        return self.mlps(x)
