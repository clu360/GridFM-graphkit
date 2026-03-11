from gridfm_graphkit.io.registries import MODELS_REGISTRY
from torch_geometric.nn import GPSConv, GINEConv
from torch import nn
import torch


@MODELS_REGISTRY.register("GPSTransformer")
class GPSTransformer(nn.Module):
    """
    GPS-based transformer used by the legacy GridFM v0.2 checkpoint.
    """

    def __init__(self, args):
        super().__init__()

        self.input_dim = args.model.input_dim
        self.hidden_dim = args.model.hidden_size
        self.output_dim = args.model.output_dim
        self.edge_dim = args.model.edge_dim
        self.pe_dim = args.model.pe_dim
        self.num_layers = args.model.num_layers

        self.heads = getattr(args.model, "attention_head", 1)
        self.dropout = getattr(args.model, "dropout", 0.0)
        self.mask_dim = getattr(args.data, "mask_dim", 6)
        self.mask_value = getattr(args.data, "mask_value", -1.0)
        self.learn_mask = getattr(args.data, "learn_mask", True)

        if not self.pe_dim < self.hidden_dim:
            raise ValueError(
                "positional encoding dimension must be smaller than model hidden dimension",
            )

        self.layers = nn.ModuleList()

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim - self.pe_dim),
            nn.LeakyReLU(),
        )
        self.input_norm = nn.BatchNorm1d(self.hidden_dim - self.pe_dim)
        self.pe_norm = nn.BatchNorm1d(self.pe_dim)

        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                nn.LeakyReLU(),
            )
            self.layers.append(
                nn.ModuleDict(
                    {
                        "conv": GPSConv(
                            channels=self.hidden_dim,
                            conv=GINEConv(nn=mlp, edge_dim=self.edge_dim),
                            heads=self.heads,
                            dropout=self.dropout,
                        ),
                        "norm": nn.BatchNorm1d(self.hidden_dim),
                    },
                ),
            )

        self.pre_decoder_norm = nn.BatchNorm1d(self.hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
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
        x_pe = self.pe_norm(pe)
        x = self.encoder(x)
        x = self.input_norm(x)
        x = torch.cat((x, x_pe), 1)

        for layer in self.layers:
            x = layer["conv"](
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
            )
            x = layer["norm"](x)

        x = self.pre_decoder_norm(x)
        return self.decoder(x)
