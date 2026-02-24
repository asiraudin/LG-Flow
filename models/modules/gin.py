import torch.nn as nn
from torch_geometric.nn import GINEConv, DirGNNConv
import torch.nn.functional as F
from loguru import logger
import torch


class DirGNNConvEdgeAttr(DirGNNConv):
    def __init__(self, conv, alpha, root_weight, in_dim, out_dim):
        super(DirGNNConvEdgeAttr, self).__init__(conv, alpha, False)
        self.root_weight = root_weight
        if self.root_weight:
            self.lin = torch.nn.Linear(in_dim, out_dim)
        else:
            self.lin = None

    def forward(self, x, edge_index, edge_attr=None):
        # Pass edge_attr into both directional convs
        x_in = self.conv_in(x, edge_index, edge_attr=edge_attr)
        x_out = self.conv_out(x, edge_index.flip([0]), edge_attr=edge_attr)

        out = self.alpha * x_out + (1 - self.alpha) * x_in
        if self.root_weight:
            out = out + self.lin(x)
        return out


class GINDirLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, edge_dim):
        super().__init__()
        self.conv = DirGNNConvEdgeAttr(
            conv=GINEConv(
                nn=nn.Sequential(
                    nn.Linear(in_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim)
                ),
                edge_dim=edge_dim
            ),
            alpha=0.5,
            root_weight=True,
            in_dim=in_dim,
            out_dim=embed_dim
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        return h


class GINLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, edge_dim):
        super().__init__()
        self.conv = GINEConv(
                nn=nn.Sequential(
                    nn.Linear(in_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim)
                ),
                edge_dim=edge_dim
            )

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        return h


class GIN(nn.Module):
    def __init__(self, directed, num_layers, in_dim, embed_dim, pe_dim, dropout=0.):
        super().__init__()
        layer = GINDirLayer if directed else GINLayer
        self.layers = nn.ModuleList(
            [
                layer(in_dim if _ == 0 else embed_dim, embed_dim, edge_dim=in_dim)
                for _ in range(num_layers)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(embed_dim, track_running_stats=False) for _ in range(num_layers)])

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2*pe_dim),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):

        for i, layer in enumerate(self.layers):
            z = layer(x, edge_index, edge_attr)
            z = self.bns[i](z)
            z = F.gelu(z)
            z = torch.dropout(z, p=self.dropout, train=self.training)
            if i != 0:
                x = z + x
            else:
                x = z

        return self.mlp(x)
