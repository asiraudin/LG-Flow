import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, heads=1, dropout=0.0):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=embed_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
            edge_dim=in_dim  # assuming edge_attr has same dim as node features
        )

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, embed_dim, pe_dim, heads=1, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            GATLayer(in_dim if _ == 0 else embed_dim, embed_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(embed_dim, track_running_stats=False) for _ in range(num_layers)])

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2 * pe_dim),
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
