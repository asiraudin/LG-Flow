import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, track_running_stats=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.bn(x.reshape(-1, x.size(-1))).reshape(x.size())
        return self.dropout(F.relu(x))


class MLP(nn.Module):
    def __init__(self, num_layers, in_dim, embed_dim, out_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MLPLayer(in_dim if _ == 0 else embed_dim, embed_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(embed_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out_proj(x)
        return self.dropout(x)
