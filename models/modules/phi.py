import torch.nn as nn
from .mlp import MLPLayer


class EigenMLP(nn.Module):
    def __init__(self, num_layers, embed_dim, out_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MLPLayer(1 if i == 0 else embed_dim, embed_dim, dropout)
                for i in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(embed_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out_proj(x)
        return self.dropout(x)
