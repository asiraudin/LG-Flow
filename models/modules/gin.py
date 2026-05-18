import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch_geometric.nn import GINEConv, DirGNNConv, GraphNorm
from typing import Any


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
    def __init__(
        self,
        directed: bool,
        num_layers: int,
        in_dim: int,
        embed_dim: int,
        pe_dim: int,
        dropout: float = 0.,
        norm: str = "batch_norm",
    ) -> None:
        super().__init__()
        layer = GINDirLayer if directed else GINLayer
        self.layers = nn.ModuleList(
            [
                layer(in_dim if _ == 0 else embed_dim, embed_dim, edge_dim=in_dim)
                for _ in range(num_layers)
            ]
        )
        if norm == "batch_norm":
            self.norms = nn.ModuleList(
                [nn.BatchNorm1d(embed_dim, track_running_stats=False) for _ in range(num_layers)]
            )
        elif norm == "graph_norm":
            self.norms = nn.ModuleList([GraphNorm(embed_dim) for _ in range(num_layers)])
        else:
            raise ValueError(f"Unsupported GIN norm: {norm}")
        self.norm = norm

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2*pe_dim),
        )
        self.dropout = dropout

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        legacy_prefix = f"{prefix}bns."
        current_prefix = f"{prefix}norms."
        remapped_keys: list[tuple[str, str]] = []

        # Older checkpoints stored the normalization layers under `bns`.
        for key in list(state_dict.keys()):
            if key.startswith(legacy_prefix):
                remapped_key = f"{current_prefix}{key[len(legacy_prefix):]}"
                if remapped_key not in state_dict:
                    state_dict[remapped_key] = state_dict.pop(key)
                    remapped_keys.append((key, remapped_key))

        if remapped_keys:
            logger.info(
                f"Remapped {len(remapped_keys)} legacy GIN checkpoint keys from `bns` to `norms` under `{prefix}`"
            )

        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm == "graph_norm" and batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)  # (num_nodes,)

        for i, layer in enumerate(self.layers):
            z = layer(x, edge_index, edge_attr)
            if self.norm == "graph_norm":
                z = self.norms[i](z, batch)
            else:
                z = self.norms[i](z)
            z = F.gelu(z)
            z = torch.dropout(z, p=self.dropout, train=self.training)
            if i != 0:
                x = z + x
            else:
                x = z

        return self.mlp(x)
