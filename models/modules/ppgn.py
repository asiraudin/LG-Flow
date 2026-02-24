import torch
from torch import nn
from .mlp import MLP


def mask2d_diag_offdiag_meanpool(x, mask):
    #  x: [B, N, N, d], mask: [B, N, N, 1]
    N = mask.size(1)
    mean_diag = torch.diagonal(x, dim1=1, dim2=2).transpose(-1, -2) # [B, N, d]
    mean_offdiag = (torch.sum(x * mask, dim=1) + torch.sum(x * mask, dim=2) - 2 * mean_diag) / (2 * N - 2)
    return torch.cat((mean_diag, mean_offdiag), dim=-1)  # [B, N, 2*d]


class PPGN(nn.Module):
    def __init__(self, num_rb_layer, num_mlp_layers, in_dims, embed_dim, pe_dim):
        super(PPGN, self).__init__()
        self.layers = torch.nn.ModuleList(
            [
                RegularBlock(
                    num_mlp_layers, in_dims if _ == 0 else embed_dim, embed_dim, embed_dim
                )
                for _ in range(num_rb_layer - 1)
            ]
        )
        self.layers.append(RegularBlock(num_mlp_layers, embed_dim, embed_dim, pe_dim))
        self.out_proj = nn.Linear(2 * pe_dim, 2 * pe_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: [B, N, N, in_dims], mask: [B, N, N, 1]
        for layer in self.layers:
            x = layer(x, mask)
        x = mask2d_diag_offdiag_meanpool(x, mask)
        x = self.out_proj(x)
        return x


class RegularBlock(nn.Module):
    def __init__(self, num_mlp_layers, in_dims, embed_dim, out_dims):
        super(RegularBlock, self).__init__()
        self.mlp = MLP(num_mlp_layers, in_dims, embed_dim, 2*out_dims)
        self.out_dims = out_dims
        self.skip = SkipConnectionBlock(num_mlp_layers, in_dims+out_dims, embed_dim, out_dims)

    # @torch.compile
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: [B, N, N, in_dims], mask: [B, N, N, 1]
        mlp = mask * self.mlp(x) # [B, N, N, 2* out_dims], mask fake nodes
        out1, out2 = mlp.split(self.out_dims, dim=-1)

        mult = torch.matmul(out1.transpose(1, -1), out2.transpose(1, -1)) # [B, out_dims, N, N]
        mult = mult.transpose(1, -1) # [B, N, N, out_dims]

        out = self.skip(x, mult)
        return out


class SkipConnectionBlock(nn.Module):
    def __init__(self, num_mlp_layers, in_dims, embed_dim, out_dims):
        super(SkipConnectionBlock, self).__init__()
        self.mlp = MLP(num_mlp_layers, in_dims, embed_dim, out_dims)

    def forward(self, x0: torch.Tensor, x: torch.Tensor):
        # X0: [B, N, N, in_dims]
        # X:  [B, N, N, out_dims]
        out = torch.cat((x0, x), dim=-1)
        out = self.mlp(out)
        return out
    