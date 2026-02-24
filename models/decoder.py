import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from torch_geometric.utils import to_dense_batch
import math
from loguru import logger


class DeepSetClassifier(nn.Module):
    def __init__(self, dim):
        super(DeepSetClassifier, self).__init__()
        self.phi = nn.Linear(1, dim)
        self.wctx = nn.Linear(dim, dim, bias=False)
        self.wself = nn.Linear(1, dim)  # per-element transform
        self.out = nn.Linear(dim, 1)

    def forward(self, z_tilde, mask):
        '''

        :param z_tilde: (bs, e, n, n)
        :param mask: (bs, n, n)
        :return:
        '''

        z_tilde = z_tilde.unsqueeze(-1)                                             # (bs, e, n, n, 1)
        mask = mask.unsqueeze(1).unsqueeze(-1)                                      # (bs, 1, n, n, 1)
        phi_out = self.phi(z_tilde.unsqueeze(-1)).squeeze()
        c = (mask * phi_out).sum(3) / mask.sum(3).clamp(min=1.0).float()           # (bs, e, n, d)
        h = F.gelu(self.wself(z_tilde) + self.wctx(c).unsqueeze(3))                 # (bs, e, n, n, d)
        return self.out(h).squeeze()


class DecoderDirected(nn.Module):
    def __init__(self, pe_dim, max_num_nodes, ds_dim, num_node_features, num_edge_features, dropout):
        super().__init__()
        self.pe_dim = pe_dim
        self.num_edge_features = num_edge_features + 1
        self.attn_proj = nn.Linear(pe_dim, 4*self.num_edge_features*self.pe_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.edge_out_proj = DeepSetClassifier(ds_dim)
        self.edge_proj_dropout = nn.Dropout(dropout)

        if num_node_features > 0:
            self.node_out_proj = nn.Linear(pe_dim, num_node_features)
        else:
            self.node_out_proj = nn.Identity()
        self.max_num_nodes = max_num_nodes

    def forward(self, x, batch, q):
        x_hat = self.node_out_proj(x)                                                          # (n_sum, x)

        k1, q1, k2, q2 = self.attn_proj(x).split(self.num_edge_features * self.pe_dim, dim=-1)         # (n_sum, e*d)
        k1 = self.attn_dropout(k1)
        q1 = self.attn_dropout(q1)
        k2 = self.attn_dropout(k2)
        q2 = self.attn_dropout(q2)

        k1, mask = to_dense_batch(k1, batch)                                                  # (b, n, e*d), (bs, n)
        q1, _ = to_dense_batch(q1, batch)
        k2, _ = to_dense_batch(k2, batch)
        q2, _ = to_dense_batch(q2, batch)
        scale_factor = 1 / math.sqrt(self.num_edge_features * self.pe_dim)

        b, n, ed = k1.shape

        k1 = k1.reshape(b, n, self.num_edge_features, -1).transpose(1, 2)    # (b, n, e, d) -> (b, e, n, d)
        q1 = q1.reshape(b, n, self.num_edge_features, -1).transpose(1, 2)
        k2 = k2.reshape(b, n, self.num_edge_features, -1).transpose(1, 2)    # (b, n, e, d) -> (b, e, n, d)
        q2 = q2.reshape(b, n, self.num_edge_features, -1).transpose(1, 2)

        z_re = torch.einsum("bend,bemd->benm", k1, q1).squeeze() * scale_factor         # (bs, e, n, n)
        z_im = torch.einsum("bend,bemd->benm", k2, q2).squeeze() * scale_factor         # (bs, e, n, n)

        c = torch.cos(2 * math.pi * q).reshape(-1, 1, 1, 1).to(z_im.device)
        s = torch.sin(2 * math.pi * q).reshape(-1, 1, 1, 1).to(z_im.device)
        z = z_re + z_im * (2 - c) / s

        # pad_size = self.max_num_nodes - n
        # z = F.pad(z, (0, pad_size, 0, pad_size))
        edge_mask = self.build_edge_mask(mask, 0)                                        # (bs, n, n)
        e_hat = self.edge_out_proj(z, edge_mask)
        e_hat = self.edge_proj_dropout(e_hat)                                                   # (bs, e, n, n)

        return e_hat.transpose(-1, 1), x_hat, edge_mask, mask

    def build_edge_mask(self, mask, pad_size):
        # mask = F.pad(mask, (0, pad_size))
        b, n = mask.shape[0], mask.shape[1]
        edge_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        diagonal_mask = torch.eye(n, dtype=torch.bool, device=edge_mask.device)
        diagonal_mask = diagonal_mask.unsqueeze(0).expand(b, -1, -1)

        edge_mask = edge_mask & ~diagonal_mask
        return edge_mask


class DecoderUnDirected(nn.Module):
    def __init__(self, pe_dim, max_num_nodes, ds_dim, num_node_features, num_edge_features, dropout):
        super().__init__()
        self.pe_dim = pe_dim
        self.num_edge_features = num_edge_features + 1
        self.attn_proj = nn.Linear(pe_dim, 2*self.num_edge_features*self.pe_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.edge_out_proj = DeepSetClassifier(ds_dim)
        self.edge_proj_dropout = nn.Dropout(dropout)

        if num_node_features > 0:
            self.node_out_proj = nn.Linear(pe_dim, num_node_features)
        else:
            self.node_out_proj = nn.Identity()
        self.max_num_nodes = max_num_nodes

    def forward(self, x, batch, q=None):
        x_hat = self.node_out_proj(x)                                                          # (n_sum, x)

        k, q = self.attn_proj(x).split(self.num_edge_features * self.pe_dim, dim=-1)         # (n_sum, e*d)
        k = self.attn_dropout(k)
        q = self.attn_dropout(q)

        k, mask = to_dense_batch(k, batch)                                                  # (b, n, e*d), (bs, n)
        q, _ = to_dense_batch(q, batch)
        scale_factor = 1 / math.sqrt(self.num_edge_features * self.pe_dim)
        b, n, ed = k.shape

        k = k.reshape(b, n, self.num_edge_features, -1).transpose(1, 2)    # (b, n, e, d) -> (b, e, n, d)
        q = q.reshape(b, n, self.num_edge_features, -1).transpose(1, 2)
        z_tilde = torch.einsum("bend,bemd->benm", q, k).squeeze() * scale_factor         # (bs, e, n, n)
        z_tilde = (z_tilde + z_tilde.transpose(-2, -1)) / 2                                    # (bs, e, n, n)

        b, n = z_tilde.shape[0], z_tilde.shape[-1]
        edge_mask = self.build_edge_mask(mask, 0)  # (bs, n, n)
        e_hat = self.edge_out_proj(z_tilde, edge_mask)
        e_hat = self.edge_proj_dropout(e_hat)                           # (bs, e, n, n)

        if e_hat.dim() < 4:
            e_hat = e_hat.unsqueeze(0)

        return e_hat.transpose(-1, 1), x_hat, edge_mask, mask

    def build_edge_mask(self, mask, pad_size):
        b, n = mask.shape[0], mask.shape[1]
        edge_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        diagonal_mask = torch.eye(n, dtype=torch.bool, device=edge_mask.device)
        diagonal_mask = diagonal_mask.unsqueeze(0).expand(b, -1, -1)

        edge_mask = edge_mask & ~diagonal_mask
        return edge_mask


class DecoderGVAE(nn.Module):
    def __init__(self, max_num_nodes):
        super().__init__()
        self.max_num_nodes = max_num_nodes

    def forward(self, x, batch, q=None):
        x_dense, mask = to_dense_batch(x, batch)                # (bs, n, d)
        p_tilde = x_dense @ x_dense.transpose(2, 1)                         # (bs, n, n)
        p_tilde = (p_tilde + p_tilde.transpose(-2, -1)) / 2     # (bs, n, n)

        b, n = p_tilde.shape[0], p_tilde.shape[-1]
        pad_size = self.max_num_nodes - n
        p_tilde = F.pad(p_tilde, (0, pad_size, 0, pad_size))
        edge_mask = self.build_edge_mask(mask, pad_size)        # (bs, n, n)

        return p_tilde, None, edge_mask, mask

    def build_edge_mask(self, mask, pad_size):
        mask = F.pad(mask, (0, pad_size))
        b, n = mask.shape[0], mask.shape[1]
        edge_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        diagonal_mask = torch.eye(n, dtype=torch.bool, device=edge_mask.device)
        diagonal_mask = diagonal_mask.unsqueeze(0).expand(b, -1, -1)

        edge_mask = edge_mask & ~diagonal_mask
        return edge_mask