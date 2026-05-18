import torch
from torch import nn
from typing import Any
from .modules import GIN
from loguru import logger


class NodeFeatureEmbedder(nn.Module):
    def __init__(self, num_node_types, num_node_features, embed_dim):
        super().__init__()
        self.num_node_types = num_node_types
        self.num_node_features = num_node_features

        if num_node_types > 0 and num_node_features > 1:
            self.mode = "mixed"
            self.type_embed = nn.Embedding(num_node_types, embed_dim)
            self.feat_proj = nn.Linear(num_node_features - 1, embed_dim)
        elif num_node_types > 0:
            self.mode = "types"
            self.embed = nn.Embedding(num_node_types, embed_dim)
        else:
            self.mode = "identity"
            self.embed = nn.Identity()

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
        legacy_weight_key = f"{prefix}weight"
        legacy_bias_key = f"{prefix}bias"
        remapped_keys: list[str] = []

        if legacy_weight_key in state_dict:
            legacy_weight = state_dict[legacy_weight_key]
            legacy_bias = state_dict.get(legacy_bias_key)

            if self.mode == "types":
                current_weight_key = f"{prefix}embed.weight"
                if (
                    legacy_weight.shape == self.embed.weight.shape
                    and legacy_bias is None
                    and current_weight_key not in state_dict
                ):
                    state_dict[current_weight_key] = state_dict.pop(legacy_weight_key)
                    remapped_keys.append(current_weight_key)
                elif (
                    legacy_weight.dim() == 2
                    and legacy_weight.size(1) == 1
                    and current_weight_key not in state_dict
                ):
                    type_ids = torch.arange(
                        self.num_node_types,
                        device=legacy_weight.device,
                        dtype=legacy_weight.dtype,
                    )  # (num_node_types,)
                    type_embed = type_ids.unsqueeze(1) * legacy_weight[:, 0].unsqueeze(0)  # (num_node_types, embed_dim)
                    if legacy_bias is not None:
                        type_embed = type_embed + legacy_bias.unsqueeze(0)  # (num_node_types, embed_dim)
                        state_dict.pop(legacy_bias_key)
                    state_dict[current_weight_key] = type_embed
                    state_dict.pop(legacy_weight_key)
                    remapped_keys.append(current_weight_key)

            elif (
                self.mode == "mixed"
                and legacy_weight.dim() == 2
                and legacy_weight.size(1) == self.num_node_features
            ):
                type_weight_key = f"{prefix}type_embed.weight"
                feat_weight_key = f"{prefix}feat_proj.weight"
                feat_bias_key = f"{prefix}feat_proj.bias"
                if type_weight_key not in state_dict and feat_weight_key not in state_dict:
                    type_ids = torch.arange(
                        self.num_node_types,
                        device=legacy_weight.device,
                        dtype=legacy_weight.dtype,
                    )  # (num_node_types,)
                    type_embed = type_ids.unsqueeze(1) * legacy_weight[:, 0].unsqueeze(0)  # (num_node_types, embed_dim)
                    state_dict[type_weight_key] = type_embed
                    state_dict[feat_weight_key] = legacy_weight[:, 1:]  # (embed_dim, num_node_features - 1)
                    remapped_keys.extend([type_weight_key, feat_weight_key])
                    if legacy_bias is not None and feat_bias_key not in state_dict:
                        state_dict[feat_bias_key] = legacy_bias
                        remapped_keys.append(feat_bias_key)
                        state_dict.pop(legacy_bias_key)
                    state_dict.pop(legacy_weight_key)

        if remapped_keys:
            logger.info(
                f"Remapped legacy node feature embedder checkpoint keys under `{prefix}` to {remapped_keys}"
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

    def forward(self, x):
        if self.mode == "identity":
            return x.float()

        if self.mode == "types":
            x_type = x if x.dim() == 1 else x[:, 0]
            return self.embed(x_type.long())

        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x_type = x[:, 0].long()
        x_feat = x[:, 1:].float()
        return self.type_embed(x_type) + self.feat_proj(x_feat)


class PaddedComplexEigenEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, bias):
        super().__init__()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(3, 2 * embed_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 2, embed_dim, bias=bias),
            torch.nn.ReLU(),
        )

    def forward(self, eigvecs_real, eigvecs_imag, eigvals):
        x = torch.stack((eigvecs_real, eigvecs_imag, eigvals), 2)
        empty_mask = torch.isnan(x)
        x[empty_mask] = 0
        return self.ffn(x)


class PaddedEigenEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, bias):
        super().__init__()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(2, 2 * embed_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 2, embed_dim, bias=bias),
            torch.nn.ReLU(),
        )

    def forward(self, eigvecs, eigvals):
        x = torch.stack((eigvecs, eigvals), 2)
        empty_mask = torch.isnan(x)
        x[empty_mask] = 0
        return self.ffn(x)


def nan_to_zero(mat):
    empty_mask = torch.isnan(mat)
    mat[empty_mask] = 0
    return mat


class EncodermLPE(nn.Module):
    def __init__(
            self,
            num_node_types,
            num_node_features,
            num_edge_features,
            global_cfg,
            phi_cfg,
            rho_cfg,
            dropout,
    ):
        super().__init__()
        self.num_vecs = global_cfg.num_vecs
        self.eps = torch.nn.Parameter(1e-12 * torch.arange(self.num_vecs).unsqueeze(0))
        self.phi_name = global_cfg.phi_model
        self.rho_name = global_cfg.rho_model

        self.node_features_embed = NodeFeatureEmbedder(
            num_node_types=num_node_types,
            num_node_features=num_node_features,
            embed_dim=phi_cfg.hidden_dim,
        )
        self.edge_features_embed = nn.Linear(num_edge_features, phi_cfg.hidden_dim) if num_edge_features > 0 else nn.Identity()

        self.phi = PaddedComplexEigenEmbedding(phi_cfg.hidden_dim, phi_cfg.bias)
        self.rho = GIN(
            True,
            rho_cfg.num_layers,
            phi_cfg.hidden_dim,
            rho_cfg.embed_dim,
            global_cfg.pe_dim,
            dropout,
            rho_cfg.norm,
        )

        self.modulation = global_cfg.modulation

    def forward(self, data):
        eigvecs_real = data.eigvecs_real[:, : self.num_vecs]  # (n_sum, num_vecs)
        eigvecs_imag = data.eigvecs_imag[:, : self.num_vecs]  # (n_sum, num_vecs)
        eigvals = data.eigvals[:, : self.num_vecs]  # (n_sum, num_vecs)
        num_vecs = min(self.num_vecs, eigvals.shape[-1])

        if self.training:
            sign_flip = torch.rand(eigvecs_real.size(1), device=eigvecs_real.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eigvecs_real = eigvecs_real * sign_flip.unsqueeze(0)
            eigvecs_imag = eigvecs_imag * sign_flip.unsqueeze(0)

        if self.modulation > 0.:
            m = (data.orbits != 0).to(eigvecs_real.dtype)  # (n_sum, 1) in {0,1}
            # logger.info(f'Number of nodes in non-trivial orbits : {m.sum()}')
            g = torch.randn(eigvecs_real.size(0), 1, device=eigvecs_real.device, dtype=eigvecs_real.dtype)  # N(0,1) per row
            factor = 1 + self.modulation * g * m
            eigvecs_real = factor * eigvecs_real
            eigvecs_imag = factor * eigvecs_imag

        eigvals = eigvals + self.eps[:, : num_vecs]
        eigen_embed = self.phi(eigvecs_real, eigvecs_imag, eigvals).sum(1)  # (n_sum, d)
        x_embed = self.node_features_embed(data.x)
        
        x_embed = x_embed + eigen_embed
        e_embed = self.edge_features_embed(data.edge_attr.float())  # (num_edges, d)
        pe = self.rho(x_embed, data.edge_index, e_embed, getattr(data, "batch", None))
        return pe


class EncoderLPE(nn.Module):
    def __init__(
            self,
            num_node_types,
            num_node_features,
            num_edge_features,
            global_cfg,
            phi_cfg,
            rho_cfg,
            dropout
    ):
        super().__init__()
        self.num_vecs = global_cfg.num_vecs
        self.eps = torch.nn.Parameter(1e-12 * torch.arange(self.num_vecs).unsqueeze(0))
        self.phi_name = global_cfg.phi_model
        self.rho_name = global_cfg.rho_model

        self.node_features_embed = NodeFeatureEmbedder(
            num_node_types=num_node_types,
            num_node_features=num_node_features,
            embed_dim=phi_cfg.hidden_dim,
        )
        self.edge_features_embed = nn.Linear(num_edge_features, phi_cfg.hidden_dim) if num_edge_features > 0 else nn.Identity()

        self.phi = PaddedEigenEmbedding(phi_cfg.hidden_dim, phi_cfg.bias)
        self.rho = GIN(
            False,
            rho_cfg.num_layers,
            phi_cfg.hidden_dim,
            rho_cfg.embed_dim,
            global_cfg.pe_dim,
            dropout,
            rho_cfg.norm,
        )

        self.modulation = global_cfg.modulation

    def forward(self, data):
        eigvecs = data.eigvecs[:, : self.num_vecs]  # (n_sum, num_vecs)
        eigvals = data.eigvals[:, : self.num_vecs]  # (n_sum, num_vecs)
        num_vecs = min(self.num_vecs, eigvals.shape[-1])

        if self.training:
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eigvecs = eigvecs * sign_flip.unsqueeze(0)

        if self.modulation > 0.:
            m = (data.orbits != 0).to(eigvecs.dtype)  # (n_sum, 1) in {0,1}
            g = torch.randn(eigvecs.size(0), 1, device=eigvecs.device, dtype=eigvecs.dtype)  # N(0,1) per row
            factor = 1 + self.modulation * g * m
            eigvecs = factor * eigvecs

        eigvals = eigvals + self.eps[:, : num_vecs]
        eigen_embed = self.phi(eigvecs, eigvals).sum(1)  # (n_sum, d)
        x_embed = self.node_features_embed(data.x)  # (n_sum, d)
        
        x_embed = x_embed + eigen_embed
        e_embed = self.edge_features_embed(data.edge_attr.float())  # (num_edges, d)
        pe = self.rho(x_embed, data.edge_index, e_embed, getattr(data, "batch", None))  # (n_sum, pe_dim)
        return pe
