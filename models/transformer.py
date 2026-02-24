import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def get_pos_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine poDiTional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: embedding dimension
        max_len: maximum length

    Returns:
        poDiTional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
    return pos_embedding


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class SelfAttention(nn.Module):
    """
    Multi-head Self-Attention using PyTorch's built-in
    scaled_dot_product_attention. No causal masking.
    """
    def __init__(self, num_head, embed_dim, dropout=0.1, bias=True):
        super().__init__()
        assert embed_dim % num_head == 0, "n_embd must be divisible by n_head"
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.dropout_p = dropout

        # Linear to produce Q, K, V together
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # Output projection
        self.out = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3*n_embd)
        q, k, v = qkv.split(C, dim=-1)
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_head, self.head_dim).transpose(1, 2)

        attn_mask = (mask.unsqueeze(1) == mask.unsqueeze(2)).unsqueeze(1)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False
        )  # (B, n_head, T, head_dim)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + dropout
        y = self.out(y)
        y = self.resid_dropout(y)
        return y


class Block(nn.Module):
    """Transformer block: LN -> Self-Attn -> LN -> MLP."""
    def __init__(self, num_heads, embed_dim, dropout=0.1, bias=True):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim, bias=bias)
        self.attn = SelfAttention(num_heads, embed_dim, dropout, bias)
        self.ln2 = LayerNorm(embed_dim, bias=bias)
        # Replace the MLP class with a simple Sequential
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=bias),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, bias=bias),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True)
        )

    def forward(self, x, c, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # Self-Attention
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.ln1(x), shift_msa, scale_msa), mask=mask)

        # Feed-forward
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.ln2(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    """
    A simplified Transformer that:
      - Expects input shape (B, T, n_embd),
      - Stacks multiple Transformer blocks,
      - Returns shape (B, T, n_embd),
      - Has no embedding layers or projection head.
    """
    def __init__(self, num_layers, num_heads, in_dim, embed_dim, dropout=0., bias=True, use_pe=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_embedder = TimestepEmbedder(hidden_size=embed_dim)
        self.in_proj = nn.Linear(in_dim*2, embed_dim)
        self.blocks = nn.ModuleList([
            Block(num_heads, embed_dim, dropout, bias) for _ in range(num_layers)
        ])
        self.ln_f = LayerNorm(embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, in_dim)
        self.use_pe = use_pe
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, t, mask, x_sc=None):
        """
        x: (bs, n*pe_dim)
        t: (bs,)
        mask: (bs, n*pe_dim)
        Returns (bs, n, pe_dim).
        """
        if t.dim() == 0:
            t = torch.full((x.shape[0], ), t.item(), device=x.device)

        t = self.time_embedder(t.squeeze())                       # (bs, d)

        if x_sc is None:
            x_sc = torch.zeros_like(x)
        x = self.in_proj(torch.cat([x, x_sc], dim=-1))                        # (bs, n, d)

        if self.use_pe:
            # Positonal embedding
            token_index = torch.cumsum(mask, dim=-1, dtype=torch.int64) - 1
            pos_emb = get_pos_embedding(token_index, self.embed_dim)
            x = x + pos_emb

        x = x * mask.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, t, mask)

        x = self.ln_f(x)
        x = self.out_proj(x) * mask.unsqueeze(-1)
        return x
