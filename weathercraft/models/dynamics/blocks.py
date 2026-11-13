from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop-path probability must be in [0, 1)")
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden_width: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BlockCausalAttention(nn.Module):
    """Exact scale-block causal attention without an ``L x L`` mask.

    Queries in one scale attend to every key in the same or a coarser scale.
    Processing each query block separately lets PyTorch use memory-efficient
    scaled-dot-product attention even when the finest pyramid is large.
    """

    def __init__(self, width: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if num_heads < 1 or width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_width = width // num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.num_heads, self.head_width).transpose(1, 2)

    def forward(self, x: torch.Tensor, block_ends: tuple[int, ...]) -> torch.Tensor:
        if not block_ends or block_ends[-1] != x.shape[1]:
            raise ValueError("block_ends must partition the complete sequence")
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query, key, value = self._heads(query), self._heads(key), self._heads(value)
        pieces, start = [], 0
        dropout = self.dropout if self.training else 0.0
        for end in block_ends:
            attended = F.scaled_dot_product_attention(
                query[:, :, start:end],
                key[:, :, :end],
                value[:, :, :end],
                dropout_p=dropout,
                is_causal=False,
            )
            pieces.append(attended)
            start = end
        merged = torch.cat(pieces, dim=2).transpose(1, 2).contiguous().flatten(2)
        return self.output(merged)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden_width = max(width, int(width * mlp_ratio))
        self.attn_norm = nn.LayerNorm(width)
        self.attn = BlockCausalAttention(width, num_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = FeedForward(width, hidden_width, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, block_ends: tuple[int, ...]) -> torch.Tensor:
        normalized = self.attn_norm(x)
        attended = self.attn(normalized, block_ends)
        x = x + self.drop_path(self.attn_dropout(attended))
        return x + self.drop_path(self.ffn(self.ffn_norm(x)))


class TransformerStack(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        if not 0.0 <= drop_path < 1.0:
            raise ValueError("drop_path must be in [0, 1)")
        rates = torch.linspace(0.0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            TransformerBlock(width, num_heads, mlp_ratio, dropout, rate) for rate in rates
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, block_ends: tuple[int, ...]) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, block_ends)
        return self.norm(x)
