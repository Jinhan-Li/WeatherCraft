from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


@dataclass
class Shape2D:
    """处理 2D 形状的工具类，可以接受 int、tuple、list。"""

    _shape: Union[int, Tuple[int, int], list]

    def __post_init__(self) -> None:
        if isinstance(self._shape, int):
            self.shape = (self._shape, self._shape)
        elif isinstance(self._shape, (list, tuple)) and len(self._shape) == 2:
            self.shape = (int(self._shape[0]), int(self._shape[1]))
        else:
            raise TypeError("Shape2D expects int or a 2-element tuple/list")

        self.height = self.shape[0]
        self.width = self.shape[1]


def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    """将特征图划分为窗口。"""

    batch, height, width, channels = x.shape
    window_h, window_w = window_size
    x = x.view(
        batch,
        height // window_h,
        window_h,
        width // window_w,
        window_w,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_h, window_w, channels)
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: Tuple[int, int],
    img_size: Tuple[int, int],
) -> torch.Tensor:
    """将窗口合并回特征图。"""

    height, width = img_size
    window_h, window_w = window_size
    channels = windows.shape[-1]
    x = windows.view(
        -1,
        height // window_h,
        width // window_w,
        window_h,
        window_w,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, channels)
    return x


def timestep_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """正弦时间步嵌入（Sinusoidal timestep embeddings）。"""

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / max(half, 1)
    )
    args = t[:, None].to(t.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    embedding = embedding.reshape(embedding.shape[0], 2, -1).flip(1).reshape(*embedding.shape)
    return embedding


class LatentEmbedding(nn.Module):
    """潜在嵌入层，用于处理时间步嵌入。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(dim, dim, bias=True)
        self.l2 = nn.Linear(dim, dim, bias=True)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return F.silu(self.l2(F.silu(self.l1(emb))))


class RMSNorm(nn.Module):
    """pre-RMSNorm。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class AdaptiveRMSNorm(nn.Module):
    """
    adaptive RMSNorm，生成 alpha / beta / gamma。

    alpha: 归一化后的缩放
    beta: 归一化后的偏移
    gamma: 残差分支门控
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)
        self.modulation = nn.Linear(dim, dim * 3, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha, beta, gamma = self.modulation(t).chunk(3, dim=-1)
        x = self.norm(x)
        x = x * (1.0 + alpha.unsqueeze(1)) + beta.unsqueeze(1)
        return x, gamma.unsqueeze(1)


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络。"""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up_proj = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up_proj)


class AxialRoPE2D(nn.Module):
    """窗口内的 2D axial rotary positional embedding。"""

    def __init__(
        self,
        head_dim: int,
        window_size: Tuple[int, int],
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.window_size = window_size

        rope_dim = head_dim - (head_dim % 4)
        half_dim = rope_dim // 2
        pair_dim = half_dim // 2
        self.rope_dim = rope_dim
        self.half_dim = half_dim
        self.pair_dim = pair_dim

        if pair_dim == 0:
            self.register_buffer("cos_y", torch.empty(1, 1, 1, 0), persistent=False)
            self.register_buffer("sin_y", torch.empty(1, 1, 1, 0), persistent=False)
            self.register_buffer("cos_x", torch.empty(1, 1, 1, 0), persistent=False)
            self.register_buffer("sin_x", torch.empty(1, 1, 1, 0), persistent=False)
            return

        window_h, window_w = window_size
        inv_freq = 1.0 / (base ** (torch.arange(pair_dim, dtype=torch.float32) / max(pair_dim, 1)))

        pos_y = torch.arange(window_h, dtype=torch.float32)
        pos_x = torch.arange(window_w, dtype=torch.float32)

        phase_y = torch.einsum("h,d->hd", pos_y, inv_freq)
        phase_x = torch.einsum("w,d->wd", pos_x, inv_freq)

        cos_y = torch.cos(phase_y).repeat_interleave(window_w, dim=0)
        sin_y = torch.sin(phase_y).repeat_interleave(window_w, dim=0)
        cos_x = torch.cos(phase_x).repeat(window_h, 1)
        sin_x = torch.sin(phase_x).repeat(window_h, 1)

        self.register_buffer("cos_y", cos_y.unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_y", sin_y.unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("cos_x", cos_x.unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_x", sin_x.unsqueeze(0).unsqueeze(0), persistent=False)

    @staticmethod
    def _apply_axis_rotary(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-1] == 0:
            return x

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope_dim == 0:
            return q, k

        q_rope, q_rest = q[..., : self.rope_dim], q[..., self.rope_dim :]
        k_rope, k_rest = k[..., : self.rope_dim], k[..., self.rope_dim :]

        q_y, q_x = q_rope[..., : self.half_dim], q_rope[..., self.half_dim :]
        k_y, k_x = k_rope[..., : self.half_dim], k_rope[..., self.half_dim :]

        q_y = self._apply_axis_rotary(q_y, self.cos_y.to(q.dtype), self.sin_y.to(q.dtype))
        q_x = self._apply_axis_rotary(q_x, self.cos_x.to(q.dtype), self.sin_x.to(q.dtype))
        k_y = self._apply_axis_rotary(k_y, self.cos_y.to(k.dtype), self.sin_y.to(k.dtype))
        k_x = self._apply_axis_rotary(k_x, self.cos_x.to(k.dtype), self.sin_x.to(k.dtype))

        q = torch.cat([q_y, q_x, q_rest], dim=-1)
        k = torch.cat([k_y, k_x, k_rest], dim=-1)
        return q, k


class Attention(nn.Module):
    """
    带 2D axial RoPE 的窗口注意力模块。

    使用标准 dot-product scale + RoPE：
    score = (q @ k^T) * (1 / sqrt(d))
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        window_size: Tuple[int, int],
        flash: bool,
    ) -> None:
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.flash = flash
        self.scale = head_dim**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)
        self.rope = AxialRoPE2D(head_dim, window_size)

    def forward(self, x: torch.Tensor, jvp: bool = False) -> torch.Tensor:
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (three h d) -> three b h n d", three=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.rope(q, k)

        if self.flash and not jvp and hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = rearrange(x, "b h n d -> b n (h d)")
        return self.wo(x)


class NeurBlock(nn.Module):
    """pre-RMSNorm + adaptive modulation + gated residual"""

    def __init__(
        self,
        dim: int,
        heads: int,
        hidden_dim: int,
        window_size: Tuple[int, int],
        flash: bool = True,
    ) -> None:
        super().__init__()
        head_dim = dim // heads
        self.attn_norm = AdaptiveRMSNorm(dim)
        self.attn = Attention(dim, heads, head_dim, window_size, flash)
        self.ffn_norm = AdaptiveRMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, jvp: bool = False) -> torch.Tensor:
        x_norm, gamma = self.attn_norm(x, t)
        x = x + gamma * self.attn(x_norm, jvp=jvp)

        x_norm, gamma = self.ffn_norm(x, t)
        x = x + gamma * self.ffn(x_norm)
        return x


class SwinTransformer(nn.Module):
    """shifted-window Swin 骨干"""

    def __init__(
        self,
        depth: int,
        dim: int,
        heads: int,
        window_size: Tuple[int, int],
        grid_size: Tuple[int, int],
        shift_size: Tuple[int, int],
        flash: bool,
    ) -> None:
        super().__init__()

        if grid_size[0] % window_size[0] != 0 or grid_size[1] % window_size[1] != 0:
            raise ValueError("grid_size must be divisible by window_size")

        self.window_size = window_size
        self.grid_size = grid_size
        self.shift_size = shift_size
        hidden_dim = int(8 / 3.0 * dim)

        self.layers = nn.ModuleList(
            [
                NeurBlock(
                    dim=dim,
                    heads=heads,
                    hidden_dim=hidden_dim,
                    window_size=window_size,
                    flash=flash,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        jvp: bool = False,
    ) -> torch.Tensor:
        shift_h, shift_w = self.shift_size
        do_shift = any(self.shift_size)

        repeat_factor = (self.grid_size[0] // self.window_size[0]) * (
            self.grid_size[1] // self.window_size[1]
        )
        t_expanded = t.repeat_interleave(repeat_factor, dim=0)

        for layer_idx, block in enumerate(self.layers):
            x_view = x.view(-1, self.grid_size[0], self.grid_size[1], x.shape[-1])
            _, height, width, channels = x_view.shape

            if do_shift and layer_idx % 2 != 0:
                x_view = torch.roll(x_view, shifts=(-shift_h, -shift_w), dims=(1, 2))

            x_windows = window_partition(x_view, self.window_size)
            x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], channels)
            x_windows = block(x_windows, t_expanded, jvp=jvp)

            x_windows = x_windows.view(-1, self.window_size[0], self.window_size[1], channels)
            x_view = window_reverse(x_windows, self.window_size, (height, width))

            if do_shift and layer_idx % 2 != 0:
                x_view = torch.roll(x_view, shifts=(shift_h, shift_w), dims=(1, 2))

            x = x_view.view(-1, height * width, channels)

        return x


class PatchEmbedding(nn.Module):
    """图像块嵌入层。"""

    def __init__(self, in_channels: int, patch_size: Tuple[int, int], dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        patch_h, patch_w = patch_size
        self.emb = nn.Linear(in_channels * patch_h * patch_w, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
        )
        return self.emb(x)


class OutputHead(nn.Module):
    """输出头，将 tokens 重构为图像。"""

    def __init__(
        self,
        dim: int,
        out_channels: int,
        patch_size: Tuple[int, int],
        grid_size: Tuple[int, int],
    ) -> None:
        super().__init__()
        patch_h, patch_w = patch_size
        grid_h, grid_w = grid_size

        self.head = nn.Sequential(
            nn.Linear(dim, out_channels * patch_h * patch_w, bias=False),
            Rearrange(
                "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                p1=patch_h,
                p2=patch_w,
                h=grid_h,
                w=grid_w,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SwinV2_Tyche(nn.Module):

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int], list],
        in_channels: int,
        out_channels: int,
        window_size: Union[int, Tuple[int, int], list],
        shift_size: Union[int, Tuple[int, int], list],
        patch_size: Union[int, Tuple[int, int], list],
        depth: int = 6,
        dim: int = 512,
        heads: int = 12,
        flash: bool = True,
        logvar: bool = False,
        timestep_weight: float = 1.0,
    ) -> None:
        super().__init__()

        image_height, image_width = Shape2D(img_resolution).shape
        patch_height, patch_width = Shape2D(patch_size).shape
        window_height, window_width = Shape2D(window_size).shape
        shift = Shape2D(shift_size).shape
        grid_h = image_height // patch_height
        grid_w = image_width // patch_width
        grid_size = (grid_h, grid_w)
        total_input_channels = in_channels * 2

        self.img_resolution = (image_height, image_width)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.timestep_weight = timestep_weight

        self.pos_embed = nn.Parameter(torch.randn(1, grid_h * grid_w, dim) * 0.02)
        self.patch_embed = PatchEmbedding(total_input_channels, (patch_height, patch_width), dim)
        self.latent_embed = LatentEmbedding(dim)
        self.latent_embed_r = LatentEmbedding(dim)
        self.logvar_embed = nn.Linear(dim, 1) if logvar else None
        self.transformer = SwinTransformer(
            depth=depth,
            dim=dim,
            heads=heads,
            window_size=(window_height, window_width),
            grid_size=grid_size,
            shift_size=shift,
            flash=flash,
        )
        self.head = OutputHead(dim, out_channels, (patch_height, patch_width), grid_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化模型权重。"""

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "modulation" in name or "head" in name:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        jvp: bool = False,
        return_logvar: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.shape[0]
        x = torch.cat((x, y), dim=1)

        x = self.patch_embed(x)
        x = x + self.pos_embed

        if t.dim() == 0 or (t.dim() == 1 and t.size(0) == 1):
            t = t.repeat(batch_size)
        t_emb = timestep_embedding(t * self.timestep_weight, x.size(2))
        t_emb = self.latent_embed(t_emb)

        if r is not None:
            if r.dim() == 0 or (r.dim() == 1 and r.size(0) == 1):
                r = r.repeat(batch_size)
            r_emb = timestep_embedding(r * self.timestep_weight, t_emb.size(1))
            r_emb = self.latent_embed_r(r_emb)
            t_emb = t_emb + r_emb

        x = self.transformer(x, t_emb, jvp=jvp)
        x = self.head(x)

        if self.logvar_embed is not None and return_logvar:
            logvar = self.logvar_embed(t_emb).squeeze(-1)
            return x, logvar

        return x
