from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

SpatialFactor = int | Sequence[int]


def normalize_spatial_factor(factor: SpatialFactor) -> tuple[int, int]:
    """Return a ``(latitude, longitude)`` factor while accepting legacy ints."""
    if isinstance(factor, int):
        values = (factor, factor)
    else:
        values = tuple(int(value) for value in factor)
        if len(values) != 2:
            raise ValueError("spatial factor must be an int or [latitude, longitude]")
    if min(values) < 1:
        raise ValueError("spatial factors must be positive")
    return values


def _groups(channels: int, maximum: int = 32) -> int:
    groups = min(channels, maximum)
    while channels % groups:
        groups -= 1
    return groups


class SphericalConv2d(nn.Module):
    """Longitude-circular and latitude-reflecting convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad:
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, self.pad, self.pad), mode="reflect")
        return self.conv(x)


class FramewiseSphericalConv2d(nn.Module):
    """Apply one spatial convolution to each within-day component."""

    def __init__(self, frames: int, in_channels: int, out_channels: int):
        super().__init__()
        self.frames, self.in_channels, self.out_channels = frames, in_channels, out_channels
        self.conv = SphericalConv2d(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels != self.frames * self.in_channels:
            raise ValueError("framewise convolution received the wrong channel count")
        hidden = self.conv(x.reshape(batch * self.frames, self.in_channels, height, width))
        return hidden.reshape(batch, self.frames * self.out_channels, height, width)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = _groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = SphericalConv2d(channels, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = SphericalConv2d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class TemporalMixBlock(nn.Module):
    """A light residual temporal mixer inserted between spatial blocks."""

    def __init__(self, channels: int, frames: int, kernel_size: int = 3):
        super().__init__()
        if frames < 2 or channels % frames or kernel_size % 2 != 1:
            raise ValueError("temporal mixing needs divisible channels and an odd kernel")
        per_frame = channels // frames
        self.frames = frames
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.temporal = nn.Conv3d(
            per_frame,
            per_frame,
            (kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=per_frame,
        )
        self.channel = nn.Conv3d(per_frame, per_frame, 1)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        hidden = self.norm(x).reshape(batch, self.frames, channels // self.frames, height, width)
        hidden = hidden.transpose(1, 2)
        hidden = self.channel(F.silu(self.temporal(hidden)))
        return x + self.gain * hidden.transpose(1, 2).reshape_as(x)


class DownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: SpatialFactor, blocks: int = 1):
        super().__init__()
        if blocks < 1:
            raise ValueError("downsample block count must be positive")
        self.factor = normalize_spatial_factor(factor)
        self.residual = (
            ResBlock(in_channels)
            if blocks == 1
            else nn.Sequential(*(ResBlock(in_channels) for _ in range(blocks)))
        )
        self.projection = SphericalConv2d(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(self.residual(x))
        return F.avg_pool2d(x, self.factor) if self.factor != (1, 1) else x


class UpStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: SpatialFactor, blocks: int = 1):
        super().__init__()
        if blocks < 1:
            raise ValueError("upsample block count must be positive")
        self.factor = normalize_spatial_factor(factor)
        self.residual = (
            ResBlock(in_channels)
            if blocks == 1
            else nn.Sequential(*(ResBlock(in_channels) for _ in range(blocks)))
        )
        self.projection = SphericalConv2d(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual(x)
        if self.factor != (1, 1):
            x = F.interpolate(x, scale_factor=self.factor, mode="nearest")
        return self.projection(x)
