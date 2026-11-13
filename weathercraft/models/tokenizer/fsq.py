"""Finite scalar quantization used by the WeatherFSQ tokenizer.

This is the small, dependency-free core of ``WeatherFSQ/fsq.py``.  Keeping the
mixed-radix index convention identical is important: with levels
``[8, 8, 8, 5, 5, 5]`` every six-dimensional code is one uint16 id in
``[0, 64000)``.
"""

from __future__ import annotations

from math import prod
from typing import Sequence

import torch
from torch import nn


def round_ste(value: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass and use the identity gradient."""

    rounded = value.round()
    return value + (rounded - value).detach()


class FiniteScalarQuantizer(nn.Module):
    """Parameter-free FSQ for channel-first tensors ``[B, D, ...]``.

    The bounding, normalization, and mixed-radix indexing equations are the
    same as the FSQ implementation used by the best WeatherFSQ checkpoint.
    """

    def __init__(self, levels: Sequence[int], force_quantization_f32: bool = True):
        super().__init__()
        levels = tuple(int(level) for level in levels)
        if not levels or any(level < 2 for level in levels):
            raise ValueError("FSQ levels must contain integers >= 2")
        basis = []
        running = 1
        for level in levels:
            basis.append(running)
            running *= level
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int32), persistent=False)
        self.register_buffer("_basis", torch.tensor(basis, dtype=torch.int32), persistent=False)
        self.codebook_dim = len(levels)
        self.codebook_size = prod(levels)
        self.force_quantization_f32 = bool(force_quantization_f32)

    @property
    def levels(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._levels.tolist())

    def bound(self, value: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        levels = self._levels.to(device=value.device, dtype=value.dtype)
        half_l = (levels - 1) * (1.0 + eps) / 2.0
        offset = torch.where((self._levels % 2).to(value.device) == 0, 0.5, 0.0).to(value)
        shift = torch.atanh(offset / half_l)
        return torch.tanh(value + shift) * half_l - offset

    def quantize(self, value: torch.Tensor) -> torch.Tensor:
        levels = self._levels.to(device=value.device)
        half_width = (levels // 2).to(dtype=value.dtype)
        return round_ste(self.bound(value)) / half_width

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert normalized codes ``[..., D]`` to composite ids."""

        if codes.shape[-1] != self.codebook_dim:
            raise ValueError(f"expected code dimension {self.codebook_dim}, got {codes.shape[-1]}")
        levels = self._levels.to(device=codes.device)
        half_width = (levels // 2).to(dtype=codes.dtype)
        level_indices = codes * half_width + half_width
        basis = self._basis.to(device=codes.device, dtype=level_indices.dtype)
        return (level_indices * basis).sum(dim=-1).to(torch.int32)

    def indices_to_level_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(torch.int64).unsqueeze(-1)
        basis = self._basis.to(device=indices.device, dtype=torch.int64)
        levels = self._levels.to(device=indices.device, dtype=torch.int64)
        return (indices // basis) % levels

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode composite ids to normalized channel-first codes."""

        if indices.numel() and (indices.min() < 0 or indices.max() >= self.codebook_size):
            raise ValueError(f"FSQ index outside [0,{self.codebook_size})")
        level_indices = self.indices_to_level_indices(indices)
        levels = self._levels.to(device=indices.device)
        half_width = (levels // 2).to(dtype=torch.float32)
        codes = (level_indices.to(torch.float32) - half_width) / half_width
        return codes.movedim(-1, 1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if value.ndim < 3 or value.shape[1] != self.codebook_dim:
            raise ValueError(
                f"expected channel-first [B,{self.codebook_dim},...], got {tuple(value.shape)}"
            )
        channel_last = value.movedim(1, -1)
        original_dtype = channel_last.dtype
        work = channel_last.float() if self.force_quantization_f32 else channel_last
        # WeatherFSQ forms mixed-radix ids while the quantized values are
        # still fp32, then restores the activation dtype.  Keeping this order
        # matters for exact compatibility with configurable level sets.
        work_codes = self.quantize(work)
        indices = self.codes_to_indices(work_codes)
        codes = work_codes.to(original_dtype)
        return codes.movedim(-1, 1), indices
