from __future__ import annotations

import torch
from torch import nn


class TemporalResidualAdapter(nn.Module):
    """Pack frames as raw values or as an anchor followed by increments."""

    def __init__(
        self,
        frames: int,
        channels: int,
        delta_scale: torch.Tensor | None = None,
        representation: str = "increments",
    ):
        super().__init__()
        if frames < 1 or channels < 1:
            raise ValueError("frames and channels must be positive")
        if representation not in {"raw", "increments"}:
            raise ValueError("representation must be 'raw' or 'increments'")
        if representation == "raw" and delta_scale is not None:
            raise ValueError("raw-frame representation cannot use delta_scale")
        self.frames = frames
        self.channels = channels
        self.representation = representation
        if frames == 1:
            scale = torch.empty(0, channels)
        elif delta_scale is None:
            scale = torch.ones(frames - 1, channels)
        else:
            scale = torch.as_tensor(delta_scale, dtype=torch.float32)
            if tuple(scale.shape) != (frames - 1, channels):
                raise ValueError(
                    f"delta_scale must be {(frames - 1, channels)}, got {tuple(scale.shape)}"
                )
            if not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("delta_scale must be finite and strictly positive")
        self.register_buffer("delta_scale", scale.clamp_min(1e-6))

    @property
    def packed_channels(self) -> int:
        return self.frames * self.channels

    def pack(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 5 or tuple(field.shape[1:3]) != (self.frames, self.channels):
            raise ValueError(
                f"expected [B,{self.frames},{self.channels},H,W], got {tuple(field.shape)}"
            )
        if self.frames == 1 or self.representation == "raw":
            residual = field
        else:
            scale = self.delta_scale[None, :, :, None, None].to(field)
            residual = torch.cat((field[:, :1], (field[:, 1:] - field[:, :-1]) / scale), dim=1)
        return residual.flatten(1, 2)

    def unpack(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 4 or packed.shape[1] != self.packed_channels:
            raise ValueError(f"expected [B,{self.packed_channels},H,W], got {tuple(packed.shape)}")
        residual = packed.unflatten(1, (self.frames, self.channels))
        if self.frames == 1 or self.representation == "raw":
            return residual
        scale = self.delta_scale[None, :, :, None, None].to(residual)
        increments = residual[:, 1:] * scale
        return torch.cat((residual[:, :1], residual[:, :1] + increments.cumsum(dim=1)), dim=1)


class LearnedTemporalSlots(nn.Module):
    """Learn a full-width 4-to-2 temporal packing before slot-wise FSQ heads."""

    def __init__(self, channels: int, slots: int, groups: int, axes: int, decoder_scales: int = 1):
        super().__init__()
        if min(channels, slots, groups, axes, decoder_scales) < 1 or channels % slots:
            raise ValueError("channels must be divisible by a positive slot count")
        self.slots, self.groups, self.axes = slots, groups, axes
        self.decoder_scales = decoder_scales
        self.slot_width = channels // slots
        # Square, identity-initialized maps retain the full pre-FSQ width.
        self.pack = nn.Conv2d(channels, channels, 1)
        self.unpack = nn.Conv2d(channels, channels, 1)
        nn.init.dirac_(self.pack.weight)
        nn.init.dirac_(self.unpack.weight)
        nn.init.zeros_(self.pack.bias)
        nn.init.zeros_(self.unpack.bias)
        self.to_latent = nn.Conv2d(self.slot_width, groups * axes, 1)
        base_decoder = nn.Conv2d(groups * axes, self.slot_width, 1)
        if decoder_scales == 1:
            self.from_latent = base_decoder
        else:
            self.from_latent = nn.Conv2d(decoder_scales * groups * axes, self.slot_width, 1)
            # At initialization, concatenating scales is exactly equivalent to
            # summing them before the original per-slot decoder projection.
            with torch.no_grad():
                self.from_latent.weight.copy_(base_decoder.weight.repeat(1, decoder_scales, 1, 1))
                self.from_latent.bias.copy_(base_decoder.bias)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = hidden.shape
        slots = self.pack(hidden).reshape(batch * self.slots, self.slot_width, height, width)
        latent = self.to_latent(slots).reshape(
            batch, self.slots, self.groups, self.axes, height, width
        )
        # The FSQ codec expects [group, axis, slot] rather than slot-major channels.
        return latent.permute(0, 2, 3, 1, 4, 5).reshape(batch, -1, height, width)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = latent.shape
        if channels != self.decoder_scales * self.slots * self.groups * self.axes:
            raise ValueError("latent channel count does not match the slot codec")
        slots = latent.reshape(
            batch, self.decoder_scales, self.groups, self.axes, self.slots, height, width
        )
        slots = slots.permute(0, 4, 1, 2, 3, 5, 6).reshape(
            batch * self.slots, self.decoder_scales * self.groups * self.axes, height, width
        )
        hidden = self.from_latent(slots).reshape(batch, -1, height, width)
        return self.unpack(hidden)
