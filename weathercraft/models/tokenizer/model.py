from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import (
    DownStage,
    FramewiseSphericalConv2d,
    ResBlock,
    SpatialFactor,
    SphericalConv2d,
    TemporalMixBlock,
    UpStage,
    _groups,
    normalize_spatial_factor,
)
from .quantizer import QuantizerOutput, ResidualVectorQuantizer, TokenPyramid
from .temporal import LearnedTemporalSlots, TemporalResidualAdapter


class SphericalAutoencoder(nn.Module):
    """WeatherFSQ-style unified encoder/decoder without skip paths."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        widths: Sequence[int],
        spatial_factors: Sequence[SpatialFactor],
        latent_channels: int,
        mid_blocks: int = 2,
        base_latent_channels: int | None = None,
        frames: int = 1,
        temporal_mix: bool = False,
        stage_blocks: int = 1,
        temporal_kernel_size: int = 3,
        slot_structure: tuple[int, int, int] | None = None,
        decoder_scales: int = 1,
    ):
        super().__init__()
        widths = tuple(map(int, widths))
        factors = tuple(normalize_spatial_factor(value) for value in spatial_factors)
        if not widths or len(widths) != len(factors) or min(mid_blocks, stage_blocks) < 1:
            raise ValueError("widths/factors must match and all block counts must be positive")
        if temporal_mix and (
            frames < 2
            or in_channels % frames
            or out_channels % frames
            or any(width % frames for width in widths)
        ):
            raise ValueError("temporal_mix requires frame-divisible input, output and widths")
        if slot_structure is not None:
            slots, groups, axes = map(int, slot_structure)
            if not temporal_mix or slots > frames or slots * groups * axes != latent_channels:
                raise ValueError("learned slots require temporal mixing and matching FSQ channels")
        if decoder_scales < 1 or (decoder_scales > 1 and slot_structure is None):
            raise ValueError("multi-scale decoder input requires learned temporal slots")
        self.conv_in = (
            FramewiseSphericalConv2d(frames, in_channels // frames, widths[0] // frames)
            if temporal_mix
            else SphericalConv2d(in_channels, widths[0])
        )

        def mixer(channels: int) -> nn.Module:
            return (
                TemporalMixBlock(channels, frames, temporal_kernel_size)
                if temporal_mix
                else nn.Identity()
            )

        self.temporal_in = mixer(widths[0])
        self.encoder_time_embedding = (
            nn.Parameter(torch.empty(1, widths[0], 1, 1)) if slot_structure else None
        )
        if self.encoder_time_embedding is not None:
            nn.init.normal_(self.encoder_time_embedding, std=0.02)
        enc_out = tuple(widths[min(i + 1, len(widths) - 1)] for i in range(len(factors)))
        enc_in = (widths[0],) + enc_out[:-1]
        self.down = nn.ModuleList(
            DownStage(a, b, f, stage_blocks) for a, b, f in zip(enc_in, enc_out, factors)
        )
        self.temporal_down = nn.ModuleList(mixer(channels) for channels in enc_out)
        bottleneck = enc_out[-1]
        self.encoder_mid = nn.ModuleList(ResBlock(bottleneck) for _ in range(mid_blocks))
        self.temporal_encoder_mid = nn.ModuleList(mixer(bottleneck) for _ in range(mid_blocks))
        self.slot_bridge = (
            LearnedTemporalSlots(bottleneck, *slot_structure, decoder_scales=decoder_scales)
            if slot_structure
            else None
        )
        self.to_latent = (
            nn.Conv2d(bottleneck, latent_channels, 1) if slot_structure is None else None
        )
        self.from_latent = (
            nn.Conv2d(latent_channels, bottleneck, 1) if slot_structure is None else None
        )
        if base_latent_channels is not None:
            base_latent_channels = int(base_latent_channels)
            if not 1 <= base_latent_channels <= latent_channels:
                raise ValueError("base_latent_channels must lie in [1, latent_channels]")
            if base_latent_channels < latent_channels:
                # The extra FSQ branch starts as an additive no-op, so the
                # wider decoder contains the one-branch model as a submodel.
                if self.slot_bridge is None:
                    nn.init.zeros_(self.from_latent.weight[:, base_latent_channels:])
                else:
                    slots = slot_structure[0]
                    if base_latent_channels % slots:
                        raise ValueError("base latent channels must divide evenly across slots")
                    branch_width = latent_channels // slots
                    for scale in range(decoder_scales):
                        start = scale * branch_width + base_latent_channels // slots
                        end = (scale + 1) * branch_width
                        nn.init.zeros_(self.slot_bridge.from_latent.weight[:, start:end])
        self.decoder_mid = nn.ModuleList(ResBlock(bottleneck) for _ in range(mid_blocks))
        self.temporal_decoder_mid = nn.ModuleList(mixer(bottleneck) for _ in range(mid_blocks))
        dec_out = tuple(reversed(enc_in))
        dec_in = (bottleneck,) + dec_out[:-1]
        self.up = nn.ModuleList(
            UpStage(a, b, f, stage_blocks) for a, b, f in zip(dec_in, dec_out, reversed(factors))
        )
        self.temporal_up = nn.ModuleList(mixer(channels) for channels in dec_out)
        self.norm_out = nn.GroupNorm(_groups(widths[0]), widths[0])
        self.decoder_time_embedding = (
            nn.Parameter(torch.empty(1, widths[0], 1, 1)) if slot_structure else None
        )
        if self.decoder_time_embedding is not None:
            nn.init.normal_(self.decoder_time_embedding, std=0.02)
        self.conv_out = (
            FramewiseSphericalConv2d(frames, widths[0] // frames, out_channels // frames)
            if temporal_mix
            else SphericalConv2d(widths[0], out_channels)
        )

    def encode(self, field: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_in(field)
        if self.encoder_time_embedding is not None:
            hidden = hidden + self.encoder_time_embedding
        hidden = self.temporal_in(hidden)
        for stage, mix in zip(self.down, self.temporal_down):
            hidden = mix(stage(hidden))
        for block, mix in zip(self.encoder_mid, self.temporal_encoder_mid):
            hidden = mix(block(hidden))
        return (
            self.to_latent(hidden) if self.slot_bridge is None else self.slot_bridge.encode(hidden)
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = (
            self.from_latent(latent)
            if self.slot_bridge is None
            else self.slot_bridge.decode(latent)
        )
        for block, mix in zip(self.decoder_mid, self.temporal_decoder_mid):
            hidden = mix(block(hidden))
        for stage, mix in zip(self.up, self.temporal_up):
            hidden = mix(stage(hidden))
        if self.decoder_time_embedding is not None:
            hidden = hidden + self.decoder_time_embedding
        return self.conv_out(F.silu(self.norm_out(hidden)))


@dataclass
class TokenizerOutput:
    reconstruction: torch.Tensor
    tokens: TokenPyramid
    latent: torch.Tensor
    quantized_latent: torch.Tensor
    vq_loss: torch.Tensor
    codebook_loss: torch.Tensor
    adapter_loss: torch.Tensor
    commitment_loss: torch.Tensor
    usage: torch.Tensor
    residual_energy: torch.Tensor
    prefix_mse: torch.Tensor
    stage_codebook_mse: torch.Tensor
    stage_adapter_mse: torch.Tensor
    contribution_energy: torch.Tensor
    final_quant_mse: torch.Tensor
    latent_energy: torch.Tensor
    quantized_energy: torch.Tensor
    final_cosine: torch.Tensor
    code_rms: torch.Tensor


class FieldTokenizer(nn.Module):
    def __init__(
        self,
        frames: int,
        channels: int,
        widths: Sequence[int],
        spatial_factors: Sequence[SpatialFactor],
        scale_shapes: Sequence[Sequence[int]],
        num_codebooks: int = 2,
        codebook_dim: int = 128,
        vocab_size: int = 4096,
        mid_blocks: int = 2,
        beta: float = 0.25,
        residual_ratio: float = 0.5,
        shared_adapters: int = 4,
        commitment_mode: str = "cumulative",
        commitment_scale_weights: Sequence[float] | None = None,
        delta_scale: torch.Tensor | None = None,
        codebook_objective: str = "cumulative_prefix",
        adapter_loss_weight: float = 1.0,
        quantizer: nn.Module | None = None,
        temporal_mix: bool = False,
        stage_blocks: int = 1,
        temporal_kernel_size: int = 3,
        temporal_representation: str = "increments",
        learnable_slots: bool = False,
        decoder_scale_fusion: str = "sum",
    ):
        super().__init__()
        self.frames, self.channels = int(frames), int(channels)
        if decoder_scale_fusion not in {"sum", "concat"}:
            raise ValueError("decoder_scale_fusion must be sum or concat")
        self.decoder_scale_fusion = decoder_scale_fusion
        self.adapter = TemporalResidualAdapter(
            self.frames, self.channels, delta_scale, representation=temporal_representation
        )
        self.quantizer = quantizer or ResidualVectorQuantizer(
            scale_shapes=scale_shapes,
            num_codebooks=num_codebooks,
            codebook_dim=codebook_dim,
            vocab_size=vocab_size,
            beta=beta,
            residual_ratio=residual_ratio,
            shared_adapters=shared_adapters,
            commitment_mode=commitment_mode,
            commitment_scale_weights=commitment_scale_weights,
            codebook_objective=codebook_objective,
            adapter_loss_weight=adapter_loss_weight,
        )
        if learnable_slots and not all(
            hasattr(self.quantizer, name)
            for name in ("num_slots", "parallel_codebooks", "fsq_axis_dim")
        ):
            raise ValueError("learnable_slots requires the structured multiscale FSQ codec")
        slot_structure = (
            (
                self.quantizer.num_slots,
                self.quantizer.parallel_codebooks,
                self.quantizer.fsq_axis_dim,
            )
            if learnable_slots
            else None
        )
        if decoder_scale_fusion == "concat" and (
            not learnable_slots or not hasattr(self.quantizer, "tokens_to_scale_latents")
        ):
            raise ValueError("concat decoder fusion requires learned slots and multiscale FSQ")
        self.autoencoder = SphericalAutoencoder(
            in_channels=self.adapter.packed_channels,
            out_channels=self.adapter.packed_channels,
            widths=widths,
            spatial_factors=spatial_factors,
            latent_channels=self.quantizer.latent_channels,
            mid_blocks=mid_blocks,
            base_latent_channels=getattr(self.quantizer, "latent_branch_channels", None),
            frames=self.frames,
            temporal_mix=temporal_mix,
            stage_blocks=stage_blocks,
            temporal_kernel_size=temporal_kernel_size,
            slot_structure=slot_structure,
            decoder_scales=len(scale_shapes) if decoder_scale_fusion == "concat" else 1,
        )

    def encode_latent(self, field: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(self.adapter.pack(field))

    @torch.no_grad()
    def encode(self, field: torch.Tensor) -> TokenPyramid:
        return self.quantizer(self.encode_latent(field)).tokens

    def decode(self, tokens: TokenPyramid, max_scales: int | None = None) -> torch.Tensor:
        if self.decoder_scale_fusion == "concat":
            latent = torch.cat(
                self.quantizer.tokens_to_scale_latents(tokens, max_scales=max_scales), dim=1
            )
        else:
            latent = self.quantizer.tokens_to_latent(tokens, max_scales=max_scales)
        return self.adapter.unpack(self.autoencoder.decode(latent))

    def forward(self, field: torch.Tensor) -> TokenizerOutput:
        latent = self.encode_latent(field)
        quantized: QuantizerOutput = self.quantizer(latent)
        decoder_latent = (
            torch.cat(quantized.scale_latents, dim=1)
            if self.decoder_scale_fusion == "concat"
            else quantized.quantized
        )
        reconstruction = self.adapter.unpack(self.autoencoder.decode(decoder_latent))
        return TokenizerOutput(
            reconstruction=reconstruction,
            tokens=quantized.tokens,
            latent=latent,
            quantized_latent=quantized.quantized,
            vq_loss=quantized.loss,
            codebook_loss=quantized.codebook_loss,
            adapter_loss=quantized.adapter_loss,
            commitment_loss=quantized.commitment_loss,
            usage=quantized.usage,
            residual_energy=quantized.residual_energy,
            prefix_mse=quantized.prefix_mse,
            stage_codebook_mse=quantized.stage_codebook_mse,
            stage_adapter_mse=quantized.stage_adapter_mse,
            contribution_energy=quantized.contribution_energy,
            final_quant_mse=quantized.final_quant_mse,
            latent_energy=quantized.latent_energy,
            quantized_energy=quantized.quantized_energy,
            final_cosine=quantized.final_cosine,
            code_rms=quantized.code_rms,
        )
