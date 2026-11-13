"""Multi-scale FSQ with configurable parallel representation groups."""

from __future__ import annotations

from math import prod
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .fsq import FiniteScalarQuantizer
from .quantizer import QuantizerOutput, TokenPyramid


def _resize(value: torch.Tensor, shape: tuple[int, int], mode: str) -> torch.Tensor:
    """Resize H/W of ``[B,group,axis,slot,H,W]`` only."""

    if tuple(value.shape[-2:]) == shape:
        return value
    batch, groups, axes, slots, height, width = value.shape
    flat = value.permute(0, 1, 3, 2, 4, 5).reshape(batch * groups * slots, axes, height, width)
    if mode == "area":
        flat = F.interpolate(flat, size=shape, mode=mode)
    else:
        flat = F.interpolate(flat, size=shape, mode=mode, align_corners=False)
    return flat.reshape(batch, groups, slots, axes, *shape).permute(0, 1, 3, 2, 4, 5)


class MultiScaleFSQQuantizer(nn.Module):
    """FSQ analogue of ClimateAR's multi-codebook, next-scale bottleneck.

    Each parallel group owns six independent FSQ axes per temporal slot.  The
    widest configured stage fixes the latent width.  A stage may activate only
    a prefix of those groups; inactive token channels are stored as zero and
    excluded from dynamics loss.  With one 60x120 stage and one group this is
    exactly the WeatherFSQ codec geometry (12 latent channels, 14,400 tokens).
    """

    def __init__(
        self,
        levels: Sequence[int],
        slots: int,
        scale_shapes: Sequence[Sequence[int]],
        codebooks_per_scale: Sequence[int] | int,
        scale_gains: Sequence[Sequence[float] | float] | None = None,
    ):
        super().__init__()
        self.levels = tuple(int(level) for level in levels)
        self.num_slots = int(slots)
        self.scale_shapes = tuple((int(h), int(w)) for h, w in scale_shapes)
        if isinstance(codebooks_per_scale, int):
            active = (int(codebooks_per_scale),) * len(self.scale_shapes)
        else:
            active = tuple(int(value) for value in codebooks_per_scale)
        if self.num_slots < 1 or not self.scale_shapes:
            raise ValueError("slots and scale_shapes must be non-empty and positive")
        if any(min(shape) < 1 for shape in self.scale_shapes):
            raise ValueError("scale_shapes must contain positive [height,width] pairs")
        if any(
            a[0] > b[0] or a[1] > b[1] for a, b in zip(self.scale_shapes, self.scale_shapes[1:])
        ):
            raise ValueError("scale_shapes must be coarse-to-fine")
        if len(active) != len(self.scale_shapes) or any(value < 1 for value in active):
            raise ValueError("codebooks_per_scale must be positive with one value per scale")

        self.codebooks_per_scale = active
        self.parallel_codebooks = max(active)
        self.fsq_axis_dim = len(self.levels)
        self.latent_branch_channels = self.num_slots * self.fsq_axis_dim
        self.latent_channels = self.parallel_codebooks * self.latent_branch_channels
        # Existing cache/dynamics call this token-channel axis num_codebooks.
        self.num_codebooks = self.parallel_codebooks * self.num_slots
        self.codebook_dim = self.fsq_axis_dim
        self.vocab_size = prod(self.levels)
        if self.vocab_size > 65536:
            raise ValueError("composite FSQ ids must fit the uint16 token cache")

        if scale_gains is None:
            gains = torch.ones(len(self.scale_shapes), self.fsq_axis_dim, dtype=torch.float32)
        else:
            rows = []
            for scale, value in enumerate(scale_gains):
                row = torch.as_tensor(value, dtype=torch.float32).flatten()
                if row.numel() == 1:
                    row = row.repeat(self.fsq_axis_dim)
                if row.numel() != self.fsq_axis_dim or (row <= 0).any():
                    raise ValueError(
                        f"scale_gains[{scale}] must be positive scalar or {self.fsq_axis_dim}-vector"
                    )
                rows.append(row)
            if len(rows) != len(self.scale_shapes):
                raise ValueError("scale_gains must have one entry per scale")
            gains = torch.stack(rows)
        self.register_buffer("scale_gains", gains, persistent=True)
        self.register_buffer(
            "active_token_channels",
            torch.tensor([value * self.num_slots for value in active], dtype=torch.int64),
            persistent=True,
        )
        self.fsq_internal_num_codebooks = 1
        self.quantizer_type = "multiscale_fsq"
        self.fsq = FiniteScalarQuantizer(self.levels)
        self.beta = 0.0
        self.adapter_loss_weight = 0.0
        self.codebook_objective = "finite_scalar"

    @property
    def prediction_vocab_sizes(self) -> tuple[int, int]:
        if len(self.levels) != 6:
            raise ValueError("A/B dynamics factorization currently requires six FSQ axes")
        return prod(self.levels[:3]), prod(self.levels[3:])

    def codebook_parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def _structured(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4:
            raise ValueError(f"expected latent [B,C,H,W], got {tuple(latent.shape)}")
        batch, channels, height, width = latent.shape
        if channels != self.latent_channels or self.scale_shapes[-1] != (height, width):
            raise ValueError(
                f"expected [B,{self.latent_channels},{self.scale_shapes[-1][0]},"
                f"{self.scale_shapes[-1][1]}], got {tuple(latent.shape)}"
            )
        # WeatherFSQ's temporal tokenizer exposes [B,axis,slot,H,W].  Prefixing
        # that exact axis-major layout with a parallel-group dimension makes
        # the first 12 flattened channels codec-equivalent to the old model.
        return latent.view(
            batch,
            self.parallel_codebooks,
            self.fsq_axis_dim,
            self.num_slots,
            height,
            width,
        )

    @staticmethod
    def _flat(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0], -1, *value.shape[-2:])

    def latent_diagnostic_view(self, latent: torch.Tensor) -> torch.Tensor:
        # Existing metrics expect [B, token-channel, 6, H, W].
        value = self._structured(latent)
        return value.permute(0, 1, 3, 2, 4, 5).flatten(1, 2)

    @staticmethod
    def _token_channel_view(value: torch.Tensor) -> torch.Tensor:
        """Map [B,G,D,S,H,W] to metric layout [B,G*S,D,H,W]."""

        return value.permute(0, 1, 3, 2, 4, 5).flatten(1, 2)

    def _gain(self, scale: int, reference: torch.Tensor) -> torch.Tensor:
        return self.scale_gains[scale].to(reference).view(1, 1, self.fsq_axis_dim, 1, 1, 1)

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        target = self._structured(latent).float()
        # Use one canonical fp32 reconstruction path in both training and
        # token decoding.  This keeps coarse-to-fine interpolation/gain math
        # reproducible under bf16 autocast while gradients still reach the
        # encoder through FSQ's straight-through rounding.
        estimate = torch.zeros_like(target)
        token_stages = []
        scale_latents = []
        residual_energy, prefix_mse, contribution_energy = [], [], []
        usage = latent.new_zeros(self.num_codebooks, len(self.scale_shapes))
        full_shape = self.scale_shapes[-1]

        for scale, (shape, active_groups) in enumerate(
            zip(self.scale_shapes, self.codebooks_per_scale)
        ):
            residual = target - estimate
            sample = _resize(residual[:, :active_groups], shape, mode="area")
            gain = self._gain(scale, sample)
            # FSQ sees [B,axis,group,slot,H,W]; group and slot remain index axes.
            value = (sample / gain).permute(0, 2, 1, 3, 4, 5)
            codes, active_ids = self.fsq(value)
            codes = codes.permute(0, 2, 1, 3, 4, 5) * gain
            decoded = self.fsq.indices_to_codes(active_ids.long()).permute(0, 2, 1, 3, 4, 5) * gain
            codes = codes + (decoded - codes).detach()
            contribution = torch.zeros_like(estimate)
            contribution[:, :active_groups] = _resize(codes, full_shape, mode="bicubic")
            estimate = estimate + contribution
            scale_latents.append(self._flat(contribution))

            ids = torch.zeros(
                latent.shape[0],
                self.parallel_codebooks,
                self.num_slots,
                *shape,
                device=latent.device,
                dtype=torch.long,
            )
            ids[:, :active_groups] = active_ids.long()
            ids = ids.flatten(1, 2)
            token_stages.append(ids)
            target_channels = self._token_channel_view(target.detach().float())
            residual_channels = self._token_channel_view(residual.detach().float())
            estimate_channels = self._token_channel_view(estimate.detach().float())
            contribution_channels = self._token_channel_view(contribution.detach().float())
            residual_energy.append(residual_channels.square().mean(dim=(0, 2, 3, 4)))
            prefix_mse.append((estimate_channels - target_channels).square().mean(dim=(0, 2, 3, 4)))
            contribution_energy.append(contribution_channels.square().mean(dim=(0, 2, 3, 4)))
            for channel in range(active_groups * self.num_slots):
                usage[channel, scale] = ids[:, channel].unique().numel() / self.vocab_size

        target_channels = self._token_channel_view(target.detach().float())
        estimate_channels = self._token_channel_view(estimate.detach().float())
        dims = (0, 2, 3, 4)
        final_quant_mse = (estimate_channels - target_channels).square().mean(dim=dims)
        latent_energy = target_channels.square().mean(dim=dims)
        quantized_energy = estimate_channels.square().mean(dim=dims)
        final_cosine = F.cosine_similarity(
            target_channels.flatten(2), estimate_channels.flatten(2), dim=2, eps=1e-8
        ).mean(0)
        zeros = usage.new_zeros(self.num_codebooks, len(self.scale_shapes))
        scalar_zero = latent.new_zeros(())
        return QuantizerOutput(
            quantized=self._flat(estimate),
            tokens=tuple(token_stages),
            loss=scalar_zero,
            codebook_loss=scalar_zero,
            adapter_loss=scalar_zero,
            commitment_loss=scalar_zero,
            usage=usage,
            residual_energy=torch.stack(residual_energy, dim=1),
            prefix_mse=torch.stack(prefix_mse, dim=1),
            stage_codebook_mse=zeros,
            stage_adapter_mse=zeros,
            contribution_energy=torch.stack(contribution_energy, dim=1),
            final_quant_mse=final_quant_mse,
            latent_energy=latent_energy,
            quantized_energy=quantized_energy,
            final_cosine=final_cosine,
            code_rms=latent.new_zeros(self.num_codebooks),
            scale_latents=tuple(scale_latents),
        )

    def tokens_to_scale_latents(
        self, tokens: TokenPyramid, max_scales: int | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Return one full-grid contribution per scale, zeroing future scales."""

        count = len(self.scale_shapes) if max_scales is None else int(max_scales)
        if not 1 <= count <= len(self.scale_shapes) or len(tokens) < count:
            raise ValueError("invalid max_scales or incomplete token pyramid")
        batch, device = int(tokens[0].shape[0]), tokens[0].device
        full_shape = self.scale_shapes[-1]
        contributions = []
        for scale, (shape, active_groups) in enumerate(
            zip(self.scale_shapes, self.codebooks_per_scale)
        ):
            contribution = torch.zeros(
                batch,
                self.parallel_codebooks,
                self.fsq_axis_dim,
                self.num_slots,
                *full_shape,
                device=device,
                dtype=torch.float32,
            )
            if scale < count:
                self._validate_tokens(tokens[scale], scale, batch=batch, device=device)
                ids = tokens[scale].view(batch, self.parallel_codebooks, self.num_slots, *shape)
                codes = self.fsq.indices_to_codes(ids[:, :active_groups].long()).permute(
                    0, 2, 1, 3, 4, 5
                )
                contribution[:, :active_groups] = _resize(
                    codes * self._gain(scale, codes), full_shape, mode="bicubic"
                )
            contributions.append(self._flat(contribution))
        return tuple(contributions)

    def tokens_to_latent(self, tokens: TokenPyramid, max_scales: int | None = None) -> torch.Tensor:
        count = len(tokens) if max_scales is None else int(max_scales)
        if not 1 <= count <= len(self.scale_shapes) or len(tokens) < count:
            raise ValueError("invalid max_scales or incomplete token pyramid")
        batch, device = int(tokens[0].shape[0]), tokens[0].device
        full_shape = self.scale_shapes[-1]
        estimate = torch.zeros(
            batch,
            self.parallel_codebooks,
            self.fsq_axis_dim,
            self.num_slots,
            *full_shape,
            device=device,
            dtype=torch.float32,
        )
        for scale in range(count):
            self._validate_tokens(tokens[scale], scale, batch=batch, device=device)
            shape, active_groups = self.scale_shapes[scale], self.codebooks_per_scale[scale]
            ids = tokens[scale].view(batch, self.parallel_codebooks, self.num_slots, *shape)
            active_ids = ids[:, :active_groups]
            codes = self.fsq.indices_to_codes(active_ids.long()).permute(0, 2, 1, 3, 4, 5)
            codes = codes * self._gain(scale, codes)
            contribution = torch.zeros_like(estimate)
            contribution[:, :active_groups] = _resize(codes, full_shape, mode="bicubic")
            estimate = estimate + contribution
        return self._flat(estimate)

    def prefix_latents(self, tokens: TokenPyramid) -> list[torch.Tensor]:
        if len(tokens) != len(self.scale_shapes):
            raise ValueError("target token pyramid must contain every configured scale")
        outputs = []
        for scale, shape in enumerate(self.scale_shapes):
            if scale == 0:
                outputs.append(
                    torch.zeros(
                        tokens[0].shape[0],
                        self.latent_channels,
                        *shape,
                        device=tokens[0].device,
                        dtype=torch.float32,
                    )
                )
            else:
                full = self._structured(self.tokens_to_latent(tokens, max_scales=scale))
                outputs.append(self._flat(_resize(full, shape, mode="area")))
        return outputs

    def context_latents(self, tokens: TokenPyramid) -> list[torch.Tensor]:
        full = self._structured(self.tokens_to_latent(tokens))
        return [self._flat(_resize(full, shape, mode="area")) for shape in self.scale_shapes]

    def split_prediction_ids(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = prod(self.levels[:3])
        return tokens.long() % first, tokens.long() // first

    def merge_prediction_ids(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first_vocab, second_vocab = self.prediction_vocab_sizes
        if first.numel() and (first.min() < 0 or first.max() >= first_vocab):
            raise ValueError("A factor id out of range")
        if second.numel() and (second.min() < 0 or second.max() >= second_vocab):
            raise ValueError("B factor id out of range")
        return first.long() + first_vocab * second.long()

    def _validate_tokens(
        self,
        tokens: torch.Tensor,
        scale: int,
        *,
        batch: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        expected = (self.num_codebooks, *self.scale_shapes[scale])
        if tokens.ndim != 4 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"scale {scale} expected [B,{expected}], got {tuple(tokens.shape)}")
        if batch is not None and tokens.shape[0] != batch:
            raise ValueError("all token scales must have the same batch size")
        if device is not None and tokens.device != device:
            raise ValueError("all token scales must be on the same device")
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError(f"scale {scale} token id outside [0,{self.vocab_size})")
