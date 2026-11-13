from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import SphericalConv2d

TokenPyramid = tuple[torch.Tensor, ...]


@dataclass
class QuantizerOutput:
    quantized: torch.Tensor
    tokens: TokenPyramid
    loss: torch.Tensor
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
    scale_latents: tuple[torch.Tensor, ...] | None = None


class ResidualAdapter(nn.Module):
    def __init__(self, channels: int, ratio: float):
        super().__init__()
        if not 0.0 <= float(ratio) <= 1.0:
            raise ValueError("residual adapter ratio must lie in [0, 1]")
        self.register_buffer("ratio", torch.tensor(float(ratio), dtype=torch.float32))
        self.conv = SphericalConv2d(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ratio = self.ratio.to(x)
        return x * (1.0 - ratio) + self.conv(x) * ratio


class ResidualVectorQuantizer(nn.Module):
    """Shared-codebook, coarse-to-fine residual VQ adapted from ClimateAR/VAR."""

    def __init__(
        self,
        scale_shapes: Sequence[Sequence[int]],
        num_codebooks: int = 2,
        codebook_dim: int = 128,
        vocab_size: int = 4096,
        beta: float = 0.25,
        residual_ratio: float = 0.5,
        shared_adapters: int = 4,
        commitment_mode: str = "cumulative",
        commitment_scale_weights: Sequence[float] | None = None,
        codebook_objective: str = "cumulative_prefix",
        adapter_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.scale_shapes = tuple((int(h), int(w)) for h, w in scale_shapes)
        if not self.scale_shapes or any(h < 1 or w < 1 for h, w in self.scale_shapes):
            raise ValueError("scale_shapes must contain positive (height, width) pairs")
        if any(
            a[0] > b[0] or a[1] > b[1] for a, b in zip(self.scale_shapes, self.scale_shapes[1:])
        ):
            raise ValueError("scale_shapes must be coarse-to-fine")
        self.num_codebooks = int(num_codebooks)
        self.codebook_dim = int(codebook_dim)
        self.vocab_size = int(vocab_size)
        self.beta = float(beta)
        self.commitment_mode = str(commitment_mode)
        self.codebook_objective = str(codebook_objective)
        self.adapter_loss_weight = float(adapter_loss_weight)
        if min(self.num_codebooks, self.codebook_dim) < 1 or self.vocab_size < 2:
            raise ValueError("num_codebooks/codebook_dim must be positive and vocab_size >= 2")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if self.commitment_mode not in {"cumulative", "final_only", "weighted"}:
            raise ValueError("commitment_mode must be cumulative, final_only, or weighted")
        if self.codebook_objective not in {"cumulative_prefix", "stage_aligned"}:
            raise ValueError("codebook_objective must be cumulative_prefix or stage_aligned")
        if self.adapter_loss_weight < 0.0:
            raise ValueError("adapter_loss_weight must be non-negative")
        if commitment_scale_weights is None:
            weights = torch.ones(len(self.scale_shapes), dtype=torch.float32)
        else:
            weights = torch.as_tensor(tuple(commitment_scale_weights), dtype=torch.float32)
            if (
                weights.numel() != len(self.scale_shapes)
                or (weights < 0).any()
                or weights.sum() <= 0
            ):
                raise ValueError(
                    "commitment_scale_weights must be non-negative with one value per scale"
                )
        self.register_buffer("commitment_scale_weights", weights / weights.sum(), persistent=False)
        if int(shared_adapters) < 1:
            raise ValueError("shared_adapters must be positive")
        self.latent_channels = self.num_codebooks * self.codebook_dim
        self.embeddings = nn.ModuleList(
            nn.Embedding(self.vocab_size, self.codebook_dim) for _ in range(self.num_codebooks)
        )
        self.transforms = nn.ModuleList(
            nn.Linear(self.codebook_dim, self.codebook_dim, bias=False)
            for _ in range(self.num_codebooks)
        )
        count = max(1, min(int(shared_adapters), len(self.scale_shapes)))
        self.adapters = nn.ModuleList(
            nn.ModuleList(ResidualAdapter(self.codebook_dim, residual_ratio) for _ in range(count))
            for _ in range(self.num_codebooks)
        )
        for embedding in self.embeddings:
            nn.init.trunc_normal_(embedding.weight, std=0.02)
        for transform in self.transforms:
            nn.init.eye_(transform.weight)

    def codebook_parameters(self) -> tuple[nn.Parameter, ...]:
        """Learned VQ parameters, separated only when a codebook LR is requested."""

        return tuple(self.embeddings.parameters()) + tuple(self.transforms.parameters())

    def adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.adapters.parameters())

    def _adapter(self, head: int, scale: int) -> nn.Module:
        adapters = self.adapters[head]
        numerator = scale * (len(adapters) - 1)
        denominator = max(1, len(self.scale_shapes) - 1)
        index = (2 * numerator + denominator) // (2 * denominator)
        return adapters[index]

    def _codes(self, head: int) -> torch.Tensor:
        return self.transforms[head](self.embeddings[head].weight)

    def _embedding(self, head: int, ids: torch.Tensor) -> torch.Tensor:
        return self.transforms[head](self.embeddings[head](ids)).permute(0, 3, 1, 2)

    @staticmethod
    def _nearest(samples: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        flat = samples.permute(0, 2, 3, 1).reshape(-1, samples.shape[1])
        distance = flat.square().sum(1, keepdim=True) + codes.square().sum(1)
        distance.addmm_(flat, codes.t(), beta=1.0, alpha=-2.0)
        return distance.argmin(1)

    def _contribution(
        self, head: int, scale: int, ids: torch.Tensor, full_shape: tuple[int, int]
    ) -> torch.Tensor:
        value = self._embedding(head, ids)
        if tuple(value.shape[-2:]) != full_shape:
            value = F.interpolate(value, size=full_shape, mode="bicubic", align_corners=False)
        return self._adapter(head, scale)(value)

    @staticmethod
    def _upsample(value: torch.Tensor, full_shape: tuple[int, int]) -> torch.Tensor:
        if tuple(value.shape[-2:]) == full_shape:
            return value
        return F.interpolate(value, size=full_shape, mode="bicubic", align_corners=False)

    def forward(self, latent: torch.Tensor) -> QuantizerOutput:
        b, channels, height, width = latent.shape
        if channels != self.latent_channels or self.scale_shapes[-1] != (height, width):
            raise ValueError(
                f"expected [B,{self.latent_channels},{self.scale_shapes[-1][0]},"
                f"{self.scale_shapes[-1][1]}], got {tuple(latent.shape)}"
            )
        heads = latent.view(b, self.num_codebooks, self.codebook_dim, height, width)
        token_lists: list[list[torch.Tensor]] = [[] for _ in self.scale_shapes]
        quantized_heads, codebook_losses, adapter_losses, commitment_losses, energies = (
            [],
            [],
            [],
            [],
            [],
        )
        prefix_mse, stage_codebook_mse, stage_adapter_mse, contribution_energy = [], [], [], []
        final_quant_mse, latent_energy, quantized_energy, final_cosine, code_rms = (
            [],
            [],
            [],
            [],
            [],
        )
        usage = latent.new_zeros(self.num_codebooks, len(self.scale_shapes))
        for q in range(self.num_codebooks):
            target = heads[:, q]
            estimate = torch.zeros_like(target)
            codes = self._codes(q)
            head_commitment_losses = []
            head_prefix_mse, head_contribution_energy = [], []
            for s, shape in enumerate(self.scale_shapes):
                residual = target.detach() - estimate.detach()
                sample = (
                    residual
                    if shape == (height, width)
                    else F.interpolate(residual, shape, mode="area")
                )
                ids = self._nearest(sample.float(), codes.detach().float()).view(b, *shape)
                if self.codebook_objective == "stage_aligned":
                    # Assignment and centroid fitting share the same native-scale
                    # target.  The detached code passed to the adapter makes the
                    # adapter loss a separate gradient path.
                    selected = self._embedding(q, ids)
                    code_loss = F.mse_loss(selected, sample.detach())
                    adapter_input = self._upsample(selected.detach(), (height, width))
                    contribution = self._adapter(q, s)(adapter_input)
                    adapter_target = self._upsample(sample.detach(), (height, width))
                    adapter_loss = F.mse_loss(contribution, adapter_target)
                else:
                    contribution = self._contribution(q, s, ids, (height, width))
                estimate = estimate + contribution
                prefix_loss = F.mse_loss(estimate, target.detach())
                if self.codebook_objective == "cumulative_prefix":
                    # Exact ClimateAR/VAR default: every prefix approximates the
                    # original latent and later prefixes revisit coarse codes.
                    code_loss = prefix_loss
                    adapter_loss = prefix_loss.new_zeros(())
                commitment_loss = F.mse_loss(estimate.detach(), target)
                codebook_losses.append(code_loss)
                if self.codebook_objective == "stage_aligned":
                    adapter_losses.append(adapter_loss)
                head_commitment_losses.append(commitment_loss)
                head_prefix_mse.append(prefix_loss.detach())
                stage_codebook_mse.append(code_loss.detach())
                stage_adapter_mse.append(adapter_loss.detach())
                head_contribution_energy.append(contribution.detach().float().square().mean())
                token_lists[s].append(ids)
                usage[q, s] = ids.unique().numel() / self.vocab_size
                energies.append(residual.square().mean().detach())
            if self.commitment_mode == "cumulative":
                commitment_losses.extend(head_commitment_losses)
            elif self.commitment_mode == "final_only":
                commitment_losses.append(head_commitment_losses[-1])
            else:
                weights = self.commitment_scale_weights.to(head_commitment_losses[0])
                commitment_losses.append(torch.sum(torch.stack(head_commitment_losses) * weights))
            target_float, estimate_float = target.detach().float(), estimate.detach().float()
            prefix_mse.extend(head_prefix_mse)
            contribution_energy.extend(head_contribution_energy)
            final_quant_mse.append((estimate_float - target_float).square().mean())
            latent_energy.append(target_float.square().mean())
            quantized_energy.append(estimate_float.square().mean())
            final_cosine.append(
                F.cosine_similarity(
                    target_float.flatten(1), estimate_float.flatten(1), dim=1, eps=1e-8
                ).mean()
            )
            code_rms.append(codes.detach().float().square().mean().sqrt())
            quantized_heads.append(target + (estimate - target).detach())
        tokens = tuple(torch.stack(per_head, dim=1) for per_head in token_lists)
        codebook_loss = torch.stack(codebook_losses).mean()
        adapter_loss = (
            torch.stack(adapter_losses).mean() if adapter_losses else codebook_loss.new_zeros(())
        )
        commitment_loss = torch.stack(commitment_losses).mean()
        return QuantizerOutput(
            quantized=torch.cat(quantized_heads, dim=1),
            tokens=tokens,
            loss=(
                codebook_loss
                + self.adapter_loss_weight * adapter_loss
                + self.beta * commitment_loss
            ),
            codebook_loss=codebook_loss,
            adapter_loss=adapter_loss,
            commitment_loss=commitment_loss,
            usage=usage,
            residual_energy=torch.stack(energies).view(self.num_codebooks, -1),
            prefix_mse=torch.stack(prefix_mse).view(self.num_codebooks, -1),
            stage_codebook_mse=torch.stack(stage_codebook_mse).view(self.num_codebooks, -1),
            stage_adapter_mse=torch.stack(stage_adapter_mse).view(self.num_codebooks, -1),
            contribution_energy=torch.stack(contribution_energy).view(self.num_codebooks, -1),
            final_quant_mse=torch.stack(final_quant_mse),
            latent_energy=torch.stack(latent_energy),
            quantized_energy=torch.stack(quantized_energy),
            final_cosine=torch.stack(final_cosine),
            code_rms=torch.stack(code_rms),
        )

    def tokens_to_latent(self, tokens: TokenPyramid, max_scales: int | None = None) -> torch.Tensor:
        count = len(tokens) if max_scales is None else int(max_scales)
        if not 1 <= count <= len(self.scale_shapes) or len(tokens) < count:
            raise ValueError("invalid max_scales or incomplete token pyramid")
        b = tokens[0].shape[0]
        device = tokens[0].device
        for scale in range(count):
            self._validate_tokens(tokens[scale], scale, batch=b, device=device)
        full_shape = self.scale_shapes[-1]
        reconstructed = []
        for q in range(self.num_codebooks):
            estimate = self.embeddings[q].weight.new_zeros(b, self.codebook_dim, *full_shape)
            for s in range(count):
                estimate = estimate + self._contribution(q, s, tokens[s][:, q].long(), full_shape)
            reconstructed.append(estimate)
        return torch.cat(reconstructed, dim=1)

    def prefix_latents(self, tokens: TokenPyramid) -> list[torch.Tensor]:
        """Target inputs: scale s receives only target scales strictly below s."""
        if len(tokens) != len(self.scale_shapes):
            raise ValueError("target token pyramid must contain every configured scale")
        full_shape = self.scale_shapes[-1]
        b = tokens[0].shape[0]
        device = tokens[0].device
        for scale, value in enumerate(tokens):
            self._validate_tokens(value, scale, batch=b, device=device)
        running = self.embeddings[0].weight.new_zeros(b, self.latent_channels, *full_shape)
        outputs = []
        for s, shape in enumerate(self.scale_shapes):
            outputs.append(
                running if shape == full_shape else F.interpolate(running, shape, mode="area")
            )
            pieces = [
                self._contribution(q, s, tokens[s][:, q].long(), full_shape)
                for q in range(self.num_codebooks)
            ]
            running = running + torch.cat(pieces, dim=1)
        return outputs

    def context_latents(self, tokens: TokenPyramid) -> list[torch.Tensor]:
        full = self.tokens_to_latent(tokens)
        return [
            full if shape == self.scale_shapes[-1] else F.interpolate(full, shape, mode="area")
            for shape in self.scale_shapes
        ]

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
