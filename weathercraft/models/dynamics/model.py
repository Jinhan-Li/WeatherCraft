from __future__ import annotations

from math import pi
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import TransformerStack
from .layout import (
    DynamicsOutput,
    FactorLogitPyramid,
    LatentCodec,
    LogitPyramid,
    ScaleLayout,
    TokenPyramid,
)


class NextScaleDynamics(nn.Module):
    """Predict a target day jointly per scale, coarse to fine.

    The codec supplies the previous day's complete latent at every resolution and
    the target day's accumulated reconstruction from strictly coarser scales.
    Conditioning accepts leap-aware annual sin/cos as ``[B,2]``; scalar or
    ``[B]`` day ordinals use ``year_days`` as a fixed-period fallback.
    Logits use layout ``[B, Q, V, H, W]`` at each scale.
    """

    def __init__(
        self,
        codec: LatentCodec,
        width: int | None = None,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        model_dim: int | None = None,
        label_smoothing: float = 0.0,
        year_days: float = 365.2425,
        freeze_codec: bool = True,
    ):
        super().__init__()
        if width is None:
            width = 512 if model_dim is None else int(model_dim)
        elif model_dim is not None and width != model_dim:
            raise ValueError("width and model_dim aliases disagree")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.label_smoothing = float(label_smoothing)
        self.codec = codec
        if freeze_codec and isinstance(codec, nn.Module):
            codec.requires_grad_(False)
        self.num_codebooks = int(codec.num_codebooks)
        self.codebook_dim = int(codec.codebook_dim)
        self.vocab_size = int(codec.vocab_size)
        self.prediction_vocab_sizes = tuple(
            int(value) for value in getattr(codec, "prediction_vocab_sizes", (self.vocab_size,))
        )
        self.factorized_output = len(self.prediction_vocab_sizes) > 1
        self.layout = ScaleLayout(codec.scale_shapes)
        active_channels = getattr(codec, "active_token_channels", None)
        if active_channels is None:
            self.active_token_channels = (self.num_codebooks,) * len(self.layout.shapes)
        else:
            if torch.is_tensor(active_channels):
                active_channels = active_channels.detach().cpu().tolist()
            self.active_token_channels = tuple(int(value) for value in active_channels)
            if len(self.active_token_channels) != len(self.layout.shapes):
                raise ValueError("codec active_token_channels must contain one entry per scale")
            if any(value < 1 or value > self.num_codebooks for value in self.active_token_channels):
                raise ValueError(
                    "codec active_token_channels entries must be in " f"[1, {self.num_codebooks}]"
                )
        self.year_days = float(year_days)
        if self.year_days <= 0:
            raise ValueError("year_days must be positive")
        if self.num_codebooks < 1 or self.codebook_dim < 1 or self.vocab_size < 2:
            raise ValueError("codec dimensions must be positive and vocab_size >= 2")
        latent_width = self.num_codebooks * self.codebook_dim
        self.input_projection = nn.Linear(latent_width * 2, width)
        self.scale_position = nn.Embedding(len(self.layout.shapes), width)
        self.row_position = nn.Embedding(max(h for h, _ in self.layout.shapes), width)
        self.column_position = nn.Embedding(max(w for _, w in self.layout.shapes), width)
        self.day_projection = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.transformer = TransformerStack(width, depth, num_heads, mlp_ratio, dropout, drop_path)
        if self.factorized_output:
            self.heads = nn.ModuleList()
            self.factor_heads = nn.ModuleList(
                nn.Linear(width, self.num_codebooks * vocab)
                for vocab in self.prediction_vocab_sizes
            )
        else:
            self.heads = nn.ModuleList(
                nn.Linear(width, self.vocab_size) for _ in range(self.num_codebooks)
            )
            self.factor_heads = nn.ModuleList()

    @property
    def scale_shapes(self) -> tuple[tuple[int, int], ...]:
        return self.layout.shapes

    def _validate_tokens(self, tokens: Sequence[torch.Tensor], name: str) -> TokenPyramid:
        if len(tokens) != len(self.scale_shapes):
            raise ValueError(f"{name} must contain {len(self.scale_shapes)} scales")
        batch = int(tokens[0].shape[0])
        checked = []
        for scale, (value, shape) in enumerate(zip(tokens, self.scale_shapes)):
            expected = (batch, self.num_codebooks, *shape)
            if value.ndim != 4 or tuple(value.shape) != expected:
                raise ValueError(f"{name}[{scale}] expected {expected}, got {tuple(value.shape)}")
            checked.append(value)
        return tuple(checked)

    def _day_embedding(
        self, target_day: torch.Tensor | float | int, batch: int, reference: torch.Tensor
    ) -> torch.Tensor:
        day = torch.as_tensor(target_day, device=reference.device, dtype=torch.float32)
        if day.ndim == 2 and tuple(day.shape) == (batch, 2):
            annual = day
        elif day.ndim == 0:
            day = day.expand(batch)
            phase = (2.0 * pi / self.year_days) * day
            annual = torch.stack((phase.sin(), phase.cos()), dim=-1)
        elif day.ndim == 1 and day.shape[0] == batch:
            phase = (2.0 * pi / self.year_days) * day
            annual = torch.stack((phase.sin(), phase.cos()), dim=-1)
        else:
            raise ValueError(
                f"target_day must be scalar, [B], or annual sin/cos [B,2], got {tuple(day.shape)}"
            )
        return self.day_projection(annual.to(reference.dtype))

    def build_teacher_inputs(
        self,
        context_tokens: Sequence[torch.Tensor],
        target_tokens: Sequence[torch.Tensor],
        target_day: torch.Tensor | float | int,
        max_scales: int | None = None,
    ) -> torch.Tensor:
        """Build the transformer sequence before self-attention."""

        if len(context_tokens) != len(self.scale_shapes):
            raise ValueError("context scale count mismatch")
        latent_width = self.num_codebooks * self.codebook_dim
        if all(value.is_floating_point() for value in context_tokens):
            batch = int(context_tokens[0].shape[0])
            context_latents = tuple(context_tokens)
            for scale, (value, shape) in enumerate(zip(context_latents, self.scale_shapes)):
                expected = (batch, latent_width, *shape)
                if tuple(value.shape) != expected:
                    raise ValueError(
                        f"context_latents[{scale}] expected {expected}, got {tuple(value.shape)}"
                    )
        else:
            context = self._validate_tokens(context_tokens, "context_tokens")
            context_latents = self.codec.context_latents(context)
        target = self._validate_tokens(target_tokens, "target_tokens")
        prefix_latents = self.codec.prefix_latents(target)
        if len(context_latents) != len(self.scale_shapes) or len(prefix_latents) != len(
            self.scale_shapes
        ):
            raise ValueError("codec must return one context/prefix latent per scale")
        count = len(self.scale_shapes) if max_scales is None else int(max_scales)
        if not 1 <= count <= len(self.scale_shapes):
            raise ValueError("max_scales must select a non-empty scale prefix")
        batch = target[0].shape[0]
        device = target[0].device
        scale_ids = self.layout.scale_ids(device)
        row_ids, column_ids = self.layout.spatial_ids(device)
        token_limit = sum(self.layout.lengths[:count])
        position = (
            self.scale_position(scale_ids[:token_limit])
            + self.row_position(row_ids[:token_limit])
            + self.column_position(column_ids[:token_limit])
        )
        pieces = []
        for scale, (shape, previous, prefix) in enumerate(
            zip(self.scale_shapes[:count], context_latents[:count], prefix_latents[:count])
        ):
            expected = (batch, latent_width, *shape)
            if tuple(previous.shape) != expected or tuple(prefix.shape) != expected:
                raise ValueError(
                    f"codec scale {scale} must return {expected}; got "
                    f"{tuple(previous.shape)} and {tuple(prefix.shape)}"
                )
            feature = (
                torch.cat((previous, prefix), dim=1)
                .permute(0, 2, 3, 1)
                .reshape(batch, -1, 2 * latent_width)
            )
            pieces.append(feature)
        sequence = self.input_projection(torch.cat(pieces, dim=1))
        day = self._day_embedding(target_day, batch, sequence).unsqueeze(1)
        return sequence + position.unsqueeze(0) + day

    def _scale_logits(self, hidden: torch.Tensor, scale: int) -> torch.Tensor:
        if self.factorized_output:
            raise RuntimeError("factorized codecs use _scale_factor_logits")
        height, width = self.scale_shapes[scale]
        block = hidden[:, self.layout.block_slice(scale)]
        per_head = torch.stack([head(block) for head in self.heads], dim=1)
        return per_head.permute(0, 1, 3, 2).reshape(
            hidden.shape[0], self.num_codebooks, self.vocab_size, height, width
        )

    def _logits(self, hidden: torch.Tensor) -> LogitPyramid:
        return tuple(self._scale_logits(hidden, scale) for scale in range(len(self.scale_shapes)))

    def _scale_factor_logits(self, hidden: torch.Tensor, scale: int) -> tuple[torch.Tensor, ...]:
        height, width = self.scale_shapes[scale]
        block = hidden[:, self.layout.block_slice(scale)]
        batch, positions = block.shape[:2]
        outputs = []
        for head, vocab in zip(self.factor_heads, self.prediction_vocab_sizes):
            value = head(block).view(batch, positions, self.num_codebooks, vocab)
            outputs.append(
                value.permute(0, 2, 3, 1).reshape(batch, self.num_codebooks, vocab, height, width)
            )
        return tuple(outputs)

    def _factor_logits(self, hidden: torch.Tensor) -> FactorLogitPyramid:
        return tuple(
            self._scale_factor_logits(hidden, scale) for scale in range(len(self.scale_shapes))
        )

    def forward(
        self,
        context_tokens: Sequence[torch.Tensor],
        target_tokens: Sequence[torch.Tensor],
        target_day: torch.Tensor | float | int,
        *,
        rollout: bool = False,
    ) -> DynamicsOutput:
        target = self._validate_tokens(target_tokens, "target_tokens")
        if rollout:
            # Each target scale sees its own previously *generated* coarse
            # scales. Detached discrete tokens also become the next day's
            # context in multi-day training (truncated BPTT through sampling).
            generated = [torch.zeros_like(value) for value in target]
            predictions = []
            for scale in range(len(target)):
                sequence = self.build_teacher_inputs(
                    context_tokens, tuple(generated), target_day, max_scales=scale + 1
                )
                hidden = self.transformer(sequence, self.layout.ends[: scale + 1])
                prediction = (
                    self._scale_factor_logits(hidden, scale)
                    if self.factorized_output
                    else self._scale_logits(hidden, scale)
                )
                predictions.append(prediction)
                with torch.no_grad():
                    if self.factorized_output:
                        active = self.active_token_channels[scale]
                        factors = tuple(value[:, :active].argmax(2) for value in prediction)
                        generated[scale][:, :active] = self.codec.merge_prediction_ids(*factors)
                    else:
                        generated[scale] = prediction.argmax(2)
            logits = tuple(predictions)
        else:
            sequence = self.build_teacher_inputs(context_tokens, target, target_day)
            hidden = self.transformer(sequence, self.layout.ends)
            logits = self._factor_logits(hidden) if self.factorized_output else self._logits(hidden)
        loss_sum = hidden.new_zeros(())
        token_count = 0
        for scale, (prediction, truth) in enumerate(zip(logits, target)):
            if self.factorized_output:
                active = self.active_token_channels[scale]
                active_truth = truth[:, :active]
                factor_targets = self.codec.split_prediction_ids(active_truth)
                if len(prediction) != len(factor_targets):
                    raise ValueError("codec factor targets and dynamics heads disagree")
                for factor_logits, factor_truth, vocab in zip(
                    prediction, factor_targets, self.prediction_vocab_sizes
                ):
                    factor_logits = factor_logits[:, :active]
                    batch, heads, _, height, width = factor_logits.shape
                    loss_sum = loss_sum + F.cross_entropy(
                        factor_logits.reshape(batch * heads, vocab, height, width),
                        factor_truth.long().reshape(batch * heads, height, width),
                        reduction="sum",
                        label_smoothing=self.label_smoothing,
                    )
                # The factor losses sum to the joint negative log-likelihood
                # of one composite FSQ token, so divide by composite tokens.
                token_count += active_truth.numel()
            else:
                batch, heads, _, height, width = prediction.shape
                loss_sum = loss_sum + F.cross_entropy(
                    prediction.reshape(batch * heads, self.vocab_size, height, width),
                    truth.long().reshape(batch * heads, height, width),
                    reduction="sum",
                    label_smoothing=self.label_smoothing,
                )
                token_count += truth.numel()
        return DynamicsOutput(
            logits=logits,
            loss=loss_sum / token_count,
            generated_tokens=tuple(generated) if rollout else None,
        )

    @staticmethod
    def _draw(
        logits: torch.Tensor, greedy: bool, temperature: float, top_k: int | None
    ) -> torch.Tensor:
        if greedy or temperature == 0:
            return logits.argmax(dim=2)
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        scores = logits / temperature
        if top_k is not None and top_k > 0:
            values = scores.topk(min(top_k, scores.shape[2]), dim=2).values
            scores = scores.masked_fill(scores < values[:, :, -1:].expand_as(scores), -torch.inf)
        elif top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative")
        probabilities = (
            scores.permute(0, 1, 3, 4, 2).reshape(-1, scores.shape[2]).float().softmax(-1)
        )
        sampled = torch.multinomial(probabilities, 1).reshape(
            scores.shape[0], scores.shape[1], *scores.shape[-2:]
        )
        return sampled

    @torch.no_grad()
    def generate(
        self,
        context_tokens: Sequence[torch.Tensor],
        target_day: torch.Tensor | float | int,
        *,
        greedy: bool = True,
        temperature: float = 1.0,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> TokenPyramid:
        if mode is not None:
            if mode not in {"greedy", "sample"}:
                raise ValueError("mode must be 'greedy' or 'sample'")
            greedy = mode == "greedy"
        if all(value.is_floating_point() for value in context_tokens):
            context = tuple(context_tokens)
            if len(context) != len(self.scale_shapes):
                raise ValueError("context_latents scale count mismatch")
            batch, device = context[0].shape[0], context[0].device
            latent_width = self.num_codebooks * self.codebook_dim
            for value, shape in zip(context, self.scale_shapes):
                if tuple(value.shape) != (batch, latent_width, *shape):
                    raise ValueError("context_latents shape mismatch")
        else:
            context = self._validate_tokens(context_tokens, "context_tokens")
            batch, device = context[0].shape[0], context[0].device
        generated = [
            torch.zeros(batch, self.num_codebooks, h, w, dtype=torch.long, device=device)
            for h, w in self.scale_shapes
        ]
        was_training = self.training
        self.eval()
        try:
            for scale in range(len(generated)):
                sequence = self.build_teacher_inputs(
                    context, tuple(generated), target_day, max_scales=scale + 1
                )
                hidden = self.transformer(sequence, self.layout.ends[: scale + 1])
                if self.factorized_output:
                    active = self.active_token_channels[scale]
                    factors = tuple(
                        self._draw(logits[:, :active], greedy, temperature, top_k)
                        for logits in self._scale_factor_logits(hidden, scale)
                    )
                    generated[scale][:, :active] = self.codec.merge_prediction_ids(*factors)
                else:
                    generated[scale] = self._draw(
                        self._scale_logits(hidden, scale), greedy, temperature, top_k
                    )
        finally:
            self.train(was_training)
        return tuple(generated)


ClimateARDynamics = NextScaleDynamics
