"""Two observed daily latents and causal coarse-to-fine target blocks."""

from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import (
    checkpoint,
    create_selective_checkpoint_contexts,
    CheckpointPolicy,
)
from .model import NextScaleDynamics
from .layout import DynamicsOutput

ATTENTION_OPERATORS = frozenset(
    (
        "aten._scaled_dot_product_flash_attention.default",
        "aten._scaled_dot_product_efficient_attention.default",
        "aten._scaled_dot_product_flash_attention_for_cpu.default",
        "aten._scaled_dot_product_cudnn_attention.default",
    )
)


def attention_checkpoint_policy(context, operator, *args, **kwargs):
    return (
        CheckpointPolicy.MUST_SAVE
        if str(operator) in ATTENTION_OPERATORS
        else CheckpointPolicy.PREFER_RECOMPUTE
    )


def joint_output(model, logits, target):
    loss = sum(
        F.cross_entropy(
            v[:, : model.active_token_channels[s]].flatten(0, 1),
            y.flatten(0, 1).long(),
            reduction="sum",
        )
        for s, (pair, truth) in enumerate(zip(logits, target))
        for v, y in zip(
            pair, model.codec.split_prediction_ids(truth[:, : model.active_token_channels[s]])
        )
    ) / sum(y[:, : model.active_token_channels[s]].numel() for s, y in enumerate(target))
    out = DynamicsOutput(logits=tuple(logits), loss=loss)
    out.standard_loss = out.objective_loss = loss
    return out


class WeatherDynamics(NextScaleDynamics):
    """Two complete daily latents are separate, uncompressed prefix sequences.

    Sequence: [H(t-1), H(t)], Q0, C0, Q1, C1, Q2.
    H is one bidirectional observed-only block. Qs predicts scale s; Cs contains
    the reconstructed target latent through scale s. Strict block causality
    prevents any query from reading its own target content at any layer.
    No temporal conditioner, pooling, resampler, or per-token autoregression.
    """

    def __init__(self, codec, activation_checkpointing=True, **kwargs):
        super().__init__(codec, **kwargs)
        self.role_position = nn.Embedding(4, self.input_projection.out_features)
        self.activation_checkpointing = bool(activation_checkpointing)

    def _history(self, context):
        if len(context) != len(self.scale_shapes):
            raise ValueError("history scale count")
        for value, shape in zip(context, self.scale_shapes):
            if tuple(value.shape[1:]) != (2, self.num_codebooks, *shape):
                raise ValueError("Two-day dynamics expects [B,2,Q,H,W] at each scale")
        return [self.codec.tokens_to_latent(tuple(v[:, day] for v in context)) for day in range(2)]

    def _embed(self, first, second, scale, role, annual):
        batch, _, h, w = first.shape
        feature = torch.cat((first, second), 1).flatten(2).transpose(1, 2)
        sequence = self.input_projection(feature)
        rows = torch.arange(h, device=first.device).repeat_interleave(w)
        cols = torch.arange(w, device=first.device).repeat(h)
        position = (
            self.row_position(rows)
            + self.column_position(cols)
            + self.scale_position.weight[scale]
            + self.role_position.weight[role]
        )
        return sequence + position[None] + annual[:, None]

    def build_sequence(self, context, target, target_day, max_scales=3):
        histories = self._history(context)
        annual = self._day_embedding(target_day, target[0].shape[0], histories[0])
        parts = [
            self._embed(v, torch.zeros_like(v), len(self.scale_shapes) - 1, day, annual)
            for day, v in enumerate(histories)
        ]
        length = sum(v.shape[1] for v in parts)
        ends, queries = [length], []
        # Retain the original target query's latest-day and strict-coarse latent
        # features, while exposing the actual known target contents as keys too.
        previous = self.codec.context_latents(tuple(v[:, 1] for v in context))
        prefix = self.codec.prefix_latents(target)
        for s in range(max_scales):
            query = self._embed(previous[s], prefix[s], s, 2, annual)
            parts.append(query)
            queries.append(slice(length, length + query.shape[1]))
            length += query.shape[1]
            ends.append(length)
            if s + 1 < max_scales:
                content = self.codec.tokens_to_latent(target, max_scales=s + 1)
                content = F.interpolate(content, size=self.scale_shapes[s], mode="area")
                value = self._embed(previous[s], content, s, 3, annual)
                parts.append(value)
                length += value.shape[1]
                ends.append(length)
        return torch.cat(parts, 1), tuple(ends), queries

    def _hidden(self, sequence, ends):
        for block in self.transformer.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                sequence = checkpoint(
                    block,
                    sequence,
                    ends,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    context_fn=partial(
                        create_selective_checkpoint_contexts, attention_checkpoint_policy
                    ),
                )
            else:
                sequence = block(sequence, ends)
        return self.transformer.norm(sequence)

    def query_logits(self, block, scale):
        batch, positions = block.shape[:2]
        h, w = self.scale_shapes[scale]
        return tuple(
            head(block)
            .view(batch, positions, self.num_codebooks, vocab)
            .permute(0, 2, 3, 1)
            .reshape(batch, self.num_codebooks, vocab, h, w)
            for head, vocab in zip(self.factor_heads, self.prediction_vocab_sizes)
        )

    def forward(self, context_tokens, target_tokens, target_day, *, rollout=False):
        if rollout:
            raise ValueError("Dynamics training uses a one-day joint CE objective")
        target = self._validate_tokens(target_tokens, "target")
        sequence, ends, queries = self.build_sequence(context_tokens, target, target_day)
        hidden = self._hidden(sequence, ends)
        logits = [self.query_logits(hidden[:, indices], s) for s, indices in enumerate(queries)]
        return joint_output(self, logits, target)

    @torch.no_grad()
    def generate(
        self, context_tokens, target_day, *, greedy=True, temperature=1.0, top_k=None, mode=None
    ):
        if mode is not None:
            if mode not in ("greedy", "sample"):
                raise ValueError(mode)
            greedy = mode == "greedy"
        generated = [torch.zeros_like(x[:, 1]) for x in context_tokens]
        was_training = self.training
        self.eval()
        try:
            for s in range(len(generated)):
                sequence, ends, queries = self.build_sequence(
                    context_tokens, generated, target_day, s + 1
                )
                hidden = self._hidden(sequence, ends)
                factors = [
                    self._draw(v, greedy, temperature, top_k)
                    for v in self.query_logits(hidden[:, queries[-1]], s)
                ]
                generated[s] = self.codec.merge_prediction_ids(*factors)
        finally:
            self.train(was_training)
        return tuple(generated)
