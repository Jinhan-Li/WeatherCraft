"""Inference-only block KV cache for the unchanged WeatherDynamics model.

H(previous),H(current) are ONE bidirectional block. Q0,C0,Q1,C1,Q2 are
successive bidirectional blocks, each attending only to itself and earlier
blocks. Queries are also keys for later blocks and must remain in the cache.
No cache is shared between forecast days: roles, annual conditioning and the
bidirectional history context change. No parameter/buffer is added to the model.
"""

import torch
from torch.nn import functional as F
from weathercraft.models.tokenizer.multiscale_fsq import _resize


class BlockKVCache:
    def __init__(self, model, prune_last_layer=True):
        if not hasattr(model, "build_sequence") or not hasattr(model, "role_position"):
            raise TypeError("Requires the unchanged two-day WeatherDynamics architecture")
        self.model = model
        self.prune_last_layer = prune_last_layer
        shapes = model.scale_shapes
        self.history_length = 2 * shapes[-1][0] * shapes[-1][1]
        self.capacity = (
            self.history_length + sum(h * w for h, w in shapes) + sum(h * w for h, w in shapes[:-1])
        )
        self.cache = []
        self.cursor = 0
        self.positions = {}
        self.parameter_versions = None
        self.cache_signature = None

    def clear(self):
        self.cache = []
        self.cache_signature = None
        self.cursor = 0
        self.positions.clear()
        self.parameter_versions = None

    def _begin(self):
        if self.model.training or torch.is_grad_enabled():
            raise RuntimeError("KV inference requires model.eval() and no_grad/inference_mode")
        versions = tuple(p._version for p in self.model.parameters())
        if versions != self.parameter_versions:
            self.positions.clear()
            self.parameter_versions = versions
        self.cursor = 0

    @property
    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for pair in self.cache for t in pair)

    def _allocate(self, key):
        signature = (key.shape[0], key.device, key.dtype, key.shape[1], key.shape[-1])
        if self.cache_signature != signature:
            self.cache = []
            shape = (key.shape[0], key.shape[1], self.capacity, key.shape[-1])
            for _ in self.model.transformer.blocks:
                self.cache.append(
                    (
                        torch.empty(shape, device=key.device, dtype=key.dtype),
                        torch.empty(shape, device=key.device, dtype=key.dtype),
                    )
                )
            self.cache_signature = signature

    def chunk(self, x, lengths):
        """Append complete new blocks, preserving rectangular SDPA semantics.

        Returns final-normalized states for the LAST new block (the query).
        At the last layer, non-query attention/FFN outputs cannot affect any
        subsequent state: their input K/V has already been cached. Skip those
        outputs, but retain their K/V. Earlier layers always compute every row.
        """
        if sum(lengths) != x.shape[1] or min(lengths) <= 0:
            raise ValueError("Chunk must contain complete nonempty attention blocks")
        start, end = self.cursor, self.cursor + x.shape[1]
        if end > self.capacity:
            raise ValueError("KV capacity exceeded")
        query_start = x.shape[1] - lengths[-1]
        last_layer = len(self.model.transformer.blocks) - 1
        for index, block in enumerate(self.model.transformer.blocks):
            attn = block.attn
            q, k, v = attn.qkv(block.attn_norm(x)).chunk(3, dim=-1)
            q, k, v = attn._heads(q), attn._heads(k), attn._heads(v)
            if index == 0:
                self._allocate(k)
            ck, cv = self.cache[index]
            ck[:, :, start:end].copy_(k)
            cv[:, :, start:end].copy_(v)
            del k, v
            prune = self.prune_last_layer and index == last_layer
            pieces, offset = [], 0
            for length in lengths:
                stop = offset + length
                if not prune or offset == query_start:
                    pieces.append(
                        F.scaled_dot_product_attention(
                            q[:, :, offset:stop],
                            ck[:, :, : start + stop],
                            cv[:, :, : start + stop],
                            dropout_p=0.0,
                            is_causal=False,
                        )
                    )
                offset = stop
            merged = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=2)
            merged = merged.transpose(1, 2).contiguous().flatten(2)
            if prune:
                x = x[:, query_start:]
            # These modules are exactly the original eval-mode operations.
            x = x + block.drop_path(block.attn_dropout(attn.output(merged)))
            x = x + block.drop_path(block.ffn(block.ffn_norm(x)))
        self.cursor = end
        return self.model.transformer.norm(x if self.prune_last_layer else x[:, query_start:])

    def _embed(self, first, second, scale, role, annual):
        m = self.model
        b, _, h, w = first.shape
        key = (scale, role, h, w, first.device)
        if key not in self.positions:
            rows = torch.arange(h, device=first.device).repeat_interleave(w)
            cols = torch.arange(w, device=first.device).repeat(h)
            self.positions[key] = (
                m.row_position(rows)
                + m.column_position(cols)
                + m.scale_position.weight[scale]
                + m.role_position.weight[role]
            )
        # Preserve addition order/dtype promotion of the original _embed.
        features = torch.cat((first, second), 1).flatten(2).transpose(1, 2)
        return m.input_projection(features) + self.positions[key][None] + annual[:, None]

    def _down(self, full, shape):
        codec = self.model.codec
        return codec._flat(_resize(codec._structured(full), shape, "area"))

    def _decode_scale(self, ids, scale):
        """Canonical codec arithmetic for one known-valid generated scale.

        Drawn IDs are validated at the public boundary. Avoid repeated
        device-to-host min/max checks and decoding earlier scales again.
        """
        codec = self.model.codec
        b = ids.shape[0]
        shape = codec.scale_shapes[scale]
        groups = codec.codebooks_per_scale[scale]
        ids = ids.view(b, codec.parallel_codebooks, codec.num_slots, *shape)[:, :groups]
        digits = codec.fsq.indices_to_level_indices(ids)
        half = (codec.fsq._levels.to(ids.device) // 2).float()
        codes = ((digits.float() - half) / half).movedim(-1, 1).permute(0, 2, 1, 3, 4, 5)
        value = torch.zeros(
            b,
            codec.parallel_codebooks,
            codec.fsq_axis_dim,
            codec.num_slots,
            *codec.scale_shapes[-1],
            device=ids.device,
            dtype=torch.float32,
        )
        value[:, :groups] = _resize(
            codes * codec._gain(scale, codes), codec.scale_shapes[-1], "bicubic"
        )
        return codec._flat(value)

    @torch.no_grad()
    def initial_latents(self, history):
        return self.model._history(history)

    @torch.no_grad()
    def generate(
        self,
        history,
        target_day,
        *,
        mode="greedy",
        temperature=1.0,
        top_k=0,
        history_latents=None,
        return_latent=False,
    ):
        """Generate one forecast day using cached attention blocks."""
        self._begin()
        if mode not in ("greedy", "sample"):
            raise ValueError(mode)
        m, codec = self.model, self.model.codec
        if history_latents is None:
            latents = self.initial_latents(history)
        else:
            if history is not None or len(history_latents) != 2:
                raise ValueError("Pass either history tokens or the owned rollout latent state")
            latents = history_latents
        annual = m._day_embedding(target_day, latents[0].shape[0], latents[0])
        previous = [self._down(latents[1], shape) for shape in m.scale_shapes]
        history_parts = [
            self._embed(value, torch.zeros_like(value), len(m.scale_shapes) - 1, role, annual)
            for role, value in enumerate(latents)
        ]
        prefix = torch.zeros_like(latents[0])
        generated, content = [], None
        for s, shape in enumerate(m.scale_shapes):
            query = self._embed(previous[s], self._down(prefix, shape), s, 2, annual)
            if s == 0:
                chunk = torch.cat((*history_parts, query), dim=1)
                lengths = (self.history_length, query.shape[1])
                del history_parts
            else:
                chunk = torch.cat((content, query), dim=1)
                lengths = (content.shape[1], query.shape[1])
            hidden = self.chunk(chunk, lengths)
            factors = m.query_logits(hidden, s)
            draws = [m._draw(value, mode == "greedy", temperature, top_k) for value in factors]
            ids = draws[0].long() + m.prediction_vocab_sizes[0] * draws[1].long()
            generated.append(ids)
            del hidden, chunk, query, factors
            if s + 1 < len(m.scale_shapes) or return_latent:
                # Add in the same order as tokens_to_latent; no across-day KV reuse.
                prefix = prefix + self._decode_scale(ids, s)
            if s + 1 < len(m.scale_shapes):
                content = self._embed(previous[s], self._down(prefix, shape), s, 3, annual)
        result = tuple(generated)
        if return_latent:
            return dict(tokens=result, latent=prefix)
        return result


class KVRollout:
    """Own-prediction state; reuse decoded latents, reset all Transformer KV daily."""

    def __init__(self, engine, observed_history):
        self.engine = engine
        self.latents = engine.initial_latents(observed_history)

    @torch.no_grad()
    def step(self, target_day, **kwargs):
        result = self.engine.generate(
            None, target_day, history_latents=self.latents, return_latent=True, **kwargs
        )
        self.latents = [self.latents[1], result["latent"]]
        return result["tokens"]
