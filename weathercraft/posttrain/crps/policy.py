"""Hard Gumbel tokens with full scale/time STE gradients through a frozen decoder."""

from functools import partial
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts
from weathercraft.models.dynamics.weather import WeatherDynamics, attention_checkpoint_policy
from weathercraft.models.tokenizer.multiscale_fsq import _resize


class ReferencePolicy(WeatherDynamics):
    def _hidden(self, sequence, ends):
        # Keep dropout/drop-path OFF during both sampling and likelihood replay.
        # Activation checkpointing remains enabled for gradient-bearing replay.
        assert not self.training
        for block in self.transformer.blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
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

    def log_probability(self, pair, target):
        # Sum both factor log probabilities, divide by the fixed count of COMPOSITE tokens.
        # This constant rescales the fair-CRPS policy gradient, without changing its direction.
        total = 0
        for logits, ids in zip(pair, self.codec.split_prediction_ids(target)):
            total = total - F.cross_entropy(
                logits.float().flatten(0, 1), ids.long().flatten(0, 1), reduction="none"
            ).reshape(target.shape).flatten(1).sum(1)
        return total

    def forward(self, history, tokens, day):
        sequence, ends, queries = self.build_sequence(history, tokens, day)
        hidden = self._hidden(sequence, ends)
        return torch.stack(
            [
                self.log_probability(self.query_logits(hidden[:, q], s), tokens[s])
                for s, q in enumerate(queries)
            ],
            dim=1,
        ) / sum(t[0].numel() for t in tokens)

    @torch.no_grad()
    def sample(self, history, day):
        tokens = [torch.zeros_like(v[:, 1]) for v in history]
        logps = []
        for s in range(len(tokens)):
            # Always use all query/content blocks. Future placeholders cannot affect this query.
            seq, ends, queries = self.build_sequence(history, tokens, day)
            hidden = self._hidden(seq, ends)
            pair = self.query_logits(hidden[:, queries[s]], s)
            factors = [self._draw(v, False, 1.0, None) for v in pair]
            tokens[s] = self.codec.merge_prediction_ids(*factors)
            logps.append(self.log_probability(pair, tokens[s]))
        return tuple(tokens), torch.stack(logps, 1) / sum(t[0].numel() for t in tokens)


def distance(x, y, latitude):
    w = torch.cos(torch.deg2rad(latitude.float())).clamp_min(0)[None, None, :, None]
    return ((x.float() - y.float()).abs() * w).sum((1, 2, 3)) / (w.sum() * x.shape[1] * x.shape[-1])


def rewards(first, second, truth, latitude):
    e1 = distance(first, truth, latitude)
    e2 = distance(second, truth, latitude)
    delta = distance(first, second, latitude)
    # NO sibling/within-pair centering: that would cancel Delta for two members.
    return (
        torch.stack((-e1 + delta, -e2 + delta)),
        0.5 * (e1 + e2 - delta),
        torch.stack((e1, e2, delta)),
    )


class Policy(ReferencePolicy):
    """Hard categorical forward, stochastic soft backward, full scale unroll."""

    def code_vectors(self, ids):
        return self.codec.fsq.indices_to_codes(ids.long()).permute(0, 2, 1, 3, 4).float()

    def contributions(self, vectors):
        assert self.codec.num_slots == 1 and all(x == 4 for x in self.codec.codebooks_per_scale)
        out = []
        for s, v in enumerate(vectors):
            z = v.unsqueeze(3)
            out.append(
                self.codec._flat(
                    _resize(z * self.codec._gain(s, z), self.scale_shapes[-1], mode="bicubic")
                )
            )
        return out

    def vector_sequence(self, context, vectors, day):
        histories = self._history(context)
        annual = self._day_embedding(day, vectors[0].shape[0], histories[0])
        parts = [
            self._embed(v, torch.zeros_like(v), len(self.scale_shapes) - 1, d, annual)
            for d, v in enumerate(histories)
        ]
        length = sum(v.shape[1] for v in parts)
        ends = [length]
        queries = []
        previous = self.codec.context_latents(tuple(v[:, 1] for v in context))
        contributions = self.contributions(vectors)
        prefix = torch.zeros_like(contributions[0])
        for s, shape in enumerate(self.scale_shapes):
            coarse = self.codec._flat(_resize(self.codec._structured(prefix), shape, mode="area"))
            query = self._embed(previous[s], coarse, s, 2, annual)
            parts.append(query)
            queries.append(slice(length, length + query.shape[1]))
            length += query.shape[1]
            ends.append(length)
            prefix = prefix + contributions[s]
            if s + 1 < len(vectors):
                content = F.interpolate(prefix, size=shape, mode="area")
                value = self._embed(previous[s], content, s, 3, annual)
                parts.append(value)
                length += value.shape[1]
                ends.append(length)
        return torch.cat(parts, 1), tuple(ends), queries

    def gumbel_codes(self, pair, scale, tau, generator):
        assert tau > 0
        vectors = []
        ids = []
        with torch.autocast(pair[0].device.type, enabled=False):
            for f, logits in enumerate(pair):
                logits = logits.float()
                u = torch.rand(
                    logits.shape, device=logits.device, generator=generator, dtype=torch.float32
                )
                u = u.clamp(torch.finfo(u.dtype).eps, 1 - torch.finfo(u.dtype).eps)
                noisy = logits - torch.log(-torch.log(u))
                index = noisy.argmax(2)
                ids.append(index)
                q = (noisy / tau).softmax(2)
                values = torch.arange(logits.shape[2], device=logits.device) * (
                    1 if f == 0 else 512
                )
                table = self.codec.fsq.indices_to_codes(values)[:, f * 3 : (f + 1) * 3].float()
                soft = torch.einsum("bqvhw,vd->bqdhw", q, table)
                hard = F.embedding(index, table).permute(0, 1, 4, 2, 3)
                vectors.append(hard + (soft - soft.detach()))
        return torch.cat(vectors, 2), self.codec.merge_prediction_ids(*ids)

    def rollout(self, history, day, tau, seed, detach_prefix=False):
        generator = torch.Generator(device=history[0].device).manual_seed(int(seed))
        vectors = [self.code_vectors(torch.zeros_like(t[:, 1])) for t in history]
        tokens = []
        leaves = []
        for s in range(len(vectors)):
            context = [v.detach() if detach_prefix else v for v in vectors]
            seq, ends, queries = self.vector_sequence(history, context, day)
            hidden = self._hidden(seq, ends)
            pair = self.query_logits(hidden[:, queries[s]], s)
            vectors[s], ids = self.gumbel_codes(pair, s, tau, generator)
            tokens.append(ids)
            leaves.append(vectors[s])
        return sum(self.contributions(vectors)), tuple(tokens), tuple(leaves)

    def latent_sequence(self, histories, vectors, day, max_scales=None):
        max_scales = len(self.scale_shapes) if max_scales is None else max_scales
        annual = self._day_embedding(day, vectors[0].shape[0], histories[0])
        parts = [
            self._embed(v, torch.zeros_like(v), len(self.scale_shapes) - 1, d, annual)
            for d, v in enumerate(histories)
        ]
        length = sum(v.shape[1] for v in parts)
        ends = [length]
        queries = []
        full = self.codec._structured(histories[-1])
        previous = [
            self.codec._flat(_resize(full, shape, mode="area")) for shape in self.scale_shapes
        ]
        contributions = self.contributions(vectors)
        prefix = torch.zeros_like(contributions[0])
        for s, shape in enumerate(self.scale_shapes[:max_scales]):
            coarse = self.codec._flat(_resize(self.codec._structured(prefix), shape, mode="area"))
            query = self._embed(previous[s], coarse, s, 2, annual)
            parts.append(query)
            queries.append(slice(length, length + query.shape[1]))
            length += query.shape[1]
            ends.append(length)
            prefix = prefix + contributions[s]
            if s + 1 < max_scales:
                content = F.interpolate(prefix, size=shape, mode="area")
                value = self._embed(previous[s], content, s, 3, annual)
                parts.append(value)
                length += value.shape[1]
                ends.append(length)
        return torch.cat(parts, 1), tuple(ends), queries

    def latent_day(self, histories, day, tau, seed):
        generator = torch.Generator(device=histories[0].device).manual_seed(int(seed))
        vectors = [
            self.code_vectors(
                torch.zeros(
                    (histories[0].shape[0], self.num_codebooks, *shape),
                    device=histories[0].device,
                    dtype=torch.long,
                )
            )
            for shape in self.scale_shapes
        ]
        ids = []
        leaves = []
        for s in range(len(vectors)):
            seq, ends, queries = self.latent_sequence(
                histories, vectors, day, s + 1 if getattr(self, "prune_future", True) else None
            )
            hidden = self._hidden(seq, ends)
            pair = self.query_logits(hidden[:, queries[s]], s)
            vectors[s], token = self.gumbel_codes(pair, s, tau, generator)
            ids.append(token)
            leaves.append(vectors[s])
        return sum(self.contributions(vectors)), tuple(ids), tuple(leaves)

    def trajectory(self, history, days, tau, seed, detach_time=False):
        histories = self._history(history)
        latents = []
        tokens = []
        leaves = []
        for lead, day in enumerate(days):
            latent, ids, lv = self.latent_day(histories, day, tau, int(seed) + 1000003 * lead)
            latents.append(latent)
            tokens.append(ids)
            leaves.append(lv)
            histories = [histories[-1], latent.detach() if detach_time else latent]
        return latents, tokens, leaves

    def _latent_only(self, first, second, day, tau, seed):
        return self.latent_day([first, second], day, tau, seed)[0]

    def forward(self, history, days, tau, seeds):
        members = []
        for seed in seeds:
            histories = self._history(history)
            for lead, day in enumerate(days):
                fn = partial(self._latent_only, tau=tau, seed=int(seed) + 1000003 * lead)
                if getattr(self, "day_checkpoint", True) and torch.is_grad_enabled():
                    latent = checkpoint(
                        fn,
                        histories[0],
                        histories[1],
                        day,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                else:
                    latent = fn(histories[0], histories[1], day)
                histories = [histories[-1], latent]
            members.append(latent)
        return torch.stack(members)
