from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Protocol, Sequence, runtime_checkable

import torch

TokenPyramid = tuple[torch.Tensor, ...]
LogitPyramid = tuple[torch.Tensor, ...]
FactorLogits = tuple[torch.Tensor, ...]
FactorLogitPyramid = tuple[FactorLogits, ...]


@runtime_checkable
class LatentCodec(Protocol):
    num_codebooks: int
    codebook_dim: int
    vocab_size: int
    scale_shapes: Sequence[Sequence[int]]

    def tokens_to_latent(
        self, tokens: TokenPyramid, max_scales: int | None = None
    ) -> torch.Tensor: ...
    def prefix_latents(self, tokens: TokenPyramid) -> list[torch.Tensor]: ...
    def context_latents(self, tokens: TokenPyramid) -> list[torch.Tensor]: ...


@dataclass
class DynamicsOutput:
    logits: LogitPyramid | FactorLogitPyramid
    loss: torch.Tensor
    generated_tokens: TokenPyramid | None = None


def _shapes(value: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    shapes = tuple((int(shape[0]), int(shape[1])) for shape in value)
    if not shapes or any(h < 1 or w < 1 for h, w in shapes):
        raise ValueError("scale_shapes must contain positive (height, width) pairs")
    if any(a[0] > b[0] or a[1] > b[1] for a, b in zip(shapes, shapes[1:])):
        raise ValueError("scale_shapes must be coarse-to-fine")
    return shapes


@dataclass(frozen=True)
class ScaleLayout:
    """Bookkeeping for a coarse-to-fine sequence of spatial scale blocks."""

    shapes: tuple[tuple[int, int], ...]

    def __init__(self, shapes: Sequence[Sequence[int]]):
        object.__setattr__(self, "shapes", _shapes(shapes))

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(prod(shape) for shape in self.shapes)

    @property
    def offsets(self) -> tuple[int, ...]:
        starts, offset = [], 0
        for length in self.lengths:
            starts.append(offset)
            offset += length
        return tuple(starts)

    @property
    def ends(self) -> tuple[int, ...]:
        """Exclusive end offset for each coarse-to-fine scale block."""
        return tuple(start + length for start, length in zip(self.offsets, self.lengths))

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    def block_slice(self, scale: int) -> slice:
        start = self.offsets[scale]
        return slice(start, start + self.lengths[scale])

    def scale_ids(self, device: torch.device | None = None) -> torch.Tensor:
        parts = [
            torch.full((n,), scale, device=device, dtype=torch.long)
            for scale, n in enumerate(self.lengths)
        ]
        return torch.cat(parts)

    def spatial_ids(self, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols = [], []
        for height, width in self.shapes:
            rows.append(torch.arange(height, device=device).repeat_interleave(width))
            cols.append(torch.arange(width, device=device).repeat(height))
        return torch.cat(rows), torch.cat(cols)


def build_scale_block_mask(
    scale_shapes: Sequence[Sequence[int]] | ScaleLayout,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a bool attention mask where True entries are forbidden.

    A query may attend every token from its own scale and all coarser scales,
    while keys belonging to a finer (future) scale are blocked.
    """

    layout = scale_shapes if isinstance(scale_shapes, ScaleLayout) else ScaleLayout(scale_shapes)
    scale = layout.scale_ids(device)
    return scale.unsqueeze(0) > scale.unsqueeze(1)


scale_block_causal_mask = build_scale_block_mask
