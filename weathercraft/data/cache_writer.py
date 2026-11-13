"""Multi-scale token-cache writer."""

from __future__ import annotations
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import numpy as np
import torch


def json_bytes(value: object) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return (text + "\n").encode()


def as_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class TokenCacheWriter:
    """Incrementally write one uint16 NPY memmap per residual-VQ scale."""

    def __init__(
        self,
        root: str | Path,
        count: int,
        scale_shapes: Sequence[Sequence[int]],
        num_codebooks: int,
        *,
        split: str = "unspecified",
        modality: str | None = None,
        vocab_size: int | None = None,
        metadata: Mapping | None = None,
        metadata_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> None:
        if overwrite:
            raise ValueError("in-place token-cache overwrite is intentionally unsupported")
        self.root, self.count = (Path(root), int(count))
        self.scale_shapes = tuple(((int(v[0]), int(v[1])) for v in scale_shapes))
        self.num_codebooks = int(num_codebooks)
        if self.count < 1 or self.num_codebooks < 1:
            raise ValueError("count and num_codebooks must be positive")
        if not self.scale_shapes or any((min(shape) < 1 for shape in self.scale_shapes)):
            raise ValueError("scale_shapes must contain positive (height, width) pairs")
        self.root.mkdir(parents=True, exist_ok=True)
        self.split, self.modality, self.vocab_size = (str(split), modality, vocab_size)
        if self.vocab_size is not None and (not 1 <= int(self.vocab_size) <= 65536):
            raise ValueError("vocab_size must fit uint16")
        self.source_metadata = dict(metadata or {})
        self.checkpoint_name = Path(checkpoint_path).name if checkpoint_path else None
        names = ["days.npy", "metadata.json"]
        names += [f"scale_{scale:02d}.npy" for scale in range(len(self.scale_shapes))]
        conflicts = [self.root / name for name in names if (self.root / name).exists()]
        if conflicts:
            raise FileExistsError(f"token cache file already exists: {conflicts[0]}")
        self._days = np.lib.format.open_memmap(
            self.root / "days.npy", mode="w+", dtype=np.int64, shape=(self.count,)
        )
        self._scales = [
            np.lib.format.open_memmap(
                self.root / f"scale_{scale:02d}.npy",
                mode="w+",
                dtype=np.uint16,
                shape=(self.count, self.num_codebooks, height, width),
            )
            for scale, (height, width) in enumerate(self.scale_shapes)
        ]
        self._written = np.zeros(self.count, dtype=bool)
        self._closed = False

    def write(self, index: int, tokens: Sequence, day) -> None:
        values = [as_array(value) for value in tokens]
        if any((value.ndim != 3 for value in values)):
            raise ValueError("one-day tokens must have shape [Q,Hs,Ws]")
        if isinstance(day, torch.Tensor):
            day = day.detach().cpu().item()
        self.write_batch(index, [value[None] for value in values], [day])

    def write_batch(self, start: int, tokens: Sequence, days) -> None:
        if self._closed:
            raise RuntimeError("token cache writer is closed")
        values = [as_array(value) for value in tokens]
        expected = [(self.num_codebooks, *shape) for shape in self.scale_shapes]
        if len(values) != len(expected) or any(
            (
                value.ndim != 4 or tuple(value.shape[1:]) != shape
                for value, shape in zip(values, expected)
            )
        ):
            raise ValueError("token pyramid conflicts with configured scale geometry")
        sizes = {len(value) for value in values}
        if len(sizes) != 1:
            raise ValueError("all token scales must have the same batch size")
        size = sizes.pop()
        start, end = (int(start), int(start) + size)
        if start < 0 or end > self.count or self._written[start:end].any():
            raise IndexError("token cache write is out of range or overlaps prior data")
        day_values = as_array(days).astype(np.int64, copy=False).reshape(-1)
        if len(day_values) != size:
            raise ValueError("day count must match token batch size")
        for destination, value in zip(self._scales, values):
            if not np.issubdtype(value.dtype, np.integer):
                raise TypeError("token ids must be integers")
            maximum = 65535 if self.vocab_size is None else int(self.vocab_size) - 1
            if value.size and (int(value.min()) < 0 or int(value.max()) > maximum):
                raise ValueError("token id cannot be represented by uint16")
            destination[start:end] = value.astype(np.uint16, copy=False)
        self._days[start:end], self._written[start:end] = (day_values, True)

    def close(self) -> Path:
        if self._closed:
            return self.root / "metadata.json"
        if not self._written.all():
            raise RuntimeError(f"cannot finalize cache: {(~self._written).sum()} days missing")
        if self.count > 1 and np.any(np.diff(self._days) <= 0):
            raise ValueError("token-cache days must be strictly increasing")
        for value in [self._days, *self._scales]:
            value.flush()
        metadata_record = {
            "format": "weathercraft-token-cache-v1",
            "complete": True,
            "split": self.split,
            "modality": self.modality,
            "count": self.count,
            "dtype": "uint16",
            "num_codebooks": self.num_codebooks,
            "scale_shapes": [list(shape) for shape in self.scale_shapes],
            "scale_files": [f"scale_{i:02d}.npy" for i in range(len(self.scale_shapes))],
            "days_file": "days.npy",
            "first_day": int(self._days[0]),
            "last_day": int(self._days[-1]),
            "vocab_size": self.vocab_size,
            "source_metadata": self.source_metadata,
            "checkpoint_name": self.checkpoint_name,
        }
        path = self.root / "metadata.json"
        path.write_bytes(json_bytes(metadata_record))
        self._closed = True
        for value in [self._days, *self._scales]:
            if getattr(value, "_mmap", None) is not None:
                value._mmap.close()
        self._days, self._scales = (None, [])
        return path

    def __enter__(self) -> "TokenCacheWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
