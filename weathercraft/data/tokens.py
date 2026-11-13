"""Memory-mapped multi-scale token datasets."""

from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .cache_writer import TokenCacheWriter
from .pairs import DayPairDataset


def seasonal_features(day_id: int) -> torch.Tensor:
    """Leap-aware annual sine/cosine for a NumPy/Unix day ordinal."""
    day = np.datetime64(int(day_id), "D")
    year = day.astype("datetime64[Y]")
    start = year.astype("datetime64[D]")
    end = (year + np.timedelta64(1, "Y")).astype("datetime64[D]")
    elapsed = float((day - start) / np.timedelta64(1, "D"))
    duration = float((end - start) / np.timedelta64(1, "D"))
    phase = 2.0 * math.pi * elapsed / duration
    return torch.tensor([math.sin(phase), math.cos(phase)], dtype=torch.float32)


class TokenDayDataset(Dataset):
    """Read a completed multi-scale token cache."""

    def __init__(self, root: str | Path, split: str | None = None) -> None:
        root = Path(root)
        self.root = root / split if split is not None and (root / split).exists() else root
        path = self.root / "metadata.json"
        self.metadata = json.loads(path.read_text(encoding="utf-8"))
        if self.metadata.get("format") != "weathercraft-token-cache-v1" or not self.metadata.get(
            "complete"
        ):
            raise ValueError("unsupported or incomplete token cache")
        if split is not None and self.metadata.get("split") != split:
            raise ValueError("requested split does not match token-cache metadata")
        self.count = int(self.metadata["count"])
        self.num_codebooks = int(self.metadata["num_codebooks"])
        self.vocab_size = self.metadata.get("vocab_size")
        self.modality = self.metadata.get("modality")
        self.scale_shapes = tuple((tuple(map(int, item)) for item in self.metadata["scale_shapes"]))
        if len(self.metadata.get("scale_files", ())) != len(self.scale_shapes):
            raise ValueError("token-cache scale file list does not match scale geometry")
        self.scale_paths = [self.root / name for name in self.metadata["scale_files"]]
        self.day_ids = np.array(
            np.load(self.root / self.metadata["days_file"], mmap_mode="r"), copy=True
        )
        if (
            self.day_ids.dtype != np.int64
            or self.day_ids.shape != (self.count,)
            or (self.count > 1 and np.any(np.diff(self.day_ids) <= 0))
        ):
            raise ValueError("token-cache days must be strictly increasing and match count")
        for scale_path, shape in zip(self.scale_paths, self.scale_shapes):
            value = np.load(scale_path, mmap_mode="r", allow_pickle=False)
            if value.dtype != np.uint16 or value.shape != (self.count, self.num_codebooks, *shape):
                raise ValueError(f"invalid token scale file: {scale_path}")
        self.split = str(self.metadata.get("split", split or "unspecified"))
        self.split_ids = np.full(self.count, self.split, dtype=object)
        self._scales = None

    def validate_schema(self, *, scale_shapes, num_codebooks: int, vocab_size: int) -> None:
        """Reject a cache produced by a different tokenizer geometry."""
        expected_shapes = tuple((tuple(map(int, shape)) for shape in scale_shapes))
        if self.scale_shapes != expected_shapes:
            raise ValueError(
                f"token-cache scale geometry {self.scale_shapes} does not match {expected_shapes}"
            )
        if self.num_codebooks != int(num_codebooks):
            raise ValueError("token-cache codebook-axis count does not match tokenizer")
        if self.vocab_size is None or int(self.vocab_size) != int(vocab_size):
            raise ValueError("token-cache vocabulary does not match tokenizer")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_scales"] = None
        return state

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict:
        if self._scales is None:
            self._scales = [
                np.load(path, mmap_mode="r", allow_pickle=False) for path in self.scale_paths
            ]
        tokens = tuple(
            (torch.from_numpy(np.array(value[index], copy=True)).long() for value in self._scales)
        )
        return {"tokens": tokens, "day": torch.tensor(int(self.day_ids[index]), dtype=torch.int64)}


class TokenPairDataset(DayPairDataset):
    """Flatten next-day token pairs for the dynamics data loader."""

    def __init__(
        self, dataset: TokenDayDataset | str | Path, split: str | None = None, **kwargs
    ) -> None:
        if not isinstance(dataset, TokenDayDataset):
            dataset = TokenDayDataset(dataset, split, **kwargs)
        super().__init__(dataset)
        self.scale_shapes = dataset.scale_shapes
        self.num_codebooks = dataset.num_codebooks
        self.vocab_size = dataset.vocab_size

    def __getitem__(self, index: int) -> dict:
        pair = super().__getitem__(index)
        return {
            "current_tokens": pair["current"]["tokens"],
            "target_tokens": pair["target"]["tokens"],
            "current_day": pair["current"]["day"],
            "target_day": pair["target"]["day"],
            "target_seasonal": seasonal_features(pair["target"]["day"].item()),
        }


class TokenRolloutDataset(Dataset):
    """Leakage-safe contiguous day windows for open-loop curriculum tuning."""

    def __init__(self, dataset: TokenDayDataset, horizon: int) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.dataset = dataset
        self.horizon = int(horizon)
        days = dataset.day_ids
        self.starts = np.asarray(
            [
                index
                for index in range(max(0, len(days) - horizon))
                if np.all(np.diff(days[index : index + horizon + 1]) == 1)
            ],
            dtype=np.int64,
        )
        if not len(self.starts):
            raise ValueError(f"no contiguous {horizon}-day trajectories in {dataset.split}")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict:
        start = int(self.starts[index])
        items = [self.dataset[position] for position in range(start, start + self.horizon + 1)]
        return {
            "current_tokens": items[0]["tokens"],
            "target_tokens": tuple(
                (
                    torch.stack([item["tokens"][scale] for item in items[1:]])
                    for scale in range(len(self.dataset.scale_shapes))
                )
            ),
            "target_seasonal": torch.stack(
                [seasonal_features(int(item["day"])) for item in items[1:]]
            ),
            "start_day": items[0]["day"],
        }


class TokenSeq2SeqDataset(Dataset):
    """Past ``history`` observed days to the next ``horizon`` target days.

    Each window stays inside one split, so validation/test targets never enter
    training. All dates must be consecutive; no index-based year-boundary jumps.
    """

    def __init__(self, dataset: TokenDayDataset, history: int = 10, horizon: int = 10) -> None:
        if history < 1 or horizon < 1:
            raise ValueError("history and horizon must be positive")
        self.dataset = dataset
        self.history, self.horizon = (int(history), int(horizon))
        days = dataset.day_ids
        length = self.history + self.horizon
        self.starts = np.asarray(
            [
                index
                for index in range(max(0, len(days) - length + 1))
                if np.all(np.diff(days[index : index + length]) == 1)
            ],
            dtype=np.int64,
        )
        if not len(self.starts):
            raise ValueError(f"no contiguous {history}->{horizon} windows in {dataset.split}")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict:
        start = int(self.starts[index])
        items = [
            self.dataset[position] for position in range(start, start + self.history + self.horizon)
        ]
        history_items = items[: self.history]
        target_items = items[self.history :]
        scales = range(len(self.dataset.scale_shapes))
        return {
            "history_tokens": tuple(
                (torch.stack([item["tokens"][scale] for item in history_items]) for scale in scales)
            ),
            "target_tokens": tuple(
                (torch.stack([item["tokens"][scale] for item in target_items]) for scale in scales)
            ),
            "target_seasonal": torch.stack(
                [seasonal_features(int(item["day"])) for item in target_items]
            ),
            "last_history_day": history_items[-1]["day"],
            "target_days": torch.stack([item["day"] for item in target_items]),
        }
