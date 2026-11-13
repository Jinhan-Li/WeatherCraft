"""Leakage-safe next-day pairs."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class DayPairDataset(Dataset):
    """Wrap a daily dataset and retain only same-split, adjacent-day pairs."""

    def __init__(self, dataset: Dataset) -> None:
        if not hasattr(dataset, "day_ids"):
            raise TypeError("wrapped dataset must expose integer day_ids")
        self.dataset = dataset
        days = np.asarray(dataset.day_ids, dtype=np.int64)
        if days.ndim != 1 or len(days) != len(dataset):
            raise ValueError("day_ids must be 1-D and match dataset length")
        labels = getattr(dataset, "split_ids", None)
        if labels is None:
            labels = np.full(len(days), str(getattr(dataset, "split", "unspecified")))
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(days):
            raise ValueError("split_ids must be 1-D and match dataset length")
        valid = (np.diff(days) == 1) & (labels[:-1] == labels[1:])
        left = np.flatnonzero(valid).astype(np.int64)
        self.pair_indices = np.stack((left, left + 1), axis=1)
        self.day_ids = days[left]
        self.target_day_ids = days[left + 1]
        self.split_ids = labels[left]
        self.split = getattr(dataset, "split", "unspecified")
        self.latitudes = getattr(dataset, "latitudes", None)

    def __len__(self) -> int:
        return len(self.pair_indices)

    def __getitem__(self, index: int) -> dict:
        current, target = map(int, self.pair_indices[index])
        return {
            "current": self.dataset[current],
            "target": self.dataset[target],
        }


__all__ = ["DayPairDataset"]
