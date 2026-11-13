"""Daily field baselines share WeatherCraft normalization and split boundaries."""

import numpy as np
import torch
from torch.utils.data import Dataset
from weathercraft.data.daily import field_dataset as normalized_days
from weathercraft.losses import latitude_weighted_mse


class DailyPairs(Dataset):
    def __init__(self, days):
        self.days = days
        self.indices = np.flatnonzero(np.diff(days.day_ids) == 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return self.days[j]["field"][0], self.days[j + 1]["field"][0]


def weighted_loss(pred, truth, lat):
    return latitude_weighted_mse(pred.float()[:, None], truth.float()[:, None], lat.float())


class ContextDays(Dataset):
    def __init__(self, days, prior=None):
        self.base = days
        self.prior = prior
        for n in ("days", "day_ids", "latitudes", "variable_names", "mean", "std"):
            setattr(self, n, getattr(days, n))
        if prior is not None:
            assert self.day_ids[0] - prior.day_ids[-1] == 1

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.base[i]

    def history(self, i):
        if i == 0:
            assert self.prior is not None, "No pre-training history outside train split"
            previous = self.prior[len(self.prior) - 1]
        else:
            assert self.day_ids[i] - self.day_ids[i - 1] == 1
            previous = self.base[i - 1]
        return torch.cat((previous["field"], self.base[i]["field"]), dim=1)


class TwoDayPairs(Dataset):
    def __init__(self, days):
        self.days = days
        self.indices = [
            int(i)
            for i in np.flatnonzero(np.diff(days.day_ids) == 1)
            if (i > 0 and days.day_ids[i] - days.day_ids[i - 1] == 1)
            or (i == 0 and days.prior is not None)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = self.indices[j]
        return self.days.history(i)[0], self.days[i + 1]["field"][0]


def field_dataset(config, split):
    days = normalized_days(config, split)
    if config["model"].get("history_days", 1) == 1:
        return days
    # The preceding split supplies observed history only, never a target.
    prior = (
        normalized_days(config, {"valid": "train", "test": "valid"}[split])
        if split != "train"
        else None
    )
    return ContextDays(days, prior)


def training_pairs(days):
    return TwoDayPairs(days) if isinstance(days, ContextDays) else DailyPairs(days)
