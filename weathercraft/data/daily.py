"""Training-only normalization and two-day observed-history contracts."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .atmosphere import AtmosphereDayDataset
from .tokens import seasonal_features


class TrainingNormalizedDays(Dataset):
    """Affine re-normalization from existing cache to train-1979:2018 statistics."""

    def __init__(self, days, stats_path):
        self.base = days
        stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        assert stats["fit_split"] == ["1979-01-01", "2018-12-31"]
        assert stats["variable_names"] == list(days.variable_names)
        self.shift = torch.tensor(stats["cache_mean"], dtype=torch.float32)[:, None, None]
        self.scale = torch.tensor(stats["cache_std"], dtype=torch.float32)[:, None, None]
        self.mean = days.mean + days.std * self.shift
        self.std = days.std * self.scale
        for name in ("days", "day_ids", "latitudes", "variable_names"):
            setattr(self, name, getattr(days, name))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = dict(self.base[i])
        item["field"] = (item["field"] - self.shift) / self.scale
        return item


def field_dataset(cfg, split):
    ds = AtmosphereDayDataset.from_config(cfg["data"], split)
    return TrainingNormalizedDays(ds, cfg["data"]["train_stats_path"])


def complete_starts(days, horizon):
    ids = days.day_ids
    return [
        i
        for i in range(len(ids) - horizon)
        if np.array_equal(ids[i : i + horizon + 1] - ids[i], np.arange(horizon + 1))
    ]


def channel_mse(pred, truth, lat):
    weights = torch.cos(torch.deg2rad(lat.float())).clamp_min(0)[None, None, :, None]
    return ((pred.float() - truth.float()).square() * weights).sum((0, 2, 3)) / (
        weights.sum() * pred.shape[0] * pred.shape[-1]
    )


def groups(names):
    result = {
        g: [i for i, n in enumerate(names) if n.startswith(g + "_")]
        for g in ("z", "t", "q", "u", "v")
    }
    result["surface"] = list(range(65, 69))
    result.update(
        {
            n: [names.index(n)]
            for n in ("z_500", "t_850", "q_700", "u_850", "v_850", "t2m", "msl", "u10", "v10")
        }
    )
    return result


class TwoDayTokenPairs(Dataset):
    def __init__(self, days, prior=None):
        self.days, self.prior = days, prior
        pair_starts = np.flatnonzero(np.diff(days.day_ids) == 1)
        self.indices = [
            int(i)
            for i in pair_starts
            if (i > 0 and days.day_ids[i] - days.day_ids[i - 1] == 1)
            or (i == 0 and prior is not None and days.day_ids[0] - prior.day_ids[-1] == 1)
        ]
        self.day_ids = days.day_ids[self.indices]
        self.target_day_ids = self.day_ids + 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        old = self.days[i - 1] if i else self.prior[len(self.prior) - 1]
        now, target = self.days[i], self.days[i + 1]
        return dict(
            current_tokens=tuple(torch.stack((a, b)) for a, b in zip(old["tokens"], now["tokens"])),
            target_tokens=target["tokens"],
            current_day=now["day"],
            target_day=target["day"],
            target_seasonal=seasonal_features(int(target["day"])),
        )


def start_history(days, index, device, prior=None):
    previous = days[index - 1] if index else prior[len(prior) - 1]
    current = days[index]
    if int(current["day"]) - int(previous["day"]) != 1:
        raise ValueError("Observed history must contain consecutive days")
    return tuple(
        torch.stack((a, b))[None].to(device) for a, b in zip(previous["tokens"], current["tokens"])
    )


def advance_history(history, prediction):
    return tuple(torch.stack((h[:, 1], p), dim=1) for h, p in zip(history, prediction))


def expand_members(history, members):
    return tuple(value.expand(members, *([-1] * (value.ndim - 1))).clone() for value in history)
