"""Daily ERA5 blocks backed by the normalized NPY cache."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_HOURS = np.asarray([0, 6, 12, 18], dtype=np.int64)


def _validate_global_coordinates(latitude: np.ndarray, longitude: np.ndarray) -> None:
    latitude, longitude = np.asarray(latitude), np.asarray(longitude)
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("latitude/longitude axes must be finite")
    if (
        latitude.min() < -90.001
        or latitude.max() > 90.001
        or not (latitude.min() < 0 < latitude.max())
    ):
        raise ValueError("latitude axis must stay within [-90,90] and cross the equator")
    lat_step, lon_step = np.diff(latitude), np.diff(longitude)
    if not len(lat_step) or not len(lon_step):
        raise ValueError("latitude/longitude axes must contain multiple points")
    if not (np.all(lat_step > 0) or np.all(lat_step < 0)) or not np.all(lon_step > 0):
        raise ValueError("latitude must be monotonic and longitude strictly increasing")
    lat_spacing, lon_spacing = np.median(np.abs(lat_step)), np.median(lon_step)
    if not np.allclose(np.abs(lat_step), lat_spacing, rtol=1e-4, atol=1e-5):
        raise ValueError("latitude axis must be uniformly spaced")
    if not np.allclose(lon_step, lon_spacing, rtol=1e-4, atol=1e-5):
        raise ValueError("longitude axis must be uniformly spaced")
    if not np.isclose(lon_spacing * len(longitude), 360.0, rtol=1e-4, atol=1e-3):
        raise ValueError("longitude axis must cover one complete global period")
    if not np.isclose(lat_spacing * len(latitude), 180.0, rtol=1e-3, atol=lat_spacing + 1e-3):
        raise ValueError("latitude axis does not cover the global domain")


def _split_bounds(config: Mapping, split: str) -> tuple[object, object]:
    """Read either ``train_start`` keys or a nested ``splits`` mapping."""
    if f"{split}_start" in config and f"{split}_end" in config:
        return config[f"{split}_start"], config[f"{split}_end"]
    ranges = config.get("splits", config.get("split_dates", {}))
    value = ranges.get(split) if isinstance(ranges, Mapping) else None
    if isinstance(value, Mapping):
        return value["start"], value["end"]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0], value[1]
    raise KeyError(f"no date bounds configured for split {split!r}")


def _path(config: Mapping, *names: str, required: bool = True):
    for name in names:
        if config.get(name) is not None:
            return config[name]
    if required:
        raise KeyError(f"one of {names!r} is required")
    return None


class AtmosphereDayDataset(Dataset):
    """Return either four 6-hour fields or one materialized daily-mean field.

    ``day`` is the NumPy/Unix day ordinal and ``frame_times`` contains Unix
    seconds.  Integer timestamps keep PyTorch's default collation lossless.
    """

    @classmethod
    def from_config(cls, config: Mapping, split: str) -> "AtmosphereDayDataset":
        start, end = _split_bounds(config, split)
        return cls(
            data_path=_path(config, "data_path", "cache_path", "path"),
            times_path=_path(config, "times_path"),
            start=start,
            end=end,
            split=split,
            latitude_path=_path(config, "latitude_path", "latitudes_path", required=False),
            longitude_path=_path(config, "longitude_path", required=False),
            metadata_path=_path(config, "metadata_path", required=False),
            stats_path=_path(config, "statistics_path", "stats_path", required=False),
            variable_names=config.get("variable_names"),
            channels=config.get("expected_channels", config.get("channels", 69)),
            height=config.get("crop_latitude", config.get("height", 120)),
            width=config.get("width", 240),
            temporal_aggregation=config.get("temporal_aggregation", "four_frames"),
        )

    def __init__(
        self,
        data_path: str | Path,
        times_path: str | Path,
        start: object,
        end: object,
        *,
        split: str = "unspecified",
        latitude_path: str | Path | None = None,
        longitude_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        stats_path: str | Path | None = None,
        variable_names: list[str] | None = None,
        channels: int = 69,
        height: int = 120,
        width: int = 240,
        temporal_aggregation: str = "four_frames",
    ) -> None:
        self.data_path, self.times_path = Path(data_path), Path(times_path)
        self.split = str(split)
        self.temporal_aggregation = str(temporal_aggregation)
        if self.temporal_aggregation not in {"four_frames", "daily_mean"}:
            raise ValueError("temporal_aggregation must be four_frames or daily_mean")
        self.channels, self.height, self.width = map(int, (channels, height, width))
        if min(self.channels, self.height, self.width) < 1:
            raise ValueError("channels, height and width must be positive")

        cache = np.load(self.data_path, mmap_mode="r", allow_pickle=False)
        expected = (self.channels, self.height + 1, self.width)
        if cache.ndim != 4 or tuple(cache.shape[1:]) != expected:
            raise ValueError(
                f"ERA5 cache must have shape [N,{expected[0]},{expected[1]},"
                f"{expected[2]}], found {tuple(cache.shape)}"
            )
        frame_count = len(cache)
        del cache

        raw_times = np.load(self.times_path, allow_pickle=False)
        try:
            times = np.asarray(raw_times).astype("datetime64[s]")
        except (TypeError, ValueError) as error:
            raise ValueError("ERA5 times must be datetime-like") from error
        if times.ndim != 1 or len(times) != frame_count or np.isnat(times).any():
            raise ValueError("ERA5 timestamps must be a finite 1-D array matching N")
        if len(times) > 1 and np.any(np.diff(times) <= np.timedelta64(0, "s")):
            raise ValueError("ERA5 timestamps must be strictly increasing")

        first = np.datetime64(start, "D")
        last = np.datetime64(end, "D")
        if first > last:
            raise ValueError("split start must not be after split end")
        dates = times.astype("datetime64[D]")
        selected = np.flatnonzero((dates >= first) & (dates <= last))
        if not len(selected):
            raise ValueError(f"split {self.split!r} contains no ERA5 frames")
        selected_times = times[selected]
        if self.temporal_aggregation == "daily_mean":
            selected_days = selected_times.astype("datetime64[D]")
            if np.any(selected_times != selected_days.astype("datetime64[s]")):
                raise ValueError("daily-mean timestamps must be UTC midnight dates")
            if len(selected_days) > 1 and np.any(np.diff(selected_days) != np.timedelta64(1, "D")):
                raise ValueError("daily-mean dates must be consecutive without duplicates")
            self._indices = selected[:, None].astype(np.int64, copy=False)
            # Noon represents the centre of the 00/06/12/18 UTC averaging window.
            self._frame_times = times[self._indices] + np.timedelta64(12, "h")
            self.days = selected_days
            self.day_ids = selected_days.astype(np.int64)
        else:
            offsets = (
                (selected_times - selected_times.astype("datetime64[D]")) / np.timedelta64(1, "h")
            ).astype(np.int64)
            if np.any(~np.isin(offsets, _HOURS)):
                raise ValueError("ERA5 frames must occur only at 00/06/12/18 UTC")

            unique_days, starts, counts = np.unique(
                selected_times.astype("datetime64[D]"),
                return_index=True,
                return_counts=True,
            )
            if np.any(counts != 4):
                raise ValueError("every selected ERA5 natural day must contain four frames")
            rows = np.stack([selected[begin : begin + 4] for begin in starts])
            if any(not np.array_equal(offsets[begin : begin + 4], _HOURS) for begin in starts):
                raise ValueError("each ERA5 day must be ordered 00/06/12/18 UTC")
            self._indices = rows.astype(np.int64, copy=False)
            self._frame_times = times[self._indices]
            self.days = unique_days
            self.day_ids = unique_days.astype(np.int64)
        self.split_ids = np.full(len(self.days), self.split, dtype=object)
        self.latitudes = self._load_latitudes(latitude_path)
        self.latitude = self.latitudes.numpy()
        self.longitude = self._load_longitudes(longitude_path)
        _validate_global_coordinates(self.latitude, self.longitude)
        metadata = {}
        if metadata_path:
            metadata_file = Path(metadata_path)
            if not metadata_file.is_file():
                raise FileNotFoundError(f"configured ERA5 metadata is missing: {metadata_file}")
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            if (
                self.temporal_aggregation == "daily_mean"
                and metadata.get("temporal_aggregation") != "four_frame_mean_00_06_12_18_utc"
            ):
                raise ValueError("daily-mean cache metadata has an unexpected aggregation")
        names = variable_names or metadata.get("variable_names", metadata.get("channel_names"))
        self.variable_names = (
            [str(name) for name in names]
            if names
            else [f"channel_{i}" for i in range(self.channels)]
        )
        if len(self.variable_names) != self.channels:
            raise ValueError("ERA5 variable_names must match the channel count")
        self.mean = self.std = None
        if stats_path:
            stats = torch.load(stats_path, map_location="cpu", weights_only=False)
            stats_names = stats.get("variable_names", stats.get("channel_names"))
            if stats_names is not None and list(stats_names) != self.variable_names:
                raise ValueError("ERA5 statistics do not match metadata variable order")
            mean = torch.as_tensor(stats["mean"], dtype=torch.float32).reshape(-1)
            std = torch.as_tensor(stats["std"], dtype=torch.float32).reshape(-1)
            if mean.shape != (self.channels,) or std.shape != (self.channels,):
                raise ValueError("ERA5 statistics must contain one scalar per channel")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("ERA5 mean/std must be finite and std strictly positive")
            self.mean = mean.reshape(self.channels, 1, 1)
            self.std = std.reshape(self.channels, 1, 1)
        self._cache = None

    def _load_latitudes(self, path: str | Path | None) -> torch.Tensor:
        if path is None:
            values = np.linspace(90.0, -90.0, self.height + 1)[: self.height]
        else:
            values = np.load(path, allow_pickle=False)
            if values.ndim != 1 or len(values) != self.height + 1:
                raise ValueError("ERA5 latitude axis must match the uncropped 121 rows")
            values = values[: self.height]
        return torch.as_tensor(np.array(values, dtype=np.float32, copy=True))

    def _load_longitudes(self, path: str | Path | None) -> np.ndarray:
        values = (
            np.linspace(0.0, 360.0, self.width, endpoint=False)
            if path is None
            else np.load(path, allow_pickle=False)
        )
        if np.asarray(values).ndim != 1 or len(values) != self.width:
            raise ValueError("ERA5 longitude axis must match field width")
        return np.array(values, dtype=np.float32, copy=True)

    def denormalize(self, field: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return field
        shape = (1,) * (field.ndim - 3) + (self.channels, 1, 1)
        return field * self.std.to(field).reshape(shape) + self.mean.to(field).reshape(shape)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache"] = None
        return state

    def __len__(self) -> int:
        return len(self._indices)

    def frame_times(self, index: int) -> np.ndarray:
        return np.array(self._frame_times[index], copy=True)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._cache is None:
            self._cache = np.load(self.data_path, mmap_mode="r", allow_pickle=False)
        indices = self._indices[index]
        field = np.array(self._cache[indices, :, : self.height, :], dtype=np.float32, copy=True)
        frame_times = self._frame_times[index].astype(np.int64)
        return {
            "field": torch.from_numpy(field),
            "day": torch.tensor(int(self.day_ids[index]), dtype=torch.int64),
            "frame_times": torch.from_numpy(np.array(frame_times, copy=True)),
        }


ERA5DayDataset = AtmosphereDayDataset


__all__ = ["AtmosphereDayDataset", "ERA5DayDataset"]
