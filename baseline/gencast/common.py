"""Shared paths, immutable data contract, atomic records; no GPU initialization."""

import json
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parent
from weathercraft.config import PROJECT_ROOT, load_config

CFG = load_config(PROJECT_ROOT / "configs/baseline/gencast.yaml")
OUT = Path(CFG["output_dir"])
PIN = "bf9785fa72a0f5c3eec5b516d5ea154ab07f12ab"
UPSTREAM = CODE / "vendor"
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
NAMES = [f"{v}_{p}" for v in ["z", "t", "q", "u", "v"] for p in LEVELS] + [
    "t2m",
    "u10",
    "v10",
    "msl",
]


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    t.replace(p)


class Data:

    def __init__(self, require_residual=True):
        root = Path(CFG["data_root"])
        cache = root / "daily"
        self.fields = np.load(cache / "fields.npy", mmap_mode="r", allow_pickle=False)
        self.days = np.load(cache / "times.npy", allow_pickle=False).astype("datetime64[D]")
        assert self.fields.shape == (len(self.days), 69, 121, 240)
        assert np.all(np.diff(self.days) == np.timedelta64(1, "D"))
        self.raw_lat = np.load(root / "latitude.npy")[:120].astype(np.float32)
        self.order = np.argsort(self.raw_lat)
        self.lat = self.raw_lat[self.order]
        self.lon = np.load(root / "longitude.npy").astype(np.float32)
        assert np.allclose(self.lon, np.arange(240) * 1.5)
        assert np.allclose(np.diff(self.lat), 1.5)
        stats = Path(CFG["normalization"])
        self.stats = json.loads(stats.read_text())
        assert self.stats["variable_names"] == NAMES
        assert self.stats["fit_split"] == CFG["split"]["train"]
        self.shift = np.array(self.stats["cache_mean"], np.float32)[:, None, None]
        self.scale = np.array(self.stats["cache_std"], np.float32)[:, None, None]
        self.metadata = json.loads((cache / "metadata.json").read_text())
        assert self.metadata["temporal_aggregation"] == "four_frame_mean_00_06_12_18_utc"
        self.residual = None
        if require_residual:
            r = json.loads(Path(CFG["residual_statistics"]).read_text())
            assert r["fit_split"] == CFG["split"]["train"] and r["pairs"] == 14609
            self.residual = np.array(r["normalized_daily_difference_std"], np.float32)
            self.physical_mean = np.array(r["physical_mean"], np.float32)
            self.physical_std = np.array(r["physical_std"], np.float32)

    def field(self, index):
        x = np.asarray(self.fields[index, :, :120, :], np.float32)
        return np.ascontiguousarray(((x - self.shift) / self.scale)[:, self.order, :])

    def centers(self, split):
        a, b = map(np.datetime64, CFG["split"][split])
        d = self.days
        return np.flatnonzero((d >= a + np.timedelta64(1, "D")) & (d <= b - np.timedelta64(1, "D")))

    def test_starts(self):
        a, b = map(np.datetime64, CFG["split"]["test"])
        return np.flatnonzero((self.days >= a) & (self.days <= b - np.timedelta64(15, "D")))

    def batch(self, indices):
        ids = np.asarray(indices)
        x = np.stack([np.stack([self.field(int(i) - 1), self.field(int(i))]) for i in ids])
        y = np.stack([self.field(int(i) + 1) for i in ids])
        phase = (self.days[ids].astype(np.int64) + 0.375) / 365.24219 % 1
        return (x, y, phase.astype(np.float32))


def epoch_indices(ids, epoch):
    a = np.random.default_rng(CFG["seed"] + epoch).permutation(ids)
    return a[: len(a) // 32 * 32]
