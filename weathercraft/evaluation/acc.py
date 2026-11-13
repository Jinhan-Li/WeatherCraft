"""Verifying-month-day, training-only uncentered spatial ACC."""

import json
from pathlib import Path
import numpy as np

CALENDAR = np.arange("2000-01-01", "2001-01-01", dtype="datetime64[D]")
LOOKUP = {str(d)[5:]: i for i, d in enumerate(CALENDAR)}


def calendar_index(day):
    return LOOKUP[str(np.datetime64(day, "D"))[5:]]


def load_climatology(directory, config):
    p = Path(directory)
    meta = json.loads((p / "metadata.json").read_text())
    assert meta["fit_split"] == ["1979-01-01", "2018-12-31"] and meta["smoothing"] is False
    a = np.load(p / "mean_normalized.npy", mmap_mode="r")
    assert a.shape == (366, 69, 120, 240)
    return (a, meta)


def score(pred, truth, climate, latitude):
    """Uncentered spatial ACC per case/channel; temporal mean is taken later.

    Shapes [..., C, H, W]. Climatology is selected at the verifying date.
    A zero anomaly norm is explicitly undefined, never silently discarded.
    """
    p, y, c = (np.asarray(v, np.float64) for v in (pred, truth, climate))
    assert p.shape == y.shape == c.shape
    assert np.isfinite(p).all() and np.isfinite(y).all() and np.isfinite(c).all()
    w = np.maximum(np.cos(np.deg2rad(np.asarray(latitude, np.float64))), 0)[:, None]
    a, b = (p - c, y - c)
    ab = (w * a * b).sum((-2, -1))
    aa = (w * a * a).sum((-2, -1))
    bb = (w * b * b).sum((-2, -1))
    denominator = np.sqrt(aa * bb)
    if np.any(denominator <= 0):
        raise FloatingPointError("Undefined ACC: zero forecast/truth anomaly norm")
    acc = ab / denominator
    assert np.all(np.abs(acc) <= 1 + 1e-12)
    mse = (w * (p - y) ** 2).sum((-2, -1)) / (w.sum() * p.shape[-1])
    return (np.clip(acc, -1, 1), mse)
