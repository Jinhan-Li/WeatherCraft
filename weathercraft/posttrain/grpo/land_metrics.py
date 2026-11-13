"""Daily, explicitly discrete event operators; no trainable event detector."""

from .runtime import *
import numpy as np

LAT = np.arange(-90, 90, 1.5, dtype=np.float64)
LON = np.arange(0, 360, 1.5, dtype=np.float64)
G = 9.80665
BLOCK_ROWS = np.where((LAT >= 35) & (LAT + 15 <= LAT[-1]) & (LAT <= 75))[0]


def persistent(flags, n):
    """Mark every day belonging to a complete run of at least n days."""
    x = np.asarray(flags, dtype=bool)
    out = np.zeros_like(x)
    for start in range(len(x) - n + 1):
        ok = x[start : start + n].all(axis=0)
        out[start : start + n] |= ok
    return out


def counts(pred, truth, weight):
    p = np.asarray(pred, bool)
    t = np.asarray(truth, bool)
    w = np.broadcast_to(weight, p.shape)
    return [float(np.sum(w * a, dtype=np.float64)) for a in (p & t, p & ~t, ~p & t)]


def csi(c):
    return float(c[0] / sum(c)) if sum(c) > 0 else 0.0


def event_maps(fields, thresholds, task, land):
    """fields: [past4+future15,2,120,240] in physical units (z,t2m)."""
    assert task == "land_temperature"
    hot = (fields[:, 1] > thresholds[1]) & land
    cold = (fields[:, 1] < thresholds[0]) & land
    return np.stack([persistent(hot, 3), persistent(cold, 3)])


def score(pred, ctx, task):
    fields = np.concatenate([ctx["past"], pred], axis=0)
    events = event_maps(fields, ctx["thresholds"], task, ctx["land"])[:, 6:15]
    truth = ctx["events"][task][:, 6:15]
    weight = ctx["weights"][task]
    cs = [counts(p, t, weight) for p, t in zip(events, truth)]
    reward = float(np.mean([csi(c) for c in cs]))
    channel = 1
    rows = np.arange(120)
    p = pred[2:11, channel][:, rows]
    t = ctx["future"][2:11, channel][:, rows]
    denom = weight.sum() * len(p)
    return dict(
        reward=reward,
        counts=cs,
        csi_by_event=[csi(c) for c in cs],
        mse=float(np.sum((p - t) ** 2 * weight) / denom),
        bias=float(np.sum((p - t) * weight) / denom),
        predicted_area=[float((p * weight).sum() / (weight.sum() * p.shape[0])) for p in events],
        true_area=[float((p * weight).sum() / (weight.sum() * p.shape[0])) for p in truth],
        scored_leads=list(range(3, 12)),
        complete_duration_labels=True,
    )


def probability_metrics(predictions, ctx, independent=True):
    m = len(predictions)
    assert m == 8
    maps = np.stack(
        [
            event_maps(
                np.concatenate([ctx["past"], p]), ctx["thresholds"], "land_temperature", ctx["land"]
            )[:, 6:15]
            for p in predictions
        ]
    )
    p = maps.mean(0)
    truth = ctx["events"]["land_temperature"][:, 6:15]
    w = ctx["weights"]["land_temperature"]
    denom = w.sum() * 9
    brier = float(np.mean([np.sum((a - b) ** 2 * w) / denom for a, b in zip(p, truth)]))
    fair_brier = (
        float(
            np.mean(
                [
                    np.sum(((a - b) ** 2 - a * (1 - a) / (m - 1)) * w) / denom
                    for a, b in zip(p, truth)
                ]
            )
        )
        if independent
        else None
    )
    x = predictions[:, 2:11, 1].astype("float64")
    y = ctx["future"][2:11, 1]
    mae = np.abs(x - y).mean(0)
    spread = sum((np.abs(x[i] - x[j]) for i in range(m) for j in range(i + 1, m)))
    crps = float(((mae - spread / (m * m)) * w).sum() / denom)
    fair_crps = float(((mae - spread / (m * (m - 1))) * w).sum() / denom) if independent else None
    return dict(
        brier=brier,
        fair_brier=fair_brier,
        crps=crps,
        fair_crps=fair_crps,
        members=m,
        independent_members=independent,
    )


class Events:

    def __init__(self, fields):
        self.truth = np.load(ASSETS / "reference/physical_z500_t2m_2013_2022.npy", mmap_mode="r")
        self.fields = fields
        self.meta = read(ASSETS / "thresholds/COMPLETE.json")
        self.thresholds = np.load(ASSETS / "thresholds/temperature_thresholds.npy", mmap_mode="r")
        self.land = np.load(ASSETS / "thresholds/land_mask.npy")
        self.weights = {"land_temperature": np.cos(np.deg2rad(LAT))[:, None] * self.land}

    def context(self, origin):
        d = np.datetime64(origin, "D")
        dates = np.arange(d - 3, d + 16, dtype="datetime64[D]")
        ii = (dates - np.datetime64("2013-01-01")).astype(int)
        assert ii.min() >= 0 and ii.max() < len(self.truth)
        x = np.asarray(self.truth[ii])
        th = np.asarray(self.thresholds[[bin366(v) for v in dates]]).transpose(1, 0, 2, 3)
        return dict(
            origin=origin,
            past=x[:4],
            future=x[4:],
            thresholds=th,
            land=self.land,
            weights=self.weights,
            events={"land_temperature": event_maps(x, th, "land_temperature", self.land)},
            persistence=np.repeat(x[3:4], 15, axis=0),
        )
