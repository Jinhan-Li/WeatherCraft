"""Cosine-latitude ensemble scores in normalized field space."""

import numpy as np


def scores(members, truth, lat):
    x = members.astype(np.float64)
    y = truth.astype(np.float64)
    m = len(x)
    w = np.maximum(np.cos(np.deg2rad(lat.astype(np.float64))), 0)

    def avg(v):
        return (v * w[None, :, None]).sum((-2, -1)) / (w.sum() * v.shape[-1])

    pair = (np.sort(x, axis=0) * (2 * np.arange(1, m + 1) - m - 1)[:, None, None, None]).sum(0)
    first = np.abs(x - y).mean(0)
    return np.stack(
        [
            avg((x.mean(0) - y) ** 2),
            avg(x.var(0, ddof=1)),
            avg(first - pair / (m * (m - 1))),
            avg(first - pair / (m * m)),
            avg((x[0] - y) ** 2),
        ]
    )
