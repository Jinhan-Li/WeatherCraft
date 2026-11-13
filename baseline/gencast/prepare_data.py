"""Fit daily increment scales on 1979-2018 only; never mutate the cache."""

import argparse
from pathlib import Path
import time
import numpy as np
from .common import CODE, CFG, Data, NAMES, write_json


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    import torch

    d = Data(False)
    p = Path(CFG["residual_statistics"])
    if p.exists():
        Data()
        print("RESIDUAL_STATS_ALREADY_VERIFIED", flush=True)
        return
    ids = np.flatnonzero(
        (d.days >= np.datetime64("1979-01-01")) & (d.days <= np.datetime64("2018-12-31"))
    )
    assert len(ids) == 14610
    total = np.zeros(69, np.float64)
    squares = total.copy()
    n = 0
    before = d.field(ids[0]).astype(np.float64)
    start = time.time()
    for j, i in enumerate(ids[1:], 1):
        after = d.field(i).astype(np.float64)
        delta = after - before
        before = after
        total += delta.sum((1, 2))
        squares += np.square(delta).sum((1, 2))
        n += 120 * 240
        if j % 1000 == 0:
            print("RESIDUAL_FIT", j, time.time() - start, flush=True)
    std = np.sqrt(np.maximum(squares / n - (total / n) ** 2, 0))
    assert np.isfinite(std).all() and (std > 0).all()
    old = torch.load(CFG["data_root"] + "/era5_69_stats.pt", map_location="cpu", weights_only=False)
    om = np.asarray(old["mean"]).reshape(69)
    os = np.asarray(old["std"]).reshape(69)
    result = dict(
        fit_split=CFG["split"]["train"],
        pairs=len(ids) - 1,
        variable_names=NAMES,
        normalized_daily_difference_mean=(total / n).tolist(),
        normalized_daily_difference_std=std.tolist(),
        physical_mean=(om + os * d.shift[:, 0, 0]).tolist(),
        physical_std=(os * d.scale[:, 0, 0]).tolist(),
        seconds=time.time() - start,
        note="Population std of one-day increments; no residual centering, following GenCast.",
    )
    write_json(p, result)
    write_json(
        Path(CFG["residual_statistics"]).with_name("gencast_data.json"),
        dict(
            shape=list(d.fields.shape),
            lat=d.lat.tolist(),
            lon=d.lon.tolist(),
            train_triples=len(d.centers("train")),
            valid_triples=len(d.centers("valid")),
            updates_per_epoch=len(d.centers("train")) // 32,
            test_starts=[str(d.days[i]) for i in d.test_starts()],
            metadata=d.metadata,
        ),
    )
    print("DATA_PREPARED", len(d.centers("train")), len(d.test_starts()), flush=True)


if __name__ == "__main__":
    main()
