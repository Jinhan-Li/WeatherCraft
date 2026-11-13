"""Build unsmoothed 366 month-day means using 1979-2018 normalized fields only."""

import argparse
from pathlib import Path
import numpy as np
from weathercraft.config import load_config, write_json
from weathercraft.data.daily import field_dataset
from .acc import calendar_index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/dynamics.yaml")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    c = load_config(a.config)
    assert c["data"]["splits"]["train"] == ["1979-01-01", "2018-12-31"]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    days = field_dataset(c, "train")
    assert len(days) == 14610
    shape = (366, 69, 120, 240)
    sums = np.lib.format.open_memmap(out / "sum_float64.npy", mode="w+", dtype="f8", shape=shape)
    sums[:] = 0
    counts = np.zeros(366, dtype="i8")
    for i in range(len(days)):
        k = calendar_index(days.days[i])
        sums[k] += days[i]["field"][0].numpy()
        counts[k] += 1
    assert (counts > 0).all()
    mean = np.lib.format.open_memmap(
        out / "mean_normalized.npy", mode="w+", dtype="f4", shape=shape
    )
    for k in range(366):
        mean[k] = sums[k] / counts[k]
    mean.flush()
    del mean
    del sums
    (out / "sum_float64.npy").unlink()
    write_json(
        out / "metadata.json",
        dict(fit_split=c["data"]["splits"]["train"], smoothing=False, counts=counts.tolist()),
    )


if __name__ == "__main__":
    main()
