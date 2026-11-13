"""Inspect ERA5, fit train-only statistics, or materialize daily means."""

import argparse
import json
from pathlib import Path
import numpy as np
from weathercraft.config import load_config, write_json
from weathercraft.data.atmosphere import AtmosphereDayDataset
from weathercraft.data.daily import field_dataset


def fit_statistics(config):
    target = Path(config["data"]["train_stats_path"])
    if target.exists():
        recorded = json.loads(target.read_text(encoding="utf-8"))
        if recorded["fit_split"] != config["data"]["splits"]["train"]:
            raise ValueError("Existing normalization uses a different training period")
        print(f"NORMALIZATION_REUSED path={target}")
        return
    days = AtmosphereDayDataset.from_config(config["data"], "train")
    count = 0
    mean = np.zeros(len(days.variable_names))
    second_moment = np.zeros_like(mean)
    for index in range(len(days)):
        values = days[index]["field"][0].numpy().reshape(len(mean), -1).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite training fields on {days.days[index]}")
        size = values.shape[1]
        current_mean = values.mean(1)
        delta = current_mean - mean
        second_moment += ((values - current_mean[:, None]) ** 2).sum(
            1
        ) + delta**2 * count * size / (count + size)
        mean += delta * size / (count + size)
        count += size
        if (index + 1) % 500 == 0:
            print(f"NORMALIZATION_PROGRESS days={index + 1}/{len(days)}", flush=True)
    standard_deviation = np.sqrt(second_moment / count)
    if not (standard_deviation > 1e-08).all():
        raise ValueError("Degenerate channel normalization")
    write_json(
        target,
        dict(
            fit_split=config["data"]["splits"]["train"],
            days=len(days),
            crop_shape=[120, 240],
            variable_names=list(days.variable_names),
            spatial_weighting="uniform_grid",
            cache_mean=mean.tolist(),
            cache_std=standard_deviation.tolist(),
        ),
    )


def daily_means(fields_path, times_path, metadata_path, output):
    """Average already-normalized 00/06/12/18 UTC fields, retaining all 121 rows."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Daily output must be a new or empty directory")
    fields = np.load(fields_path, mmap_mode="r", allow_pickle=False)
    times = np.load(times_path, allow_pickle=False).astype("datetime64[s]")
    if fields.ndim != 4 or fields.shape[1:] != (69, 121, 240) or len(times) != len(fields):
        raise ValueError(
            "Expected normalized six-hour fields [N,69,121,240] and matching timestamps"
        )
    if len(times) % 4 or (
        len(times) > 1 and (not np.all(np.diff(times) == np.timedelta64(6, "h")))
    ):
        raise ValueError("Daily aggregation needs consecutive complete 6-hour records")
    days = times[::4].astype("datetime64[D]")
    expected = days[:, None].astype("datetime64[s]") + np.arange(4)[None] * np.timedelta64(6, "h")
    if not np.array_equal(times.reshape(-1, 4), expected):
        raise ValueError("Each day must be ordered 00,06,12,18 UTC")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    names = metadata.get("variable_names", metadata.get("channel_names"))
    if names is None or len(names) != 69:
        raise ValueError("Input metadata must provide the 69-channel order")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "fields.incomplete.npy"
    array = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(days), 69, 121, 240)
    )
    for index in range(len(days)):
        chunk = np.asarray(fields[index * 4 : (index + 1) * 4], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise ValueError(f"Nonfinite input on {days[index]}")
        array[index] = chunk.mean(0, dtype=np.float32)
        if (index + 1) % 500 == 0:
            print(f"DAILY_MEAN_PROGRESS days={index + 1}/{len(days)}", flush=True)
    array.flush()
    del array
    temporary.replace(output / "fields.npy")
    np.save(output / "times.npy", days.astype("datetime64[s]"))
    metadata.update(
        temporal_aggregation="four_frame_mean_00_06_12_18_utc",
        variable_names=names,
        source_fields=str(Path(fields_path).resolve()),
    )
    write_json(output / "metadata.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "statistics", "daily"])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--fields", help="Already-normalized six-hour NPY fields for daily aggregation"
    )
    parser.add_argument("--times", help="Matching six-hour timestamps")
    parser.add_argument("--metadata", help="Source channel metadata")
    parser.add_argument("--output", help="New daily-mean output directory")
    args = parser.parse_args()
    if args.action == "daily":
        if not all((args.fields, args.times, args.metadata, args.output)):
            parser.error("daily requires --fields, --times, --metadata and --output")
        daily_means(args.fields, args.times, args.metadata, args.output)
    else:
        config = load_config(args.config)
        if args.action == "statistics":
            fit_statistics(config)
        else:
            for split in ("train", "valid", "test"):
                days = field_dataset(config, split)
                sample = days[0]["field"]
                if not np.isfinite(sample.numpy()).all():
                    raise ValueError("Nonfinite example")
                print(
                    f"{split}: days={len(days)} dates={days.days[0]}..{days.days[-1]} shape={tuple(sample.shape)}"
                )


if __name__ == "__main__":
    main()
