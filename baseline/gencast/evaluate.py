"""Eight-member own-prediction trajectories, matched starts; streaming scores."""

import argparse
from pathlib import Path
import json
import pickle
import time
import numpy as np
from .common import CODE, CFG, OUT, Data, NAMES, write_json


def spatial_mean(x, lat):
    w = np.maximum(np.cos(np.deg2rad(lat.astype(np.float64))), 0)
    return (x * w[None, :, None]).sum((-2, -1)) / (w.sum() * x.shape[-1])


def ensemble_scores(members, truth, lat):
    """Per-channel scores; fair CRPS uses independent stochastic members."""
    m = len(members)
    assert m > 1
    x = members.astype(np.float64)
    y = truth.astype(np.float64)
    avg = x.mean(0)
    var = x.var(0, ddof=1)
    sorted_x = np.sort(x, axis=0)
    pair_sum = (sorted_x * (2 * np.arange(1, m + 1) - m - 1)[:, None, None, None]).sum(0)
    first = np.abs(x - y).mean(0)
    mse = spatial_mean((avg - y) ** 2, lat)
    spread = spatial_mean(var, lat)
    return np.stack(
        [
            mse,
            spread,
            spatial_mean(first - pair_sum / (m * (m - 1)), lat),
            spatial_mean(first - pair_sum / (m * m), lat),
            spatial_mean((x[0] - y) ** 2, lat),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=OUT / "train/best.pkl")
    parser.add_argument("--output", type=Path, default=OUT / "evaluation")
    args = parser.parse_args()
    import jax
    from .model import transforms

    assert jax.local_device_count() == 8 and all((x.platform == "gpu" for x in jax.local_devices()))
    d = Data()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "test_results.json").exists():
        raise FileExistsError("Use a new evaluation output directory")
    with args.checkpoint.open("rb") as f:
        ckpt = pickle.load(f)
    params = jax.device_put_replicated(ckpt["params"], jax.local_devices())
    sample = jax.pmap(transforms(d)["sample"].apply)
    starts = d.test_starts()
    leads = CFG["leads"]
    initial_days = [str(d.days[i]) for i in starts]
    totals = np.zeros((len(leads), 6, 69), np.float64)
    count = 0
    started = time.time()
    progress = out / "progress.json"
    if progress.exists():
        prev = json.loads(progress.read_text())
        assert (
            prev["initial_days"] == initial_days
        ), "Progress belongs to a different evaluation subset"
        count = prev["count"]
        totals = np.array(prev["totals"])
    for position in range(count, len(starts)):
        origin = int(starts[position])
        initial = d.field(origin)
        h = np.stack([d.field(origin - 1), initial])
        history = np.broadcast_to(h, (8, 1) + h.shape).copy()
        phase = np.full(
            (8, 1), (d.days[origin].astype(np.int64) + 0.375) / 365.24219 % 1, np.float32
        )
        for lead in range(1, 16):
            key = jax.random.fold_in(jax.random.PRNGKey(20261024), origin * 16 + lead)
            keys = jax.random.split(key, 8)
            pred = np.asarray(
                sample(params, keys, history, np.zeros((8, 1, 69, 120, 240), np.float32), phase)
            )
            if not np.isfinite(pred).all():
                raise FloatingPointError(f"Nonfinite test origin {origin} lead {lead}")
            if lead in leads:
                j = leads.index(lead)
                truth = d.field(origin + lead)
                totals[j, :5] += ensemble_scores(pred[:, 0], truth, d.lat)
                totals[j, 5] += spatial_mean((initial.astype(np.float64) - truth) ** 2, d.lat)
            history = np.stack([history[:, :, -1], pred], axis=2)
            phase = phase + 1 / 365.24219
        count = position + 1
        write_json(
            progress,
            dict(
                count=count,
                totals=totals.tolist(),
                initial_days=initial_days,
                last_initial_day=str(d.days[origin]),
                seconds_this_run=time.time() - started,
            ),
        )
        print("TEST_PROGRESS", count, len(starts), str(d.days[origin]), flush=True)
    result = dict(
        model="gencast",
        selected_epoch=ckpt["epoch"],
        initial_days=[str(d.days[i]) for i in starts],
        start_count=count,
        leads=leads,
        ensemble_members=8,
        sampler_steps=20,
        denoiser_evaluations_per_day=39,
        forecast={},
        raw_persistence={},
        probabilistic={},
    )
    for j, lead in enumerate(leads):
        scores = totals[j] / count
        for label, mse in [("forecast", scores[0]), ("raw_persistence", scores[5])]:
            result[label][str(lead)] = dict(
                normalized_rmse=float(np.sqrt(mse.mean())),
                per_variable_normalized_rmse={
                    n: float(np.sqrt(mse[k])) for k, n in enumerate(NAMES)
                },
                per_variable_physical_rmse={
                    n: float(np.sqrt(mse[k]) * d.physical_std[k]) for k, n in enumerate(NAMES)
                },
            )
        fair_mse = scores[0] - scores[1] / 8
        result["probabilistic"][str(lead)] = {
            n: dict(
                fair_crps_normalized=float(scores[2, k]),
                fair_crps_physical=float(scores[2, k] * d.physical_std[k]),
                empirical_crps_physical=float(scores[3, k] * d.physical_std[k]),
                spread_physical=float(np.sqrt(scores[1, k]) * d.physical_std[k]),
                fair_ensemble_mean_mse_normalized=float(fair_mse[k]),
                fair_ensemble_mean_rmse_physical=(
                    float(np.sqrt(fair_mse[k]) * d.physical_std[k]) if fair_mse[k] >= 0 else None
                ),
                spread_skill_ratio=(
                    float(np.sqrt(scores[1, k] / fair_mse[k])) if fair_mse[k] > 0 else None
                ),
                member0_rmse_physical=float(np.sqrt(scores[4, k]) * d.physical_std[k]),
            )
            for k, n in enumerate(NAMES)
        }
    write_json(out / "test_results.json", result)
    write_json(
        out / "COMPLETE", dict(test_starts=count, ensemble_members=8, selected_epoch=ckpt["epoch"])
    )


if __name__ == "__main__":
    main()
