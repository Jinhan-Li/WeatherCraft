import random, time
import numpy as np
import torch
from torch import distributed as dist
from . import experiment as e
from weathercraft.data.daily import start_history
from weathercraft.data.tokens import seasonal_features
from weathercraft.evaluation.metrics import scores


def origin_indices():
    selection = np.array(
        [
            0,
            11,
            22,
            33,
            45,
            56,
            67,
            78,
            90,
            101,
            112,
            123,
            135,
            146,
            157,
            168,
            180,
            191,
            202,
            213,
            225,
            236,
            247,
            258,
            270,
            281,
            292,
            303,
            315,
            326,
            337,
            349,
        ]
    )
    return selection


def evaluate(m, tok, days, prior, fields, indices, label, rank, world, device):
    saved = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
    )
    leads = e.config()["posttraining"]["validation_leads"]
    started = time.monotonic()
    rows = []
    try:
        with torch.no_grad():
            for i in indices[rank::world]:
                i = int(i)
                assert i + max(leads) < len(days)
                forecasts = {l: [] for l in leads}
                observed = start_history(days, i, device, prior)
                for member in range(8):
                    h = observed
                    torch.manual_seed(1311000 + int(days.day_ids[i]) * 100 + member)
                    for lead in range(1, max(leads) + 1):
                        day = seasonal_features(int(days.day_ids[i]) + lead)[None].to(device)
                        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                            tokens, _ = m.sample(h, day)
                        if lead in forecasts:
                            field = tok.decode(tokens)[:, 0].float()
                            assert torch.isfinite(field).all()
                            forecasts[lead].append(field[0].cpu().numpy())
                        h = tuple(
                            torch.stack((old[:, 1], new), dim=1) for old, new in zip(h, tokens)
                        )
                metrics = []
                persistence = []
                lat = fields.latitudes.numpy()
                w = np.maximum(np.cos(np.deg2rad(lat.astype(np.float64))), 0)
                for lead in leads:
                    truth = fields[i + lead]["field"][0].numpy()
                    metric = scores(np.stack(forecasts[lead]), truth, lat)
                    assert np.isfinite(metric).all()
                    metrics.append(metric.tolist())
                    diff = (fields[i]["field"][0].numpy().astype(np.float64) - truth) ** 2
                    persistence.append(
                        (
                            (diff * w[None, :, None]).sum((-2, -1)) / (w.sum() * diff.shape[-1])
                        ).tolist()
                    )
                rows.append(dict(origin_index=i, metrics=metrics, persistence_mse=persistence))
                print(
                    "VALIDATION_PROGRESS",
                    label,
                    rank,
                    len(rows),
                    len(indices[rank::world]),
                    flush=True,
                )
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        joined = sorted([v for part in gathered for v in part], key=lambda x: x["origin_index"])
        assert [v["origin_index"] for v in joined] == sorted(map(int, indices))
        avg = np.array([v["metrics"] for v in joined]).mean(0)
        k = leads.index(e.config()["posttraining"]["lead_days"])
        result = dict(
            label=label,
            rows=joined,
            origins=len(indices),
            members=8,
            leads=leads,
            fair_crps=float(avg[k, 2].mean()),
            rmse=float(np.sqrt(avg[k, 0].mean())),
            spread=float(np.sqrt(avg[k, 1].mean())),
            per_channel_mean=avg.tolist(),
            per_lead=[
                dict(
                    lead=l,
                    fair_crps=float(a[2].mean()),
                    rmse=float(np.sqrt(a[0].mean())),
                    spread=float(np.sqrt(a[1].mean())),
                )
                for l, a in zip(leads, avg)
            ],
            variable_names=list(fields.variable_names),
            physical_std=fields.std.flatten().tolist(),
            seconds=time.monotonic() - started,
        )
        if rank == 0:
            e.write(e.RUN / "evaluation" / f"{label}.json", result)
        return result
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
        torch.cuda.set_rng_state(saved[3])
