"""Ocean aggregate metrics and the ERA5-forced control."""

from .runtime import *
import torch.distributed as dist
from .ocean_metrics import score, probability


def aggregate(self, rows, label):
    ms = [m for r in rows for m in r["members"][:8]]
    out = dict(
        label=label,
        updates=self.update,
        origins=[r["origin"] for r in rows],
        origin_results=rows,
        member_csi=float(np.mean([m["reward"] for m in ms])),
        q95_csi=float(np.mean([m["q95_csi"] for m in ms])),
        strict_event_csi=float(np.mean([m["strict_event_csi"] for m in ms])),
        limited_event_csi=float(np.mean([m["limited_event_csi"] for m in ms])),
        member_rmse=float(np.sqrt(np.mean([m["mse"][2:] for m in ms]))),
        member_bias=float(np.mean([m["bias"][2:] for m in ms])),
        weather_normalized_rmse=float(
            np.sqrt(np.mean([np.asarray(r["weather_mse"])[:8] for r in rows]))
        ),
        sensitivity_csi=float(np.mean([m["sensitivity"]["reward"] for m in ms])),
        utc=now(),
    )
    for name in ["ensemble", "persistence"]:
        out[name + "_csi"] = float(np.mean([r[name]["reward"] for r in rows]))
        out[name + "_rmse"] = float(np.sqrt(np.mean([r[name]["mse"][2:] for r in rows])))
    if len(rows[0]["members"]) == 9:
        out.update(
            greedy_csi=float(np.mean([r["members"][8]["reward"] for r in rows])),
            greedy_rmse=float(np.sqrt(np.mean([r["members"][8]["mse"][2:] for r in rows]))),
        )
    for name in ["brier", "fair_brier", "crps", "fair_crps"]:
        out[name] = float(np.mean([r["probability"][name] for r in rows]))
    return out


@torch.no_grad()
def oracle_evaluate(self, split):
    directory = OUTPUT / "ocean/era5_forcing"
    path = directory / f"{split}.json"
    dates = self.origins[split]
    if path.exists():
        assert read(path)["origins"] == dates
        return read(path)
    local = []
    for slot in range(self.rank, len(dates), 8):
        p = directory / f"{split}_origin{slot:03}.json"
        origin = dates[slot]
        if p.exists():
            row = read(p)
            assert row["origin"] == origin
        else:
            ctx = self.events.context(origin)
            state = torch.from_numpy(ctx.pop("initial")[None]).to(self.device)
            sst = []
            for lead in range(1, 11):
                d = np.datetime64(origin, "D") + lead
                f = self.fields.physical(d)[[66, 67, 65, 68]]
                state = self.ocean.step(state, f)
                sst.append(self.ocean.physical_sst(state, ctx["climate_sst"][lead - 1]))
                self.progress("era5_forcing", split=split, origin=origin, lead=lead)
            row = dict(origin=origin, score=score(np.asarray(sst), ctx))
            write(p, row)
        local.append(row)
    gathered = [None] * 8
    dist.all_gather_object(gathered, local)
    rows = sorted([r for part in gathered for r in part], key=lambda r: dates.index(r["origin"]))
    assert len(rows) == len(dates)
    result = dict(
        origins=dates,
        origin_results=rows,
        csi=float(np.mean([r["score"]["reward"] for r in rows])),
        sst_rmse=float(np.sqrt(np.mean([r["score"]["mse"][2:] for r in rows]))),
        forcing="observed ERA5 physical daily mean, same interpolation and ocean state",
        utc=now(),
    )
    if self.rank == 0:
        write(path, result)
    dist.barrier()
    return result
