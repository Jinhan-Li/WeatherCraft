"""Native grid, original CF packing and independently fitted frozen percentiles."""

from .runtime import *
import h5py
from .ocean_metrics import event5
import numpy as np
import importlib.util

LAT = 90 - np.arange(720) * 0.25
LON = -180 + np.arange(1440) * 0.25
LAT_WEATHER = -90 + np.arange(120) * 1.5
PREP = PREPARED
PERIODS = ("1993_2018", "1993_2012")


def climate_indices(date):
    d = dt.date.fromisoformat(str(date))
    if (d.month, d.day) == (2, 29):
        return [(58, 0.5), (59, 0.5)]
    return [((dt.date(2001, d.month, d.day) - dt.date(2001, 1, 1)).days, 1.0)]


class OceanData:

    def __init__(self):
        self.directory = DATA / "glorys_daily"
        self.req = read(self.directory / "REQUEST.json")
        with h5py.File(OCEAN_ASSETS / "native_assets/land_mask.h5") as f:
            self.mask = np.asarray(f["fields"][0], bool)
        assert self.mask.shape == (93, 720, 1440)
        self.mean = np.load(OCEAN_ASSETS / "native_assets/mean_s_t_ssh.npy")[0, :93].astype(
            "float32"
        )
        self.std = np.load(OCEAN_ASSETS / "native_assets/std_s_t_ssh.npy")[0, :93].astype("float32")
        self.climate = np.load(DATA / "ocean_climatology/climate_mean_s_t_ssh.npy", mmap_mode="r")
        self.q = {p: np.load(PREP / f"{p}_thresholds.npy", mmap_mode="r") for p in PERIODS}

    def clim(self, date, surface=False):
        result = None
        for idx, weight in climate_indices(date):
            a = (
                np.array(
                    self.climate[idx, 69, :720] if surface else self.climate[idx, :93, :720],
                    dtype="float32",
                )
                * weight
            )
            result = a if result is None else result + a
        return result

    def decode(self, date, surface=False):
        arrays = []
        with np.load(self.directory / "daily" / f"{date}.npz") as z:
            for name in ["thetao"] if surface else ["so", "uo", "vo", "thetao", "zos"]:
                raw = z[name][:1] if surface else z[name]
                att = self.req["variable_attributes"][name]
                x = raw.astype("float32") * np.float32(att["scale_factor"]) + np.float32(
                    att["add_offset"]
                )
                x[raw == self.req["fill_values"][name]] = np.nan
                arrays.append(x)
        a = np.concatenate(arrays)
        out = np.full((len(a), 720, 1440), np.nan, "float32")
        out[:, :681] = a[:, ::-1]
        return out[0] if surface else out

    def context(self, origin):
        d = dt.date.fromisoformat(origin)
        pastdates = [str(d + dt.timedelta(days=i)) for i in range(-3, 1)]
        future = [str(d + dt.timedelta(days=i)) for i in range(1, 11)]
        x = self.decode(origin)
        cl = self.clim(origin)
        valid = self.mask & np.isfinite(x) & np.isfinite(cl)
        initial = np.zeros_like(x)
        np.divide(x - cl - self.mean, self.std, out=initial, where=valid)
        initial[~valid] = 0
        assert np.isfinite(initial).all()
        persist = np.repeat(x[69:70], 10, axis=0)
        missing = int((self.mask & ~valid).sum())
        initial_surface_valid = valid[69].copy()
        del x, cl, valid
        truth = np.stack([self.decode(v, True) for v in future])
        past = np.stack([self.decode(v, True) for v in pastdates])
        clim = np.stack([self.clim(v, True) for v in future])
        thresholds = {
            p: np.asarray(self.q[p][:, [bin366(v) for v in pastdates + future]]) for p in PERIODS
        }
        mask = (
            initial_surface_valid
            & np.isfinite(truth).all(0)
            & np.isfinite(past).all(0)
            & np.isfinite(clim).all(0)
        )
        for q in thresholds.values():
            mask &= np.isfinite(q).all((0, 1))
        weight = np.cos(np.deg2rad(LAT)).clip(0)[:, None] * mask
        assert weight.sum() > 0
        contexts = {}
        for p, q in thresholds.items():
            p90 = (past > q[0, :4]) & mask
            te = event5(np.concatenate([p90, (truth > q[0, 4:]) & mask]))[4:]
            contexts[p] = dict(
                truth=truth,
                thresholds=q[:, 4:],
                valid=mask,
                weight=weight,
                past90=p90,
                truth_event90=te,
            )
        return dict(
            origin=origin,
            initial=initial,
            truth=truth,
            climate_sst=clim,
            persistence=persist,
            contexts=contexts,
            valid=mask,
            weight=weight,
            scored_cells=int(mask.sum()),
            missing_initial_native_wet=missing,
        )


def remap_periodic(field, src_lat, src_lon, dst_lat, dst_lon):
    """Coordinate-aware bilinear remap; periodic longitude, edge-clamped latitude.

    Both native axes must be supplied from verified preprocessing metadata. Shape
    alone is not evidence for latitude direction or longitude origin.
    """
    field = np.asarray(field, dtype=np.float32)
    slat = np.asarray(src_lat, dtype=np.float64)
    slon = np.asarray(src_lon, dtype=np.float64) % 360
    dlat = np.asarray(dst_lat, dtype=np.float64)
    dlon = np.asarray(dst_lon, dtype=np.float64) % 360
    if field.shape[-2:] != (len(slat), len(slon)):
        raise ValueError("Source grid/field mismatch")
    if not all((np.isfinite(x).all() for x in [slat, slon, dlat, dlon])):
        raise ValueError("Nonfinite coordinates")
    ilat = np.argsort(slat)
    ilon = np.argsort(slon)
    slat = slat[ilat]
    slon = slon[ilon]
    if len(slat) < 2 or len(slon) < 2 or np.any(np.diff(slat) <= 0) or np.any(np.diff(slon) <= 0):
        raise ValueError("Duplicate or insufficient coordinates")
    if not np.allclose(np.diff(np.r_[slon, slon[0] + 360]), 360 / len(slon), atol=1e-05):
        raise ValueError("Expected a global regular periodic longitude grid")
    f = field[..., ilat, :][..., ilon]
    yy = np.clip(dlat, slat[0], slat[-1])
    iy = np.searchsorted(slat, yy, side="right") - 1
    iy = np.clip(iy, 0, len(slat) - 2)
    wy = ((yy - slat[iy]) / (slat[iy + 1] - slat[iy])).astype(np.float32)
    xx = (dlon - slon[0]) % 360
    step = 360 / len(slon)
    ix = np.floor(xx / step).astype(int) % len(slon)
    wx = (xx / step - np.floor(xx / step)).astype(np.float32)
    low = f[..., iy, :] * (1 - wy[:, None]) + f[..., iy + 1, :] * wy[:, None]
    return low[..., ix] * (1 - wx) + low[..., (ix + 1) % len(slon)] * wx


class OceanRuntime:

    def __init__(self, device, data):
        self.device = device
        from . import native_ocean as m

        self.model = m.Triton(params=None)
        ck = torch.load(
            OCEAN_ASSETS / "weights/ocean_best_ckpt_final.tar",
            map_location="cpu",
            weights_only=False,
        )
        self.model.load_state_dict(
            {k.removeprefix("module."): v for k, v in ck["model_state"].items()}, strict=True
        )
        del ck
        self.model.to(device).eval().requires_grad_(False)
        self.mean = torch.from_numpy(
            np.load(OCEAN_ASSETS / "native_assets/mean_s_t_ssh.npy").astype("float32")
        ).to(device)
        self.std = torch.from_numpy(
            np.load(OCEAN_ASSETS / "native_assets/std_s_t_ssh.npy").astype("float32")
        ).to(device)
        self.mask = torch.from_numpy(data.mask).to(device)
        self.remap = remap_periodic

    @torch.inference_mode()
    def step(self, state, forcing):
        mapped = self.remap(forcing, LAT_WEATHER, np.arange(240) * 1.5, LAT, LON)
        f = torch.from_numpy(mapped[None]).to(self.device)
        f = (f - self.mean[:, 93:]) / self.std[:, 93:]
        assert torch.isfinite(state).all() and torch.isfinite(f).all()
        out = self.model(torch.cat([state, f], 1))
        assert out.shape == state.shape and torch.isfinite(out).all()
        return out.masked_fill(~self.mask, 0)

    def physical_sst(self, state, climate):
        return (state[0, 69] * self.std[0, 69] + self.mean[0, 69]).cpu().numpy() + climate
