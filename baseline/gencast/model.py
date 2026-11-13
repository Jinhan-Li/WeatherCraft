"""Thin WeatherCraft adapter around pinned, unmodified official GenCast modules.

Only boundary adaptation: 69 variables, daily residual scales, cropped grid,
calendar inputs, GPU-compatible official attention, block rematerialization.
"""

import sys
import numpy as np
from .common import CFG, UPSTREAM, LEVELS

sys.path.insert(0, str(UPSTREAM))
import jax
import jax.numpy as jnp
import haiku as hk
import xarray as xr
from graphcast import (
    gencast,
    denoiser,
    graphcast,
    casting,
    losses,
    samplers_utils,
    sparse_transformer,
    xarray_jax,
)

ATM = [
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
]
SURF = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
VARS = tuple(ATM + SURF)

# Preserve the official spectral noise: evaluate it on the full 121-row sphere,
# then take the exact existing WeatherCraft crop. No weather field is padded/regridded.
_original_noise = samplers_utils.spherical_white_noise_like


def cropped_spherical_noise(template):
    lat = np.arange(-90, 90.001, 1.5, dtype=np.float32)
    full = {}
    for name, a in template.items():
        shape = list(a.shape)
        shape[a.dims.index("lat")] = len(lat)
        coords = {k: v for k, v in a.coords.items() if k != "lat"}
        coords["lat"] = lat
        full[name] = xarray_jax.DataArray(
            jnp.zeros(shape, dtype=a.dtype), dims=a.dims, coords=coords
        )
    return _original_noise(xr.Dataset(full)).sel(lat=template.lat)


samplers_utils.spherical_white_noise_like = cropped_spherical_noise


# Official area formula, applied to a subset of its supported full polar grid.
def cropped_latitude_weights(data):
    lat = data.coords["lat"]
    delta = 1.5
    assert np.allclose(np.diff(lat.values), delta)
    w = np.cos(np.deg2rad(lat.values)) * np.sin(np.deg2rad(delta / 2))
    w[np.isclose(np.abs(lat.values), 90)] = np.sin(np.deg2rad(delta / 4)) ** 2
    return xr.DataArray(w / w.mean(), dims=["lat"], coords={"lat": lat})


losses.normalized_latitude_weights = cropped_latitude_weights
_original_sum = losses.sum_per_variable_losses


def available_variable_losses(per_variable_losses, weights):
    # Omit weights for SST/precipitation, neither exists in WeatherCraft's targets.
    return _original_sum(
        per_variable_losses, {k: v for k, v in weights.items() if k in per_variable_losses}
    )


losses.sum_per_variable_losses = available_variable_losses

if CFG["rematerialize"]:
    _block_call = sparse_transformer.Block.__call__

    def rematerialized_block(self, x, global_norm_conditioning):
        # Construct the Haiku remat transform inside the active hk.transform.
        return hk.remat(lambda a, c: _block_call(self, a, global_norm_conditioning=c))(
            x, global_norm_conditioning
        )

    sparse_transformer.Block.__call__ = rematerialized_block


def fields_to_dataset(x, lat, lon, times):
    # x: B,T,C,H,W in the existing WeatherCraft channel order.
    coords = dict(lat=lat, lon=lon, level=LEVELS, time=np.array(times, dtype="timedelta64[h]"))
    vals = {
        name: xarray_jax.Variable(
            ("batch", "time", "level", "lat", "lon"), x[:, :, i * 13 : (i + 1) * 13]
        )
        for i, name in enumerate(ATM)
    }
    vals.update(
        {
            name: xarray_jax.Variable(("batch", "time", "lat", "lon"), x[:, :, 65 + i])
            for i, name in enumerate(SURF)
        }
    )
    return xr.Dataset(vals, coords=coords)


def dataset_to_fields(ds):
    arr = [
        xarray_jax.unwrap_data(ds[n].transpose("batch", "time", "level", "lat", "lon")) for n in ATM
    ]
    arr += [
        xarray_jax.unwrap_data(ds[n].transpose("batch", "time", "lat", "lon"))[:, :, None]
        for n in SURF
    ]
    return jnp.concatenate(arr, axis=2)[:, 0]


def calendar(phase, offsets, lon):
    phase = phase[:, None] + jnp.array(offsets)[None, :] / 365.24219
    day = jnp.broadcast_to(
        jnp.array(0.375 + lon / 360)[None, None, :], (len(phase), len(offsets), len(lon))
    )
    vals = {}
    for name, z, dims in [
        ("year_progress", phase, ("batch", "time")),
        ("day_progress", day, ("batch", "time", "lon")),
    ]:
        vals[name + "_sin"] = xarray_jax.Variable(dims, jnp.sin(2 * jnp.pi * z))
        vals[name + "_cos"] = xarray_jax.Variable(dims, jnp.cos(2 * jnp.pi * z))
    return xr.Dataset(
        vals, coords={"time": np.array(offsets) * np.timedelta64(24, "h"), "lon": lon}
    )


def construct():
    return gencast.GenCast(
        task_config=graphcast.TaskConfig(
            input_variables=VARS + graphcast.GENERATED_FORCING_VARS,
            target_variables=VARS,
            forcing_variables=graphcast.GENERATED_FORCING_VARS,
            pressure_levels=tuple(LEVELS),
            input_duration="48h",
        ),
        denoiser_architecture_config=denoiser.DenoiserArchitectureConfig(
            mesh_size=CFG["mesh_size"],
            latent_size=CFG["latent_size"],
            hidden_layers=1,
            sparse_transformer_config=denoiser.SparseTransformerConfig(
                attention_k_hop=CFG["attention_k_hop"],
                d_model=CFG["latent_size"],
                num_layers=CFG["transformer_layers"],
                num_heads=CFG["attention_heads"],
                ffw_hidden=CFG["ffw_hidden"],
                attention_type=CFG["attention_type"],
            ),
        ),
        noise_encoder_config=denoiser.NoiseEncoderConfig(),
        noise_config=gencast.NoiseConfig(),
        sampler_config=gencast.SamplerConfig(
            churn_max_noise_level=80.0, num_noise_levels=CFG["sampler_steps"]
        ),
    )


def transforms(data, precision=None):
    precision = precision or CFG["precision"]
    lat, lon = data.lat, data.lon
    residual = jnp.array(data.residual)[None, :, None, None]

    def run(x, y, phase, mode):
        inputs = fields_to_dataset(x, lat, lon, [-24, 0]).assign(calendar(phase, [-1, 0], lon))
        forcings = calendar(phase, [1], lon)
        targets = fields_to_dataset(((y - x[:, -1]) / residual)[:, None], lat, lon, [24])
        dtype = jnp.bfloat16 if precision == "bfloat16" else jnp.float32
        inputs = inputs.astype(dtype)
        targets = targets.astype(dtype)
        forcings = forcings.astype(dtype)
        predictor = construct()
        if mode == "loss":
            loss, diag = predictor.loss(inputs, targets, forcings)
            return xarray_jax.unwrap_data(loss).astype(jnp.float32).mean()
        if mode == "sample":
            pred = predictor(inputs, targets, forcings)
        else:
            # Deterministic denoising path for batch/gradient equivalence audits.
            sigma = xarray_jax.DataArray(jnp.ones(x.shape[0], dtype), dims=["batch"])
            pred = predictor._preconditioned_denoiser(inputs, targets, sigma, forcings)
        return x[:, -1] + dataset_to_fields(pred).astype(jnp.float32) * residual

    def wrapped(mode):
        def f(x, y, phase):
            if precision == "bfloat16":
                with casting.bfloat16_variable_view():
                    # Form/normalize residuals in FP32 before casting network
                    # inputs, as in official InputsAndResiduals+Bfloat16Cast.
                    return run(x, y, phase, mode)
            return run(x, y, phase, mode)

        return hk.transform(f)

    return {mode: wrapped(mode) for mode in ["loss", "sample", "denoise"]}
