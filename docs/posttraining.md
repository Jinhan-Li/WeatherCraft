# Post-training

CRPS and event-reward GRPO are separate branches initialized from the pretrained
weather dynamics. Both freeze the tokenizer. No weights or generated results
are bundled.

## Fair-CRPS

Set tokenizer, dynamics, and token-cache paths in the two configurations:

```bash
torchrun --standalone --nproc_per_node=8 -m weathercraft.posttrain.crps.train \
  --config configs/crps/one_day.yaml
torchrun --standalone --nproc_per_node=8 -m weathercraft.posttrain.crps.train \
  --config configs/crps/two_day.yaml
```

Stage 1 uses 500 updates. Stage 2 starts from its selected `best.pt` with a new
optimizer and uses 750 updates. The configurations specify validation checkpoints
on the same 32 origins in 2019; selection minimizes terminal-lead fair-CRPS.
Each update uses two sampled trajectories, global batch 32, AdamW LR 1e-6,
zero weight decay, gradient clipping 1, and straight-through temperature 1.
Hard Gumbel sampling supplies a biased differentiable estimator; gradients flow
through every generated scale/day and the frozen decoder's inputs.

`--resume` restores optimizer, scheduler, update cursor, and per-rank RNG states
from this distribution's `last.pt`. Use a fresh run directory when changing the model or training configuration. Test evaluation is separate:

```bash
torchrun --standalone --nproc_per_node=8 -m weathercraft.evaluation.forecast \
  --config configs/crps/two_day.yaml --checkpoint /path/to/best.pt \
  --output runs/evaluation/crps
```

## Event-reward GRPO

```bash
torchrun --standalone --nproc_per_node=8 -m weathercraft.posttrain.grpo.train land vanilla
torchrun --standalone --nproc_per_node=8 -m weathercraft.posttrain.grpo.train land tree
torchrun --standalone --nproc_per_node=8 -m weathercraft.posttrain.grpo.train ocean vanilla
torchrun --standalone --nproc_per_node=8 -m weathercraft.posttrain.grpo.train ocean tree
```

Run methods sequentially. Task settings and fixed origin dates are in
`weathercraft/posttrain/grpo/config.json` and `origins.json`. Each method uses
100 updates, eight members, and a frozen reference policy. Tree GRPO branches
at days 1/4/7 into 2/4/8 paths and assigns sibling-relative credit. Validation
selects the policy; the training entry point then evaluates the selected policy
on the fixed test dates.

Land reward is the mean warm/cold persistent-temperature CSI over days 3–11 of
a 15-day rollout (1981–2010 q90/q10; duration at least three days). Ocean reward
is daily SST q90 exceedance CSI over days 3–10, using a frozen ocean model and
1993–2018 thresholds. Thresholds and masks remain fixed during optimization.

### External assets

The following environment variables override project-relative defaults:

| Variable | Default | Contents |
|---|---|---|
| `WEATHERCRAFT_GRPO_ASSETS` | `data/posttrain/assets` | weather weights, normalization, tokens, land labels |
| `WEATHERCRAFT_GRPO_DATA` | `data` | physical daily ERA5, packed GLORYS, ocean climatology |
| `WEATHERCRAFT_OCEAN_ASSETS` | `data/posttrain/ocean_assets` | frozen ocean weights, normalization, mask |
| `WEATHERCRAFT_OCEAN_THRESHOLDS` | `data/posttrain/ocean_thresholds` | calendar-day SST percentiles |
| `WEATHERCRAFT_GRPO_OUTPUT` | `runs/grpo` | checkpoints and metrics |

Weather assets:

```text
weights/base.pt
weights/tokenizer.pt
normalization/train_stats.json
normalization/era5_69_stats.pt
reference/initial_tokens.npz            # dates, scale0, scale1, scale2
reference/physical_z500_t2m_2013_2022.npy
thresholds/COMPLETE.json
thresholds/temperature_thresholds.npy  # [366, 2, 120, 240], q10/q90
thresholds/land_mask.npy
```

The physical ERA5 loader reads `era5_daily_physical/YYYY.npy` under the data root:
source-order z/q/t/u/v pressure levels then u10/v10/t2m/msl, north-to-south latitude
and longitude -180..178.5. It reorders/crops/rolls into the weather model grid.
Initial tokens must be encoded with the matching tokenizer and normalization.

Ocean assets contain `weights/ocean_best_ckpt_final.tar`, and `native_assets/`
with `mean_s_t_ssh.npy`, `std_s_t_ssh.npy`, `land_mask.h5`. The data root contains
`glorys_daily/REQUEST.json` (packing metadata), `glorys_daily/daily/YYYY-MM-DD.npz`
and `ocean_climatology/climate_mean_s_t_ssh.npy`. Percentiles are
`1993_2018_thresholds.npy` and `1993_2012_thresholds.npy`, float32
`[2,366,720,1440]` for q90/q95 on latitude 90..-89.75, longitude -180..179.75.
They use ±5 calendar days and 31-day circular smoothing. These external assets
must be prepared before running; this source package does not include their
download or experiment-specific preparation jobs.
