# Data

The default loader expects daily means of ERA5 00/06/12/18 UTC fields on a
1.5-degree grid. Set `ERA5_ROOT` or edit `configs/data.yaml`.

```text
era5/
  daily/fields.npy       float32 [days, 69, 121, 240], normalized source fields
  daily/times.npy        daily UTC-midnight timestamps
  daily/metadata.json   variable_names and temporal_aggregation
  latitude.npy          121 ascending latitudes, -90 to 90
  longitude.npy         240 longitudes, 0 to 358.5
  era5_69_stats.pt       mean, std, variable_names for source normalization
```

Channels are `z`, `t`, `q`, `u`, `v`, each at pressure levels
50/100/150/200/250/300/400/500/600/700/850/925/1000 hPa, followed by
`t2m`, `u10`, `v10`, `msl`. The model uses the first 120 latitude rows.
The supplied `configs/normalization.json` records the additional training-only
normalization of the daily fields. Use the corresponding source normalization; do not substitute physical fields
for normalized fields.

For an existing normalized six-hour cache:

```bash
python -m weathercraft.data.prepare daily --fields /path/to/fields.npy \
  --times /path/to/times.npy --metadata /path/to/metadata.json \
  --output /path/to/era5/daily
python -m weathercraft.data.prepare inspect
```

For a newly prepared dataset, set `train_stats_path` to a new file and run
`python -m weathercraft.data.prepare statistics`. Fit statistics on the training
split only, before tokenizer training. Use the same normalization and tokenizer throughout training and inference.

The forecast test uses all complete windows through day 15: 1,081 origins from
2020-01-01 to 2022-12-16. Two-day input history can use the final preceding-split
observation. Reported lead metrics average over origins and normalized channels,
not over the five displayed lead times.

Optional ACC needs a training-only calendar-day climatology:

```bash
python -m weathercraft.evaluation.climatology --config configs/dynamics.yaml \
  --output cache/climatology
```

GRPO has additional physical-data and frozen-ocean inputs described in
[Post-training](posttraining.md).
