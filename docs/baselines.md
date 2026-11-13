# Baselines

All configurations use the shared daily 69-channel data and train/validation/test
splits. These are daily-data adaptations of the named models.

| Model | Configuration | Input days |
|---|---|---:|
| ConvLSTM | `configs/baseline/convlstm.yaml` | 2 |
| SimVP | `configs/baseline/simvp.yaml` | 2 |
| SFNO | `configs/baseline/sfno.yaml` | 2 |
| STORM | `configs/baseline/storm.yaml` | 2 |
| Pangu | `configs/baseline/pangu.yaml` | 1 |
| OneForecast | `configs/baseline/oneforecast.yaml` | 1 |
| FuXi | `configs/baseline/fuxi.yaml` | 2 |
| FengWu | `configs/baseline/fengwu.yaml` | 2 |
| GraphCast | `configs/baseline/graphcast.yaml` | 2 |
| GenCast | `configs/baseline/gencast.yaml` | 2 |
| Tyche | `configs/baseline/tyche.yaml` | 1 |

## Deterministic models

```bash
torchrun --standalone --nproc_per_node=8 -m baseline.deterministic.train \
  --config configs/baseline/simvp.yaml
torchrun --standalone --nproc_per_node=8 -m baseline.deterministic.evaluate \
  --config configs/baseline/simvp.yaml --checkpoint /path/to/best.pt \
  --climatology cache/climatology --output runs/evaluation/simvp
```

Substitute the desired configuration. GraphCast and OneForecast require a
PyTorch-compatible DGL installation and `torchdata`, `trimesh`, and `scipy`.
Install DGL using its CUDA/PyTorch version-specific instructions.
SFNO includes the required spherical-harmonic source subset; it restores the
missing pole for its transform and crops back to the model grid. STORM retains
the multiscale core with daily-grid padding/cropping and corrected decoder
upsampling.

## Tyche

```bash
TYCHE_CONFIG=configs/baseline/tyche.yaml torchrun --standalone --nproc_per_node=8 \
  -m baseline.tyche.train
torchrun --standalone --nproc_per_node=8 -m baseline.tyche.evaluate \
  --checkpoint /path/to/best_model.pth --output runs/evaluation/tyche
```

Training uses FP32 MeanFlow/JVP and explicit distributed gradient averaging.
Set `TYCHE_RESUME=1` to resume the matching run.

## GenCast

Use a separate environment for the pinned JAX dependencies:

```bash
pip install -r baseline/gencast/requirements.txt
python -m baseline.gencast.prepare_data
python -m baseline.gencast.train
python -m baseline.gencast.evaluate --checkpoint /path/to/best.pkl \
  --output runs/evaluation/gencast
```

These entry points use eight local GPUs and `configs/baseline/gencast.yaml`.
Probabilistic evaluations use eight independent sampled members; deterministic
evaluations use one prediction. All rollouts feed back their own predictions.
