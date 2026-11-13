from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models.tokenizer import FieldTokenizer
from .models.tokenizer.quantizer import ResidualVectorQuantizer
from .models.tokenizer.multiscale_fsq import MultiScaleFSQQuantizer
from .models.dynamics import WeatherDynamics
from .validation import validate_dynamics_config, validate_tokenizer_config


def _delta_scale(config: dict[str, Any]) -> torch.Tensor | None:
    path = config.get("delta_scale_path")
    if not path:
        return None
    values = np.load(Path(path))
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def _build_quantizer(config: dict[str, Any]):
    kind = config["type"]
    if kind == "residual_vq":
        return ResidualVectorQuantizer(
            scale_shapes=config["scale_shapes"],
            num_codebooks=config.get("num_codebooks", 2),
            codebook_dim=config.get("codebook_dim", 128),
            vocab_size=config.get("vocab_size", 4096),
            beta=config.get("beta", 0.25),
            residual_ratio=config.get("residual_ratio", 0.5),
            shared_adapters=config.get("shared_adapters", 4),
            commitment_mode=config.get("commitment_mode", "cumulative"),
            commitment_scale_weights=config.get("commitment_scale_weights"),
            codebook_objective=config.get("codebook_objective", "cumulative_prefix"),
            adapter_loss_weight=config.get("adapter_loss_weight", 1.0),
        )
    if kind == "multiscale_fsq":
        return MultiScaleFSQQuantizer(
            levels=config["levels"],
            slots=config.get("slots", 2),
            scale_shapes=config["scale_shapes"],
            codebooks_per_scale=config.get("codebooks_per_scale", 1),
            scale_gains=config.get("scale_gains"),
        )
    raise ValueError(f"unknown tokenizer quantizer type: {kind!r}")


def build_tokenizer(config: dict[str, Any]) -> FieldTokenizer:
    validate_tokenizer_config(config)
    model = config["model"]
    quantizer_config = model["quantizer"]
    quantizer = _build_quantizer(quantizer_config)
    return FieldTokenizer(
        frames=model["frames"],
        channels=model["channels"],
        widths=model["widths"],
        spatial_factors=model["spatial_factors"],
        scale_shapes=quantizer.scale_shapes,
        mid_blocks=model.get("mid_blocks", 2),
        stage_blocks=model.get("stage_blocks", 1),
        temporal_mix=model.get("temporal_mix", False),
        temporal_kernel_size=model.get("temporal_kernel_size", 3),
        temporal_representation=model.get("temporal_representation", "increments"),
        learnable_slots=model.get("learnable_slots", False),
        decoder_scale_fusion=model.get("decoder_scale_fusion", "sum"),
        delta_scale=_delta_scale(model),
        quantizer=quantizer,
    )


def load_tokenizer_checkpoint(
    model: FieldTokenizer,
    path: str | Path,
    config: dict[str, Any] | None = None,
    strict: bool = True,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=strict)
    return checkpoint


def build_dynamics(config: dict[str, Any], tokenizer: FieldTokenizer) -> WeatherDynamics:
    validate_dynamics_config(config)
    return WeatherDynamics(codec=tokenizer.quantizer, **config["model"])
