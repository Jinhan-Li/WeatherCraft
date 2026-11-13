"""Small, explicit configuration checks used before model construction."""

from __future__ import annotations

from datetime import date
from math import prod
from typing import Mapping


def _positive_int(value, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_spatial_factor(value, name: str) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{name} entries must be integers or [latitude, longitude]")
        return tuple(_positive_int(item, name) for item in value)
    factor = _positive_int(value, name)
    return factor, factor


def _validate_train(config: Mapping) -> None:
    if not config:
        return
    for name in ("epochs", "batch_size", "grad_accumulation"):
        _positive_int(config.get(name, 1), f"train.{name}")
    if config.get("amp", "bf16") not in {"fp32", "bf16", "fp16"}:
        raise ValueError("train.amp must be fp32, bf16, or fp16")
    learning_rate = float(config.get("learning_rate", 0.0))
    minimum = float(config.get("min_learning_rate", 0.0))
    if learning_rate <= 0.0 or not 0.0 <= minimum <= learning_rate:
        raise ValueError("learning rates must satisfy 0 <= min_learning_rate <= learning_rate")


def _validate_splits(data: Mapping) -> None:
    splits = data.get("splits")
    if not isinstance(splits, Mapping):
        return
    intervals = []
    for name in ("train", "valid", "test"):
        if name not in splits:
            raise ValueError(f"data.splits is missing {name!r}")
        value = splits[name]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"data.splits.{name} must be [start, end]")
        start, end = (date.fromisoformat(str(item)) for item in value)
        if start > end:
            raise ValueError(f"data.splits.{name} starts after it ends")
        intervals.append((name, start, end))
    for (_, _, left_end), (right_name, right_start, _) in zip(intervals, intervals[1:]):
        if left_end >= right_start:
            raise ValueError(f"data split overlap before {right_name!r}")


def validate_tokenizer_config(config: Mapping) -> None:
    modality = str(config.get("modality", "")).lower().replace("-", "_")
    if modality not in {"atmosphere", "ocean_current", "ocean"}:
        raise ValueError("modality must be atmosphere, ocean_current or ocean")
    model, data = config.get("model", {}), config.get("data", {})
    frames = _positive_int(model.get("frames", 0), "model.frames")
    channels = _positive_int(model.get("channels", 0), "model.channels")
    expected_channels = int(data.get("expected_channels", channels))
    if channels != expected_channels:
        raise ValueError("model.channels must match data.expected_channels")
    if modality == "atmosphere":
        aggregation = data.get("temporal_aggregation", "four_frames")
        if (frames, aggregation) not in {(4, "four_frames"), (1, "daily_mean")}:
            raise ValueError(
                "atmosphere requires four 6-hour frames or one materialized daily mean"
            )
    elif frames != 1:
        raise ValueError("ocean-current data requires one frame")
    if modality == "ocean_current" and list(data.get("variable_names", ())) != ["ugos", "vgos"]:
        raise ValueError("ocean-current v1 requires channel order [ugos, vgos]")
    if modality == "ocean" and len(set(data.get("variable_names", ()))) != channels:
        raise ValueError("ocean variable names must be unique and match channels")
    widths = tuple(_positive_int(value, "model.widths") for value in model.get("widths", ()))
    factors = tuple(
        _positive_spatial_factor(value, "model.spatial_factors")
        for value in model.get("spatial_factors", ())
    )
    if not widths or len(widths) != len(factors):
        raise ValueError("model.widths and spatial_factors must have equal non-zero length")
    _positive_int(model.get("mid_blocks", 2), "model.mid_blocks")
    _positive_int(model.get("stage_blocks", 1), "model.stage_blocks")
    temporal_mix = model.get("temporal_mix", False)
    if not isinstance(temporal_mix, bool):
        raise ValueError("model.temporal_mix must be a boolean")
    representation = model.get("temporal_representation", "increments")
    if representation not in {"raw", "increments"}:
        raise ValueError("model.temporal_representation must be raw or increments")
    if representation == "raw" and model.get("delta_scale_path"):
        raise ValueError("raw-frame representation cannot use delta_scale_path")
    learnable_slots = model.get("learnable_slots", False)
    if not isinstance(learnable_slots, bool):
        raise ValueError("model.learnable_slots must be a boolean")
    if learnable_slots and not temporal_mix:
        raise ValueError("learnable_slots requires temporal_mix")
    if temporal_mix:
        kernel = _positive_int(model.get("temporal_kernel_size", 3), "model.temporal_kernel_size")
        if frames < 2 or any(width % frames for width in widths) or kernel % 2 != 1:
            raise ValueError(
                "temporal_mix needs multiple frames, frame-divisible widths and odd kernel"
            )
    quantizer = model.get("quantizer", {})
    quantizer_type = quantizer.get("type")
    if quantizer_type not in {"residual_vq", "multiscale_fsq"}:
        raise ValueError("model.quantizer.type must be residual_vq or multiscale_fsq")
    if learnable_slots and quantizer_type != "multiscale_fsq":
        raise ValueError("learnable_slots requires multiscale_fsq")
    decoder_scale_fusion = model.get("decoder_scale_fusion", "sum")
    if decoder_scale_fusion not in {"sum", "concat"}:
        raise ValueError("model.decoder_scale_fusion must be sum or concat")
    if decoder_scale_fusion == "concat" and (
        not learnable_slots or quantizer_type != "multiscale_fsq"
    ):
        raise ValueError("concat decoder fusion requires learned slots and multiscale_fsq")
    shapes = tuple(tuple(map(int, shape)) for shape in quantizer.get("scale_shapes", ()))
    if not shapes or any(len(shape) != 2 or min(shape) < 1 for shape in shapes):
        raise ValueError("quantizer.scale_shapes must contain positive [height,width] pairs")
    if any(a[0] > b[0] or a[1] > b[1] for a, b in zip(shapes, shapes[1:])):
        raise ValueError("quantizer scales must be coarse-to-fine")
    height = int(
        data.get("crop_latitude", data.get("height", 120 if modality == "atmosphere" else 720))
    )
    width = int(data.get("width", 240 if modality == "atmosphere" else 1440))
    latitude_divisor = prod(factor[0] for factor in factors)
    longitude_divisor = prod(factor[1] for factor in factors)
    if (
        height % latitude_divisor
        or width % longitude_divisor
        or shapes[-1] != (height // latitude_divisor, width // longitude_divisor)
    ):
        raise ValueError("finest quantizer scale must match the encoder downsampling geometry")
    if quantizer_type == "residual_vq":
        for name in ("num_codebooks", "codebook_dim", "vocab_size"):
            _positive_int(quantizer.get(name, 0), f"model.quantizer.{name}")
        commitment_mode = quantizer.get("commitment_mode", "cumulative")
        if commitment_mode not in {"cumulative", "final_only", "weighted"}:
            raise ValueError(
                "model.quantizer.commitment_mode must be cumulative, final_only, or weighted"
            )
        if commitment_mode == "weighted":
            weights = quantizer.get("commitment_scale_weights")
            if not isinstance(weights, (list, tuple)) or len(weights) != len(shapes):
                raise ValueError(
                    "weighted commitment requires one commitment_scale_weight per scale"
                )
            if any(float(value) < 0 for value in weights) or sum(map(float, weights)) <= 0:
                raise ValueError("commitment_scale_weights must be non-negative with positive sum")
        codebook_objective = quantizer.get("codebook_objective", "cumulative_prefix")
        if codebook_objective not in {"cumulative_prefix", "stage_aligned"}:
            raise ValueError(
                "model.quantizer.codebook_objective must be cumulative_prefix or stage_aligned"
            )
        if float(quantizer.get("adapter_loss_weight", 1.0)) < 0:
            raise ValueError("model.quantizer.adapter_loss_weight must be non-negative")
        if int(quantizer["vocab_size"]) > 65536:
            raise ValueError("vocab_size must fit the uint16 token cache")
    else:
        levels = tuple(
            _positive_int(value, "model.quantizer.levels") for value in quantizer.get("levels", ())
        )
        if not levels or any(level < 2 for level in levels):
            raise ValueError("multiscale_fsq levels must contain integers >= 2")
        if prod(levels) > 65536:
            raise ValueError("composite FSQ vocabulary must fit the uint16 token cache")
        slots = _positive_int(quantizer.get("slots", 0), "model.quantizer.slots")
        if modality == "atmosphere" and frames == 1 and slots != 1:
            raise ValueError("one-frame daily atmosphere requires one latent slot")
        codebooks = quantizer.get("codebooks_per_scale", 1)
        values = (codebooks,) * len(shapes) if isinstance(codebooks, int) else tuple(codebooks)
        if len(values) != len(shapes) or any(int(value) < 1 for value in values):
            raise ValueError("codebooks_per_scale must be positive with one value per scale")
        gains = quantizer.get("scale_gains")
        if gains is not None:
            if not isinstance(gains, (list, tuple)) or len(gains) != len(shapes):
                raise ValueError("scale_gains must contain one entry per scale")
            for scale, gain in enumerate(gains):
                row = gain if isinstance(gain, (list, tuple)) else (gain,)
                if len(row) not in {1, len(levels)} or any(float(value) <= 0 for value in row):
                    raise ValueError(
                        f"scale_gains[{scale}] must be positive scalar or one value per FSQ axis"
                    )
        if float(config.get("loss", {}).get("vq_weight", 0.0)) != 0.0:
            raise ValueError("multiscale_fsq has no auxiliary VQ loss; set loss.vq_weight to 0")
    expected_loss = (
        "latitude_weighted_mse" if modality == "atmosphere" else "masked_latitude_weighted_mse"
    )
    if config.get("loss", {}).get("reconstruction") != expected_loss:
        raise ValueError(f"loss.reconstruction must be {expected_loss!r} for {modality}")
    _validate_splits(data)
    _validate_train(config.get("train", {}))
    train = config.get("train", {})
    if quantizer_type == "multiscale_fsq" and train.get("codebook_learning_rate") is not None:
        raise ValueError(
            "multiscale_fsq has no learned codebook and cannot use codebook_learning_rate"
        )
    if (
        train.get("codebook_learning_rate") is not None
        and float(train["codebook_learning_rate"]) <= 0
    ):
        raise ValueError("train.codebook_learning_rate must be positive")
    if float(train.get("codebook_weight_decay", 0.0)) < 0:
        raise ValueError("train.codebook_weight_decay must be non-negative")
    hold_steps = train.get("codebook_hold_steps")
    after_hold_lr = train.get("codebook_after_hold_learning_rate")
    if (hold_steps is None) != (after_hold_lr is None):
        raise ValueError(
            "codebook_hold_steps and codebook_after_hold_learning_rate must be configured together"
        )
    if hold_steps is not None:
        if int(hold_steps) < 0 or float(after_hold_lr) <= 0:
            raise ValueError(
                "two-stage codebook schedule requires non-negative hold steps and positive LR"
            )
        initial_lr = float(train.get("codebook_learning_rate", 0.0))
        if initial_lr <= 0 or float(after_hold_lr) > initial_lr:
            raise ValueError("two-stage codebook LR must not increase after the hold phase")
        if train.get("codebook_constant_lr", True):
            raise ValueError("two-stage codebook schedule requires codebook_constant_lr: false")


def validate_dynamics_config(config: Mapping) -> None:
    for name in ("tokenizer_config", "tokenizer_checkpoint", "token_cache"):
        if not config.get(name):
            raise ValueError(f"dynamics config requires {name}")
    model = config.get("model", {})
    if model.get("freeze_codec", True) is not True:
        raise ValueError("v1 dynamics requires freeze_codec=true")
    for name in ("depth", "model_dim", "num_heads"):
        _positive_int(model.get(name, 0), f"model.{name}")
    if int(model["model_dim"]) % int(model["num_heads"]):
        raise ValueError("model.model_dim must be divisible by model.num_heads")
    _validate_train(config.get("train", {}))
