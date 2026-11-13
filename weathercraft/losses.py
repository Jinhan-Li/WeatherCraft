from __future__ import annotations

import torch


def latitude_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitudes: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE for [B,T,C,H,W], optionally restricted to valid ocean cells."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must share shape [B,T,C,H,W]")
    if latitudes.numel() != target.shape[-2]:
        raise ValueError("latitude count does not match field height")
    weights = torch.cos(torch.deg2rad(latitudes.to(target))).clamp_min(0.0)[
        None, None, None, :, None
    ]
    weights = weights.expand_as(target)
    if valid_mask is not None:
        mask = torch.broadcast_to(
            valid_mask.to(dtype=torch.bool, device=target.device), target.shape
        )
        weights = weights * mask
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("reconstruction loss received no valid cells")
    return ((prediction - target).square() * weights).sum() / denominator


def tokenizer_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitudes: torch.Tensor,
    vq_loss: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    vq_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    reconstruction = latitude_weighted_mse(prediction, target, latitudes, valid_mask)
    total = reconstruction + float(vq_weight) * vq_loss
    return {"loss": total, "reconstruction": reconstruction, "vq": vq_loss}
