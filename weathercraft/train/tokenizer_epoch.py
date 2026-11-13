from __future__ import annotations
from contextlib import nullcontext
from collections.abc import Mapping, Sequence
import torch
from torch import distributed as dist
from tqdm.auto import tqdm
from ..losses import tokenizer_loss


def atmosphere_metric_groups(
    variable_names: Sequence[str],
    selected: Sequence[str] = ("z_500", "t_850", "q_700", "u_850", "v_850", "t2m"),
) -> dict[str, tuple[int, ...]]:
    """Use the cache's channel order for WeatherFSQ-compatible diagnostics."""
    names = tuple(map(str, variable_names))
    if len(set(names)) != len(names):
        raise ValueError("atmosphere variable names must be unique")
    prefixes = ("z", "t", "q", "u", "v")
    groups = {
        prefix: tuple((i for i, name in enumerate(names) if name.startswith(prefix + "_")))
        for prefix in prefixes
    }
    groups["surface"] = tuple(
        (
            i
            for i, name in enumerate(names)
            if not any((name.startswith(prefix + "_") for prefix in prefixes))
        )
    )
    if any((not indices for indices in groups.values())) or sum(map(len, groups.values())) != len(
        names
    ):
        raise ValueError("atmosphere groups must cover every variable exactly once")
    groups.update({name: (names.index(name),) for name in selected if name in names})
    return groups


def _gradient_norm(parameters) -> torch.Tensor:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return torch.zeros(())
    return torch.stack(values).sum().sqrt()


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_tokenizer_epoch(
    model: torch.nn.Module,
    loader,
    latitudes: torch.Tensor,
    device: torch.device,
    precision: str,
    vq_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler: torch.amp.GradScaler | None = None,
    grad_accumulation: int = 1,
    grad_clip: float = 1.0,
    progress: bool = False,
    progress_description: str = "",
    variable_metrics: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, float]:
    if grad_accumulation < 1:
        raise ValueError("grad_accumulation must be positive")
    training = optimizer is not None
    model.train(training)
    module = model.module if hasattr(model, "module") else model
    quantizer = module.quantizer
    totals = torch.zeros(6, device=device)
    residual_total = torch.zeros(
        quantizer.num_codebooks, len(quantizer.scale_shapes), device=device
    )
    prefix_total = torch.zeros_like(residual_total)
    stage_codebook_total = torch.zeros_like(residual_total)
    stage_adapter_total = torch.zeros_like(residual_total)
    contribution_total = torch.zeros_like(residual_total)
    code_rms_total = torch.zeros(quantizer.num_codebooks, device=device)
    diagnostic_weight = torch.zeros((), device=device)
    quant_moments = torch.zeros(quantizer.num_codebooks, 5, device=device, dtype=torch.float64)
    code_counts = torch.zeros(
        quantizer.num_codebooks,
        len(quantizer.scale_shapes),
        quantizer.vocab_size,
        device=device,
        dtype=torch.int64,
    )
    active_channels = getattr(quantizer, "active_token_channels", None)
    if active_channels is None:
        active_mask = torch.ones(
            quantizer.num_codebooks, len(quantizer.scale_shapes), device=device, dtype=torch.bool
        )
    else:
        active_channels = torch.as_tensor(active_channels, device=device, dtype=torch.long)
        if active_channels.shape != (len(quantizer.scale_shapes),):
            raise ValueError("quantizer.active_token_channels must contain one value per scale")
        active_mask = (
            torch.arange(quantizer.num_codebooks, device=device)[:, None] < active_channels[None, :]
        )
    frame_sse = torch.zeros(module.frames, device=device, dtype=torch.float64)
    frame_weight = torch.zeros(module.frames, device=device, dtype=torch.float64)
    variable_metrics = variable_metrics or {}
    variable_names = tuple(variable_metrics)
    variable_indices = [tuple(map(int, variable_metrics[name])) for name in variable_names]
    if any(
        (
            not indices or min(indices) < 0 or max(indices) >= module.channels
            for indices in variable_indices
        )
    ):
        raise ValueError("variable metric indices must select valid channels")
    variable_sse = torch.zeros(len(variable_names), device=device, dtype=torch.float64)
    variable_weight = torch.zeros_like(variable_sse)
    delta_sse = torch.zeros(max(0, module.frames - 1), device=device, dtype=torch.float64)
    delta_weight = torch.zeros(max(0, module.frames - 1), device=device, dtype=torch.float64)
    gradient_total = torch.zeros((), device=device)
    base_gradient_total = torch.zeros((), device=device)
    codebook_gradient_total = torch.zeros((), device=device)
    adapter_gradient_total = torch.zeros((), device=device)
    gradient_updates = torch.zeros((), device=device)
    if scaler is None:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=training and device.type == "cuda" and (precision == "fp16")
        )
    if training:
        optimizer.zero_grad(set_to_none=True)
    batch_count = len(loader)
    tail = batch_count % grad_accumulation
    progress_bar = tqdm(
        loader,
        total=batch_count,
        desc=progress_description,
        disable=not progress,
        dynamic_ncols=True,
        leave=True,
    )
    for step, batch in enumerate(progress_bar):
        update = (step + 1) % grad_accumulation == 0 or step + 1 == batch_count
        optimizer_updated = False
        field = batch["field"].to(device, non_blocking=True)
        mask = batch.get("valid_mask")
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        gradient_context = nullcontext() if training else torch.no_grad()
        sync_context = (
            model.no_sync()
            if training and hasattr(model, "no_sync") and (not update)
            else nullcontext()
        )
        with sync_context:
            with gradient_context:
                with _autocast(device, precision):
                    output = model(field)
                    losses = tokenizer_loss(
                        output.reconstruction,
                        field,
                        latitudes,
                        output.vq_loss,
                        valid_mask=mask,
                        vq_weight=vq_weight,
                    )
                    if not all((torch.isfinite(value.detach()) for value in losses.values())):
                        raise FloatingPointError("tokenizer loss became NaN or Inf")
                    window = tail if tail and step >= batch_count - tail else grad_accumulation
                    scaled = losses["loss"] / window
                if progress and (step % 10 == 0 or step + 1 == batch_count):
                    postfix = {
                        "loss": f"{float(losses['loss'].detach()):.4f}",
                        "rec": f"{float(losses['reconstruction'].detach()):.4f}",
                    }
                    if getattr(quantizer, "quantizer_type", None) != "multiscale_fsq":
                        postfix["vq"] = f"{float(losses['vq'].detach()):.4f}"
                    if training:
                        postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
                    progress_bar.set_postfix(postfix, refresh=False)
                if training:
                    scaler.scale(scaled).backward()
        if training:
            if update:
                scaler.unscale_(optimizer)
                codebook_parameters = tuple(quantizer.codebook_parameters())
                codebook_ids = {id(parameter) for parameter in codebook_parameters}
                base_parameters = tuple(
                    (
                        parameter
                        for parameter in model.parameters()
                        if id(parameter) not in codebook_ids
                    )
                )
                codebook_gradient_norm = _gradient_norm(codebook_parameters).to(device)
                base_gradient_norm = _gradient_norm(base_parameters).to(device)
                adapter_gradient_norm = (
                    _gradient_norm(quantizer.adapter_parameters()).to(device)
                    if quantizer.codebook_objective == "stage_aligned"
                    else field.new_zeros(())
                )
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("tokenizer gradient norm became NaN or Inf")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_updated = not scaler.is_enabled() or scaler.get_scale() >= previous_scale
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and optimizer_updated:
                    scheduler.step()
                if (
                    optimizer_updated
                    and progress
                    and (step + 1 == grad_accumulation or (step + 1) % 50 == 0)
                ):
                    print(
                        "TOKENIZER_TRAIN_STEP_OK",
                        progress_description,
                        "batch",
                        step + 1,
                        "loss",
                        float(losses["loss"].detach()),
                        "grad",
                        float(gradient_norm),
                        "next_lr",
                        optimizer.param_groups[0]["lr"],
                        flush=True,
                    )
            else:
                gradient_norm = field.new_zeros(())
                base_gradient_norm = field.new_zeros(())
                codebook_gradient_norm = field.new_zeros(())
                adapter_gradient_norm = field.new_zeros(())
        else:
            gradient_norm = field.new_zeros(())
            base_gradient_norm = field.new_zeros(())
            codebook_gradient_norm = field.new_zeros(())
            adapter_gradient_norm = field.new_zeros(())
        batch_weight = field.shape[0]
        totals += (
            torch.stack(
                (
                    losses["loss"].detach(),
                    losses["reconstruction"].detach(),
                    losses["vq"].detach(),
                    output.codebook_loss.detach(),
                    output.adapter_loss.detach(),
                    output.commitment_loss.detach(),
                )
            )
            * batch_weight
        )
        diagnostic_weight += batch_weight
        residual_total += output.residual_energy.detach() * batch_weight
        prefix_total += output.prefix_mse.detach() * batch_weight
        stage_codebook_total += output.stage_codebook_mse.detach() * batch_weight
        stage_adapter_total += output.stage_adapter_mse.detach() * batch_weight
        contribution_total += output.contribution_energy.detach() * batch_weight
        code_rms_total += output.code_rms.detach() * batch_weight
        with torch.no_grad():
            shape = output.latent.shape
            diagnostic_view = getattr(quantizer, "latent_diagnostic_view", None)
            if diagnostic_view is None:
                latent_heads = (
                    output.latent.detach()
                    .float()
                    .view(shape[0], quantizer.num_codebooks, quantizer.codebook_dim, *shape[-2:])
                )
                quantized_heads = output.quantized_latent.detach().float().view_as(latent_heads)
            else:
                latent_heads = diagnostic_view(output.latent.detach().float())
                quantized_heads = diagnostic_view(output.quantized_latent.detach().float())
                expected = (shape[0], quantizer.num_codebooks, quantizer.codebook_dim, *shape[-2:])
                if (
                    tuple(latent_heads.shape) != expected
                    or tuple(quantized_heads.shape) != expected
                ):
                    raise ValueError(
                        f"quantizer.latent_diagnostic_view returned an invalid shape: expected {expected}, got {tuple(latent_heads.shape)} and {tuple(quantized_heads.shape)}"
                    )
            dimensions = (0, 2, 3, 4)
            quant_moments[:, 0] += (
                (quantized_heads - latent_heads).square().sum(dim=dimensions).double()
            )
            quant_moments[:, 1] += latent_heads.square().sum(dim=dimensions).double()
            quant_moments[:, 2] += quantized_heads.square().sum(dim=dimensions).double()
            quant_moments[:, 3] += (latent_heads * quantized_heads).sum(dim=dimensions).double()
            quant_moments[:, 4] += latent_heads[0, 0].numel() * latent_heads.shape[0]
        for scale, tokens in enumerate(output.tokens):
            for head in range(quantizer.num_codebooks):
                if not bool(active_mask[head, scale]):
                    continue
                code_counts[head, scale] += torch.bincount(
                    tokens[:, head].reshape(-1).long(), minlength=quantizer.vocab_size
                )
        with torch.no_grad():
            weights = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)[None, None, None, :, None]
            weights = weights.expand_as(field)
            if mask is not None:
                weights = weights * torch.broadcast_to(mask.to(dtype=torch.bool), field.shape)
            squared = (output.reconstruction.float() - field.float()).square()
            frame_sse += (squared * weights).sum(dim=(0, 2, 3, 4), dtype=torch.float64)
            frame_weight += weights.sum(dim=(0, 2, 3, 4), dtype=torch.float64)
            if variable_names:
                channel_sse = (squared * weights).sum(dim=(0, 1, 3, 4), dtype=torch.float64)
                channel_weight = weights.sum(dim=(0, 1, 3, 4), dtype=torch.float64)
                for index, channels in enumerate(variable_indices):
                    variable_sse[index] += channel_sse[list(channels)].sum()
                    variable_weight[index] += channel_weight[list(channels)].sum()
            if module.frames > 1:
                predicted_delta = output.reconstruction[:, 1:] - output.reconstruction[:, :-1]
                target_delta = field[:, 1:] - field[:, :-1]
                delta_weights = torch.minimum(weights[:, 1:], weights[:, :-1])
                delta_sse += (
                    (predicted_delta.float() - target_delta.float()).square() * delta_weights
                ).sum(dim=(0, 2, 3, 4), dtype=torch.float64)
                delta_weight += delta_weights.sum(dim=(0, 2, 3, 4), dtype=torch.float64)
        if optimizer_updated:
            gradient_total += gradient_norm.detach().float()
            base_gradient_total += base_gradient_norm.detach().float()
            codebook_gradient_total += codebook_gradient_norm.detach().float()
            adapter_gradient_total += adapter_gradient_norm.detach().float()
            gradient_updates += 1
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(residual_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(prefix_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(stage_codebook_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(stage_adapter_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(contribution_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(code_rms_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(diagnostic_weight, op=dist.ReduceOp.SUM)
        dist.all_reduce(quant_moments, op=dist.ReduceOp.SUM)
    if dist.is_initialized():
        dist.all_reduce(code_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(gradient_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(base_gradient_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(codebook_gradient_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(adapter_gradient_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(gradient_updates, op=dist.ReduceOp.SUM)
        dist.all_reduce(frame_sse, op=dist.ReduceOp.SUM)
        dist.all_reduce(frame_weight, op=dist.ReduceOp.SUM)
        dist.all_reduce(variable_sse, op=dist.ReduceOp.SUM)
        dist.all_reduce(variable_weight, op=dist.ReduceOp.SUM)
        if module.frames > 1:
            dist.all_reduce(delta_sse, op=dist.ReduceOp.SUM)
            dist.all_reduce(delta_weight, op=dist.ReduceOp.SUM)
    usage = (code_counts > 0).float().mean(-1)
    probabilities = code_counts.double() / code_counts.sum(-1, keepdim=True).clamp_min(1)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    normalized_entropy = entropy / torch.log(entropy.new_tensor(float(quantizer.vocab_size)))
    perplexity = entropy.exp()
    top1_fraction = probabilities.amax(-1)
    residual_total /= diagnostic_weight.clamp_min(1)
    prefix_total /= diagnostic_weight.clamp_min(1)
    stage_codebook_total /= diagnostic_weight.clamp_min(1)
    stage_adapter_total /= diagnostic_weight.clamp_min(1)
    contribution_total /= diagnostic_weight.clamp_min(1)
    code_rms_total /= diagnostic_weight.clamp_min(1)
    totals /= diagnostic_weight.clamp_min(1)
    metrics = dict(zip(("loss", "reconstruction", "vq", "vq_codebook"), map(float, totals[:4])))
    metrics["vq_commitment_raw"] = float(totals[5])
    if quantizer.codebook_objective == "stage_aligned":
        metrics["vq_adapter"] = float(totals[4])
        metrics["vq_adapter_weighted"] = metrics["vq_adapter"] * quantizer.adapter_loss_weight
    metrics["vq_commitment_weighted"] = metrics["vq_commitment_raw"] * quantizer.beta
    active_usage = usage[active_mask]
    active_normalized_entropy = normalized_entropy[active_mask]
    active_entropy = entropy[active_mask]
    active_top1_fraction = top1_fraction[active_mask]
    metrics["usage"] = float(active_usage.mean())
    metrics["normalized_entropy"] = float(active_normalized_entropy.mean())
    metrics["perplexity"] = float(active_entropy.mean().exp())
    metrics["top1_fraction"] = float(active_top1_fraction.mean())
    metrics["grad_norm"] = float(gradient_total / gradient_updates.clamp_min(1))
    metrics["base_grad_norm"] = float(base_gradient_total / gradient_updates.clamp_min(1))
    metrics["codebook_grad_norm"] = float(codebook_gradient_total / gradient_updates.clamp_min(1))
    if quantizer.codebook_objective == "stage_aligned":
        metrics["adapter_grad_norm"] = float(adapter_gradient_total / gradient_updates.clamp_min(1))
    error, latent_sum, quantized_sum, dot, elements = quant_moments.sum(0)
    metrics["final_quant_mse"] = float(error / elements.clamp_min(1))
    metrics["final_quant_relative_mse"] = float(error / latent_sum.clamp_min(1e-30))
    metrics["final_quant_relative_l2"] = float(torch.sqrt(error / latent_sum.clamp_min(1e-30)))
    metrics["final_quant_cosine"] = float(
        dot / torch.sqrt((latent_sum * quantized_sum).clamp_min(1e-30))
    )
    metrics["latent_rms"] = float(torch.sqrt(latent_sum / elements.clamp_min(1)))
    metrics["quantized_rms"] = float(torch.sqrt(quantized_sum / elements.clamp_min(1)))
    metrics["quantized_to_latent_rms"] = metrics["quantized_rms"] / max(
        metrics["latent_rms"], 1e-30
    )
    for frame in range(module.frames):
        metrics[f"frame{frame}_reconstruction"] = float(
            frame_sse[frame] / frame_weight[frame].clamp_min(1)
        )
    for index, name in enumerate(variable_names):
        metrics[name] = float(variable_sse[index] / variable_weight[index].clamp_min(1))
    for delta in range(max(0, module.frames - 1)):
        metrics[f"delta{delta}_reconstruction"] = float(
            delta_sse[delta] / delta_weight[delta].clamp_min(1)
        )
    for head in range(quantizer.num_codebooks):
        head_error, head_latent, head_quantized, head_dot, head_elements = quant_moments[head]
        metrics[f"final_quant_mse_q{head}"] = float(head_error / head_elements.clamp_min(1))
        metrics[f"final_quant_relative_mse_q{head}"] = float(
            head_error / head_latent.clamp_min(1e-30)
        )
        metrics[f"final_quant_cosine_q{head}"] = float(
            head_dot / torch.sqrt((head_latent * head_quantized).clamp_min(1e-30))
        )
        metrics[f"latent_rms_q{head}"] = float(torch.sqrt(head_latent / head_elements.clamp_min(1)))
        metrics[f"quantized_rms_q{head}"] = float(
            torch.sqrt(head_quantized / head_elements.clamp_min(1))
        )
        metrics[f"code_rms_q{head}"] = float(code_rms_total[head])
        for scale in range(len(quantizer.scale_shapes)):
            metrics[f"active_q{head}_scale{scale}"] = int(active_mask[head, scale])
            metrics[f"usage_q{head}_scale{scale}"] = float(usage[head, scale])
            metrics[f"entropy_q{head}_scale{scale}"] = float(entropy[head, scale])
            metrics[f"normalized_entropy_q{head}_scale{scale}"] = float(
                normalized_entropy[head, scale]
            )
            metrics[f"perplexity_q{head}_scale{scale}"] = float(perplexity[head, scale])
            metrics[f"top1_fraction_q{head}_scale{scale}"] = float(top1_fraction[head, scale])
            metrics[f"residual_energy_q{head}_scale{scale}"] = float(residual_total[head, scale])
            metrics[f"prefix_mse_q{head}_scale{scale}"] = float(prefix_total[head, scale])
            if quantizer.codebook_objective == "stage_aligned":
                metrics[f"stage_codebook_mse_q{head}_scale{scale}"] = float(
                    stage_codebook_total[head, scale]
                )
                metrics[f"stage_adapter_mse_q{head}_scale{scale}"] = float(
                    stage_adapter_total[head, scale]
                )
            metrics[f"prefix_relative_mse_q{head}_scale{scale}"] = float(
                prefix_total[head, scale]
                / (head_latent / head_elements.clamp_min(1)).clamp_min(1e-30)
            )
            metrics[f"contribution_rms_q{head}_scale{scale}"] = float(
                contribution_total[head, scale].clamp_min(0).sqrt()
            )
    if getattr(quantizer, "quantizer_type", None) == "multiscale_fsq":
        keys = [
            "loss",
            "reconstruction",
            "usage",
            "normalized_entropy",
            "perplexity",
            "top1_fraction",
            "grad_norm",
            "final_quant_mse",
            "final_quant_relative_mse",
            "final_quant_cosine",
            "latent_rms",
            "quantized_rms",
        ]
        keys += [f"frame{frame}_reconstruction" for frame in range(module.frames)]
        keys += list(variable_names)
        for head in range(quantizer.num_codebooks):
            keys.append(f"final_quant_relative_mse_q{head}")
            for scale in range(len(quantizer.scale_shapes)):
                if not bool(active_mask.all()):
                    keys.append(f"active_q{head}_scale{scale}")
                keys.extend(
                    (f"usage_q{head}_scale{scale}", f"normalized_entropy_q{head}_scale{scale}")
                )
        return {key: metrics[key] for key in keys}
    return metrics
