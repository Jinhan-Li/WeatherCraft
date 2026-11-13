"""Original optimizer semantics plus scale/branch/factor diagnostics."""

from contextlib import nullcontext
import torch
from torch import distributed as dist
from torch.nn import functional as F


def run_dynamics_epoch(
    model,
    loader,
    device,
    precision,
    optimizer=None,
    scheduler=None,
    scaler=None,
    grad_accumulation=1,
    grad_clip=1.0,
    epoch=0,
):
    training = optimizer is not None
    model.train(training)
    module = model.module if hasattr(model, "module") else model
    assert grad_accumulation >= 1
    if scaler is None:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=training and device.type == "cuda" and precision == "fp16"
        )
    totals = None
    names = None
    count = 0
    gradient_sum = 0.0
    updates = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        current = tuple(v.to(device) for v in batch["current_tokens"])
        truth = tuple(v.to(device) for v in batch["target_tokens"])
        day = batch["target_seasonal"].to(device)
        update = (step + 1) % grad_accumulation == 0 or step + 1 == len(loader)
        tail = len(loader) % grad_accumulation
        window = tail if tail and step >= len(loader) - tail else grad_accumulation
        sync = (
            model.no_sync()
            if training and hasattr(model, "no_sync") and not update
            else nullcontext()
        )
        autocast = (
            torch.autocast(
                device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16
            )
            if precision != "fp32"
            else nullcontext()
        )
        with sync, torch.set_grad_enabled(training):
            with autocast:
                out = model(current, truth, day)
            if not torch.isfinite(out.loss):
                raise FloatingPointError("Nonfinite dynamics objective")
            if training:
                scaler.scale(out.loss / window).backward()
        if training and update:
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=True
            )
            previous = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not scaler.is_enabled() or scaler.get_scale() >= previous:
                if scheduler is not None:
                    scheduler.step()
                gradient_sum += float(grad)
                updates += 1
            if step == 0 or (step + 1) % 50 == 0:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(
                        "TRAIN_STEP_OK",
                        "epoch",
                        epoch,
                        "batch",
                        step + 1,
                        "of",
                        len(loader),
                        "objective",
                        float(out.objective_loss.detach()),
                        "standard_CE",
                        float(out.standard_loss.detach()),
                        "grad",
                        float(grad),
                        "nextLR",
                        optimizer.param_groups[0]["lr"],
                        flush=True,
                    )
        with torch.no_grad():
            values = {
                "loss": out.loss.detach().float(),
                "standard_joint_ce": out.standard_loss.detach().float(),
                "objective_loss": out.objective_loss.detach().float(),
            }
            correct = torch.zeros(module.num_codebooks, device=device)
            counts = torch.zeros_like(correct)
            fcorrect = torch.zeros(2, device=device)
            fcounts = torch.zeros_like(fcorrect)
            for s, (logits, target) in enumerate(zip(out.logits, truth)):
                target = target[:, : module.active_token_channels[s]]
                factors = module.codec.split_prediction_ids(target)
                predictions = tuple(v[:, : target.shape[1]].argmax(2) for v in logits)
                composite = module.codec.merge_prediction_ids(*predictions)
                correct[: target.shape[1]] += (composite == target).sum((0, 2, 3))
                counts[: target.shape[1]] += target.shape[0] * target.shape[2] * target.shape[3]
                scale_ce = 0.0
                for f, (v, y, pred) in enumerate(zip(logits, factors, predictions)):
                    v = v[:, : target.shape[1]]
                    ce = F.cross_entropy(v.float().flatten(0, 1), y.flatten(0, 1).long())
                    scale_ce = scale_ce + ce
                    fcorrect[f] += (pred == y).sum()
                    fcounts[f] += y.numel()
                    for q in range(target.shape[1]):
                        prefix = f"scale_{s}_branch_{q}_factor_{f}"
                        values[prefix + "_ce"] = F.cross_entropy(v[:, q].float(), y[:, q].long())
                        values[prefix + "_accuracy"] = (pred[:, q] == y[:, q]).float().mean()
                values[f"scale_{s}_ce"] = scale_ce
            values["accuracy"] = correct.sum() / counts.sum()
            values.update(
                {f"slot_{q}_accuracy": correct[q] / counts[q] for q in range(len(correct))}
            )
            values.update({f"factor_{f}_accuracy": fcorrect[f] / fcounts[f] for f in range(2)})
            if names is None:
                names = list(values)
                totals = torch.zeros(len(names), dtype=torch.float64, device=device)
            assert list(values) == names
            totals += torch.stack([values[k] for k in names]).double() * len(day)
            count += len(day)
        del out
    tail = torch.tensor([count, gradient_sum, updates], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(totals)
        dist.all_reduce(tail)
    result = dict(zip(names, (totals / tail[0]).cpu().tolist()))
    result["grad_norm"] = float(tail[1] / tail[2].clamp_min(1))
    return result
