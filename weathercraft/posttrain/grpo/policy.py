"""Composite-token GRPO on sampled coarse prefixes, with fixed reference KL."""

import torch
from torch.nn import functional as F


def selected_logp(model, logits, targets):
    out = []
    for pair, target in zip(logits, targets):
        values = []
        for v, y in zip(pair, model.codec.split_prediction_ids(target)):
            lp = -F.cross_entropy(v.float().flatten(0, 1), y.long().flatten(0, 1), reduction="none")
            values.append(lp.reshape(target.shape))
        out.append(sum(values).flatten(1))
    return torch.cat(out, dim=1)


def replay(model, context, target, season):
    module = model.module if hasattr(model, "module") else model
    return selected_logp(module, model(context, target, season).logits, target)


def gradient_mode(model):
    model.eval()
    model.training = True
    assert not any((x.training for x in model.modules() if x is not model))


@torch.no_grad()
def sample(model, context, season):
    model.eval()
    generated = [torch.zeros_like(x[:, 1]) for x in context]
    lps = []
    for s in range(len(generated)):
        sequence, ends, queries = model.build_sequence(context, generated, season, s + 1)
        hidden = model._hidden(sequence, ends)
        logits = model.query_logits(hidden[:, queries[-1]], s)
        factors = [model._draw(v, False, 1.0, 0) for v in logits]
        generated[s] = model.codec.merge_prediction_ids(*factors)
        vals = []
        for v, y in zip(logits, factors):
            val = -F.cross_entropy(
                v.float().flatten(0, 1), y.long().flatten(0, 1), reduction="none"
            )
            vals.append(val.reshape(generated[s].shape))
        lps.append(sum(vals).flatten(1))
    return (tuple(generated), torch.cat(lps, dim=1))


def advantages(reward, floor=0.0001):
    r = reward.detach().float()
    std = r.std(unbiased=False)
    return (r - r.mean()) / std.clamp_min(floor) if std >= floor else torch.zeros_like(r)


def objective(lp, old, reference, adv, clip=0.2, beta=0.02):
    ratio = (lp - old.detach()).exp()
    delta = reference.detach() - lp
    kl = torch.expm1(delta) - delta
    assert torch.isfinite(ratio).all() and torch.isfinite(kl).all()
    a = adv.detach()[:, None]
    pg = -torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
    loss = pg + beta * kl.mean()
    metrics = torch.stack(
        [
            pg.detach(),
            kl.mean().detach(),
            ((ratio - 1).abs() > clip).float().mean().detach(),
            ratio.mean().detach(),
        ]
    )
    return (loss, metrics)


def node_count(lead):
    assert 1 <= lead <= 15
    return 2 if lead <= 3 else 4 if lead <= 6 else 8


def leaf_indices(count, node):
    assert count in (2, 4, 8) and 0 <= node < count
    width = 8 // count
    return list(range(node * width, (node + 1) * width))


def branch_credit(rewards, floor=0.0001):
    """Sibling LOO baseline, with one common root scale (not binary +/-1)."""
    r = rewards.detach().float()
    assert r.shape == (8,) and torch.isfinite(r).all()
    std = r.std(unbiased=False)
    values, advantages = ({}, {})
    for count in (2, 4, 8):
        q = r.reshape(count, 8 // count).mean(1)
        sibling = q.reshape(-1, 2).flip(1).flatten()
        a = (q - sibling) / std.clamp_min(floor)
        if std < floor:
            a = torch.zeros_like(q)
        values[count] = q
        advantages[count] = a
    return (values, advantages, std)


def scale_weights(sizes, alpha=0.5, device=None):
    n = torch.as_tensor(sizes, dtype=torch.float32, device=device)
    assert n.numel() == 3 and (n > 0).all() and (0 <= alpha <= 1)
    w = n.pow(1.0 - alpha)
    return w / w.sum()


def tree_objective(lp, old, reference, adv, sizes, alpha=0.5, clip=0.2, beta=0.02):
    """Composite-token ratios, scale means, then weighted sum; no joint ratio."""
    assert sum(sizes) == lp.shape[1] and lp.shape == old.shape == reference.shape
    ratio = (lp - old.detach()).exp()
    delta = reference.detach() - lp
    kl = torch.expm1(delta) - delta
    assert torch.isfinite(ratio).all() and torch.isfinite(kl).all()
    a = adv.detach()[:, None]
    pg = -torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    w = scale_weights(sizes, alpha, lp.device)

    def aggregate(x):
        means = torch.stack([v.mean(1) for v in x.split(sizes, dim=1)], dim=1)
        return (means * w).sum(1).mean()

    p, k = (aggregate(pg), aggregate(kl))
    metrics = torch.stack(
        [
            p.detach(),
            k.detach(),
            aggregate(((ratio - 1).abs() > clip).float()).detach(),
            aggregate(ratio).detach(),
        ]
    )
    return (p + beta * k, metrics)
