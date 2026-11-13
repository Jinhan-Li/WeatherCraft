import numpy as np


def event5(a):
    a = np.asarray(a, bool)
    out = np.zeros_like(a)
    for j in range(len(a) - 4):
        out[j : j + 5] |= a[j : j + 5].all(0)
    return out


def counts(p, t, w):
    return np.array(
        [np.sum((p & t) * w), np.sum((p & ~t) * w), np.sum((~p & t) * w)], dtype="float64"
    )


def csi(c):
    return float(c[0] / sum(c)) if sum(c) > 0 else 0.0


def score_one(pred, c):
    valid = c["valid"]
    w = c["weight"]
    q = c["thresholds"]
    truth = c["truth"]
    assert np.isfinite(pred[:, valid]).all()
    daily = np.array(
        [[counts(pred[i] > q[k, i], truth[i] > q[k, i], w) for i in range(10)] for k in range(2)]
    )
    pooled = daily[:, 2:10].sum(1)
    pe = event5(np.concatenate([c["past90"], (pred > q[0]) & valid]))[4:]
    strict = counts(pe[2:6], c["truth_event90"][2:6], w[None])
    limited = counts(pe[2:10], c["truth_event90"][2:10], w[None])
    diff = np.where(valid, pred - truth, 0).astype("float64")
    mse = (diff**2 * w).sum((1, 2)) / w.sum()
    bias = (diff * w).sum((1, 2)) / w.sum()
    return dict(
        reward=csi(pooled[0]),
        q95_csi=csi(pooled[1]),
        counts90=pooled[0].tolist(),
        counts95=pooled[1].tolist(),
        daily_counts=daily.tolist(),
        strict_event_csi=csi(strict),
        strict_event_counts=strict.tolist(),
        limited_event_csi=csi(limited),
        limited_event_counts=limited.tolist(),
        mse=mse.tolist(),
        bias=bias.tolist(),
        predicted_event_fraction=float(((pred[2:] > q[0, 2:]) * w).sum() / (8 * w.sum())),
        true_event_fraction=float(((truth[2:] > q[0, 2:]) * w).sum() / (8 * w.sum())),
    )


def score(pred, ctx):
    result = score_one(pred, ctx["contexts"]["1993_2018"])
    result["sensitivity"] = score_one(pred, ctx["contexts"]["1993_2012"])
    return result


def probability(pred, ctx, independent=True):
    c = ctx["contexts"]["1993_2018"]
    valid = c["valid"]
    w = c["weight"][valid]
    den = 8 * w.sum()
    x = pred[:, 2:, valid].astype("float64")
    y = c["truth"][2:, valid]
    q = c["thresholds"][0, 2:][:, valid]
    p = (x > q).mean(0)
    truth = y > q
    bs = (p - truth) ** 2
    mae = np.abs(x - y).mean(0)
    spread = np.zeros_like(y, dtype="float64")
    for i in range(8):
        for j in range(i + 1, 8):
            spread += np.abs(x[i] - x[j])
    return dict(
        brier=float((bs * w).sum() / den),
        fair_brier=float(((bs - p * (1 - p) / 7) * w).sum() / den) if independent else None,
        crps=float(((mae - spread / 64) * w).sum() / den),
        fair_crps=float(((mae - spread / 56) * w).sum() / den) if independent else None,
        independent_members=independent,
        members=8,
    )


def check_cached_ocean_path(reference, cached, valid, weight, reference_reward, cached_reward):
    """Physical aggregate equivalence, not bitwise FP32 reproducibility.

    0.1 mK per-lead RMS and 1 mK maximum are execution-error bounds, far
    below observed SST forecast errors (~0.38 K). A separate threshold-aware diagnostic bounds raw CSI changes. Max error is retained to reject isolated cache corruption.
    """
    assert reference.shape == cached.shape and reference.shape[1:] == valid.shape == weight.shape
    assert reference.shape[0] == 10 and valid.any()
    a, b = (reference[:, valid], cached[:, valid])
    assert np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite ocean cache/replay"
    w = np.asarray(weight[valid], dtype=np.float64)
    assert np.isfinite(w).all() and (w >= 0).all() and (w.sum() > 0)
    d = a.astype(np.float64) - b.astype(np.float64)
    per_lead_rmse = np.sqrt((d * d * w[None]).sum(1) / w.sum())
    per_lead_max = np.abs(d).max(1)
    reward_error = abs(float(reference_reward) - float(cached_reward))
    assert np.isfinite(reference_reward) and np.isfinite(cached_reward)
    checks = dict(
        max_absolute_sst_difference=float(per_lead_max.max()),
        area_weighted_sst_rmse_difference=float(np.sqrt(np.mean(per_lead_rmse**2))),
        per_lead_sst_rmse_difference=per_lead_rmse.tolist(),
        per_lead_max_absolute_sst_difference=per_lead_max.tolist(),
        absolute_reward_difference=reward_error,
        per_lead_rmse_limit_degrees_C=0.0001,
        max_absolute_limit_degrees_C=0.001,
        reward_limit=1e-05,
        bitwise_equivalence_claimed=False,
    )
    assert per_lead_rmse.max() <= 0.0001 and per_lead_max.max() <= 0.001, checks
    return checks


def check_threshold_sensitive_reward(reference, cached, ctx, replay_score, cached_score, group_std):
    """Verify raw CSI differences are small, physically bounded threshold flips.

    CSI is discontinuous. A fixed sub-micro CSI equality test is not a
    meaningful cache equivalence test when micro-K differences cross q90.
    This diagnostic never changes the training mask, threshold, or reward.
    """
    valid = ctx["valid"]
    w = ctx["weight"][valid].astype(np.float64)
    a = reference[2:10, valid]
    b = cached[2:10, valid]
    threshold = ctx["thresholds"][0, 2:10][:, valid]
    assert a.shape == b.shape == threshold.shape
    flips = (a > threshold) != (b > threshold)
    near = (np.abs(a - threshold) <= 0.0001) | (np.abs(b - threshold) <= 0.0001)
    assert not (flips & ~near).any(), "Classifications changed away from threshold boundary"
    area = float((flips * w[None]).sum())
    fraction = area / (8 * w.sum())
    u1 = sum(replay_score["counts90"])
    u2 = sum(cached_score["counts90"])
    bound = area / min(u1, u2) if min(u1, u2) > 0 else 0.0 if area == 0 else 1.0
    error = abs(replay_score["reward"] - cached_score["reward"])
    limit = min(1e-05, 0.01 * max(float(group_std), 0.0001))
    assert error <= bound + 1e-12 and error <= limit and (fraction <= 1e-05), (
        error,
        bound,
        limit,
        fraction,
    )
    idx = np.argwhere(flips)
    examples = [
        dict(
            lead=int(i) + 3,
            valid_flat_index=int(j),
            replay=float(a[i, j]),
            cached=float(b[i, j]),
            q90=float(threshold[i, j]),
        )
        for i, j in idx[:20]
    ]
    return dict(
        threshold_sensitive_check_passed=True,
        flipped_cell_days=int(flips.sum()),
        flipped_area_fraction=float(fraction),
        flip_area_fraction_limit=1e-05,
        no_flips_away_from_0_1mK_boundary=True,
        raw_reward_unchanged=True,
        possible_csi_error_bound_from_flip_area=bound,
        absolute_reward_difference=error,
        reward_limit=limit,
        group_reward_std=float(group_std),
        reward_error_fraction_of_group_std=error / max(float(group_std), 0.0001),
        threshold_flip_examples=examples,
    )
