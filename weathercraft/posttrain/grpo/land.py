"""Land rollouts and aggregate evaluation metrics."""

from .runtime import *
import time
from .land_metrics import score, probability_metrics
from .policy import sample, replay, advantages, branch_credit, node_count


@torch.no_grad()
def rollout(self, origin, seed, training=False, greedy=False):
    start = time.monotonic()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    self.model.eval()
    day = int(np.datetime64(origin, "D").astype(int))
    tree = training and self.method == "tree"
    initial = self.initial(origin)
    history = (
        initial if tree else tuple((x.expand(8, *[-1] * (x.ndim - 1)).clone() for x in initial))
    )
    gh = initial if greedy else None
    paths = [[] for _ in range(1 if tree else 8)]
    errors = [[] for _ in paths]
    gp = []
    ge = []
    trans = []
    parity = []
    for lead in range(1, 16):
        if tree and lead in (1, 4, 7):
            history = tuple((x.repeat_interleave(2, dim=0) for x in history))
            paths = [list(p) for p in paths for _ in range(2)]
            errors = [list(p) for p in errors for _ in range(2)]
        count = history[0].shape[0]
        assert count == (node_count(lead) if tree else 8)
        season = self.seasonal(day + lead)[None].to(self.device).expand(count, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if training:
                parts = [
                    sample(self.model, tuple((x[j : j + 2] for x in history)), season[j : j + 2])
                    for j in range(0, count, 2)
                ]
                pred = tuple((torch.cat([p[0][s] for p in parts]) for s in range(3)))
                slp = torch.cat([p[1] for p in parts])
                decoded = torch.cat(
                    [
                        self.t.decode(tuple((v[j : j + 2] for v in pred)))[:, 0].float()
                        for j in range(0, count, 2)
                    ]
                )
            else:
                pred, slp = sample(self.model, history, season)
                decoded = self.t.decode(pred)[:, 0].float()
            if greedy:
                g = self.model.generate(gh, season[:1], mode="greedy")
                gh = advance(gh, g)
                gd = self.t.decode(g)[:, 0].float()
        assert torch.isfinite(decoded).all()
        if training:
            old = []
            ref = []
            for j in range(0, count, 2):
                h = tuple((x[j : j + 2] for x in history))
                a = tuple((x[j : j + 2] for x in pred))
                s = season[j : j + 2]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    old.append(replay(self.model, h, a, s))
                    ref.append(replay(self.ref, h, a, s))
            old = torch.cat(old)
            ref = torch.cat(ref)
            err = (old - slp).abs().float()
            check = dict(
                lead=lead, mean=float(err.mean()), p99=float(torch.quantile(err.flatten(), 0.99))
            )
            assert check["mean"] <= 0.01 and check["p99"] <= 0.05, check
            parity.append(check)
            trans.append(
                dict(
                    history=history,
                    target=pred,
                    season=season,
                    old=old,
                    reference=ref,
                    count=count,
                    lead=lead,
                )
            )
        phys = (decoded[:, [7, 65]] * self.sd + self.mu).cpu().numpy()
        truth = torch.from_numpy(self.fields.normalized(np.datetime64(day + lead, "D"))).to(
            self.device
        )
        mse = (
            (((decoded - truth) ** 2 * self.w).sum((2, 3)) / (self.w.sum() * 240))
            .mean(-1)
            .cpu()
            .tolist()
        )
        for j in range(count):
            paths[j].append(phys[j])
            errors[j].append(mse[j])
        if greedy:
            gp.append((gd[:, [7, 65]] * self.sd + self.mu)[0].cpu().numpy())
            gmse = ((gd - truth) ** 2 * self.w).sum((2, 3)) / (self.w.sum() * 240)
            ge.append(float(gmse.mean()))
        history = advance(history, pred)
        self.progress("rollout", origin=origin, lead=lead, training=training, nodes=count)
    predictions = np.asarray(paths)
    ctx = self.events.context(origin)
    members = [score(p, ctx, "land_temperature") for p in predictions]
    if greedy:
        members.append(score(np.asarray(gp), ctx, "land_temperature"))
    row = dict(
        origin=origin,
        seed=seed,
        members=members,
        ensemble=score(predictions.mean(0), ctx, "land_temperature"),
        persistence=score(ctx["persistence"], ctx, "land_temperature"),
        probability=probability_metrics(predictions, ctx, independent=not tree),
        weather_mse=errors + ([ge] if greedy else []),
        parity=parity,
        seconds=time.monotonic() - start,
        reward_std=float(np.std([m["reward"] for m in members[:8]])),
        unique_member_days=sum((t["count"] for t in trans)) if training else 120,
    )
    rewards = torch.tensor([m["reward"] for m in members[:8]], device=self.device)
    if tree:
        q, adv, std = branch_credit(rewards)
        row["tree_credit"] = dict(
            q={str(k): v.cpu().tolist() for k, v in q.items()},
            advantages={str(k): v.cpu().tolist() for k, v in adv.items()},
            root_std=float(std),
        )
        for t in trans:
            t["adv"] = adv[t["count"]]
    elif training:
        adv = advantages(rewards)
        row["advantages"] = adv.cpu().tolist()
        for t in trans:
            t["adv"] = adv
    return (trans, row)


def aggregate(self, rows, label):
    return dict(
        label=label,
        updates=self.update,
        origins=[r["origin"] for r in rows],
        origin_results=rows,
        member_csi=float(np.mean([m["reward"] for r in rows for m in r["members"][:8]])),
        warm_csi=float(np.mean([m["csi_by_event"][0] for r in rows for m in r["members"][:8]])),
        cold_csi=float(np.mean([m["csi_by_event"][1] for r in rows for m in r["members"][:8]])),
        member_rmse=float(np.sqrt(np.mean([m["mse"] for r in rows for m in r["members"][:8]]))),
        member_bias=float(np.mean([m["bias"] for r in rows for m in r["members"][:8]])),
        predicted_area=np.mean(
            [m["predicted_area"] for r in rows for m in r["members"][:8]], axis=0
        ).tolist(),
        true_area=np.mean([r["members"][0]["true_area"] for r in rows], axis=0).tolist(),
        ensemble_rmse=float(np.sqrt(np.mean([r["ensemble"]["mse"] for r in rows]))),
        persistence_rmse=float(np.sqrt(np.mean([r["persistence"]["mse"] for r in rows]))),
        greedy_rmse=(
            float(np.sqrt(np.mean([r["members"][8]["mse"] for r in rows])))
            if len(rows[0]["members"]) == 9
            else None
        ),
        ensemble_csi=float(np.mean([r["ensemble"]["reward"] for r in rows])),
        greedy_csi=(
            float(np.mean([r["members"][8]["reward"] for r in rows]))
            if len(rows[0]["members"]) == 9
            else None
        ),
        persistence_csi=float(np.mean([r["persistence"]["reward"] for r in rows])),
        brier=float(np.mean([r["probability"]["brier"] for r in rows])),
        fair_brier=float(np.mean([r["probability"]["fair_brier"] for r in rows])),
        fair_crps=float(np.mean([r["probability"]["fair_crps"] for r in rows])),
        crps=float(np.mean([r["probability"]["crps"] for r in rows])),
        weather_normalized_rmse=float(
            np.sqrt(np.mean([np.asarray(r["weather_mse"])[:8] for r in rows]))
        ),
        utc=now(),
    )
